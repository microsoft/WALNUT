import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pickle
import sys
import argparse
import gc
import copy
from logger import get_logger
from tqdm import tqdm

import apex

from models import * 
from mlc import sync_backward
from mlc_utils import tocuda, DummyScheduler, save_checkpoint

parser = argparse.ArgumentParser(description='Pretrain Weak Classifiers')
parser.add_argument('--dataset', default='mnist', type=str, choices=['imdb', 'mnist', 'scr', 'sst', 'twitter', 'ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'sogou', 'yahoo', 'imdb2'])
parser.add_argument('--data_path', type=str, help='Root for the datasets.')
parser.add_argument('--method', default='weak', type=str, choices=['weak'])
parser.add_argument('--corruption_type', default='unif', type=str, choices=['unif', 'flip'])
parser.add_argument('--corruption_level', default='-1', type=float, help='Corruption level')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_seed', type=int, default=1)
parser.add_argument('--num_epochs', default=25, type=int)
parser.add_argument('--every', default=5, type=int, help='Eval interval (default: 10 iters)')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--init_lr', default=3e-5, type=float, help='init lr')
parser.add_argument('--reg_str', default=0, type=float, help='reg str')
parser.add_argument('--wdecay', default=3e-4, type=float, help='weight decay')

# bert args
parser.add_argument('--bert', default=False, action='store_true', help='Use bert')
parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help='Bert model name to use')
parser.add_argument('--bert_seqlen', default=128, type=int, help='sequence size for BERT (defaut: 128)')
parser.add_argument('--keep_left', default=False, action='store_true', help='Keep left part of text for bert')
parser.add_argument('--warmup_steps', default=1000, type=int, help='Linear warm up steps (Default: 1000)')

parser.add_argument('--runid', default='exp', type=str)
parser.add_argument('--acc', default=False, action='store_true', help='Use acc on dev set to save models')
parser.add_argument('--queue_size', default=5, type=int, help='Number of iterations before to compute mean loss_g')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')

# distributed training
parser.add_argument('--local_rank', type=int, default=-1, help='local rank (-1 for local training)')
parser.add_argument('--workers', type=int, default=0, help='workers for data loader (default: 0)')
parser.add_argument('--amp', type=int, default=-1, choices=[-1, 0, 1, 2, 3], help='Apex AMP level (Default: -1, no amp)')
parser.add_argument('--apexddp', default=False, action='store_true', help='Use Apex DDP (Default: Pytorch DDP')

args = parser.parse_args()
args.trainweak = True # Used for data helpers
args.weaklabel = -1 # hard coded to avoid confusion 
args.keep_right = not args.keep_left

# //////////////////////////// set logging ///////////////////////
logfile = 'logs/' + '_'.join([args.method, args.dataset, args.corruption_type, args.runid, str(args.num_epochs)]) +'.log'
logger = get_logger(logfile, args.local_rank)
# //////////////////////////////////////////////////////////////////

logger.info(args)
logger.info('CUDA available:' + str(torch.cuda.is_available()))

if args.local_rank !=-1:
    torch.cuda.set_device(args.local_rank)
    logger.info('Start multi-GPU training...')
    torch.distributed.init_process_group(
        init_method='env://',
        backend='nccl',
    )
else:
    torch.cuda.set_device(0) # local GPU

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.cuda.synchronize()

# //////////////////////// defining model ////////////////////////
reg_str = args.reg_str
batch_size = args.bs
init_lr = args.init_lr #0.0003

def get_data(dataset, gold_fraction, corruption_prob, get_C):
    if dataset == 'mnist':
        sys.path.append('MNIST')

        from data_helper_mnist import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, args)#, merge_valset=args.mergeval)
    elif dataset == 'imdb':
        sys.path.append('IMDB')
        
        from data_helper_imdb import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, args)
    elif dataset == 'scr':
        sys.path.append('SCR')

        from data_helper_scr import prepare_data
        return prepare_data(args.bs)
    elif dataset == 'sst':
        sys.path.append('SST')

        from data_helper_sst import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, args)#, merge_valset=args.mergeval)
    elif dataset == 'twitter':
        sys.path.append('Twitter')

        from data_helper_twitter import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, args)
    elif dataset in ['ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'sogou', 'yahoo', 'imdb2']:
        sys.path.append('TEXT')
        # max_len and vocab_size fixed to 200 and 10000, refer to data_helper for details
        dir_dict = {'ag': 'ag_news_csv',
                    'amazon2': 'amazon_review_polarity_csv',
                    'amazon5': 'amazon_review_full_csv',
                    'dbpedia': 'dbpedia_csv',
                    'yelp2': 'yelp_review_polarity_csv',
                    'yelp5': 'yelp_review_full_csv',
                    'sogou': 'sogou_news_csv',
                    'yahoo': 'yahoo_answers_csv',
                    'imdb2': 'imdb2_csv'
                 }
        train_csv = 'data/text/%s/train.csv' % dir_dict[dataset]
        test_csv = 'data/text/%s/test.csv' % dir_dict[dataset]
        
        from data_helper_csv import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, train_csv, test_csv, args)

def build_models(dataset, num_classes):
    if dataset == 'mnist':
        model = ThreeLayerNet(784, 128, num_classes)#.cuda()
    elif dataset == 'imdb':
        vocab_size, emb_dim, h_dim = 5000, 128, 128
        # main net
        model = LSTMNet(vocab_size, emb_dim, h_dim, num_classes)#.cuda()
    elif dataset == 'scr':
        pass
    elif dataset == 'sst':
        vocab_size, emb_dim, h_dim = 10000, 100, 32
        # main net
        model = WordAveragingLinear(vocab_size, emb_dim, num_classes)#.cuda()
    elif dataset == 'twitter':
        emb_dim, h_dim = 50, 256
        # main net
        model = ThreeLayerNet(3*emb_dim, h_dim, num_classes)#.cuda()
    elif dataset in ['ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'sogou', 'yahoo', 'imdb2']:
        if args.bert:
            model = BertMain(args.bert_name, num_classes)
        else:
            vocab_size, emb_dim, h_dim = 10000, 300, 512
            # main net
            model = WordAveragingLinear(vocab_size, emb_dim, num_classes)#.cuda()

    logger.info('========== Model ==========')
    logger.info(model)

    return model

def setup_training(net):
    # ////////////////// set up optimizers, schedulers, AMP and DDP //////
    # set up optimizers and schedulers
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=args.wdecay, amsgrad=True, 
                                    eps=1e-6 if args.amp > -1 else 1e-8)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=args.wdecay, momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(net.parameters(), weight_decay=args.wdecay, 
                                            eps=1e-6 if args.amp > -1 else 1e-8)

    scheduler = DummyScheduler(optimizer)

    # set up AMP
    if args.amp > -1: # AMP on
        net, optimizer = apex.amp.initialize(net, optimizer, opt_level='O%1d' % args.amp, keep_batchnorm_fp32=True)

    # set up DDP
    # set up multi-GPU training 
    if args.local_rank != -1:#  multi-GPU job
        if args.apexddp: # Apex distributed training
            from apex.parallel import DistributedDataParallel as DDP
            net = DDP(net, delay_allreduce=True)
        else: # PyTorch distributed training
            from torch.nn.parallel import DistributedDataParallel as DDP
            net = DDP(net,
                      device_ids=[args.local_rank],
                      output_device=args.local_rank,
                      broadcast_buffers=True,
                      find_unused_parameters=True)

    return net, optimizer, scheduler

def uniform_mix_C(num_classes, mixing_ratio):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(num_classes, corruption_prob):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(1)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

# //////////////////////// defining model ////////////////////////
reg_str = args.reg_str
num_epochs = args.num_epochs
batch_size = args.bs

def train_and_test(model, gold_loader, silver_loader, test_loader, all_loader, method='glc', exp_id=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # save init point and use later to retrain the classifier
    model.cuda()

    net, optimizer, scheduler = setup_training(model)
    net.train()

    acc_interval = (args.num_classes - 1.0) / (4 * args.num_classes)
    random_acc = 1.0 / args.num_classes
    acc_list = [(random_acc + i * acc_interval) for i in range(0, 4)] # three acc levels from random acc to 1
    
    acc_idx = 1
    
    for epoch in range(args.num_epochs):
        num_iterations = len(gold_loader.loader)
        for i in tqdm(range(num_iterations)):
            if acc_idx > 3: # has obtained all 3 weak classifiers already
                return
            
            *data_g, target_g = next(gold_loader)
            data_g, target_g = tocuda(data_g), tocuda(target_g)

            logit_g = net(data_g)
            loss = F.cross_entropy(logit_g, target_g)

            optimizer.zero_grad()
            sync_backward(loss, optimizer, args)
            optimizer.step()

            # eval on test
            if i % args.every == 0:
                net.eval()
                correct = torch.zeros(1).cuda()
                nsamples = torch.zeros(1).cuda()
    
                for idx, (*data, target) in enumerate(test_loader):
                    data, target = tocuda(data), tocuda(target)
                    output = net(data)
                    # accuracy
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.data).sum().item()
                    nsamples += len(target)

                if args.local_rank != -1: # DDP
                    gather_c = [torch.zeros(1).cuda()] * dist.get_world_size()
                    gather_n = [torch.zeros(1).cuda()] * dist.get_world_size()

                    dist.all_gather(gather_c, correct)
                    dist.all_gather(gather_n, nsamples)

                    test_acc = (sum(gather_c) / sum(gather_n)).item()
                else:
                    test_acc = (correct / nsamples).item()

                if args.local_rank <= 0:
                    itr = epoch * num_iterations + i
                    if test_acc >  acc_list[acc_idx-1] and test_acc <= acc_list[acc_idx] and acc_list[acc_idx] - test_acc < test_acc - acc_list[acc_idx-1]: # lie more towards acc_list[acc_idx], save this classifier
                        logger.info('Saving model and pred with test acc: %.3f' % test_acc)
                        torch.save(net, 'pretrained_weak/%s_%d.pth' %(args.dataset, acc_idx))
                        '''
                        save_checkpoint({
                            'net': net.state_dict(),
                            'acc': test_acc
                        }, 'pretrained_weak/%s_%d.pth' %(args.dataset, acc_idx))
                        '''

                        # use the classifier to classify all the train points
                        pred_list = []
                        net.eval()
                        with torch.no_grad():
                            for idx, (*data, ) in enumerate(all_loader):
                                data = tocuda(data)
                                output = net(data)
                                # accuracy
                                pred = output.data.max(1)[1]
                                pred_list.append(pred)

                        net.train()
                        pred_all = torch.cat(pred_list, dim=0).squeeze().cpu().numpy()
                        # dump weak prediction to file
                        with open('pretrained_weak/%s_%s.weak' % (args.dataset, acc_idx), 'wb') as f:
                            pickle.dump(pred_all, f)

                        # move to next acc target
                        acc_idx += 1
                        
            scheduler.step()

# //////////////////////// run experiments ////////////////////////
def run():
    corruption_fnctn = uniform_mix_C if args.corruption_type == 'unif' else flip_labels_C
    filename = '_'.join(['pretrained_weak', args.dataset, args.corruption_type, args.runid, str(args.num_epochs), str(args.seed), str(args.data_seed)])
    results = {}

    gold_fractions = [0.99] # use pretty much all clean data to pretrain classifiers
    corruption_levels = [0]

    for gold_fraction in gold_fractions:
        for corruption_level in corruption_levels:
            # ///////////////////////// load data /////////////////////////////
            gold_loader, silver_loader, valid_loader, test_loader, all_loader, num_classes = get_data(args.dataset, gold_fraction, corruption_level, corruption_fnctn)

            args.num_classes = num_classes

            # //////////////////////// build main_net and meta_net/////////////
            exp_id = '_'.join([args.method, filename, str(gold_fraction), str(corruption_level)])
            net = build_models(args.dataset, num_classes)
            train_and_test(net, gold_loader, silver_loader, test_loader, all_loader, method=args.method, exp_id=exp_id)

if __name__ == '__main__':
    run()
        
