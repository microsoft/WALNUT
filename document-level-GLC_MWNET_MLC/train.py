import numpy as np
import pickle
import copy
import sys
import argparse
import logging
import gc
from logger import get_logger
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from mlc import step_mlc_debug, step_refine_net
from mlc_utils import clone_parameters, tocuda, DummyScheduler, save_checkpoint, pseudo_eval

from models import *       # import all models
from meta_models import *  # import all meta_models

parser = argparse.ArgumentParser(description='MLC training code')
parser.add_argument('--dataset', type=str, choices=['imdb', 'mnist', 'sst', 'twitter', 'ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'yahoo', 'imdb2'], default='mnist')

parser.add_argument('--method', default='mlc_debug', type=str, choices=['mlc_debug'])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_seed', type=int, default=1)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--every', default=1000, type=int, help='Eval interval (default: 1000 iters)')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--gold_bs', type=int, default=32)

# not used actually
parser.add_argument('--cls_dim', type=int, default=32, help='Label embedding dim (Default: 32)') 

parser.add_argument('--grad_clip', default=5.0, type=float, help='max grad norm (default: 5.0)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=3e-4, type=float, help='lr for main net')
parser.add_argument('--meta_lr', default=3e-5, type=float, help='lr for meta net')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--tau', default=1, type=float, help='tau')
parser.add_argument('--reg_str', default=0, type=float, help='reg str')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--refine_iterations', default=0, type=int)
parser.add_argument('--refine_reset', default=False, action='store_true', help='Reset main net in refine  (default: False)')
parser.add_argument('--refine_method', default='refine_mix', type=str, choices=['refine', 'refine_mix'])
parser.add_argument('--refine_lr', default=3e-4, type=float, help='lr for main net in refine process')
parser.add_argument('--corruption_type', default='unif', type=str, choices=['unif', 'flip'])
parser.add_argument('--corruption_level', default='-1', type=float, help='Corruption level')
parser.add_argument('--weaklabel', default=-1, type=int, help='Weak label index (Default: -1, no label from weak classifiers', choices=[-1, 1, 2, 3])
parser.add_argument('--gold_fraction', default='-1', type=float, help='Gold fraction')

parser.add_argument('--cosine', default=False, action='store_true', help='Use cosine scheduler for main net and meta net (default: False)')
parser.add_argument('--cosine_ratio', default=1., type=float, help='Cosine annealing period (fraction of total iterations) (default: 1.0)')

parser.add_argument('--meta_eval', default=False, action='store_true', help='Set meta net to eval mode in training (Default: False, turns off dro\
pout only)')

# not used 
parser.add_argument('--valid_ratio', default=0., type=float, help='Ratio of seprate validation set (default: 0)')

# bert args
parser.add_argument('--bert', default=False, action='store_true', help='Use bert')
parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help='Bert model name to use')
parser.add_argument('--bert_seqlen', default=128, type=int, help='sequence size for BERT (defaut: 128)')
parser.add_argument('--keep_left', default=False, action='store_true', help='Keep left part of text for bert')
parser.add_argument('--warmup_steps', default=1000, type=int, help='Linear warm up steps (Default: 1000)')

parser.add_argument('--runid', default='exp', type=str)
parser.add_argument('--acc', default=False, action='store_true', help='Use acc on dev set to save models')
parser.add_argument('--queue_size', default=5, type=int, help='Number of iterations before to compute mean loss_g')
#parser.add_argument('--debug', default=False, action='store_true', help='Debug mode for experimental ideas. (default: False)')

############## MAGIC STEPS ##################
parser.add_argument('--magic_max', default=1, type=float, help='Max magic coefficient for proxy_g (default: 1.0)')
parser.add_argument('--magic_min', default=1, type=float, help='Min magic coefficient for proxy_g (default: 1.0)')
parser.add_argument('--magic_ratio', default=1., type=float, help='Min magic coefficient for proxy_g (default: 1.0)')
############## MAGIC STPES ##################

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
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
args.keep_right = not args.keep_left

# //////////////////////////// set logging ///////////////////////
logfile = 'logs/' + '_'.join([args.dataset, args.method, args.corruption_type, args.runid, str(args.num_iterations), 'refine', str(args.refine_iterations)]) +'.log'
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


hard_loss_f = F.cross_entropy
from mlc_utils import soft_cross_entropy as soft_loss_f

# //////////////////////// defining model ////////////////////////

def get_data(dataset, gold_fraction, corruption_prob, get_C):
    if dataset == 'mnist':
        sys.path.append('MNIST')

        from data_helper_mnist import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, args)#, merge_valset=args.mergeval)
    elif dataset == 'imdb':
        sys.path.append('IMDB')
        
        from data_helper_imdb import prepare_data
        return prepare_data(gold_fraction, corruption_prob, get_C, args)
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
    cls_dim = args.cls_dim # Default 32
    if dataset == 'mnist':
        # main net
        model = ThreeLayerNet(784, 128, num_classes).cuda()

        main_net = model
        f_params, f_aux = None, None

        # meta net
        meta_net = NoiseNet(784, cls_dim, 128, 10, args)#.cuda()
    elif dataset == 'imdb':
        vocab_size, emb_dim, h_dim = 5000, 128, 128
        # main net
        model = LSTMNet(vocab_size, emb_dim, h_dim, num_classes)#.cuda()

        main_net = model
        f_params, f_aux = None, None

        # meta net
        meta_net = LSTMMeta(vocab_size, emb_dim, cls_dim, h_dim, num_classes, args)#.cuda()
    elif dataset == 'sst':
        vocab_size, emb_dim, h_dim = 10000, 100, 32
        # main net
        model = WordAveragingLinear(vocab_size, emb_dim, num_classes)#.cuda()

        main_net = model
        f_params, f_aux = None, None

        # meta net
        meta_net = WordAvgMeta(vocab_size, emb_dim, cls_dim, num_classes, args)#.cuda()
    elif dataset == 'twitter':
        emb_dim, h_dim = 50, 256
        # main net
        model = ThreeLayerNet(3*emb_dim, h_dim, num_classes)#.cuda()
        
        main_net = model
        f_params, f_aux = None, None

        # meta net
        meta_net = NoiseNet(3*emb_dim, cls_dim, h_dim, num_classes, args)#.cuda()

    elif dataset in ['ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'sogou', 'yahoo', 'imdb2']:
        if args.bert:
            model = BertMain(args.bert_name, num_classes)
            main_net = model
            meta_net = BertMeta(args.bert_name, cls_dim, num_classes, args)
            f_params, f_aux = None, None
        else:
            vocab_size, emb_dim, h_dim = 10000, 300, 512
            # main net
            model = WordAveragingLinear(vocab_size, emb_dim, num_classes)#.cuda()

            main_net = model
            f_params, f_aux = None, None

            # meta net
            meta_net = LCN(vocab_size, emb_dim, cls_dim, h_dim, num_classes, args)#.cuda()

    if isinstance(main_net, torch.nn.Module):
        main_net = main_net.cuda()
    meta_net = meta_net.cuda()

    logger.info('========== Main model ==========')
    logger.info(model)
    logger.info('========== Meta model ==========')
    logger.info(meta_net)

    return main_net, f_params, f_aux, meta_net

def setup_training(main_net, f_params, meta_net):
    # ////////////////// set up optimizers, schedulers, AMP and DDP //////
    # set up optimizers and schedulers
    # meta net optimizer
    if False: #args.bert: # adamw seems like MLC doesn't like AdamW
        from pytorch_transformers import AdamW, WarmupConstantSchedule
        optimizer = AdamW(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.wdecay,
                          eps=1e-6 if args.amp > -1 else 1e-8)
        scheduler = WarmupConstantSchedule(optimizer, args.warmup_steps)
    else:
        optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr,
                                     weight_decay=args.wdecay, # meta should have wdecay or not??
                                     amsgrad=True,
                                     eps=1e-6 if args.amp > -1 else 1e-8) # meta params: meta network to transform (x, y') -> y

    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_ratio * args.num_iterations, eta_min=1e-6)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1e4, gamma=0.5)
    else:
        scheduler = DummyScheduler(optimizer)

    # main net optimizer
    # Adam seems to work much better than SGD
    main_params = main_net.parameters()

    if False: #args.bert: # adamw
        main_opt = AdamW(main_params, lr=args.main_lr, weight_decay=args.wdecay,
                         eps=1e-6 if args.amp > -1 else 1e-8)
        main_schdlr = WarmupConstantSchedule(main_opt, args.warmup_steps)
    else:
        if args.optimizer == 'adam':
            main_opt = torch.optim.Adam(main_params, lr=args.main_lr, weight_decay=args.wdecay, amsgrad=True, 
                                        eps=1e-6 if args.amp > -1 else 1e-8)
        elif args.optimizer == 'sgd':
            main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)
        elif args.optimizer == 'adadelta':
            main_opt = torch.optim.Adadelta(main_params, weight_decay=args.wdecay, 
                                            eps=1e-6 if args.amp > -1 else 1e-8)
            
    if args.cosine:
        main_schdlr = torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, args.cosine_ratio * args.num_iterations, eta_min=1e-6)
    else:
        main_schdlr = DummyScheduler(main_opt)
    
    # set up AMP
    if args.amp > -1: # AMP on # loss scale dynamic
        [main_net, meta_net], [main_opt, optimizer] = apex.amp.initialize([main_net, meta_net], [main_opt, optimizer], opt_level='O%1d' % args.amp, 
                                                                          loss_scale=1.0,
                                                                          keep_batchnorm_fp32=True)

    # set up DDP
    # set up multi-GPU training 
    if args.local_rank != -1:#  multi-GPU job
        if args.apexddp: # Apex distributed training
            from apex.parallel import DistributedDataParallel as DDP
            if isinstance(main_net, torch.nn.Module):
                main_net = DDP(main_net, delay_allreduce=True)

            meta_net = DDP(meta_net, delay_allreduce=True)
        else: # PyTorch distributed training
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(main_net, torch.nn.Module):
                main_net = DDP(main_net,
                               device_ids=[args.local_rank],
                               output_device=args.local_rank,
                               broadcast_buffers=True,
                               find_unused_parameters=True)

            meta_net = DDP(meta_net,
                           device_ids=[args.local_rank],
                           output_device=args.local_rank,
                           broadcast_buffers=True,
                           find_unused_parameters=True)

    return main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler
    
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
    np.random.seed(args.data_seed)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


# //////////////////////// run experiments ////////////////////////
def run():
    corruption_fnctn = uniform_mix_C if args.corruption_type == 'unif' else flip_labels_C
    filename = '_'.join([args.dataset, args.method, args.corruption_type, args.runid, str(args.num_iterations), 'refine', str(args.refine_iterations), str(args.seed), str(args.data_seed)])
    if args.weaklabel > 0:
        filename += '_weak%d' % args.weaklabel

    results = {}

    gf_dict = {'yelp2': 20.0 / 560000,
               'yelp5': 2500.0 / 650000,
               'amazon2': 20.0 / 3600000,
               'amazon5': 2500.0 / 3000000,
               'dbpedia': 140.0 / 560000,
               'yahoo': 700.0 / 1400000,
               'imdb2': 20.0 / 25000,
               'ag': 0.0005,
               }

    #for gold_fraction in ([gf_dict[args.dataset]] + [0.0001, 0.001] if args.bert else [0.001, 0.01, 0.05]):
    if args.bert:
        gold_fractions = [gf_dict[args.dataset]]
    else:
        gold_fractions = [0.05, 0.001, 0.01]

    if args.gold_fraction != -1:
        assert args.gold_fraction >=0 and args.gold_fraction <=1, 'Wrong gold fraction!'
        gold_fractions = [args.gold_fraction]

    corruption_levels = ([0.2, 0.4, 0.6, 0.8, 0, 1] if args.bert else [1, 0, 0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    if args.corruption_level != -1: # specied one corruption_level
        assert args.corruption_level >= 0 and args.corruption_level <=1, 'Wrong noise level!'
        corruption_levels = [args.corruption_level]

    if args.weaklabel != -1:
        corruption_levels = [0]
    
    for gold_fraction in gold_fractions:
    #for gold_fraction in ([gf_dict[args.dataset]] if args.bert else [0.001]):        
        results[gold_fraction] = {}
        #for corruption_level in ([0.2, 0.4, 0.6, 0.8, 0, 1] if args.bert else [1, 0.9, 0.8, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
        for corruption_level in corruption_levels:
            # //////////////////////// load data //////////////////////////////
            gold_loader, silver_loader, valid_loader, test_loader, num_classes = get_data(args.dataset, gold_fraction, corruption_level, corruption_fnctn)
            
            # //////////////////////// build main_net and meta_net/////////////
            main_net, f_params, f_aux, meta_net = build_models(args.dataset, num_classes)
            
            # //////////////////////// train and eval model ///////////////////
            exp_id = '_'.join([filename, str(gold_fraction), str(corruption_level)])
            test_acc, baseline_acc = train_and_test(main_net, f_params, f_aux, meta_net, gold_loader, silver_loader, valid_loader, test_loader, exp_id)
        
            results[gold_fraction][corruption_level] = {}
            results[gold_fraction][corruption_level]['method'] = test_acc
            results[gold_fraction][corruption_level]['baseline'] = baseline_acc
            logger.info(' '.join(['Gold fraction:', str(gold_fraction), '| Corruption level:', str(corruption_level),
                  '| Method acc:', str(results[gold_fraction][corruption_level]['method']),
                                  '| Baseline acc:', str(results[gold_fraction][corruption_level]['baseline'])]))
            logger.info('')

            gc.collect()

    with open('out/' + filename, 'wb') as file:
        pickle.dump(results, file)
    logger.info("Dumped results_ours in file: " + filename)


####################################################################################################
###  training code 
####################################################################################################
def train_and_test(main_net, f_params, f_aux, meta_net, gold_loader, silver_loader, valid_loader, test_loader, exp_id=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.local_rank <=0: # single GPU or GPU 0
        writer = SummaryWriter(args.logdir + '/' + exp_id)

    main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler = setup_training(main_net, f_params, meta_net)

    # //////////////////////// set up training ////////////////////////
    meta_net.train()
    if args.meta_eval:
        pseudo_eval(meta_net)

    main_net.train()
    
    # get on pretrain model
    '''
    data_g, target_g = gold_loader.next()
    data_g, target_g = data_g.cuda(), target_g.cuda()
    
    logit_g = main_net(data_g, f_params, f_aux)
    loss_g = F.cross_entropy(logit_g, target_g)
    '''

    # back up main net initial state if reset in refinement
    if args.refine_iterations > 0 and args.refine_reset:
        init_params = copy.deepcopy(main_net.state_dict())
        init_opt_sd = copy.deepcopy(main_opt.state_dict())
        init_schdlr_sd = copy.deepcopy(main_schdlr.state_dict())
    
    best_params = None
    best_main_opt_sd = None
    best_main_schdlr_sd = None

    best_meta_params = None
    best_meta_opt_sd = None
    best_meta_schdlr_sd = None
    best_loss_g = float('inf')
    best_val_metric = float('inf')
    val_metric_queue = deque() 
    # set done

    args.dw_prev = [0 for param in meta_net.parameters() if param.requires_grad] # 0 for previous iteration
    args.steps = 0

    for i in tqdm(range(args.num_iterations + args.refine_iterations)):
        *data_g, target_g = next(gold_loader)#.next()
        *data_s, target_s_ = next(silver_loader)#.next()
        data_g, target_g = tocuda(data_g), tocuda(target_g)
        data_s, target_s_ = tocuda(data_s), tocuda(target_s_)

        if i < args.num_iterations: # bi-level optimization stage
            # magic scheduling

            # cosine scheduling
            args.magic = args.magic_min + 0.5 * (args.magic_max - args.magic_min) * ( 1 + math.cos((1.0 * i / (args.magic_ratio * args.num_iterations))*math.pi))

            # linear scheduling
            #args.magic = max(args.magic_max - (args.magic_max - args.magic_min) * ( 1.0 * i / (args.magic_ratio * args.num_iterations) ), args.magic_min)

            # mlc_debug
            loss_g, loss_s = step_mlc_debug(main_net, main_opt, hard_loss_f,
                                            meta_net, optimizer, soft_loss_f,
                                            data_s, target_s_, data_g, target_g,
                                            main_schdlr.get_lr()[0], args)

            # step_mlc does not call scheduler
            scheduler.step()
            main_schdlr.step()

        args.steps += 1

        #  tracking  and recording 
        if len(val_metric_queue) == args.queue_size:  # keep at most this number of records
            # remove the oldest record
            val_metric_queue.popleft()

        if args.acc:
            # forward
            main_net.eval()
            output = main_net(data_g)
            main_net.train()

            pred = output.data.max(1)[1]
            val_acc = pred.eq(target_g.data).sum().item() / len(target_g)

            # data_g acc
            val_metric_queue.append(-val_acc)
        else:
            # data_g loss
            val_metric_queue.append(loss_g.item())
            val_acc = 0
        
        avg_val_metric = sum(list(val_metric_queue)) / len(val_metric_queue)
        if avg_val_metric < best_val_metric or (avg_val_metric == best_val_metric and loss_g.item() < best_loss_g): 
            best_val_metric = avg_val_metric
            best_loss_g = loss_g.item()
            
            # record current state dict
            best_params = copy.deepcopy(main_net.state_dict())

            best_main_opt_sd = copy.deepcopy(main_opt.state_dict())
            best_main_schdlr_sd = copy.deepcopy(main_schdlr.state_dict())

            best_meta_params = copy.deepcopy(meta_net.state_dict())
            best_meta_opt_sd = copy.deepcopy(optimizer.state_dict())
            best_meta_schdlr_sd = copy.deepcopy(scheduler.state_dict())

        if i % args.every == 0:
            correct = torch.zeros(1).cuda()
            nsamples = torch.zeros(1).cuda()

            main_net.eval()

            for idx, (*data, target) in enumerate(test_loader):
                data, target = tocuda(data), tocuda(target)
        
                # forward
                output = main_net(data)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()
                nsamples += len(target)

            main_net.train()

            if args.local_rank != -1: # DDP
                gather_c = [torch.zeros(1).cuda()] * dist.get_world_size()
                gather_n = [torch.zeros(1).cuda()] * dist.get_world_size()

                dist.all_gather(gather_c, correct)
                dist.all_gather(gather_n, nsamples)

                test_acc = (sum(gather_c) / sum(gather_n)).item()
            else:
                test_acc = correct / nsamples
            
            if args.local_rank <=0: # single GPU or GPU 0
                writer.add_scalar('train/loss_g', loss_g.item(), i)
                writer.add_scalar('train/loss_s', loss_s.item(), i)

                ''' get entropy of predictions from meta-net '''
                pseudo_target_s = meta_net(data_s, target_s_).detach()
                entropy = -(pseudo_target_s * torch.log(pseudo_target_s+1e-10)).sum(-1).mean()

                writer.add_scalar('train/meta_entropy', entropy.item(), i)
                writer.add_scalar('test/acc', test_acc, i)

                main_lr = main_schdlr.get_lr()[0]
                meta_lr = scheduler.get_lr()[0]
                writer.add_scalar('train/main_lr', main_lr, i)
                writer.add_scalar('train/meta_lr', meta_lr, i)
                writer.add_scalar('train/magic', args.magic, i)
                alpha = meta_net.module.get_alpha().item() if isinstance(meta_net, torch.nn.parallel.DistributedDataParallel) or isinstance(meta_net, apex.parallel.DistributedDataParallel) else meta_net.get_alpha().item()

                tqdm.write('Iteration %d val loss: %.8f\ttrain loss: %.8f\tval acc: %.4f\ttest acc: %.4f\talpha: %.3f\tMeta entropy: %.3f\tLoss_g Mean: %.3f\tMain LR: %.6f\tMeta LR: %.6f\tMagic: %.1f' %( i, loss_g.item(), loss_s.item(), val_acc, test_acc, alpha, entropy.item(), avg_val_metric, main_lr, meta_lr, args.magic))
                #logger.info('Iteration %d val loss: %.8f\ttrain loss: %.8f\ttest acc: %.4f\talpha: %.3f' %( i, loss_g.item(), loss_s.item(), test_acc, alpha))
            #tqdm.write('** Meta net grad norm max: %.8f **' % net_grad_norm_max(noise_net, 2))

        
    ####################### save best models so far ###################
    if args.local_rank <=0: # only need to save once
        logger.info('Saving best models...')
        save_checkpoint({
            'main_net': best_params,
            'main_opt': best_main_opt_sd,
            'main_schdlr': best_main_schdlr_sd,
            'meta_net': best_meta_params,
            'meta_opt': best_meta_opt_sd,
            'meta_schdlr': best_meta_schdlr_sd
        }, 'models/%s.pth' % exp_id)

    # //////////////////////// evaluate method ////////////////////////
    correct = torch.zeros(1).cuda()
    nsamples = torch.zeros(1).cuda()

    # forward
    main_net.load_state_dict(best_params)
    main_net.eval()

    for idx, (*data, target) in enumerate(test_loader):
        data, target = tocuda(data), tocuda(target)

        # forward
        output = main_net(data)
        
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

    if args.local_rank <=0: # single GPU or GPU 0
        writer.add_scalar('test/acc', test_acc, i) # this test_acc should be roughly the best as it's taken from the best iteration

    return test_acc, 0#baseline_acc

if __name__ == '__main__':
    run()
