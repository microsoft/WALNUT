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
from collections import deque

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
#import apex

from models import * # import all models
from mlc_utils import tocuda, DummyScheduler

from datetime import datetime
from datasets import load_dataset
from datasets import load_metric
from dataset import NoiseDataset, FakeNewsDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AdamW, set_seed
import os

parser = argparse.ArgumentParser(description='GLC Loss Correction Experiments')
#parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'cifarmix', 'clothing1m', 'imdb', 'mnist', 'scr', 'sst', 'twitter', 'ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'sogou', 'yahoo', 'imdb2'])
parser.add_argument('--method', default='glc', type=str, choices=['glc', 'forward', 'forward_gold', 'ideal', 'confusion'])
parser.add_argument('--dataset', type=str, choices=['imdb', "agnews", "yelp", "political", "gossipcop"], default='imdb')
parser.add_argument("--file_path", type=str, default='data', help="base directory of the data")
parser.add_argument("--weak_ratio", default=0.8, type=float, help="splittion of weak data and clean data")
parser.add_argument("--clean_ratio", default=20, type=float, help="number of clean samples for imdb/agnews/yelp dataset")
parser.add_argument("--n_high_cov", default=1, type=float, help="number of valid weak labeling functions for weak data")
parser.add_argument("--lstm_embed_dim", default=50, type=int, choices=[50, 100, 200, 300])
parser.add_argument("--is_overwrite_file",action="store_true")
parser.add_argument('--model_name', default='distilbert-base-uncased', type=str, choices=['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'bilstm', 'cnn', 'bert-large-uncased', 'roberta-large'])
parser.add_argument('--weaklabel', default=-1, type=int, help='Weak label index (Default: -1, no label from weak classifiers', choices=[-1, 1, 2, 3])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--every', default=1000, type=int, help='Eval interval (default: 1000 iters)')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--cls_dim', type=int, default=64, help='Label embedding dim (Default: 64)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for optimizers')
parser.add_argument('--main_lr', default=1e-5, type=float, help='init lr')
parser.add_argument('--reg_str', default=0, type=float, help='reg str')
parser.add_argument('--wdecay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--is_debug', action="store_true",help='Debug Mode')
parser.add_argument('--queue_size', default=1, type=int, help='Number of iterations before to compute mean loss_g')

parser.add_argument('--runid', default='exp', type=str)

parser.add_argument('--cosine', default="False", type=str, help='Use cosine scheduler for main net and meta net (default: False)')
parser.add_argument('--cosine_ratio', default=1., type=float, help='Cosine annealing period (fraction of total iterations) (default: 1.0)')
parser.add_argument('--warmup_steps', default=1000, type=int, help='Linear warm up steps (Default: 1000)')
parser.add_argument('--resume', default="False", type=str, help='Resume from last checkpoint')

# i/o
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')

# distributed training
parser.add_argument('--local_rank', type=int, default=-1, help='local rank (-1 for local training)')
parser.add_argument('--workers', type=int, default=0, help='workers for data loader (default: 0)')
parser.add_argument('--amp', type=int, default=-1, choices=[-1, 0, 1, 2, 3], help='Apex AMP level (Default: -1, no amp)')

args = parser.parse_args()

if args.model_name in ['bilstm', 'cnn']:
    args.is_roberta = False
else:
    args.is_roberta = True
# set seed for reproduction
set_seed(args.seed)
            
# methods operates on nn.Module main net
methods_on_net = ['mlc_debug', 'mlc_newton', 'mlc_mix_debug', 'mlc_l2w', 'l2w', 'mlc_C', 'hmlc_debug', 'mlc_mix_l2w', 'mwnet', 'hmlc', 'hmlc_fast', 'hmlc_K', 'hmlc_K_mix', 'l2w_fast']

# //////////////////////////// set logging ///////////////////////
now = datetime.now()
date_time = now.strftime("%Y_%m_%d-%H_%M")
filename = date_time + '_' + '_'.join([args.dataset, args.model_name, args.method, args.runid, str(args.epochs), str(args.seed)])
logfolder = 'logs/'
logfile = logfolder + filename + '.log'
os.makedirs(logfolder, exist_ok=True)
logger = get_logger(logfile, args.local_rank)

os.makedirs('models', exist_ok=True)
os.makedirs('out', exist_ok=True)
# //////////////////////////////////////////////////////////////////

# //////////////////////////// set data path ///////////////////////
args.glove_data_path = args.file_path+"/glove_embeds/glove.6B.{}d.txt.pkl".format(getattr(args,"lstm_embed_dim", 50))
args.file_path = args.file_path+"/{}/".format(args.dataset)+ "{}_organized_nb.pt".format(args.dataset)
args.random_seed = args.seed

logger.info(args)
logger.info('CUDA available:' + str(torch.cuda.is_available()))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#torch.cuda.set_device(device) 

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.cuda.synchronize()



from functools import partial


def get_data_loaders(args, glove_map):
    if args.dataset == "political" or args.dataset == "gossipcop":
        #dataset_fn = FakeNewsDataset
        dataset_fn = partial(FakeNewsDataset, elmo_map=glove_map)
    else:
        dataset_fn = partial(NoiseDataset, glove_map=glove_map)

    train_gold_dataset = dataset_fn(args, is_only_clean=True, train_status="train")
    train_silver_dataset = dataset_fn(args, is_only_clean=False, train_status="train")
    dev_dataset = dataset_fn(args, train_status="val")
    test_dataset = dataset_fn(args, train_status="test")

    num_classes = train_gold_dataset.num_class
    label_list = train_gold_dataset.label_list

    from data_utils import DataIterator
    gold_loader = DataIterator(DataLoader(train_gold_dataset, batch_size=args.bs, shuffle=True))
    silver_loader = DataLoader(train_silver_dataset, batch_size=args.bs, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    return gold_loader, silver_loader, dev_loader, test_loader, num_classes, label_list

def build_models(num_classes, method, model_name, cls_dim, glove_weight):
    if args.is_roberta:
        model = HFSC(model_name, num_classes, is_debug=args.is_debug)
        print("Hello We are using Roberta Model")
        hx_dim = 768
    else:
        hx_dim = 128
        model = LSTM_text(num_classes, h_dim=hx_dim, embed_weight=glove_weight, is_ner=False)

    if method in ['mlc', 'hmlc_K', 'hmlc_K_mix']:
        meta_model = MetaNet(hx_dim, cls_dim, 128, num_classes, args)
    elif method == 'mwnet':
        meta_model = VNet(1, 128, 1)
    else: # non meta-learning based methods
        meta_model = None

    # model.resize_token_embeddings(len(tokenizer))
    #import pdb; pdb.set_trace()

    model.to(device)
    if meta_model is not None:
        meta_model.to(device)
    
    logger.info('========== Main model ==========')
    logger.info(model)
    logger.info('========== Meta model ==========')
    logger.info(meta_model)

    return model, meta_model

def setup_training(main_net, exp_id=None):
    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # main net optimizer
    # Adam seems to work much better than SGD
    main_params = main_net.parameters() if isinstance(main_net, nn.Module) else f_params

    if False: #args.bert: # adamw
        main_opt = torch.optim.AdamW(main_params, lr=args.main_lr,
                                     weight_decay=args.wdecay, eps=args.opt_eps)
        main_schdlr = WarmupLinearSchedule(main_opt, args.warmup_steps, args.num_iterations)
    else:
        if args.optimizer == 'adam':
            main_opt = torch.optim.Adam(main_params, lr=args.main_lr, weight_decay=args.wdecay, amsgrad=True, eps=args.opt_eps)
        elif args.optimizer == 'sgd':
            main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)
        elif args.optimizer == 'adadelta':
            main_opt = torch.optim.Adadelta(main_params, weight_decay=args.wdecay, eps=args.opt_eps)
        if args.dataset in ['cifar10', 'cifar100', 'cifarmix']:
            # follow MW-Net setting
            main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[80,100], gamma=0.1)
        elif args.dataset in ['clothing1m']:
            main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[5], gamma=0.1)
        else:
            if args.cosine:
                main_schdlr = torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, args.cosine_ratio * args.epochs, eta_min=1e-6)
            else:
                main_schdlr = DummyScheduler(main_opt)


    # ============== resume from last checkpoint ================
    if args.resume and os.path.exists('models/%s_latest.pth' % exp_id):
        saved_states = torch.load('models/%s_latest.pth' % exp_id, torch.device('cuda', args.local_rank) if args.local_rank !=-1 else torch.device('cpu'))
        np.random.set_state(saved_states['np_rng'])
        torch.set_rng_state(saved_states['pt_rng'].cpu())
        torch.cuda.set_rng_state(saved_states['cuda_rng'].cpu(), args.local_rank if args.local_rank !=-1 else 0)

        last_epoch = saved_states['last_epoch']
        main_net.load_state_dict(saved_states['main_net'])
        main_opt.load_state_dict(saved_states['main_opt'])
        main_schdlr.load_state_dict(saved_states['main_schdlr'])
        optimizer.load_state_dict(saved_states['meta_opt'])
        scheduler.load_state_dict(saved_states['meta_schdlr'])
        if args.amp > 0:
            apex.amp.load_state_dict(saved_states['amp'])
                                
        logger.info('=== Resumed from last checkpoint at Epoch %d ===' % last_epoch)
    else:
        last_epoch = -1

    return main_net, main_opt, main_schdlr#, last_epoch
    
def trim_input(data):
    bert_ids = data[0]
    bert_mask = data[1]
    
    max_length = (bert_mask !=0).max(0)[0].nonzero().numel()
    
    if max_length < bert_ids.shape[1]:
        bert_ids = bert_ids[:, :max_length]
        bert_mask = bert_mask[:, :max_length]
        
    return [bert_ids, bert_mask]


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def test(model, test_loader, label_list=None): # this could be eval or test
    # //////////////////////// evaluate method ////////////////////////
    test_metric = load_metric('accuracy')
    label_list = []
    prediction_list = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model((input_ids, attention_mask,),)

            predictions = torch.argmax(outputs, -1)

            predictions = predictions.tolist()
            labels = labels.tolist()
            true_predictions = [prediction for prediction, label in zip(predictions, labels) if label != -100]
            true_labels = [label for prediction, label in zip(predictions, labels) if label != -100]
            label_list.extend(true_labels)
            prediction_list.extend(true_predictions)
            # test_metric.add_batch(predictions=true_predictions, references=true_labels)
            # test_metric.add_batch(predictions=torch.tensor(true_predictions), references=torch.tensor(true_labels))
    model.train()
    # res = test_metric.compute()
    # print(res)
    acc = accuracy_score(y_true=label_list, y_pred=prediction_list)
    f1 = f1_score(y_true=label_list, y_pred=prediction_list, average="macro")
    cm = ",".join(map(str,confusion_matrix(y_true=label_list, y_pred=prediction_list).ravel()))
    return acc, f1, cm

def train_and_test(model, meta_net, gold_loader, silver_loader, valid_loader, test_loader, exp_id=None, label_list=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # save init point and use later to retrain the classifier
    model_bkp = copy.deepcopy(model)
    model.cuda()

    if args.local_rank <= 0:
        writer = SummaryWriter(args.logdir + '/' + exp_id)

    net, optimizer, scheduler = setup_training(model)
    
    # //////////////////////// train for estimation ////////////////////////
    logger.info('=========== Phase 1 Train with nosy data ==============')
    net.train()
    itr = 0
    for epoch in range(args.epochs):
        for i, batch_s in enumerate(silver_loader):
            data_s = [batch_s['input_ids'], batch_s['attention_mask']]
            target_s_ = batch_s['label']
            data_s, target_s_ = tocuda(data_s), tocuda(target_s_)

            # forward
            output = net(data_s)
            loss = F.cross_entropy(output, target_s_)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            itr += 0
            if i % args.every ==0:
                tqdm.write('Epoch %d Iteration %d loss: %.16f' % (epoch, i, loss.item()))
                if args.local_rank <= 0: # single GPU or GPU 0
                    writer.add_scalar('train/pretrain_loss', loss.item(), itr)

        # per epoch op
        scheduler.step()

    # //////////////////////// baseline acc //////////////////////////
    net.eval()
    correct = torch.zeros(1).cuda()
    nsamples = torch.zeros(1).cuda()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model((input_ids, attention_mask,),)
            # accuracy
            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels.data).sum().item()
            nsamples += len(labels)

    if args.local_rank != -1: # DDP
        gather_c = [torch.zeros(1).cuda()] * dist.get_world_size()
        gather_n = [torch.zeros(1).cuda()] * dist.get_world_size()

        dist.all_gather(gather_c, correct)
        dist.all_gather(gather_n, nsamples)

        baseline_acc = (sum(gather_c) / sum(gather_n)).item()
    else:
        baseline_acc = (correct / nsamples).item()
        
    logger.info('Baseline acc: %.4f' % baseline_acc)
    
    # //////////////////////// estimate C ////////////////////////
    logger.info('=========== Phase 2 Estimate C ==============')    
    if args.method == 'glc':
        probs = []
        golds = []
        
        for batch in gold_loader.loader:
            # for idx, (*data, target) in enumerate(gold_loader.loader):
            data = [batch['input_ids'], batch['attention_mask']]
            target = batch['label']
            data, target = tocuda(data), tocuda(target)
            probs.append(F.softmax(net(data), -1).detach())
            golds.append(target)

        probs = torch.cat(probs, 0)
        golds = torch.cat(golds, 0)

        if args.local_rank != -1: # DDP
            gather_probs = [torch.zeros_like(probs)] * dist.get_world_size()
            gather_golds = [torch.zeros_like(golds)] * dist.get_world_size()

            dist.all_gather(gather_probs, probs)
            dist.all_gather(gather_golds, golds)

            prob_all = torch.cat(gather_probs, 0).cpu().numpy()
            gold_all = torch.cat(gather_golds, 0).cpu().numpy()
        else:
            prob_all = probs.cpu().numpy()
            gold_all = golds.cpu().numpy()

        num_classes = probs.size(-1)
        C_hat = np.zeros((num_classes, num_classes))
        
        for label in range(num_classes):
            indices = np.arange(len(gold_all))[gold_all == label]
            C_hat[label] = np.mean(prob_all[indices], axis=0, keepdims=True)

    '''
    elif method == 'forward' or method == 'forward_gold':
        probs = F.softmax(net(V(torch.from_numpy(dataset['x']).cuda(), volatile=True))).data.cpu().numpy()

        C_hat = np.zeros((num_classes,num_classes))
        for label in range(num_classes):
            class_probs = probs[:,label]
            thresh = np.percentile(class_probs, 97, interpolation='higher')
            class_probs[class_probs >= thresh] = 0

            C_hat[label] = probs[np.argsort(class_probs)][-1]

    elif method == 'ideal': C_hat = C

    elif method == 'confusion':
        # directly estimate confusion matrix on gold
        probs = F.softmax(net(V(torch.from_numpy(gold['x']).cuda(), volatile=True))).data.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        C_hat = np.zeros([num_classes, num_classes])

        for i in range(len(gold['y'])):
            C_hat[gold['y'][i], preds[i]] += 1

        C_hat /= (np.sum(C_hat, axis=1, keepdims=True) + 1e-7)

        C_hat = C_hat * 0.99 + np.full_like(C_hat, 1/num_classes) * 0.01
    '''


    #print('True C:', np.round(C, decimals=3))
    logger.info('C_hat:\n' +  str(np.round(C_hat, decimals=3)))

    C_hat = torch.from_numpy(C_hat).cuda().float()

    # //////////////////////// retrain with correction ////////////////////////
    # re-init net, opt, and scheduler
    logger.info('=========== Phase 3 Retrain ==============')        
    del net, optimizer, scheduler
    model_bkp.cuda()
    net, optimizer, scheduler = setup_training(model_bkp)
    net.train()

    method = args.method
    if method == 'glc' or method == 'ideal' or method == 'confusion' or method == 'forward_gold':
        itr = 0
        best_val_metric = float('inf')
        val_metric_queue = deque()         
        for epoch in range(args.epochs):
            for i, batch_s in enumerate(silver_loader):
            #for i, (*data_s, target_s_) in enumerate(silver_loader):
                batch_g = next(gold_loader)
                data_s = [batch_s['input_ids'], batch_s['attention_mask']]
                target_s_ = batch_s['label']
                
                data_g = [batch_g['input_ids'], batch_g['attention_mask']]
                target_g = batch_g['label']
                
                data_s, target_s_ = tocuda(data_s), tocuda(target_s_)
                data_g, target_g = tocuda(data_g), tocuda(target_g)

                output_s = net(data_s)
                output_g = net(data_g)
                
                loss_g = F.cross_entropy(output_g, target_g, reduction='sum')
                pre1 = C_hat.t()[target_s_.data]
                pre2 = F.softmax(output_s, -1) * pre1
                loss_s = -(torch.log(pre2.sum(1))).sum(0)

                # backward
                loss = (loss_g + loss_s) / (len(target_g) + len(target_s_))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                itr += 1
                if i % args.every == 0:
                    if args.local_rank <= 0:
                        writer.add_scalar('train/loss', loss.item(), itr)
                        tqdm.write('Epoch %d Iteration %d loss: %.16f' % (epoch, i, loss.item()))
            # Per epoch ops
            scheduler.step()

            # eval on validation
            val_acc, val_f1, val_cm = test(net, valid_loader, label_list=label_list)
            test_acc, test_f1, test_cm = test(net, test_loader, label_list=label_list)
            # val_acc = val_perf[metric_name]
            # test_acc = test_perf[metric_name]

            logger.info('Val perf: %.4f\tTest perf: %.4f' % (val_acc, test_acc))
            if args.local_rank <=0: # single GPU or GPU 0
                writer.add_scalar('train/val_acc', val_acc, epoch)
                writer.add_scalar('train/val_macro_f1', val_f1, epoch)
                writer.add_scalar('test/test_acc', test_acc, epoch)
                writer.add_scalar('test/test_macro_f1', test_f1, epoch)
                writer.add_text('test/test_cm', test_cm, epoch)
                writer.add_text('train/val_cm', val_cm, epoch)

            if len(val_metric_queue) == args.queue_size:  # keep at most this number of records
                # remove the oldest record
                val_metric_queue.popleft()

            val_metric_queue.append(-val_f1)

            avg_val_metric = sum(list(val_metric_queue)) / len(val_metric_queue)
            if avg_val_metric < best_val_metric:
                best_val_metric = avg_val_metric

                if isinstance(net, nn.Module):
                    # record current state dict
                    best_params = copy.deepcopy(net.state_dict())
                else:
                    best_params = copy.deepcopy(f_params)

                best_main_opt_sd = copy.deepcopy(optimizer.state_dict())
                best_main_schdlr_sd = copy.deepcopy(scheduler.state_dict())


                # dump best to file also
                ####################### save best models so far ###################
                if args.local_rank <=0: # only need to save once
                    logger.info('Saving best models...')
                    torch.save({
                        'epoch': epoch,
                        'val_metric': best_val_metric,
                        'main_net': best_params,
                        'main_opt': best_main_opt_sd,
                        'main_schdlr': best_main_schdlr_sd,
                    }, 'models/%s_best.pth' % exp_id)

            if args.local_rank <=0: # single GPU or GPU 0
                writer.add_scalar('train/val_acc_best', -best_val_metric, epoch) # write current best val_acc to tensorboard
                            
    # //////////////////////// evaluating method ////////////////////////
    if isinstance(net, nn.Module):
        net.load_state_dict(best_params)

    metric_name = 'accuracy' if label_list is None else 'overall_f1'
    test_acc, test_f1, test_cm = test(net, test_loader, label_list=label_list) # evaluate best params picked from validation

    '''
    if args.local_rank <=0: # single GPU or GPU 0
        writer.add_scalar('test/acc', test_acc, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        writer.add_scalar('test/macro_f1', test_f1, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        writer.add_text('test/cm', test_cm, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        # logger.info('Test results: {}'.format(test_perf))
        logger.info('Test {}: {:.4f}'.format(metric_name, test_acc))
    '''

    return {'acc':test_acc, 'f1':test_f1, 'cm':test_cm}

# //////////////////////// run experiments ////////////////////////
def run():
    dataset = args.dataset
    # get glove_embedding for LSTM model
    glove_data = pickle.load(open(args.glove_data_path, 'rb'))
    glove_map = {i[0]: index + 1 for index, i in enumerate(glove_data)}
    glove_weight = np.stack([np.zeros((glove_data[0][1].size)), *[i[1] for i in glove_data]], axis=0)

    gold_loader, silver_loader, dev_loader, test_loader, num_classes, label_list = get_data_loaders(args, glove_map)
    main_net, meta_net = build_models(num_classes, args.method, args.model_name, args.cls_dim, glove_weight=glove_weight)

    exp_id = 'exp_' + filename
    results = train_and_test(main_net, meta_net, gold_loader, silver_loader, dev_loader, test_loader, exp_id, label_list=label_list)

    with open('out/' + filename, 'wb') as file:
        pickle.dump(results, file)
    logger.info("Dumped results_ours in file: " + filename)

if __name__ == '__main__':
    run()
        
