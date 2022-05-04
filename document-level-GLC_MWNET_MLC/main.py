import numpy as np
import pickle
import copy
import sys
import argparse
import logging
import os
import gc
from logger import get_logger
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AdamW, set_seed

from mlc import step_mwnet, step_hmlc_K, step_l2w_fast
from mlc_utils import clone_parameters, tocuda, DummyScheduler, pseudo_eval, soft_cross_entropy

from models import *       # import all models and F
from meta_models import *  # import all meta_models
from datetime import datetime

from dataset import NoiseDataset, FakeNewsDataset
import os

parser = argparse.ArgumentParser(description='WALNUT Baseline Training Framework')
# fake news and other noise dataset required parameters
parser.add_argument('--dataset', type=str, choices=['imdb', "agnews", "yelp", "political", "gossipcop"], default='imdb')
parser.add_argument("--file_path", type=str, default='data', help="base directory of the data")
parser.add_argument("--weak_ratio", default=0.8, type=float, help="splittion of weak data and clean data")
parser.add_argument("--clean_ratio", default=20, type=float, help="number of clean samples for imdb/agnews/yelp dataset")
parser.add_argument("--n_high_cov", default=1, type=float, help="number of valid weak labeling functions for weak data")
parser.add_argument("--lstm_embed_dim", default=50, type=int, choices=[50, 100, 200, 300])
parser.add_argument("--is_overwrite_file",action="store_true")
parser.add_argument('--model_name', default='distilbert-base-uncased', type=str, choices=['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'bilstm', 'cnn', 'bert-large-uncased', 'roberta-large'])
parser.add_argument('--method', default='hmlc_K_mix', type=str, choices=['mlc', 'mlc_fast', 'mlc_mix', 'mlc_debug', 'mlc_newton', 'hmlc_debug', 'mlc_mix_l2w', 'mwnet', 'hmlc_fast', 'hmlc_K', 'hmlc_K_mix', 'l2w_fast'])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--max_seq_length', default=512, type=int, help='maximum sequence length')
parser.add_argument('--epochs', '-e', type=int, default=75, help='Number of epochs to train.')
parser.add_argument('--every', default=100, type=int, help='Eval interval (default: 100 iters)')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--cls_dim', type=int, default=64, help='Label embedding dim (Default: 64)')
parser.add_argument('--grad_clip', default=0.0, type=float, help='max grad norm (default: 0, no clip)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
parser.add_argument('--main_lr', default=1e-5, type=float, help='lr for main net')
parser.add_argument('--meta_lr', default=3e-4, type=float, help='lr for meta net')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'adadelta'])
parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for optimizers')
parser.add_argument('--sampler', default='sm', type=str, choices=['sm', 'gs', 'max'])
parser.add_argument('--tau', default=1, type=float, help='tau')
parser.add_argument('--reg_str', default=0, type=float, help='reg str')
parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay (default: 5e-4)')

# noise parameters
parser.add_argument('--skip', default="False", action="store_true", help='Skip link for LCN (default: False)')
parser.add_argument('--sparsemax', default="False", type=str, help='Use softmax instead of softmax for meta model (default: False)')

parser.add_argument('--cosine', default="False", type=str, help='Use cosine scheduler for main net and meta net (default: False)')
parser.add_argument('--cosine_ratio', default=1., type=float, help='Cosine annealing period (fraction of total iterations) (default: 1.0)')

parser.add_argument('--tie', default="False", type=str, help='Tie label embedding to the output classifier output embedding of metanet (default: False)')

parser.add_argument('--mse', default="False", type=str, help='MSE loss for soft labels (default: False)')
parser.add_argument('--lcnmode', default='full', type=str, choices=['simple', 'ssl', 'full'])

parser.add_argument('--meta_eval', default="False", type=str, help='Set meta net to eval mode in training (Default: False, turns off dropout only)')
parser.add_argument('--is_debug', action="store_true",help='Debug Mode')

# use a seprate validation set from training to be used for model selection / early stopping
#parser.add_argument('--valid_ratio', default=0., type=float, help='Ratio of seprate validation set (default: 0)')

# bert args
parser.add_argument('--warmup_steps', default=1000, type=int, help='Linear warm up steps (Default: 1000)')

parser.add_argument('--runid', default='exp', type=str)
parser.add_argument('--resume', default="False", type=str, help='Resume from last checkpoint')
# acc is used by default for evaluation
#parser.add_argument('--acc', default="False", type=str, help='Use acc on dev set to save models') 
parser.add_argument('--queue_size', default=1, type=int, help='Number of iterations before to compute mean loss_g')
#parser.add_argument('--debug', default="False", type=str, help='Debug mode for experimental ideas. (default: False)')


############## MAGIC STEPS ##################
parser.add_argument('--magic_max', default=1, type=float, help='Max magic coefficient for proxy_g (default: 1.0)')
parser.add_argument('--magic', default=1, type=int, help='step for meta model update')
parser.add_argument('--magic_min', default=1, type=float, help='Min magic coefficient for proxy_g (default: 1.0)')
parser.add_argument('--magic_ratio', default=1., type=float, help='Min magic coefficient for proxy_g (default: 1.0)')
############## MAGIC STPES ##################

# i/o
parser.add_argument('--logdir', type=str, default=os.getenv("AMLT_OUTPUT_DIR", "runs"), help='Log folder.')
parser.add_argument('--local_rank', type=int, default='-1', help='Log folder.')

parser.add_argument('--amp', default=-1, type=int, help='use APEX AMP')
args = parser.parse_args()

# //////////////////////////// Reformulate the Argparser ///////////////////////
for key, value in vars(args).items():
    if type(value) is str:
        if value.lower() == "false":
            setattr(args, key, False)
        elif value.lower() == "true":
            setattr(args, key, True)

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
        if 'large' in model_name: # XX-large
            hx_dim = 1024
        else:
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

def setup_training(main_net, meta_net, exp_id=None):
    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # meta net optimizer
    if False: #args.bert: # adamw seems like MLC doesn't like AdamW
        from transformers import WarmupLinearSchedule
        optimizer = torch.optim.AdamW(meta_net.parameters(), lr=args.meta_lr,
                                      weight_decay=args.wdecay, eps=args.opt_eps)
        scheduler = WarmupLinearSchedule(optimizer, args.warmup_steps, args.num_iterations)
    else:
        optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr,
                                     weight_decay=0, #args.wdecay, # meta should have wdecay or not??
                                     amsgrad=True, eps=args.opt_eps)

        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cosine_ratio * args.epochs, eta_min=1e-6)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1e4, gamma=0.5)
        else:
            scheduler = DummyScheduler(optimizer)

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
        meta_net.load_state_dict(saved_states['meta_net'])
        optimizer.load_state_dict(saved_states['meta_opt'])
        scheduler.load_state_dict(saved_states['meta_schdlr'])
        if args.amp > 0:
            apex.amp.load_state_dict(saved_states['amp'])
                                
        logger.info('=== Resumed from last checkpoint at Epoch %d ===' % last_epoch)
    else:
        last_epoch = -1

    return main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch
    
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
    

####################################################################################################
###  training code 
####################################################################################################
def train_and_test(main_net, meta_net, gold_loader, silver_loader, valid_loader, test_loader, exp_id=None, label_list=None):
    if args.local_rank <=0: # single GPU or GPU 0
        writer = SummaryWriter(args.logdir + '/' + exp_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch = setup_training(main_net, meta_net, exp_id)

    # //////////////////////// switching on training mode ////////////////////////
    meta_net.train()
    main_net.train()

    # ============== resume best checkpoint for validation ================
    if args.resume and os.path.exists('models/%s_best.pth' % exp_id):
        best_states = torch.load('models/%s_best.pth' % exp_id, torch.device('cuda', args.local_rank) if args.local_rank !=-1 else torch.device('cpu'))
        best_epoch = best_states['epoch']
        
        best_params = best_states['main_net']
        best_main_opt_sd = best_states['main_opt']
        best_main_schdlr_sd = best_states['main_schdlr']

        best_meta_params = best_states['meta_net']
        best_meta_opt_sd = best_states['meta_opt']
        best_meta_schdlr_sd = best_states['meta_schdlr']

        best_val_metric = best_states['val_metric']
        
        logger.info('=== Best model loaded from Epoch %d  ===' % best_epoch)
    else:
        best_params = None
        best_main_opt_sd = None
        best_main_schdlr_sd = None

        best_meta_params = None
        best_meta_opt_sd = None
        best_meta_schdlr_sd = None
        #best_loss_g = float('inf')
        best_val_metric = float('inf')
        
    val_metric_queue = deque() 
    # set done

    args.dw_prev = [0 for param in meta_net.parameters() if param.requires_grad] # 0 for previous iteration
    args.steps = 0

    for epoch in tqdm(range(last_epoch+1, args.epochs)):# change to epoch iteration
        logger.info('Epoch %d:' % epoch)

        # magic scheduling
        # cosine scheduling
        #args.magic = args.magic_min + 0.5 * (args.magic_max - args.magic_min) * ( 1 + math.cos((1.0 * epoch / (args.magic_ratio * args.epochs))*math.pi))

        # linear scheduling
        # ATTENTION: set the magic step as the hyper-parameters.
        # args.magic = 1 #max(args.magic_max - (args.magic_max - args.magic_min) * ( 1.0 * i / (args.magic_ratio * args.num_iterations) ), args.magic_min)
        for i, batch_s in enumerate(silver_loader):
            data_s = [batch_s['input_ids'], batch_s['attention_mask']]
            target_s = batch_s['label']
            #print (data_s, target_s)

            #for i, (*data_s, target_s) in enumerate(silver_loader):
            #*data_g, target_g = next(gold_loader)#.next()
            batch_g = next(gold_loader)
            data_g = [batch_g['input_ids'], batch_g['attention_mask']]
            target_g = batch_g['label']

            if False: #args.bert:
                data_g = trim_input(data_g)
                data_s = trim_input(data_s)
            # ATTENTION: edition
            data_g, target_g = tocuda(data_g), tocuda(target_g)
            data_s, target_s_ = tocuda(data_s), tocuda(target_s)
            target_s = tocuda(target_s)
            # target_s_ = target_s

            # bi-level optimization stage
            eta = main_schdlr.get_lr()[0]
            enable_cudnn = all(["lstm" not in n for n, _ in main_net.named_parameters()])
            with torch.backends.cudnn.flags(enabled=enable_cudnn):

                if args.method == 'hmlc_K':
                    loss_g, loss_s = step_hmlc_K(main_net, main_opt,
                                                 meta_net, optimizer,
                                                 data_s, target_s_, data_g, target_g,
                                                 None, None,
                                                 eta, args)
                elif args.method == 'hmlc_K_mix':
                    gbs = int(target_g.size(0) / 2)
                    if type(data_g) is list:
                        data_c = [x[gbs:] for x in data_g]
                        data_g = [x[:gbs] for x in data_g]
                    else:
                        data_c = data_g[gbs:]
                        data_g = data_g[:gbs]

                    target_c = target_g[gbs:]
                    target_g = target_g[:gbs]
                    loss_g, loss_s = step_hmlc_K(main_net, main_opt,
                                                 meta_net, optimizer,
                                                 data_s, target_s_, data_g, target_g,
                                                 data_c, target_c,
                                                 eta, args)

                elif args.method == 'l2w_fast':
                    loss_g, loss_s = step_l2w_fast(main_net, main_opt,
                                                   data_s, target_s_, data_g, target_g,
                                                   eta, args)
                elif args.method == 'mwnet':
                    loss_g, loss_s = step_mwnet(main_net, main_opt, meta_net, optimizer,
                                                data_s, target_s_, data_g, target_g,
                                                eta, args)

            args.steps += 1
            if i % args.every == 0:
                if args.local_rank <=0: # single GPU or GPU 0
                    writer.add_scalar('train/loss_g', loss_g.item(), args.steps)
                    writer.add_scalar('train/loss_s', loss_s.item(), args.steps)

                    ''' get entropy of predictions from meta-net '''
                    if args.method in ['hmlc_debug', 'hmlc_fast', 'hmlc_K', 'hmlc_K_mix']:
                        logit_s, x_s_h = main_net(data_s, return_h=True)
                        # FIXME: target_s_ contains -100. Taking care for token classification
                        # pseudo_target_s = meta_net(x_s_h.detach(), target_s_).detach()
                        if logit_s.dim() == 2:  # sequence level
                            pseudo_target_s = meta_net(x_s_h.detach(), target_s)
                        else:  # token level
                            h_dim = x_s_h.size(-1)
                            # set IGNORED_INDEX to 0, this will be counted in loss computation
                            target_s__ = target_s * (target_s >= 0)
                            # FIXME: pass to cuda() only if needed.
                            pseudo_target_s = meta_net(x_s_h.detach().view(-1, h_dim).cuda(), target_s__.view(-1).cuda())


                    elif args.method in ['mwnet', 'l2w_fast']:
                        pseudo_target_s = torch.ones(1) # entropy N/A to mwnet and l2w_fast
                    else:
                        pseudo_target_s = meta_net(data_s, target_s_).detach()


                    entropy = -(pseudo_target_s * torch.log(pseudo_target_s+1e-10)).sum(-1).mean()

                    writer.add_scalar('train/meta_entropy', entropy.item(), args.steps)

                    main_lr = main_schdlr.get_lr()[0]
                    meta_lr = scheduler.get_lr()[0]
                    writer.add_scalar('train/main_lr', main_lr, args.steps)
                    writer.add_scalar('train/meta_lr', meta_lr, args.steps)
                    writer.add_scalar('train/magic', args.magic, args.steps)
                    #alpha = meta_net.module.get_alpha().item() if isinstance(meta_net, torch.nn.parallel.DistributedDataParallel) or isinstance(meta_net, apex.parallel.DistributedDataParallel) else meta_net.get_alpha().item()

                    logger.info('Iteration %d loss_s: %.4f\tloss_g: %.4f\tMeta entropy: %.3f\tMain LR: %.8f\tMeta LR: %.8f' %( i, loss_s.item(), loss_g.item(), entropy.item(), main_lr, meta_lr))
                    logger.info('Iteration %d' % i)

        # PER EPOCH PROCESSING
        # import pdb; pdb.set_trace()
        # lr scheduler
        main_schdlr.step()        
        #scheduler.step()

        # evaluation on validation set
        # metric_name = 'accuracy' if label_list is None else 'overall_f1'
        val_acc, val_f1, val_cm = test(main_net, valid_loader, label_list=label_list)
        test_acc, test_f1, test_cm = test(main_net, test_loader, label_list=label_list)
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

        val_metric_queue.append(-val_acc)

        avg_val_metric = sum(list(val_metric_queue)) / len(val_metric_queue)
        if avg_val_metric < best_val_metric:
            best_val_metric = avg_val_metric

            if isinstance(main_net, nn.Module):
                # record current state dict
                best_params = copy.deepcopy(main_net.state_dict())
            else:
                best_params = copy.deepcopy(f_params)

            best_main_opt_sd = copy.deepcopy(main_opt.state_dict())
            best_main_schdlr_sd = copy.deepcopy(main_schdlr.state_dict())

            best_meta_params = copy.deepcopy(meta_net.state_dict())
            best_meta_opt_sd = copy.deepcopy(optimizer.state_dict())
            best_meta_schdlr_sd = copy.deepcopy(scheduler.state_dict())

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
                    'meta_net': best_meta_params,
                    'meta_opt': best_meta_opt_sd,
                    'meta_schdlr': best_meta_schdlr_sd
                }, 'models/%s_best.pth' % exp_id)
                
        if args.local_rank <=0: # single GPU or GPU 0
            writer.add_scalar('train/val_acc_best', -best_val_metric, epoch) # write current best val_acc to tensorboard
            
        # ====== save current checkpoint per epoch ==========
        if args.local_rank <= 0: # only dump for first GPU
            torch.save({
                'np_rng': np.random.get_state(),
                'pt_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state(),
                'last_epoch': epoch,

                'main_net': main_net.state_dict(),
                'main_opt': main_opt.state_dict(),
                'main_schdlr': main_schdlr.state_dict(),
                'meta_net': meta_net.state_dict(),
                'meta_opt': optimizer.state_dict(),
                'meta_schdlr': scheduler.state_dict(),
                'amp': apex.amp.state_dict() if args.amp > 0 else None
                }, 'models/%s_latest.pth' % exp_id)

    # //////////////////////// evaluating method ////////////////////////
    if isinstance(main_net, nn.Module):
        main_net.load_state_dict(best_params)

    metric_name = 'accuracy' if label_list is None else 'overall_f1'
    test_acc, test_f1, test_cm = test(main_net, test_loader, label_list=label_list) # evaluate best params picked from validation

    if args.local_rank <=0: # single GPU or GPU 0
        writer.add_scalar('test/acc', test_acc, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        writer.add_scalar('test/macro_f1', test_f1, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        writer.add_text('test/cm', test_cm, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        # logger.info('Test results: {}'.format(test_perf))
        logger.info('Test {}: {:.4f}'.format(metric_name, test_acc))

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
