import numpy as np
import pickle
import copy
import sys
import argparse
import logging
import os
import gc
import pandas as pd
from logger import get_logger
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
from datasets import ClassLabel, Sequence, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AdamW, set_seed

from mlc import mlc_hard_ce
from mlc_utils import clone_parameters, tocuda, DummyScheduler, pseudo_eval, soft_cross_entropy
from utils import token_level_stratified_sampling

from models import *       # import all models and F
from meta_models import *  # import all meta_models

from datetime import datetime
from data_utils import load_precomputed_split

parser = argparse.ArgumentParser(description='WALNUT GLC Baseline Training Framework')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'cifarmix', 'clothing1m', 'imdb', 'mnist', 'scr', 'sst', 'twitter', 'ag', 'amazon2', 'amazon5', 'dbpedia', 'yelp2', 'yelp5', 'sogou', 'yahoo', 'imdb2', 'spouse', 'CoNLL', 'BC5CDR', 'NCBI', 'LaptopReview'], default='spouse')
parser.add_argument("--fix_tagging", action="store_true", help="Convert to IO tagging format")
parser.add_argument('--method', default='glc', type=str, choices=['glc', 'forward', 'forward_gold', 'ideal', 'confusion'])
parser.add_argument('--model_name', default='distilbert-base-uncased', type=str, choices=['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'bilstm', 'bert-large-uncased', 'roberta-large'])
parser.add_argument("--label_name", help="label name", type=str, default='ner_tags')  # mv
parser.add_argument("--lstm_embed_dim", default=50, type=int, choices=[50, 100, 200, 300])
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
parser.add_argument('--skip', default=False, action='store_true', help='Skip link for LCN (default: False)')
parser.add_argument('--sparsemax', default=False, action='store_true', help='Use softmax instead of softmax for meta model (default: False)')

parser.add_argument('--cosine', default=False, action='store_true', help='Use cosine scheduler for main net and meta net (default: False)')
parser.add_argument('--cosine_ratio', default=1., type=float, help='Cosine annealing period (fraction of total iterations) (default: 1.0)')

parser.add_argument('--tie', default=False, action='store_true', help='Tie label embedding to the output classifier output embedding of metanet (default: False)')

parser.add_argument('--mse', default=False, action='store_true', help='MSE loss for soft labels (default: False)')
parser.add_argument('--lcnmode', default='full', type=str, choices=['simple', 'ssl', 'full'])

parser.add_argument('--meta_eval', default=False, action='store_true', help='Set meta net to eval mode in training (Default: False, turns off dropout only)')

# use a seprate validation set from training to be used for model selection / early stopping
#parser.add_argument('--valid_ratio', default=0., type=float, help='Ratio of seprate validation set (default: 0)')

# bert args
parser.add_argument('--warmup_steps', default=1000, type=int, help='Linear warm up steps (Default: 1000)')

parser.add_argument('--runid', default='exp', type=str)
parser.add_argument('--resume', default=False, action='store_true', help='Resume from last checkpoint')
# acc is used by default for evaluation
#parser.add_argument('--acc', default=False, action='store_true', help='Use acc on dev set to save models') 
parser.add_argument('--queue_size', default=1, type=int, help='Number of iterations before to compute mean loss_g')
#parser.add_argument('--debug', default=False, action='store_true', help='Debug mode for experimental ideas. (default: False)')

############## MAGIC STEPS ##################
parser.add_argument('--magic_max', default=1, type=float, help='Max magic coefficient for proxy_g (default: 1.0)')
parser.add_argument('--magic_min', default=1, type=float, help='Min magic coefficient for proxy_g (default: 1.0)')
parser.add_argument('--magic_ratio', default=1., type=float, help='Min magic coefficient for proxy_g (default: 1.0)')
############## MAGIC STPES ##################

# i/o
parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')
parser.add_argument('--local_rank', type=int, default='-1', help='local rank.')

parser.add_argument('--amp', default=0, type=int, help='use APEX AMP')

parser.add_argument("--train_frac", default=1.0, type=float, help="Fraction of the training data to keep")
parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default='../data')
args = parser.parse_args()

if args.model_name in ['bilstm', 'cnn']:
    args.is_roberta = False
else:
    args.is_roberta = True
    
token_level_datasets = ['CoNLL', 'BC5CDR', 'NCBI', 'LaptopReview']
if args.train_frac > 1:
    args.train_frac = int(args.train_frac)

# methods operates on nn.Module main net
methods_on_net = ['mlc_debug', 'mlc_newton', 'mlc_mix_debug', 'mlc_l2w', 'l2w', 'mlc_C', 'hmlc_debug', 'mlc_mix_l2w', 'mwnet', 'hmlc', 'hmlc_fast', 'hmlc_K', 'hmlc_K_mix', 'l2w_fast']

# //////////////////////////// set logging ///////////////////////
now = datetime.now()
date_time = now.strftime("%Y_%m_%d-%H_%M")
filename = date_time + '_' + '_'.join([args.dataset, args.model_name, args.method, args.runid, str(args.epochs), str(args.seed)])
exp_id =  'exp_' + filename
logfolder = 'logs/'
logfile = logfolder + filename + '.log'
os.makedirs(logfolder, exist_ok=True)
logger = get_logger(logfile, args.local_rank)

os.makedirs('models', exist_ok=True)
os.makedirs('out', exist_ok=True)
# //////////////////////////////////////////////////////////////////
args.glove_data_path = args.datapath + '/glove_embeds/glove.6B.{}d.txt.pkl'.format(getattr(args,"lstm_embed_dim", 50))
glove_data = pickle.load(open(args.glove_data_path, 'rb'))
glove_map = {i[0]: index + 1 for index, i in enumerate(glove_data)}
glove_weight = np.stack([np.zeros((glove_data[0][1].size)), *[i[1] for i in glove_data]], axis=0)

logger.info(args)
logger.info('CUDA available:' + str(torch.cuda.is_available()))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#torch.cuda.set_device(device) 

torch.backends.cudnn.enabled = args.is_roberta
torch.backends.cudnn.benchmark = args.is_roberta
torch.cuda.synchronize()

# modify here to add NER tasks, currently deaults to sequence classification
def get_task_type(dataset):
    # sequence classification: 0 (including pair of sequence as well)
    # token classification: 1
    if dataset in token_level_datasets:
        return 1
    return 0

def get_data_loaders(dataset, model_name):
    # NOTE separate loader for bilstm 
    if model_name == 'bilstm':
        return get_data_loaders_bilstm(dataset, model_name)
    
    if 'roberta' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_all_tokens = True

    def tokenize_and_align_labels(examples, label_name):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length', max_length=args.max_seq_length)

        labels = []
        for i, label in enumerate(examples[label_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs[label_name] = labels
        return tokenized_inputs

    if dataset not in token_level_datasets:
        # standard classification
        datasets = load_dataset('{}/{}.py'.format(args.datapath, dataset))
        num_classes = 2
        label_list = None
        clean_dataset = datasets['train_clean']
        train_dataset = datasets['train_weak']

        # downsampling for debugging reasons
        # clean_dataset = clean_dataset.train_test_split(train_size=0.2, shuffle=True, seed=42)['train']
        # train_dataset = train_dataset.train_test_split(train_size=0.2, shuffle=True, seed=42)['train']

        train_dataset = train_dataset.rename_column("label", "weaklabel")
        clean_labels = clean_dataset['label']
        train_dataset = train_dataset.map(lambda x, ind: {'label': clean_labels[ind]}, with_indices=True)

        dev_dataset = datasets['validation']
        test_dataset = datasets['test']

        gold_dataset = train_dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
        gold_dataset.set_format(type='torch', columns=['input_ids',  'attention_mask', 'label'])
        gold_loader = DataLoader(gold_dataset, batch_size=args.bs, shuffle=True)

        silver_dataset = train_dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
        silver_dataset.set_format(type='torch', columns=['input_ids',  'attention_mask', 'weaklabel'])
        silver_loader = DataLoader(silver_dataset, batch_size=args.bs, shuffle=True)

        dev_dataset = dev_dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
        dev_dataset.set_format(type='torch', columns=['input_ids',  'attention_mask', 'label'])
        dev_loader = DataLoader(dev_dataset, batch_size=args.bs, shuffle=False)

        test_dataset = test_dataset.map(lambda e: tokenizer(e['sent'], truncation=True, padding='max_length', max_length=args.max_seq_length), batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

        from data_utils import DataIterator

        gold_loader = DataIterator(gold_loader)

        return gold_loader, silver_loader, dev_loader, test_loader, num_classes, label_list

    else:
        # token-level classification
        if dataset == 'CoNLL':
            savepath = os.path.join(args.datapath, "CoNLL/conll2003_weak_labels")
            assert os.path.exists(savepath), 'need to first run data/CoNLL/download_CoNLL.sh'
            from datasets import load_from_disk
            datasets = load_from_disk(savepath)
            datasets = datasets.rename_column("majority_label", "mv")
        else:
            datasets = load_dataset('{}/{}/{}.py'.format(args.datapath, dataset, dataset))
        datasets = datasets.rename_column("mv", "weaklabel")
        datasets = datasets.rename_column("ner_tags", "label")
        label_list = datasets['train'].features["label"].feature.names
        num_classes = len(label_list)

        # train_dataset = datasets['train']
        dev_dataset = datasets['validation']
        test_dataset = datasets['test']

        precomputed_indices_file = os.path.join(args.datapath, "{}/{}_indices.pkl".format(args.dataset, args.dataset))
        logger.info("Loading pre-computed indices for seed={} in {}".format(args.seed, precomputed_indices_file))
        split = load_precomputed_split(datasets['train'], seed=args.seed, precomputed_indices_file=precomputed_indices_file)
        # split = train_dataset.train_test_split(train_size=args.train_frac, shuffle=True, seed=args.seed)

        gold_dataset = split['train']
        gold_dataset = gold_dataset.map(lambda x: tokenize_and_align_labels(x, 'label'), batched=True)
        gold_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        gold_loader = DataLoader(gold_dataset, batch_size=args.bs, shuffle=True)

        silver_dataset = split['test']
        silver_dataset = silver_dataset.map(lambda x: tokenize_and_align_labels(x, 'weaklabel'), batched=True)
        silver_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'weaklabel'])
        silver_loader = DataLoader(silver_dataset, batch_size=args.bs, shuffle=True)

        logger.info("GOLD dataset = {}, SILVER dataset =  {}".format(len(gold_dataset), len(silver_dataset)))

        dev_dataset = dev_dataset.map(lambda x: tokenize_and_align_labels(x, 'label'), batched=True)
        dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        dev_loader = DataLoader(dev_dataset, batch_size=args.bs, shuffle=False)

        test_dataset = test_dataset.map(lambda x: tokenize_and_align_labels(x, 'label'), batched=True)
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

        from data_utils import DataIterator

        gold_loader = DataIterator(gold_loader)

        print("tokenizer: {}".format(len(tokenizer)))
        return gold_loader, silver_loader, dev_loader, test_loader, num_classes, label_list

def get_data_loaders_bilstm(dataset, model_name):
    label_all_tokens = True

    # Load dataset
    dpath = os.path.join(args.datapath, "{}/{}.py".format(dataset, dataset))
    if dataset == 'CoNLL':
        #savepath = "/home/giannis/walnut/datasets/conll2003_weak_labels"
        #datasets = load_from_disk(savepath)
        savepath = os.path.join(args.datapath, "CoNLL/conll2003_weak_labels")
        assert os.path.exists(savepath), 'need to first run data/CoNLL/download_CoNLL.sh'
        datasets = load_from_disk(savepath)
        datasets = datasets.rename_column("majority_label", "mv")
    else:
        datasets = load_dataset(dpath) #, download_mode="force_redownload")

    datasets = datasets.rename_column("mv", "weaklabel")
    datasets = datasets.rename_column("ner_tags", "label")
    label_names = datasets['train'].features['label'].feature.names
    #label_names = datasets['test'].features["label"].feature.names
    name2ind = {n:i for i,n in enumerate(label_names)}
    ind2name = {i:n for n,i in name2ind.items()}

    def fix_tag(tag):
        if tag == 'B':
            return 'I'
        return tag.replace('B-', 'I-')

    def fix_tagging(example):
        columns = list(set(['ner_tags', args.label_name]))
        for column in columns:
            example[column] = [name2ind[fix_tag(ind2name[tag])] for tag in example[column]]
        return example

    def print_stats(datasets):
        for label_type in ['label']:
            for method in ['train', 'validation', 'test']:
                labels = datasets[method][label_type]
                labels = [y for ys in labels for y in ys]
                df = pd.DataFrame()
                df['tags'] = labels
                df['tags'] = df['tags'].map(lambda x: label_names[x])
                logger.info("{}-{}\n{}".format(label_type, method, df['tags'].value_counts()))

    logger.info("Original datasets:\n{}".format(datasets))

    if args.fix_tagging:
        datasets = datasets.map(fix_tagging)
        #label_names = [fix_tag(tag) for tag in label_names]

    # datasets['train'] = datasets['train'].shuffle(seed=args.seed)
    # Use custom token-level stratified sampling function and save indices
    precomputed_indices_file = os.path.join(args.datapath, "{}/{}_indices.pkl".format(dataset, dataset))
    logger.info("Loading pre-computed indices for seed={} in {}".format(args.seed, precomputed_indices_file))
    split = token_level_stratified_sampling(datasets['train'], train_size=args.train_frac, shuffle=True, seed=args.seed, label_names=label_names, precomputed_indices_file=precomputed_indices_file, label_type='label')
    print (split)
    datasets['gold'] = split['train']
    datasets['silver'] = split['test'] # clean (train) vs. weak (test)
    #datasets['train'] = split['train'] if args.label_name == 'ner_tags' else split['test']
    #datasets['train'] = datasets['train'].rename_column(args.label_name, "labels")

    # For the validation and test sets, set "labels" column to clean labels
    #datasets['validation'] = datasets['validation'].rename_column("ner_tags", "labels")
    #datasets['test'] = datasets['test'].rename_column("ner_tags", "labels")

    logger.info("Pre-processed datasets for training base model:\n{}".format(datasets))
    print_stats(datasets)

    label_list = datasets['test'].features['label'].feature.names
    logger.info(label_list)

    if args.is_roberta:
        if 'roberta' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        from lstm_util import glove_tokenize_text, bilstmTokenizer
        from functools import partial
        tokenizer = bilstmTokenizer(glove_map=glove_map)
    label_all_tokens = True

#    def tokenize_and_align_labels(examples, label_name):
#        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length', max_length=args.max_seq_length)
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[label_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs[label_name] = labels
        return tokenized_inputs

    def tokenize_and_align_labels_lstm(examples):
        tokenized_inputs = tokenizer.glove_tokenize(examples["tokens"], labels=examples['label'])
        tokenized_inputs2 = tokenizer.glove_tokenize(examples["tokens"], labels=examples['weaklabel'], label_name='weaklabels')
        tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'][0]
        tokenized_inputs['attention_mask'] = tokenized_inputs['attention_mask'][0]
        #labels = examples["labels"]
        tokenized_inputs['label'] = tokenized_inputs['labels'][0]
        tokenized_inputs['weaklabel'] = tokenized_inputs2['weaklabels'][0]
        #except:
        return tokenized_inputs

    #import pdb; pdb.set_trace()
    if args.model_name == 'bilstm':
        tokenized_datasets = datasets.map(tokenize_and_align_labels_lstm)
    else:
        tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    for split in ['train', 'validation', 'test', 'gold', 'silver']:
        for i, x in enumerate(tokenized_datasets[split]):
            if len(x['input_ids']) == 0:
                print('error empty {} {}'.format(method, x))

            len_tokens = len(x['input_ids'])
            len_labels = len(x['label'])
            assert len_tokens == len_labels, 'tokens={}!=labels={}\n{}\n\n{}'.format(len_tokens,len_labels,x['input_ids'],x['label'])

    #import pdb; pdb.set_trace()
    gold_dataset = tokenized_datasets['gold']
    gold_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'weaklabel'])
    gold_loader = DataLoader(gold_dataset, batch_size=args.bs, shuffle=True)

    silver_dataset = tokenized_datasets['silver']
    silver_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'weaklabel'])
    silver_loader = DataLoader(silver_dataset, batch_size=args.bs, shuffle=True)

    #print (silver_dataset)
    #print(silver_loader)
    #for batch in silver_loader:
    #    print (batch)

    logger.info("GOLD dataset = {}, SILVER dataset =  {}".format(len(gold_dataset), len(silver_dataset)))

    dev_dataset = tokenized_datasets['validation']
    dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dev_loader = DataLoader(dev_dataset, batch_size=args.bs, shuffle=False)

    test_dataset = tokenized_datasets['test']
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)


    from data_utils import DataIterator
    gold_loader = DataIterator(gold_loader)

    num_classes = len(label_names) # get num_classes
    return gold_loader, silver_loader, dev_loader, test_loader, num_classes, label_list    

def build_models(dataset, num_classes, method, model_name, cls_dim):
    task_type = get_task_type(dataset)

    if args.is_roberta:
        model = HFTC(model_name, num_classes)
        hx_dim = 768
    else:
        hx_dim = 128
        model = LSTM_text_TC(num_classes, h_dim=hx_dim, embed_weight=glove_weight, is_ner=True)    

    if method == 'mlc':
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



def test(model, test_loader, label_list=None): # this could be eval or test
    # //////////////////////// evaluate method ////////////////////////
    if label_list is None:
        test_metric = load_metric('accuracy')
    else:
        # token-level classification
        test_metric = load_metric('seqeval')

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

            if label_list is None:
                test_metric.add_batch(predictions=predictions, references=labels)
            else:
                predictions = predictions.tolist()
                labels = labels.tolist()
                true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
                true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels) ]
                test_metric.add_batch(predictions=true_predictions, references=true_labels)

    model.train()
    res = test_metric.compute()
    print(res)
    return res
    

####################################################################################################
###  training code 
####################################################################################################
def train_and_test(main_net, meta_net, gold_loader, silver_loader, valid_loader, test_loader, exp_id=None, label_list=None):
    if args.local_rank <=0: # single GPU or GPU 0
        writer = SummaryWriter(args.logdir + '/' + exp_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # save init point and use later to retrain the classifier
    model_bkp = copy.deepcopy(main_net)
    main_net.cuda()

    if args.local_rank <= 0:
        writer = SummaryWriter(args.logdir + '/' + exp_id)

    net, optimizer, scheduler = setup_training(main_net)        
    #main_net, meta_net, main_opt, optimizer, main_schdlr, scheduler, last_epoch = setup_training(main_net, meta_net, exp_id)

    # //////////////////////// switching on training mode ////////////////////////
    net.train()

    logger.info('=========== Phase 1 Train with nosy data ==============')
    net.train()
    itr = 0
    for epoch in range(args.epochs):
        for i, batch_s in enumerate(silver_loader):
            data_s = [batch_s['input_ids'], batch_s['attention_mask']]
            target_s_ = batch_s['weaklabel']
            data_s, target_s_ = tocuda(data_s), tocuda(target_s_)

            # forward
            output = net(data_s)
            loss = mlc_hard_ce(output, target_s_)

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
    baseline_res = test(net, test_loader, label_list=label_list)
    baseline_acc = baseline_res['overall_f1']

    logger.info('Baseline acc: %.4f' % baseline_acc)
    
    # //////////////////////// estimate C ////////////////////////
    logger.info('=========== Phase 2 Estimate C ==============')    
    if args.method == 'glc':
        probs = []
        golds = []
        
        for batch in gold_loader.loader:
            data = [batch['input_ids'], batch['attention_mask']]
            target = batch['label']
            data, target = tocuda(data), tocuda(target)
            prob = F.softmax(net(data), -1).detach()
            bs, seqlen, _ = prob.size()
            prob = prob.view(bs*seqlen, -1)
            target = target.view(-1)
            # remove IGNORED_INDEX
            target_masked = target[target>=0]
            prob_masked = prob[target>=0, :]
            probs.append(prob)
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
            if len(indices) == 0: # no such weak labels in data, back off to uniform in C_hat
                C_hat[label, :] = 1.0 / num_classes
            else:
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
                target_s_ = batch_s['weaklabel']
                
                data_g = [batch_g['input_ids'], batch_g['attention_mask']]
                target_g = batch_g['label']
                
                data_s, target_s_ = tocuda(data_s), tocuda(target_s_)
                data_g, target_g = tocuda(data_g), tocuda(target_g)

                output_s = net(data_s)
                bs, seqlen, _ = output_s.size()
                output_g = net(data_g)
                
                loss_g = mlc_hard_ce(output_g, target_g, return_raw_loss=True).sum()
                target_s_ = target_s_.view(-1)
                target_s_mask = target_s_[target_s_>=0] # mask
                output_s_mask = output_s.view(bs*seqlen, -1)[target_s_>=0, :]
                pre1 = C_hat.t()[target_s_mask.data]
                pre2 = F.softmax(output_s_mask, -1) * pre1
                loss_s = -(torch.log(pre2.sum(1))).sum(0)

                # backward
                loss = (loss_g + loss_s) / ((target_g!=-100).sum() + (target_s_!=-100).sum())
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
            val_perf = test(net, valid_loader, label_list=label_list)
            test_perf = test(net, test_loader, label_list=label_list)
            val_acc = val_perf['overall_f1']
            test_acc = test_perf['overall_f1']

            if len(val_metric_queue) == args.queue_size:  # keep at most this number of records
                # remove the oldest record
                val_metric_queue.popleft()

            val_metric_queue.append(-val_acc)

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
    test_res = test(net, test_loader, label_list=label_list) # evaluate best params picked from validation

    '''
    if args.local_rank <=0: # single GPU or GPU 0
        writer.add_scalar('test/acc', test_acc, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        writer.add_scalar('test/macro_f1', test_f1, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        writer.add_text('test/cm', test_cm, args.steps) # this test_acc should be roughly the best as it's taken from the best iteration
        # logger.info('Test results: {}'.format(test_perf))
        logger.info('Test {}: {:.4f}'.format(metric_name, test_acc))
    '''
    return test_res

# //////////////////////// run experiments ////////////////////////
def run():
    dataset = args.dataset
    gold_loader, silver_loader, dev_loader, test_loader, num_classes, label_list = get_data_loaders(dataset, args.model_name)
    main_net, meta_net = build_models(dataset, num_classes, args.method, args.model_name, 64)

    results = train_and_test(main_net, meta_net, gold_loader, silver_loader, dev_loader, test_loader, exp_id, label_list=label_list)

    with open('out/' + filename, 'wb') as file:
        pickle.dump(results, file)
    logger.info("Dumped results_ours in file: " + filename)

if __name__ == '__main__':
    run()
