import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
from torch.utils.data import dataset
from transformers import RobertaTokenizer, DistilBertTokenizer
import os
from collections import Counter
#from snorkel.labeling.model import LabelModel
import argparse
from itertools import chain
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
import re

import transformers
from functools import partial

#try:
#import texthero as hero
#import spacy
#nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# ATTENTION: define the max_length
def tokenizer_text(text_list, tokenizer, max_length=200):
    try:
        encode = tokenizer(text_list, padding='max_length', max_length=max_length, truncation=True)
    except:
        encode = tokenizer(text_list)
    return encode

#class bilstmTokenizer(transformers.PreTrainedTokenizerFast):
class bilstmTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self, glove_map=None):
        super(bilstmTokenizer, self).__init__()
        self.glove_map = glove_map
        self.tokenizer_fn = partial(glove_tokenize_text, self.glove_map)
        self.pad_token = 0

    def encode(self, x, **kwargs):
        import pdb; pdb.set_trace()
        return x

    def save_vocabulary(self, save_directory, **kwargs):
        return ('test',)

    def pad(self, encoded_inputs, labels=None, **kwargs):
        # adapted from source: https://huggingface.co/transformers/_modules/transformers/tokenization_utils_base.html#PreTrainedTokenizerBase
        #if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
        encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}
        return encoded_inputs

    def glove_tokenize(self, txt, labels, label_name='labels'):
        return self.tokenizer_fn(txt, label_list=labels, label_name=label_name)

def glove_tokenize_text_old(glove_map, text_list):
    def lemmatize(texts):
        output = []
        for doc in texts:
            s = [token.lemma_ for token in nlp(doc)]
            output.append(' '.join(s))
        return output
    max_length = 200
    cleaned_text = lemmatize(hero.clean(pd.Series(text_list)))
    tokenized_text = []
    for i in cleaned_text:
        sentence = []
        for w in i.split(" "):
            # 0 for unknown words
            word = glove_map.get(w, 0)
            if word != 0:
                sentence.append(word)
        sentence = sentence[:max_length] + [0] * (max_length - len(sentence))
        tokenized_text.append(sentence)
    attention_mask = [[1 if j != 0 else 0 for j in i] for i in tokenized_text]
    output = {"input_ids":tokenized_text, "attention_mask":attention_mask}
    return output

def glove_tokenize_text(glove_map, text_list, label_list, label_name='labels'):
    max_length = 200
    #import pdb; pdb.set_trace()
    text_list = [text_list]
    label_list = [label_list]
    tokenized_text = []
    aligned_labels = []

    for i,tokens in enumerate(text_list):
        sentence = []
        labels = []
        for j,w in enumerate(tokens):
            # 0 for unknown words
            word = glove_map.get(w, 0)
            if word != 0:
                sentence.append(word)
                labels.append(label_list[i][j])
        sentence = sentence[:max_length] + [0] * (max_length - len(sentence))
        labels = labels[:max_length] + [-100] * (max_length - len(labels))
        assert len(sentence) == len(labels), "len(sentence)={} != len(labels)={}".format(len(sentence), len(labels))
        tokenized_text.append(sentence)
        aligned_labels.append(labels)
    attention_mask = [[1 if j != 0 else 0 for j in i] for i in tokenized_text]
    output = {"input_ids":tokenized_text, "attention_mask":attention_mask, label_name: aligned_labels}
    return output

class LSTM_text(nn.Module):
    def __init__(self, num_classes, h_dim, embed_weight: np.ndarray, is_ner):
        super(LSTM_text, self).__init__()
        C = num_classes
        self.num_labels = num_classes
        # one for padded word
        V = embed_weight.shape[0]
        D = embed_weight.shape[1]

        self.embed = nn.Embedding(V, D, padding_idx=0).from_pretrained(torch.FloatTensor(embed_weight), freeze=False)

        self.lstm = nn.LSTM(D, hidden_size=h_dim//2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.head_count = 1
        # self.final_prediction = nn.Linear(self.args.hidden_size, C*self.head_count)
        # self.fc2 = nn.Linear(len(Ks) * Co, self.args.hidden_size)

        self.final_prediction = nn.Linear(h_dim, C)

        self.class_num = num_classes

        # self.loss_no_reduction = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
        def myCrossEntropyLoss(outputs, labels):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            """
            import pdb; pdb.set_trace()
            labels = labels.view(-1)
            mask = (labels >= 0).float()
            labels[labels < 0] = 0

            outputs = outputs.view(-1, outputs.shape[-1])
            outputs = outputs[range(outputs.shape[0]), labels] * mask
            # outputs = outputs * mask

            # pick the values corresponding to labels and multiply by mask
            #
            """
            # loss = loss_fct(outputs, labels)
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            return loss
        self.loss_reduction = myCrossEntropyLoss
        #self.loss_reduction = nn.CrossEntropyLoss(ignore_index=-1)
        #self.loss_reduction = self.loss_fn

        self.pad_idx = 0
        self.is_ner = is_ner
        self.pooling_fn = "mean"
        self.pad_len = 200


    def forward(self, input_ids, attention_mask=None, labels=None, return_h=False):
        x=input_ids
        #if type(x) is list or type(x) is tuple: # giannis: also consider type=tuple
        if type(x) is tuple:
            x, attention_mask = x
            x_length = torch.sum(attention_mask, dim=1)
        else:
            #import pdb; pdb.set_trace()
            x_length = torch.sum(torch.tensor(x) != self.pad_idx, dim=1)
        #import pdb; pdb.set_trace()
        device=torch.device('cuda:0') # giannis FIXME: provide the right device
        x = torch.tensor(x).to(device)
        embed_x = self.embed(x)
        sorted_seq_len, permIdx = x_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_embed_x = embed_x[permIdx]

        sorted_seq_len[sorted_seq_len==0] = 1 # FIXME: fix issue of empty sequences (len==0)
        packed_words = pack_padded_sequence(sorted_embed_x, sorted_seq_len.cpu(), True)

        self.lstm.flatten_parameters()
        lstm_out, (hidden, cell) = self.lstm(packed_words, None)

        if self.is_ner:
            #lstm_out = pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.pad_len) # giannis: fixed bug
            feature_out = self.dropout(lstm_out)
        else:
            hidden = hidden.transpose(1, 0)
            # take the last layer hidden state.
            hidden = hidden[:, -2:].reshape(hidden.shape[0], -1)
            feature_out = self.dropout(hidden)
        # reorder the elements
        feature_out = feature_out[recover_idx]
        logit = self.final_prediction(feature_out)

        # will also add loss: https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForTokenClassification
        # import pdb; pdb.set_trace()
        if labels is not None:
            # loss = self.loss_fn(logit, labels)
            loss = self.loss_reduction(logit, labels)
        #output = (feature_out.detach(), logit,)
        #output = (logit)
        #output += (loss,)
        #output = (logit, loss)
        output = (loss, logit)
        return output

        if return_h:
            return logit, feature_out
        else:
            return logit
        # logit = logit.reshape(-1, self.class_num)
        # y = y.reshape(-1, )
        # output = (feature_out.detach(), logit,)
        # if y is not None:
        #     if is_reduction:
        #         loss = self.loss_reduction(logit, y)
        #     else:
        #         loss = self.loss_no_reduction(logit, y)
        #     output += (loss, )
        # return output

    def loss_fn(self, outputs, labels):

        #reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.view(-1)

        #mask out 'PAD' tokens
        mask = (labels >= 0).float()

        #the number of tokens is the sum of elements in mask
        #num_tokens = int(torch.sum(mask).data[0])
        num_tokens = int(torch.sum(mask).data)

        outputs = outputs.view(-1, outputs.shape[-1])
        labels[labels<0] = 0
        #pick the values corresponding to labels and multiply by mask
        outputs = outputs[range(outputs.shape[0]), labels]*mask

        #cross entropy loss for all non 'PAD' tokens
        return -torch.sum(outputs)/num_tokens
