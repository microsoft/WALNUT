import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split as tts
from datasets import DatasetDict
from collections import Counter

def load_label_names(class_name_fpath):
    with open(class_name_fpath, 'r') as f:
        label_names = f.readlines()[0]
    label_names = label_names.replace('[', '').replace(']', '').split(',')
    label_names = [x.strip().replace('\'', '') for x in label_names]
    return label_names

# stratified split
def token_level_stratified_sampling(dataset, label_type='ner_tags', train_size=10, seed=42, label_names=None, shuffle=True,
                                    precomputed_indices_file=None, create_clean_weak_splits=False):
    """
    Stratified sampling for token-level classification
    We compute a single label for each sequence based on the distinct tags appearing in the sequence
    Note: this approach ignores counts of tags in a sequence
    # Input:
    :param dataset: a huggingface dataset
    :param label_type:
    :param train_size:
    :param seed:
    :param label_names:
    :param shuffle:
    :param precomputed_indices_file:
    :return: a huggingface DatasetDict with a "train" (clean) and "test" (weak) Dataset Object
    """

    inds = np.arange(len(dataset))  # indices: order of data, not their 'id'
    # inds = dataset['id']
    labels = dataset[label_type]
    distinct_labels = [tuple(sorted(set([label_names[y] for y in ys]))) for ys in labels]
    distinct_label_strings = ['_'.join(y) for y in distinct_labels]
    c = Counter(distinct_label_strings)
    distinct_label_strings = [l if c[l] > 1 else 'lowfreq' for l in distinct_label_strings]  # ignore combinations with freq = 1 to avoid errors in train_test_split
    if len(distinct_label_strings) > train_size:
        # I need to make sure that the number of labels for stratified sampling is <= number of data
        most_common = [x[0] for x in c.most_common()[:train_size-1]]
        distinct_label_strings = [l if l in most_common else 'lowfreq2' for l in distinct_label_strings]

    if precomputed_indices_file:
        # Load pre-computed indices from a file. If indices do not exist, then create them and update the file
        if not os.path.exists(precomputed_indices_file) and create_clean_weak_splits:
            test = {}
            joblib.dump(test, precomputed_indices_file)
        assert os.path.exists(precomputed_indices_file), "file with precomputed indices does not exist. use --create_clean_weak_splits if it's the first time you are running experiments"
        inds_dict = joblib.load(precomputed_indices_file)
        index_str = "seed{}".format(seed)
        if not index_str in inds_dict:
            print("Creating new entry for seed={} in {}".format(seed, precomputed_indices_file))
            clean_inds, weak_inds = tts(inds, stratify=distinct_label_strings, train_size=train_size, random_state=seed, shuffle=shuffle)
            inds_dict[index_str] = {"clean": clean_inds, "weak": weak_inds}
            joblib.dump(inds_dict, precomputed_indices_file)
        else:
            clean_inds, weak_inds = inds_dict[index_str]["clean"], inds_dict[index_str]["weak"]
    else:
        # Compute indices on the fly
        clean_inds, weak_inds = tts(inds, stratify=distinct_label_strings, train_size=train_size, random_state=seed, shuffle=shuffle)

    split = DatasetDict()
    split['train'] = dataset.select(clean_inds)
    split['test'] = dataset.select(weak_inds)
    assert len(set(split['train']['id']) & set(split['test']['id'])) == 0, "issue with splitting data"

    return split


"""
BiLSTM util functions
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import transformers
from functools import partial


def tokenizer_text(text_list, tokenizer, max_length=200):
    try:
        encode = tokenizer(text_list, padding='max_length', max_length=max_length, truncation=True)
    except:
        encode = tokenizer(text_list)
    return encode


class bilstmTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self, glove_map=None):
        super(bilstmTokenizer, self).__init__()
        self.glove_map = glove_map
        self.tokenizer_fn = partial(glove_tokenize_text, self.glove_map)
        self.pad_token = 0

    def encode(self, x, **kwargs):
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


def glove_tokenize_text(glove_map, text_list, label_list, label_name='labels'):
    max_length = 200
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
        self.final_prediction = nn.Linear(h_dim, C)

        self.class_num = num_classes
        self.device = 'cuda:0' # FIXME: provide the right device

        def TokenCrossEntropyLoss(outputs, labels):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(outputs.view(-1, self.num_labels), labels.view(-1))
            return loss
        self.loss_reduction = TokenCrossEntropyLoss

        self.pad_idx = 0
        self.is_ner = is_ner
        self.pooling_fn = "mean"
        self.pad_len = 200


    def forward(self, input_ids, attention_mask=None, labels=None, return_h=False):
        x=input_ids
        if type(x) is tuple:
            x, attention_mask = x
            x_length = torch.sum(attention_mask, dim=1)
        else:
            x_length = torch.sum(torch.tensor(x) != self.pad_idx, dim=1)

        device=torch.device(self.device)
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
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.pad_len) 
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
        if labels is not None:
            loss = self.loss_reduction(logit, labels)
        output = (loss, logit)
        return output

        if return_h:
            return logit, feature_out
        else:
            return logit
