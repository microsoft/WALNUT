import torch
from torch.utils.data import dataset
from transformers import RobertaTokenizer, DistilBertTokenizer
import os
from collections import Counter
import argparse
from itertools import chain
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np

# ATTENTION: define the max_length
def tokenizer_text(text_list, tokenizer, max_length=200):
    encode = tokenizer(text_list, padding='max_length', max_length=max_length, truncation=True)
    return encode

from collections import Iterable
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item


def label_aggregation(lf_labels):
    if type(lf_labels) is torch.Tensor:
        lf_labels = lf_labels.numpy()
    majority_count = [sorted(Counter(i).items(), key=lambda x: -x[1]) for i in lf_labels.tolist()]
    #  -1 will be ignored in the cross_entropy function
    final_label = [([j[0] for j in i if j[0] != -1]+[-1])[0] for i in majority_count]
    return final_label

def clean_ratio_split(y, weak_ratio, clean_ratio, random_state=123):
    index_list = list(range(len(y)))
    clean_index_pool, weak_index = train_test_split(index_list, random_state=random_state, stratify=y, test_size=weak_ratio)
    # attention: clean_ratio = # clean samples / (# clean samples + # weak samples)
    if clean_ratio > 1:
        num_clean = int(clean_ratio)
    else:
        num_clean = int(clean_ratio * len(weak_index) / (1 - clean_ratio))
    # assert num_clean / len(weak_index) - clean_ratio < 0.0001
    if clean_ratio > 0:

        _, clean_index = train_test_split(clean_index_pool, random_state=random_state, stratify=[y[i] for i in clean_index_pool],
                                          test_size=num_clean)
        return weak_index, clean_index
    else:
        # just the place holder for clean data
        return weak_index, []

def load_fake_news(file_path):
    data = pd.read_csv(file_path)
    data.dropna(how='any', inplace=True)
    data = json.loads(data.to_json(orient='columns'))
    new_data = {}
    for key, value in data.items():
        value = list(value.values())
        new_data[key] = value
    return new_data


class NoiseDataset(torch.utils.data.Dataset):
    # only for agnews/yelp/imdb datasets
    def __init__(self, hparams, train_status, is_only_clean=None):
        file_path = hparams.file_path
        if hparams.is_transformer:
            read_file = file_path.replace(".pt", "") + "_" + train_status + ".torch" + ".add"
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            read_file = file_path.replace(".pt", "") + "_" + train_status + "_dis.torch" + ".add"
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        if is_only_clean is None:
            is_only_clean = getattr(hparams, "is_only_clean", False)
        self.is_training = train_status == "train"

        data_interest = torch.load(read_file)

        if train_status == "train":
            # clean_data, weak_data
            self.num_class = len(set(data_interest['label'].tolist()))
            self.label_list = list(range(self.num_class))
            weak_index, clean_index = clean_ratio_split(data_interest['label'], hparams.weak_ratio, hparams.clean_ratio)
            assert len(weak_index) > len(clean_index)
            # reject weak samples when they are less lf labels
            lf_labels = data_interest['lf']
            valid_lf_count = torch.sum(torch.where(lf_labels == -1, torch.zeros_like(lf_labels),
                                                   torch.ones_like(lf_labels)), dim=1)
            lf_index = torch.nonzero(valid_lf_count > hparams.n_high_cov).squeeze(1).tolist()
            weak_index = list(set(weak_index).intersection(set(lf_index)))
            lf_count = valid_lf_count[lf_index].tolist()
            weak_data = {}
            clean_data = {}
            if len(clean_index) > 0:
                clean_data['input_ids'] = [data_interest['input_ids'][i] for i in clean_index]
                clean_data['attention_mask'] = [data_interest['attention_mask'][i] for i in clean_index]
                clean_data['label'] = data_interest['label'][clean_index].tolist()
                clean_data['lf'] = data_interest['lf'][clean_index].tolist()


            # replace the true label with aggergate weak labels
            agg_label = label_aggregation(data_interest['lf'][weak_index])
            # replace the clean label with the aggregated weak label
            weak_data['input_ids'] = [data_interest['input_ids'][i] for i in weak_index]
            weak_data['attention_mask'] = [data_interest['attention_mask'][i] for i in weak_index]
            weak_data['label'] = agg_label
            weak_data['lf'] = data_interest['lf'][weak_index].tolist()

            if is_only_clean:
                self.data_interest = clean_data
            else:
                self.data_interest = weak_data
        else:
            self.data_interest = data_interest

    def __len__(self):
        return len(self.data_interest['input_ids'])

    def __getitem__(self, item):
        input_ids = torch.tensor(self.data_interest['input_ids'][item], dtype=torch.long)
        attention_mask = torch.tensor(self.data_interest['attention_mask'][item])
        label = torch.tensor(self.data_interest['label'][item], dtype=torch.long)
        weak_labels = torch.tensor(self.data_interest['lf'][item], dtype=torch.long)
        return {"input_ids":input_ids, "attention_mask": attention_mask, "label":label, "weaklabel":weak_labels}

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        data_parser.add_argument("--file_path", type=str, required=True)
        data_parser.add_argument("--n_high_cov", type=int, default=2)
        data_parser.add_argument("--is_transformer", action="store_true")
        data_parser.add_argument("--weak_ratio", default=0.8, type=float)
        data_parser.add_argument("--clean_ratio", default=0, type=float)
        data_parser.add_argument("--random_state", default=123, type=float)
        return data_parser


class FakeNewsDataset(torch.utils.data.Dataset):
    # only for fake news dataset
    def __init__(self, hparams, train_status, is_only_clean=None):
        file_path = hparams.file_path
        # dummy code to redirect the file path
        if hparams.is_transformer:
            read_file = file_path + train_status + ".torch"
        else:
            read_file = file_path + train_status + "_dis.torch"
        if is_only_clean is None:
            is_only_clean = getattr(hparams, "is_only_clean", False)
        self.is_training = train_status == "train"

        data_interest = torch.load(read_file)


        if train_status == "train":
            clean_data = data_interest['clean']
            self.num_class = len(set(clean_data['label']))
            self.label_list = list(range(self.num_class))
            # placeholder
            clean_data['lf'] = [[-1, -1, -1]] * len(clean_data['label'])
            weak_data = data_interest['weak']

            weak_labels = np.array([list(map(int, weak_data[key])) for key in weak_data if "_label" in key]).T
            agg_label = label_aggregation(weak_labels)
            # replace the clean label with the aggregated weak label
            weak_data['lf'] = weak_labels.tolist()
            weak_data['label'] = agg_label

            # check whether only clean or only weak

            if is_only_clean:
                self.data_interest = clean_data
            else:
                self.data_interest = weak_data
        else:
            # placeholder
            data_interest['lf'] = [[-1, -1, -1]] * len(data_interest['label'])
            self.data_interest = data_interest

    def __len__(self):
        return len(self.data_interest['input_ids'])

    def __getitem__(self, item):
        # ATTENTION: 256 tokens take much longer time than expectation.
        input_ids = torch.tensor(self.data_interest['input_ids'][item][:256], dtype=torch.long)
        attention_mask = torch.tensor(self.data_interest['attention_mask'][item][:256])
        label = torch.tensor(self.data_interest['label'][item], dtype=torch.long)
        weak_labels = torch.tensor(self.data_interest['lf'][item], dtype=torch.long)
        return {"input_ids":input_ids, "attention_mask": attention_mask, "label":label, "weaklabel":weak_labels}

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        data_parser.add_argument("--file_path", type=str, required=True)
        data_parser.add_argument("--n_high_cov", type=int, default=2)
        data_parser.add_argument("--is_transformer", action="store_true")
        data_parser.add_argument("--weak_ratio", default=0.8, type=float)
        data_parser.add_argument("--clean_ratio", default=0, type=float)
        data_parser.add_argument("--random_state", default=123, type=float)
        return data_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = NoiseDataset.add_model_specific_args(parser)
    args = parser.parse_args()
    for train_status in ['train','val', 'test']:
        dataset = NoiseDataset(hparams=args, train_status=train_status)
        print("There are {} samples".format(len(dataset)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        for batch in dataloader:
            print(batch)


