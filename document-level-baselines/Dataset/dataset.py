import pandas
import torch
from torch.utils.data import dataset
from transformers import AutoTokenizer
import os
from collections import Counter
from snorkel.labeling.model import LabelModel
import argparse
from itertools import chain
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
import re
try:
    import texthero as hero
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except:
    print("This will not support LSTM module")

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 


# ATTENTION: define the max_length
def tokenizer_text(text_list, tokenizer, max_length=200):
    try:
        encode = tokenizer(text_list, padding='max_length', max_length=max_length, truncation=True)
    except:
        encode = tokenizer(text_list)
    return encode

def glove_tokenize_text(elmo_map, text_list):
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
            word = elmo_map.get(w, 0)
            if word != 0:
                sentence.append(word)
        sentence = sentence[:max_length] + [0] * (max_length - len(sentence))
        tokenized_text.append(sentence)
    attention_mask = [[1 if j != 0 else 0 for j in i] for i in tokenized_text]
    output = {"input_ids":tokenized_text, "attention_mask":attention_mask}

    return output


from collections import Iterable
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item

def load_json_file(file_path):
    elements = []
    with open(file_path, 'r') as f1:
        for line in f1.readlines():
            elements.append(json.loads(line))
    return elements

def load_fake_news(file_path):
    data = pd.read_csv(file_path)
    data.dropna(how='any', inplace=True)
    data = json.loads(data.to_json(orient='columns'))
    new_data = {}
    for key, value in data.items():
        value = list(value.values())
        new_data[key] = value
    return new_data

from functools import partial
class NoiseDataset(torch.utils.data.Dataset):
    # only for agnews/yelp/imdb datasets
    def __init__(self, hparams, train_status, is_only_clean=None, elmo_map=None):
        if is_only_clean is None:
            is_only_clean = getattr(hparams, "is_only_clean", False)
        file_path = hparams.file_path
        if train_status == "train":
            weak_data_path = file_path + f"/{hparams.task_name}/{hparams.task_name}_train_{hparams.random_seed}.weak.json"
            clean_data_path = file_path + f"/{hparams.task_name}/{hparams.task_name}_train_{hparams.clean_ratio}_{hparams.random_seed}.clean.json"
            weak_df = pd.DataFrame(load_json_file(weak_data_path))
            agg_fn_str = getattr(hparams, "agg_fn_str", "most_vote")
            if agg_fn_str == "most_vote":
                weak_df['label'] = weak_df['major_agg_label']
            elif agg_fn_str == "snorkel":
                weak_df['label'] = weak_df['snorkel_agg_label']
            else:
                raise NotImplementedError

            is_not_only_weak = os.path.exists(clean_data_path)
            if is_not_only_weak:
                clean_df = pd.DataFrame(load_json_file(clean_data_path))

            if getattr(hparams, "is_concat_weak_clean", False):
                interested_df = pd.concat([weak_df, clean_df])
            else:
                if is_only_clean and is_not_only_weak:
                    interested_df = clean_df
                else:
                    interested_df = weak_df
        else:
            data_path = file_path + f"/{hparams.task_name}/{hparams.task_name}_{train_status}.clean.json"
            interested_df = pd.DataFrame(load_json_file(data_path))
        if hparams.is_debug:
            interested_df = interested_df[:100]
        if hparams.is_transformer:
            tokenizer = AutoTokenizer.from_pretrained(hparams.transformer_model_name)
        else:
            tokenizer = partial(glove_tokenize_text, elmo_map)

        tokenized_text = tokenizer_text(tokenizer=tokenizer, text_list=interested_df['text'].tolist())
        interested_df['input_ids'] = tokenized_text['input_ids']
        interested_df['attention_mask'] = tokenized_text['attention_mask']
        print(interested_df.columns)
        self.data_interest = interested_df.to_dict(orient='records')
        try:
            labels = [i['label'] for i in self.data_interest]
        except:
            th = 1
        self.label_list = sorted(set(labels))
        self.num_class = len(set(labels))
        print("There are {} classes!".format(self.num_class))

    def __len__(self):
        return len(self.data_interest)

    def __getitem__(self, item):
        input_ids = torch.tensor(self.data_interest[item]['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(self.data_interest[item]['attention_mask'])
        label = torch.tensor(self.data_interest[item]['label'], dtype=torch.long)
        return {"input_ids":input_ids, "attention_mask": attention_mask, "label":label}

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # data_parser.add_argument("--file_path", type=str)
        data_parser.add_argument("--is_agg_weak", action="store_true")
        data_parser.add_argument("--is_overwrite_file", action="store_true")
        data_parser.add_argument("--agg_fn_str", choices=['most_vote', 'snorkel'], default="most_vote")
        data_parser.add_argument("--snorkel_ckpt_file", type=str, default="")
        data_parser.add_argument("--n_high_cov", type=int, default=1)
        data_parser.add_argument("--is_transformer", action="store_true")
        data_parser.add_argument("--transformer_model_name", default="roberta-base")
        data_parser.add_argument("--weak_ratio", default=0.8, type=float)
        data_parser.add_argument("--clean_ratio", default=0, type=float)
        data_parser.add_argument("--random_seed", default=123, type=int)
        return data_parser




