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

def glove_tokenize_text(glove_map, text_list):
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


from collections import Iterable
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item


def label_aggregation(lf_labels, agg_fn_str, class_count, snorkel_ckpt_file=None):
    if type(lf_labels) is torch.Tensor:
        lf_labels = lf_labels.numpy()

    if agg_fn_str == "most_vote":
        majority_count = [sorted(Counter(i).items(), key=lambda x: -x[1]) for i in lf_labels.tolist()]
        #  -1 will be ignored in the cross_entropy function
        final_label = [([j[0] for j in i if j[0] != -1]+[-1])[0] for i in majority_count]
    elif agg_fn_str == "snorkel":
        # snorkel model
        cardinality = class_count
        label_model = LabelModel(verbose=True, device='cuda' if torch.cuda.is_available() else 'cpu', cardinality=cardinality)
        label_model.fit(lf_labels, seed=123, lr=0.0001, n_epochs=100)
        final_label = label_model.predict(lf_labels).tolist()
        label_model.save(snorkel_ckpt_file)
    else:
        raise NotImplementedError
    return final_label

def clean_ratio_split(y, weak_ratio, clean_ratio, random_state):
    index_list = list(range(len(y)))
    clean_index_pool, weak_index = train_test_split(index_list, random_state=random_state, stratify=y, test_size=weak_ratio)
    # attention: clean_ratio = # clean samples / (# clean samples + # weak samples)
    if clean_ratio > 1:
        # ATTENTION: Clean Ratio
        num_clean = int(clean_ratio) * len(set(y.tolist()))
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

def fake_news_clean_select(y, clean_ratio, random_state):
    index_list = list(range(len(y)))
    if clean_ratio > 1:
        # ATTENTION: Clean Ratio
        num_clean = int(clean_ratio) * len(set(y))
    elif clean_ratio == -100:
        print("ATTENTION: We Are USING ALL The CLEAN DATASET")
        num_clean = len(y)
    else:
        raise NotImplementedError

    _, clean_index = train_test_split(index_list, random_state=random_state, stratify=y, test_size=num_clean)

    return clean_index


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
    def __init__(self, hparams, train_status, is_only_clean=None, glove_map=None):
        file_path = hparams.file_path
        if hparams.is_roberta:
            read_file = file_path.replace(".pt", "") + "_" + train_status + "_{}.torch".format(hparams.model_name)
            tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
        else:
            read_file = file_path.replace(".pt", "") + "_" + train_status + "_glove.torch"
            tokenizer = partial(glove_tokenize_text, glove_map)
            # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        if is_only_clean is None:
            is_only_clean = getattr(hparams, "is_only_clean", False)
        self.is_training = train_status == "train"
        if hparams.is_overwrite_file or os.path.exists(read_file) is False:
            # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            data = torch.load(file_path)
            if train_status == "train":
                data_interest = {}
                data_interest['text'] = data['labeled']['text'] + data['unlabeled']['text']
                for key, value in data['labeled'].items():
                    if type(value) is list:
                        data_interest[key] = data['labeled'][key] + data['unlabeled'][key]
                    else:
                        data_interest[key] = torch.cat([data['labeled'][key], data['unlabeled'][key]])
            elif train_status == "test":
                data_interest = data["test"]
            else:
                data_interest = data['validation']

            data_interest = {**tokenizer_text(data_interest['text'], tokenizer), **data_interest}
            torch.save(data_interest, read_file)
        else:
            data_interest = torch.load(read_file)

        is_agg_weak = getattr(hparams, "is_agg_weak", True)
        is_flat = getattr(hparams, "is_flat", False)

        assert is_agg_weak + is_flat != 2, "Don't flat and aggregate weak labels at the same time."

        if train_status == "train":
            index_file = file_path.replace(".pt", "") + "_" + train_status + "_{}".format(
                hparams.clean_ratio) + "_{}".format(hparams.random_seed) + ".index"
            if os.path.exists(index_file):
                index_data = torch.load(index_file)
                clean_index = index_data['clean_index']
                weak_index = index_data['weak_index']
            # clean_data, weak_data
            else:
                weak_index, clean_index = clean_ratio_split(data_interest['label'], hparams.weak_ratio, hparams.clean_ratio, hparams.random_seed)
                assert len(weak_index) > len(clean_index)
                # reject weak samples when they are less lf labels
                lf_labels = data_interest['lf']
                valid_lf_count = torch.sum(torch.where(lf_labels == -1, torch.zeros_like(lf_labels),
                                                   torch.ones_like(lf_labels)), dim=1)
                lf_index = torch.nonzero(valid_lf_count > hparams.n_high_cov).squeeze(1).tolist()
                weak_index = list(set(weak_index).intersection(set(lf_index)))
                lf_count = valid_lf_count[lf_index].tolist()
                index_data = {"weak_index": weak_index, "clean_index": clean_index}
                torch.save(index_data, index_file)
            weak_data = {}
            clean_data = {}
            if len(clean_index) > 0:
                clean_data['input_ids'] = [data_interest['input_ids'][i] for i in clean_index]
                clean_data['attention_mask'] = [data_interest['attention_mask'][i] for i in clean_index]
                clean_data['label'] = data_interest['label'][clean_index].tolist()
                clean_data['lf'] = data_interest['lf'][clean_index].tolist()

            if is_agg_weak:
                # replace the true label with aggergate weak labels
                agg_label = label_aggregation(data_interest['lf'][weak_index], "most_vote",
                                                            class_count=len(torch.unique(data_interest['label'])))
                # replace the clean label with the aggregated weak label
                weak_data['input_ids'] = [data_interest['input_ids'][i] for i in weak_index]
                weak_data['attention_mask'] = [data_interest['attention_mask'][i] for i in weak_index]
                weak_data['label'] = agg_label
                weak_data['lf'] = data_interest['lf'][weak_index].tolist()

            if is_flat:
                weak_data['input_ids'] = list(chain.from_iterable([[data_interest['input_ids'][i]]*count
                                                                   for i, count in zip(weak_index, lf_count)]))
                weak_data['attention_mask'] = list(chain.from_iterable([[data_interest['attention_mask'][i]] * count
                                                                        for i, count in zip(weak_index, lf_count)]))
                weak_data['label'] = list(chain.from_iterable([[data_interest['label'].tolist()[i]] * count
                                                                        for i, count in zip(weak_index, lf_count)]))
                weak_data['lf'] = list(chain.from_iterable([[data_interest['lf'].tolist()[i]] * count
                                                                        for i, count in zip(weak_index, lf_count)]))


            # check whether only clean or only weak
            if getattr(hparams, "is_concat_weak_clean", True):
                if len(clean_data) == 0:
                    # no clean data
                    self.data_interest = weak_data
                else:
                    all_data = {}
                    for key in weak_data:
                        try:
                            all_data[key] = weak_data[key] + clean_data[key]
                        except:
                            th = 1
                    self.data_interest = all_data
            else:
                if is_only_clean:
                    self.data_interest = clean_data
                else:
                    self.data_interest = weak_data
        else:
            self.data_interest = data_interest
        self.label_list = sorted(set(self.data_interest['label']))
        self.num_class = len(set(self.data_interest['label']))

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
        data_parser.add_argument("--is_agg_weak", action="store_true")
        data_parser.add_argument("--is_overwrite_file", action="store_true")
        data_parser.add_argument("--agg_fn_str", choices=['most_vote', 'snorkel'], default="most_vote")
        data_parser.add_argument("--snorkel_ckpt_file", type=str, default="")
        data_parser.add_argument("--n_high_cov", type=int, default=1)
        data_parser.add_argument("--is_roberta", action="store_true")
        data_parser.add_argument("--weak_ratio", default=0.8, type=float)
        data_parser.add_argument("--clean_ratio", default=0, type=float)
        data_parser.add_argument("--random_seed", default=123, type=int)
        return data_parser

class FakeNewsDataset(torch.utils.data.Dataset):
    # only for fake news dataset
    def __init__(self, hparams, train_status, is_only_clean=None, elmo_map=None):

        # dummy code to redirect the file path
        if "_organized_nb.pt" in hparams.file_path:
            hparams.file_path = "/".join(hparams.file_path.split("/")[:-1])
        file_path = hparams.file_path
        if hparams.is_roberta:
            read_file = file_path + "/" + train_status + "_{}.torch".format(hparams.model_name)
            tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
        else:
            read_file = file_path + "/" + train_status + "_glove.torch"
            tokenizer = partial(glove_tokenize_text, elmo_map)
        if is_only_clean is None:
            is_only_clean = getattr(hparams, "is_only_clean", False)
        self.is_training = train_status == "train"
        if hparams.is_overwrite_file or os.path.exists(read_file) is False:
            # weak data
            # data = torch.load(file_path)
            if train_status == "train":
                weak_data = load_fake_news(file_path+"/noise_0.1.csv")
                clean_data = load_fake_news(file_path+"/gold_0.1.csv")
                # TODO: support the fake news dataset.
                # if clean_ratio > 0:
                #     clean_data
                weak_data = {**tokenizer_text(weak_data['news'], tokenizer), **weak_data}
                clean_data = {**tokenizer_text(clean_data['news'], tokenizer), **clean_data}
                data_interest = {"clean": clean_data, "weak": weak_data}
            elif train_status == "test":
                data_interest = load_fake_news(file_path+"/test.csv")
                data_interest = {**tokenizer_text(data_interest['news'], tokenizer), **data_interest}
            else:
                data_interest = load_fake_news(file_path + "/val.csv")
                data_interest = {**tokenizer_text(data_interest['news'], tokenizer), **data_interest}

            # data_interest = {**tokenizer_text(data_interest['text'], tokenizer), **data_interest}
            torch.save(data_interest, read_file)
        else:
            data_interest = torch.load(read_file)

        is_agg_weak = getattr(hparams, "is_agg_weak", True)
        is_flat = getattr(hparams, "is_flat", False)

        assert is_agg_weak + is_flat != 2, "Don't flat and aggregate weak labels at the same time."

        if train_status == "train":
            clean_data = data_interest['clean']
            # placeholder
            clean_data['lf'] = [[-1, -1, -1]] * len(clean_data['label'])
            weak_data = data_interest['weak']
            index_file = os.path.join(file_path, file_path.split("/")[-1] + "_" + train_status + "_{}".format(
                hparams.clean_ratio) + "_{}".format(hparams.random_seed) + ".index")
            if os.path.exists(index_file):
                index_data = torch.load(index_file)
                clean_index = index_data['clean_index']
            # clean_data, weak_data
            else:
                print("ATTENTION YOU ARE CREATEING NEW INDEX FILES.")
                clean_index = fake_news_clean_select(clean_data['label'], hparams.clean_ratio, hparams.random_seed)
                assert len(clean_index) - len(set(clean_index)) == 0
                torch.save({"clean_index":clean_index}, index_file)
            # reorgnaize the clean data
            clean_data = {key:[clean_data[key][i] for i in clean_index] for key in clean_data}

            if is_agg_weak:
                # replace the true label with aggergate weak labels
                weak_labels = np.array([list(map(int, weak_data[key])) for key in weak_data if "_label" in key]).T
                agg_label = label_aggregation(weak_labels, "most_vote",
                                                            class_count=2)
                # replace the clean label with the aggregated weak label
                weak_data['lf'] = weak_labels.tolist()
                weak_data['label'] = agg_label

            # check whether only clean or only weak
            if getattr(hparams, "is_concat_weak_clean", True):
                if len(clean_data) == 0:
                    # no clean data
                    self.data_interest = weak_data
                else:
                    all_data = {}
                    print(weak_data.keys())
                    for key in weak_data:
                        if key in clean_data:
                            all_data[key] = weak_data[key] + clean_data[key]

                    self.data_interest = all_data
            else:
                if is_only_clean:
                    self.data_interest = clean_data
                else:
                    self.data_interest = weak_data
        else:
            # placeholder
            data_interest['lf'] = [[-1, -1, -1]] * len(data_interest['label'])
            self.data_interest = data_interest
        print("STATUS {} ".format(train_status) + "For {}".format(self.data_interest.keys()))
        self.label_list = sorted(set(self.data_interest['label']))
        self.num_class = len(set(self.data_interest['label']))
    def __len__(self):
        return len(self.data_interest['input_ids'])

    def __getitem__(self, item):
        # ATTENTION: 256 tokens take much longer time than expectation.
        input_ids = torch.tensor(self.data_interest['input_ids'][item], dtype=torch.long)
        attention_mask = torch.tensor(self.data_interest['attention_mask'][item])
        label = torch.tensor(self.data_interest['label'][item], dtype=torch.long)
        weak_labels = torch.tensor(self.data_interest['lf'][item], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label, "weaklabel": weak_labels}
        # return input_ids, attention_mask, label

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        data_parser.add_argument("--file_path", type=str)
        data_parser.add_argument("--is_agg_weak", action="store_true")
        data_parser.add_argument("--is_overwrite_file", action="store_true")
        data_parser.add_argument("--agg_fn_str", choices=['most_vote', 'snorkel'], default="most_vote")
        data_parser.add_argument("--snorkel_ckpt_file", type=str, default="")
        data_parser.add_argument("--is_concat_weak_clean", action="store_true")
        data_parser.add_argument("--is_only_clean", action="store_true")
        data_parser.add_argument("--is_transformer", action="store_true")
        data_parser.add_argument("--transformer_model_name", default="roberta-base", type=str)
        return data_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = NoiseDataset.add_model_specific_args(parser)
    args = parser.parse_args()
    for task in ['yelp', 'imdb', 'agnews']:
        for random_seed in [123, 456, 789, 10, 49]:
            for clean_ratio in [5, 10, 20, 25, 50, 100, 200]:
                setattr(args, "random_seed", random_seed)
                setattr(args, "clean_ratio", clean_ratio)
                file_path = "../data/{}/{}_organized_nb.pt".format(task, task)
                setattr(args, "file_path", file_path)
                dataset = NoiseDataset(hparams=args, train_status="train")
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        # for batch in dataloader:
        #     print(batch)


