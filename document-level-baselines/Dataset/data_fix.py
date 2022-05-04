from glob import glob
import pandas as pd
import json
import os

import torch
data_path = "/home/kshu/PycharmProjects/MetaReduce-main/data/weak_dataset"
# for file in glob(data_path+"/*.json*"):
#     elements = []
#     with open(file, 'r') as f1:
#         for line in f1.readlines():
#             line = json.loads(line)
#             line['text'] = line['news']
#             del line['news']
#             elements.append(line)
#     os.rename(file, file+".human")
#     with open(file, 'w') as f1:
#         for line in elements:
#             f1.write(json.dumps(line) + "\n")


# for file in glob(data_path+"/*.weak.json"):
#     elements = []
#     with open(file, 'r') as f1:
#         for line in f1.readlines():
#             line = json.loads(line)
#             line['snorkel_agg_label'] = line['snorkel_label']
#             del line['snorkel_label']
#             elements.append(line)
#     os.rename(file, file+".human")
#     with open(file, 'w') as f1:
#         for line in elements:
#             f1.write(json.dumps(line) + "\n")

# with open(data_path + "/gossipcop_train_25_123.human.clean.json", 'r') as f1:
#     elements = []
#     for line in f1.readlines():
#         line = json.loads(line)
#         del line['is_weak']
#         elements.append(line)
#     with open(data_path + "/gossipcop_train_25_123.clean.json", 'w') as f1:
#         for line in elements:
#             f1.write(json.dumps(line) + "\n")

def load_data(data):
    data_interest = {}
    data_interest['text'] = data['labeled']['text'] + data['unlabeled']['text']
    for key, value in data['labeled'].items():
        if type(value) is list:
            data_interest[key] = data['labeled'][key] + data['unlabeled'][key]
        else:
            data_interest[key] = torch.cat([data['labeled'][key], data['unlabeled'][key]])
    return data_interest


yelp_data=load_data(torch.load(data_path + "/../yelp/yelp_organized_nb.pt"))
agnews_data=load_data(torch.load(data_path + "/../agnews/agnews_organized_nb.pt"))
imdb_data=load_data(torch.load(data_path + "/../imdb/imdb_organized_nb.pt"))


for file in glob(data_path + "/*/*train*.weak.json"):
    if "gossipcop" in file:
        continue
    else:
        if "yelp" in file:
            data_interest = yelp_data
        elif "agnews" in file:
            data_interest = agnews_data
        elif "imdb" in file:
            data_interest = imdb_data
        else:
            raise  NotImplementedError
        with open(file, 'r') as f1:
            elements = []
            for line in f1.readlines():
                line = json.loads(line)
                if "is_weak" in line.keys():
                    del line['is_weak']
                line['clean_label'] = data_interest['label'][line['index']].item()
                elements.append(line)
        os.rename(file, file+".human.old")
        with open(file, 'w') as f1:
            for line in elements:
                f1.write(json.dumps(line) + "\n")
