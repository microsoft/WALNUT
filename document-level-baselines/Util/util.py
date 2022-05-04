from sklearn.metrics import accuracy_score, f1_score, \
    recall_score, precision_score, confusion_matrix

import numpy as np
import json
import logging
from multiprocessing import Process
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

def load_json_file(file_name):
    data = []
    with open(file_name, 'r') as f1:
        for line in f1.readlines():
            data.append(json.loads(line))
    return data


def seperate_score(predict_Y, test_Y, average):
    f1 = f1_score(y_pred=predict_Y, y_true=test_Y,average=average)
    cf_m = ",".join(map(str, confusion_matrix(y_pred=predict_Y, y_true=test_Y).ravel()))
    precision = precision_score(y_true=test_Y, y_pred=predict_Y, average=average)
    recall = recall_score(y_true=test_Y, y_pred=predict_Y, average=average)
    return f1, precision, recall

def evaluation(logits, test_Y):
    is_clf = type(test_Y[0]) == np.int64
    preds = np.argmax(logits, axis=1) if is_clf else np.squeeze(logits)
    acc = accuracy_score(y_pred=preds, y_true=test_Y)
    cf_m = ",".join(map(str, confusion_matrix(y_pred=preds, y_true=test_Y).ravel()))
    f1_macro = f1_score(y_true=test_Y, y_pred=preds, average="macro")
    f1_micro = f1_score(y_true=test_Y, y_pred=preds, average="micro")
    return {"acc":acc, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'cf_m':cf_m}


def multiprocess_function(num_process, function_ref, args):
    jobs = []
    logging.info("Multiprocessing function %s started..." % function_ref.__name__)
    print("Multiprocessing function %s started..." % function_ref.__name__)

    for idx in range(num_process):
        process = Process(target=function_ref, args=(idx,) + args)
        process.daemon = True
        jobs.append(process)
        process.start()

    for i in range(num_process):
        jobs[i].join()

    logging.info("Multiprocessing function %s completed..." % function_ref.__name__)
    print("Multiprocessing function %s completed..." % function_ref.__name__)

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def rename_label(list_dict, task_name):
    return_list = []
    for i in list_dict:
        if "class" in i:
            i['label'] = i['class']
        if task_name == "cola" and "text" in i:
            i['sentence'] = i['text']

        return_list.append(i)
    return return_list
def transformer_load_dataset(path, task_name, aug_path=None, use_test=False, **kwargs):
    train_aug = rename_label(load_json_file(aug_path.format(task_name)), task_name)
    score_key = [i for i in train_aug[0].keys() if 'score' in i.lower() or 'readability' in i.lower()]
    read_score_interval = list(map(float, kwargs['readability'].split(",")))
    train_aug = [i for i in train_aug if all([read_score_interval[1] > i[key] > read_score_interval[0]
                                              for key in score_key])]
    train_raw = rename_label(load_json_file(path.format(task_name, "train")), task_name)
    train_all = pd.DataFrame(train_raw + train_aug)
    train_all = Dataset.from_pandas(train_all)
    val = load_dataset('glue', task_name)['validation']
    data_dict = {"train":train_all, 'validation': val}
    if use_test:
        test_data = pd.DataFrame(rename_label(load_json_file(path.format(task_name, "test")), task_name))
        test_data = Dataset.from_pandas(test_data)
        data_dict['test'] = test_data
    data = DatasetDict(data_dict)
    return data





