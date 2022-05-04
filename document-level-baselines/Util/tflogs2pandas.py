#!/usr/bin/env python3
import glob
import os
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import click
import pprint
import re
import json
import seaborn as sns; sns.set()
from tqdm import tqdm
import numpy as np
import yaml
from argparse import ArgumentParser

# Extraction function
val_type="Test"

def compare_hyper(dict1, dict2, hyper):
    if type(dict1) is str or type(dict2) is str:
        return False
    dict1_copy = {**dict1}
    dict2_copy = {**dict2}
    dict1_copy.pop(hyper)
    dict2_copy.pop(hyper)
    dict1_tuples = set(list(dict1_copy.items()))
    dict2_tuples = set(list(dict2_copy.items()))
    if len(dict1_tuples.union(dict2_tuples).difference(dict2_tuples)) == 0:
        return True
    else:
        return False

def tflog2pandas(path: str, hyper:str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    # read yml
    model_config = os.path.join("/".join(path.split("/")[:-1]), "hparams.yaml")
    special_tag = ""
    if os.path.exists(model_config):
        with open(model_config) as f1:
            docs = yaml.load_all(f1, Loader=yaml.FullLoader)
            model_config = next(docs)
        try:
            special_tag = "{}_{}".format(hyper, model_config[hyper])
        except:
            th = 1
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            if 'Test' not in tag and val_type not in tag:
                continue
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            # step = list(map(lambda x: x.step, event_list))
            step = np.argsort([i.wall_time for i in event_list])
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data, special_tag, model_config

def file_max_test(df, compare_metric_list):
    metric_df = df[df['metric'].apply(lambda x: all([m in x for m in compare_metric_list]))]
    step = metric_df.iloc[metric_df['value'].argmax(), :]['step']
    return df[df['step'] == step]

def file_max(df, compare_metric_list):
    metric_df = df[df['metric'].apply(lambda x: all([m in x for m in compare_metric_list]))]
    max_value = metric_df['value'].max()
    return df[(df['step'] == 1.) & (df['metric'].apply(lambda x: "Test" in x))], max_value

def many_logs2pandas(event_paths, compare_metric_list=("acc", val_type), hyper=""):
    all_logs = {}
    for path in tqdm(event_paths):
        model_name = path.split("/")[-3]

        if "DEFAULT_" in path:
            version = path.split("/")[-5]
        else:
            version = path.split("/")[-2]
        # version = path.split("/")[5]
        # version = path.split("/")
        log, special_tag, model_config = tflog2pandas(path, hyper)
        if len(special_tag) > 0:
            model_name = special_tag

        if len(hyper) > 0:
            try:
                model_name += "_" + hyper + "_" + str(model_config[hyper])
            except:
                continue

        if log is not None and len(log) > 0:
            try:
                log, interested = file_max(log, compare_metric_list)
            except:
                continue
            if interested is np.nan:
                print("\n[ATTENTION] Cannot load validation data from {} File\n Please check it manually!".format(path))
                interested = 0
            log['ex_name'] = model_name
            log['version'] = version
            if model_name in all_logs.keys():
                interested_dict = all_logs[model_name]
                if interested > interested_dict['metric']:
                    interested_dict['metric'] = interested
                    interested_dict['data'] = log
                    interested_dict['version'] = version
                    all_logs[model_name] = interested_dict
                # interested = interested_df[interested_df['metric'].apply(lambda x:
                #                                      all([m in x for m in compare_metric_list]))]['value'].values[0]
                # if interested < log[log['metric'].apply(lambda x:
                #                                     all([m in x for m in compare_metric_list]))]['value'].values[0]:
                #     all_logs[model_name] = log
            else:
                all_logs[model_name] = {"metric": interested, "data":log, "version": version}

    # pick up the best one
    # all_logs = pd.concat(all_logs.values())
    versions = [i['version'] for i in all_logs.values()]
    all_logs = pd.concat([i['data'] for i in all_logs.values()])
    return all_logs, versions


def df2json(df):
    json_df = df.to_dict("records")
    return_dic = {}
    for i in json_df:
        if i['ex_name'] in return_dic:
            return_dic[i['ex_name']][i['metric']] = i['value']
        else:
            return_dic[i['ex_name']] = {i['metric']: i['value']}
    new_df = pd.DataFrame.from_dict(return_dic, orient='index')
    new_df['ex_name'] = new_df.index
    return new_df



def main(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str, hyper="", tag=""):
    """This is a enhanced version of https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b

    This script exctracts variables from all logs from tensorflow event files ("event*"),
    writes them to Pandas and finally stores them a csv-file or pickle-file including all (readable) runs of the logging directory.

    Example usage:

    # create csv file from all tensorflow logs in provided directory (.) and write it to folder "./converted"
    tflogs2pandas.py . --write-csv --no-write-pkl --o converted

    # creaste csv file from tensorflow logfile only and write into and write it to folder "./converted"
    tflogs2pandas.py tflog.hostname.12345 --write-csv --no-write-pkl --o converted
    """
    pp = pprint.PrettyPrinter(indent=4)
    event_paths = glob.glob(os.path.join(logdir_or_logfile + "/**/", "event*"), recursive=True)
    event_paths = [i for i in event_paths if "version_" in i]
    # event_paths = [i for i in event_paths if "DEFAULT_d711a_00011" in i or "DEFAULT_d711a_00024" in i]
    # event_paths = event_paths[:4]
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        # pp.pprint(event_paths)
        # group by this hyper-parameter
        all_logs, versions = many_logs2pandas(event_paths, hyper=hyper)
        pp.pprint("Head of created dataframe")
        pp.pprint(all_logs)
        pp.pprint(versions)

        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            print("saving to csv file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file_{}_{}.csv".format(val_type, tag))
            # print(out_file)
            # intergrate_data(all_logs,fig_dir=fig_dir)
            all_logs.to_csv(out_file, index=None)
        if write_pkl:
            print("saving to pickle file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.pkl")
            print(out_file)
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")
    return all_logs

def handle_ray(path, output_dir, hyper=""):
    all_logs = main(path, write_csv=True, write_pkl=False, out_dir=output_dir, hyper=hyper)


if __name__ == "__main__":
    args = ArgumentParser()
    #
    args.add_argument("--exp_name", default="mix_meta_datameta_model", type=str)
    args.add_argument("--log_dir", default="/home/yli29/ray_results", type=str)
    args.add_argument("--hyper", default="", type=str)
    args = args.parse_args()
    # path = "/home/yli29/DANoise/tb_logs/{}".format(exp_name)
    path = "{}/{}".format(args.log_dir, args.exp_name)
    # path = "/home/gmou/ray_results/{}/".format(exp_name)
    print("Type is {}".format(val_type))
    output_dir = "../logs/{}_1".format(args.exp_name)
    handle_ray(path, output_dir, args.hyper)


