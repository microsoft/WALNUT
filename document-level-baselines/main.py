import pytorch_lightning as pl
import shutil
from pytorch_lightning.loggers import TensorBoardLogger
from Trainer import BaselineTrainer, MetaTrainer
from argparse import ArgumentParser, Namespace
from Util import read_yaml, multiprocess_function, chunkify
from itertools import product
import logging
import traceback
from glob import glob
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import numpy as np
import pickle
from torch.multiprocessing import spawn
import os

label_count = {"agnews": 4,
               "yelp": 2,
               'gossipcop': 2,
               'political': 2,
               'imdb': 2
               }
def nn_clf_method(idx, hparams_chunk):
    cuda_list = list(range(torch.cuda.device_count()))
    try:
        cuda_index = [cuda_list[idx % len(cuda_list)]]
    except:
        cuda_index = 0
    # print("For Thread {}, there are {} visable device, we are using {}".format(idx, len(cuda_list), cuda_index))
    # cuda_index = 1
    hparams_list = hparams_chunk[idx]
    for hparams in hparams_list:
        # try:
        pl.seed_everything(hparams.random_seed)
        model, auto = activate_model_init(hparams)
        logger = TensorBoardLogger('meta_logs', name=hparams.log_name, version=hparams.version_no)
        ckpt = glob(logger.log_dir+"/checkpoints/*.ckpt")
        if os.path.exists(logger.log_dir) and os.path.exists(logger.log_dir+"/checkpoints") is False:
            print("********* FINISH THIS TRAIL**********************")
            continue
        else:
            if len(ckpt) == 0:
                ckpt = None
            else:
                ckpt = ckpt[-1]
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            save_top_k=1,
            mode='max')
        if hparams.is_fp16:
            trainer = pl.Trainer(
                gpus=cuda_index,
                max_epochs=hparams.epochs if hparams.is_debug is False else 5,
                logger=logger,
                callbacks=[checkpoint_callback],
                gradient_clip_val=getattr(hparams, "gradient_clip_val", 0),
                precision=16,
                amp_level="O1",
                resume_from_checkpoint= ckpt
            )
        else:
            trainer = pl.Trainer(
                gpus=cuda_index,
                max_epochs=hparams.epochs if hparams.is_debug is False else 5,
                logger=logger,
                callbacks=[checkpoint_callback],
                gradient_clip_val=getattr(hparams, "gradient_clip_val", 0),
                resume_from_checkpoint=ckpt
            )
        trainer.fit(model)
        # utilize the last model
        trainer.test(model)
        trainer.test()
        shutil.rmtree(logger.log_dir + "/checkpoints")


def activate_model_init(hparams):
    if hparams.model_name == 'baseline':
        # load elmo_data embedding
        elmo_data = pickle.load( open("./glove_embeds/glove.6B.50d.txt.pkl", 'rb'))
        model = BaselineTrainer(hparams, elmo_data)
        auto = True
    else:
        model = MetaTrainer(hparams)
        auto = False
    return model, auto

def hyper_monitor(hparams):
    tune_config, _, general_config = read_yaml(hparams.model_config_path)
    tune_config = tune_config.items()
    train_name = hparams.special_tag + general_config['model_name'] + "_" + hparams.task_name
    # set this based on the dataset
    general_config['class_num'] = label_count[hparams.task_name]
    setattr(hparams, "log_name", train_name)
    parameter_name = [i[0] for i in tune_config]
    parameter_value = [i[1]['grid_search'] for i in tune_config]
    parameter_search = list(product(*parameter_value))
    parameter_search = [Namespace(**{**{name: value for name, value in zip(parameter_name, i)}, **general_config, **vars(hparams), "version_no":index}) for index, i
                        in enumerate(parameter_search)]
    num_process = int(1 / hparams.gpus_per_trial)
    parameter_search_chunk = chunkify(parameter_search, num_process)
    try:
        # multiprocess_function(num_process, function_ref=nn_clf_method, args=(parameter_search_chunk, ))
        spawn(nprocs=num_process, fn=nn_clf_method, args=(parameter_search_chunk, ))
    except Exception as e:
        logging.error(e)
        print(traceback.format_exc())
        print(str(e))


if __name__ == '__main__':
    args = ArgumentParser()
    # args.add_argument("--random_seed", default=123, type=int)
    args.add_argument("--model_config_path", default="./config/clean.yml")
    args.add_argument("--special_tag", default="")
    args.add_argument("--gpus", default=1, type=int)
    args.add_argument("--gpus_per_trial", default=1, type=float)
    args.add_argument("--task_name", type=str, default="agnews")
    args.add_argument("--is_fp16", action='store_true')
    args.add_argument("--is_transformer", action='store_true')
    args.add_argument("--is_overwrite_file", action='store_true')
    args.add_argument("--transformer_model_name", default="roberta-base", type=str)
    args.add_argument("--file_path", default="./data/weak_dataset", type=str)
    args.add_argument("--is_debug", action="store_true")
    args = args.parse_args()
    hyper_monitor(args)

