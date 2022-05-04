# Meta-Learning for Noise Reduce

## Include Encoder: LSTM and Transformers models.
- LSTM model
```shell
export TASK=gossipcop/agnews/imdb/yelp
sh run_lstm.sh $TASK 
```

- Transformer model
```shell
export TASK=gossipcop/agnews/imdb/yelp
export Transformer=roberta-base/distilbert-base-uncased/bert-base-uncased
sh run_lstm.sh $TASK $Transformer 
```



## Dataset:
The dataset is from [here](https://github.com/weakrules/Denoise-multi-weak-sources). 
You should download the "*.pt" file listed in this [Google Drive](https://drive.google.com/drive/folders/1MJe1BJYNPudfmpFxCeHwYqXMx53Kv4h_?usp=sharing).
#### Prepare the dataset:
1. Download the "*.pt" file from aforementioned Google Drive. 
2. Make the directory under **./data/** directory. 
3. Make sure the name of directory is consistent with the name listed in Google Drive. 


## Run Different Tasks:
You should specify the value of **"--task_name agnews"** when run the python scripts. 


## Experiment Result:
You can find the logged tensorborad file under glue_tb_logs directory. 

To view the logs, run the following command:

```shell script
    tensorboard --logdir="./"

```

To check the best validation result, please run the command:
```shell script
  python tflogs2pandas.py --exp_name roberta-base_clean+ab_23 --log_dir ./meta_logs
```