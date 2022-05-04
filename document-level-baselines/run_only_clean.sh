trail=22

#for task_name in yelp   ; do
##gossipcop
#
## distilbert-base-uncased
##roberta-base
##bert-base-uncased
#for transformer_model_name in   roberta-base   ; do
#  python -u main.py \
#    --model_config_path ./config/clean_Ro_all.yml \
#    --special_tag ${transformer_model_name}_clean_all_$trail \
#    --is_transformer \
#    --gpus_per_trial 1 \
#    --task_name $task_name \
#    --transformer_model_name $transformer_model_name
#done
#
##python -u main.py \
##--model_config_path ./config/clean_all.yml \
##--special_tag lstm_clean_all_$trail \
##--gpus_per_trial 0.2 \
##--task_name $task_name
#
#
#done
#
#for task_name in yelp imdb; do
#for transformer_model_name in   bert-base-uncased   ; do
#  python -u main.py \
#    --model_config_path ./config/clean_Ro_all.yml \
#    --special_tag ${transformer_model_name}_clean_all_$trail \
#    --is_transformer \
#    --gpus_per_trial 1 \
#    --task_name $task_name \
#    --transformer_model_name $transformer_model_name
#done
#done

python -u main.py \
--model_config_path ./config/clean_all.yml \
--special_tag lstm_clean_all_$trail \
--gpus_per_trial 0.2 \
--task_name yelp \
--is_debug