trail=20
transformer_model_name=$2
parallel=1

python -u main.py \
--model_config_path ./config/clean_Ro.yml \
--special_tag ${transformer_model_name}_clean+ab_$trail \
--is_transformer \
--gpus_per_trial $parallel \
--task_name $1 \
--transformer_model_name $transformer_model_name \
--is_debug

python -u main.py \
--model_config_path ./config/clean_weak_most_Ro.yml \
--special_tag ${transformer_model_name}_clean+weak_ab_$trail \
--is_transformer \
--gpus_per_trial $parallel \
--task_name $1 \
--transformer_model_name $transformer_model_name \
--is_debug

python -u main.py \
--model_config_path ./config/weak_most_Ro.yml \
--special_tag ${transformer_model_name}_weak_$trail \
--is_transformer \
--gpus_per_trial $parallel \
--task_name $1 \
--transformer_model_name $transformer_model_name \
--is_debug

python -u main.py \
--model_config_path ./config/weak_snorkel_Ro.yml \
--special_tag ${transformer_model_name}_weak_snorkel_$trail \
--is_transformer \
--gpus_per_trial $parallel \
--task_name $1 \
--transformer_model_name $transformer_model_name \
--is_debug


python -u main.py \
--model_config_path ./config/clean_weak_snorkel_Ro.yml \
--special_tag ${transformer_model_name}_clean+weak_snorkel_ab_$trail \
--is_transformer \
--gpus_per_trial $parallel \
--task_name $1 \
--transformer_model_name $transformer_model_name \
--is_debug




