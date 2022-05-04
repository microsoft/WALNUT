trail=23
python main.py \
--model_config_path ./config/weak_most.yml \
--special_tag lstm_weak-21 \
--gpus_per_trial 0.25 \
--task_name agnews \
--is_debug


python main.py \
--model_config_path ./config/clean.yml \
--special_tag lstm_clean-21 \
--gpus_per_trial 0.2 \
--task_name agnews \
--is_debug

python main.py \
--model_config_path ./config/clean_weak_most.yml \
--special_tag lstm_clean+weak-21 \
--gpus_per_trial 1 \
--task_name agnews \
--is_debug


python main.py \
--model_config_path ./config/weak_snorkel.yml \
--special_tag lstm_weak_snorkel-21 \
--gpus_per_trial 0.5 \
--task_name agnews \
--is_debug


python main.py \
--model_config_path ./config/clean_weak_snorkel.yml \
--special_tag lstm_clean+weak_snorkel-20 \
--gpus_per_trial 0.5 \
--task_name agnews \
--is_debug






