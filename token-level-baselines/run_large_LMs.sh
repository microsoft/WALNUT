#!/bin/bash

# Running larger LMs: bert-large-uncased and roberta-large

# BERT Large
baselogdir="$(date +'%h%d_%H-%M')_BERT_LARGE_Clean_100"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --base_model bert-large-uncased --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_BERT_LARGE_Clean"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --walnut_mode --base_model bert-large-uncased --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_BERT_LARGE_Weak"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --label_name mv --walnut_mode --base_model bert-large-uncased --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_BERT_LARGE_Weak_Snorkel"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --label_name nb --walnut_mode --base_model bert-large-uncased --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_BERT_LARGE_Clean_Plus_Weak"
for seed in 0 20 7 1993 42; do
  for dataset in NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --clean_plus_weak --label_name mv --walnut_mode --base_model bert-large-uncased --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_BERT_LARGE_Clean_Plus_Weak_Snorkel"
for seed in 0 20 7 1993 42; do
  for dataset in NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --clean_plus_weak --label_name nb --walnut_mode --base_model bert-large-uncased --batch_size 8
  done
done


# RoBERTA Large
baselogdir="$(date +'%h%d_%H-%M')_RoBERTa_LARGE_Clean_100"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --base_model roberta-large --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_RoBERTa_LARGE_Clean"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --walnut_mode --base_model roberta-large --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_RoBERTa_LARGE_Weak"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --label_name mv --walnut_mode --base_model roberta-large --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_RoBERTa_LARGE_Weak_Snorkel"
for seed in 0 20 7 1993 42; do
  for dataset in CoNLL; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --label_name nb --walnut_mode --base_model roberta-large --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_RoBERTa_LARGE_Clean_Plus_Weak"
for seed in 0 20 7 1993 42; do
  for dataset in NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --clean_plus_weak --label_name mv --walnut_mode --base_model roberta-large --batch_size 8
  done
done

baselogdir="$(date +'%h%d_%H-%M')_RoBERTa_LARGE_Clean_Plus_Weak_Snorkel"
for seed in 0 20 7 1993 42; do
  for dataset in NCBI BC5CDR LaptopReview; do
    logdir="${baselogdir}/seed${seed}"
    python main.py --dname ${dataset} --savefolder ${logdir} --seed $seed --clean_plus_weak --label_name nb --walnut_mode --base_model roberta-large --batch_size 8
  done
done
