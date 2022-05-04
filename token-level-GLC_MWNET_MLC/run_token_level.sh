#!/bin/bash
# Supervised
#python main.py --dataset BC5CDR
#python main.py --dataset NCBI
#python main.py --dataset LaptopReview

for dataset in NCBI BC5CDR LaptopReview; do
  for seed in 0 20 7 1993 42; do
    python main.py --dataset ${dataset} --seed ${seed}
  done
done