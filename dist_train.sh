#!/usr/bin/env bash

MODEL=$1
mkdir ./output/${MODEL}
touch ./output/${MODEL}/run_log.txt
shift
python -m torch.distributed.launch --nproc_per_node=8 --master_port=123 --use_env main.py \
    --data-path your_path_to_data \
    --model $MODEL \
    --batch-size 128   \
    --lr 1e-3 \
    --drop-path 0.1 \
    --epoch 300 \
    --dist-eval \
    --output_dir /output/ \
     "$@"        \
     2>&1 | tee -a ./output/${MODEL}/run_log.txt
