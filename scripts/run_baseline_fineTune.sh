#!/bin/bash

cd ..

python ./main.py \
  --do_train \
  --do_predict \
  --split test \
  --use_pretrained \
  --model_name baseline \
  --lr_init 1e-4 \
  --l2_wd 1e-5 \
  --num_epochs 20 \
  --train_batch_size 16 \
  --loss_fn_name BCE \
  --write_outputs \
  --eval_steps 10000 \
  --metric_avg samples \
  --save_dir /mnt/disks/large/output/baseline_fineTune
