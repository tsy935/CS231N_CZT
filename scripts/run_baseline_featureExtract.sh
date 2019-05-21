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
  --num_epochs 15 \
  --train_batch_size 32 \
  --loss_fn_name BCE \
  --write_outputs \
  --eval_steps 20000 \
  --feature_extracting \
  --metric_avg samples \
  --max_pos_weight 10 \
  --load_path /mnt/disks/large/output/baseline_featureExtract/train/train-10/best.pth.tar \
  --save_dir /mnt/disks/large/output/baseline_featureExtract
