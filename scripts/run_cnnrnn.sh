#!/bin/bash

cd ..

python ./main.py \
  --do_train \
  --do_predict \
  --split test \
  --use_pretrained \
  --model_name cnn-rnn \
  --lr_init 1e-4 \
  --l2_wd 1e-5 \
  --num_epochs 10 \
  --train_batch_size 32 \
  --loss_fn_name BCE \
  --write_outputs \
  --eval_steps 20000 \
  --metric_avg samples \
  --max_pos_weight 10 \
  --feature_extracting \
  --prob_path_thresh 1e-2 \
  --load_path /mnt/disks/large/output/cnnrnn/train/train-06/best.pth.tar \
  --save_dir /mnt/disks/large/output/cnnrnn
