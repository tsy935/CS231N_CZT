#!/bin/bash

cd ..

python ./main.py \
  --do_predict \
  --split dev \
  --use_pretrained \
  --model_name cnn-rnn \
  --lr_init 1e-3 \
  --l2_wd 1e-5 \
  --loss_fn_name BCE \
  --write_outputs \
  --metric_avg samples \
  --max_pos_weight 10 \
  --feature_extracting \
  --prob_path_thresh 2e-3 \
  --load_path /mnt/disks/large/output/cnnrnn/train/train-15/best.pth.tar \
  --best_val_results /mnt/disks/large/output/cnnrnn/train/train-15/best_val_results \
  --save_dir /mnt/disks/large/output/cnnrnn
