#!/bin/bash

cd ..

python ./main.py \
  --do_train \
  --do_predict \
  --split dev \
  --use_pretrained \
  --model_name baseline_hog \
  --lr_init 5e-5 \
  --l2_wd 1e-5 \
  --num_epochs 25 \
  --train_batch_size 32 \
  --loss_fn_name BCE \
  --write_outputs \
  --eval_steps 10000 \
  --metric_avg samples \
  --max_pos_weight 10 \
  --feature_extracting \
  --prob_path_thresh 2.5 \
  --resnet_path /mnt/disks/large/output/vm1_featureExtract/best.pth.tar \
  --save_dir /mnt/disks/large/output/baseline_hog \
  --baseline_thresh_prop_power 0.05 \
  --thresh_search 

