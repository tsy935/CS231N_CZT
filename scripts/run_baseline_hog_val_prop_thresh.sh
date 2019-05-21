#!/bin/bash

cd ..

python ./main.py \
  --do_predict \
  --split dev \
  --use_pretrained \
  --model_name baseline_hog \
  --loss_fn_name BCE \
  --write_outputs \
  --metric_avg samples \
  --max_pos_weight 10 \
  --feature_extracting \
  --load_path  /mnt/disks/large/output/baseline_hog/train/train-03/best.pth.tar \
  --best_val_results /mnt/disks/large/output/baseline_hog/train/train-03/best_val_results\
  --save_dir /mnt/disks/large/output/baseline_hog_threshold_search_prop
