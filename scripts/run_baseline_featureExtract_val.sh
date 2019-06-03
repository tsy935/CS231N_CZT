#!/bin/bash

cd ..

python ./main.py \
  --do_predict \
  --split test \
  --use_pretrained \
  --model_name baseline \
  --loss_fn_name BCE \
  --write_outputs \
  --metric_avg samples \
  --max_pos_weight 10 \
  --feature_extracting \
  --thresh_search \
  --load_path /mnt/disks/large/output/vm1_featureExtract/best.pth.tar \
  --best_val_results /mnt/disks/large/output/vm1_featureExtract/best_val_results \
  --save_dir /mnt/disks/large/output/baseline_final
