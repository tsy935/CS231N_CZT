#!/bin/bash

cd ..

python ./main.py \
  --do_predict \
  --split dev \
  --use_pretrained \
  --model_name hog_cnn-rnn \
  --loss_fn_name BCE \
  --write_outputs \
  --metric_avg samples \
  --feature_extracting \
  --prob_path_thresh 2.5 \
  --beam_search \
  --beam_size 5 \
  --load_path /mnt/disks/large/output/hog_cnnrnn/train/train-04/best.pth.tar \
  --best_val_results /mnt/disks/large/output/hog_cnnrnn/train/train-04/best_val_results \
  --save_dir /mnt/disks/large/output/hog_cnnrnn
