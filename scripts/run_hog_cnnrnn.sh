#!/bin/bash

cd ..

python ./main.py \
  --do_train \
  --do_predict \
  --split test \
  --use_pretrained \
  --model_name hog_cnn-rnn \
  --lr_init 5e-4 \
  --l2_wd 1e-5 \
  --num_epochs 25 \
  --train_batch_size 32 \
  --loss_fn_name BCE \
  --write_outputs \
  --eval_steps 500 \
  --metric_avg samples \
  --max_pos_weight 10 \
  --feature_extracting \
  --prob_path_thresh 2.5 \
  --save_dir /mnt/disks/large/output/hog_cnnrnn
