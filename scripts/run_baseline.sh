#!/bin/bash

cd ..

python ./main.py \
  --do_train True \
  --do_predict True \
  --split test \
  --use_pretrained True \
  --model_name baseline \
  --lr_init 5e-5 \
  --l2_wd 0. \
  --num_epochs 10 \
  --train_batch_size 16 \
  --loss_fn_name BCE \
  --write_outputs True \
  --eval_steps 7000 \
  --feature_extracting True \
  --metric_avg samples \
  --save_dir /mnt/disks/large/output/baseline
