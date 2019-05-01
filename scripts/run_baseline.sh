#!/bin/bash

cd ..

python ./main.py \
  --do_train True \
  --do_predict True \
  --split test \
  --use_pretrained True \
  --model_name baseline \
  --lr_init 1e-4 \
  --l2_wd 1e-5 \
  --num_epochs 20 \
  --train_batch_size 16 \
  --loss_fn_name BCE \
  --write_outputs True \
  --eval_steps 5000 \
  --feature_extracting True \
  --metric_avg samples \
  --save_dir /mnt/disks/large/output/baseline
