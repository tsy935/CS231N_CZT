#!/bin/bash

cd ..

python ./main.py \
  --do_train True \
  --do_predict True \
  --split test \
  --use_pretrained True \
  --model_name baseline \
  --lr_init 0.001 \
  --l2_wd 0. \
  --num_epochs 3 \
  --train_batch_size 16 \
  --loss_fn_name BCE \
  --write_outputs True \
  --eval_steps 16 \
  --save_dir /mnt/disks/large/output/baseline
