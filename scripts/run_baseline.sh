#!bin/bash

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
  --feature_extracting False \
  --train_batch_size 32 \
  --loss_fn_name BCE \
  --write_outputs True \
  --save_dir /mnt/disks/large/output/baseline
