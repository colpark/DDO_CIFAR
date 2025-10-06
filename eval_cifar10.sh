#!/bin/bash

# Evaluation script for DDO on CIFAR-10 dataset

# Set these paths according to your setup
EXP_PATH=${1:-./experiments/cifar10_ddo}
DATA_PATH=${2:-./data}
FID_PATH=${3:-./fid_stats/cifar10}

python main.py --command_type=test \
  --exp_path=${EXP_PATH} \
  --data=${DATA_PATH} --dataset=cifar10 --train_img_height=32 --input_dim=3 --coord_dim=2 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=32 --eval_batch_size=512 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=${FID_PATH} \
  --checkpoint_file=checkpoint_fid.pt \
  --eval_fid
