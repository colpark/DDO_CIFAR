#!/bin/bash

# Training script for DDO on CIFAR-10 dataset
# CIFAR-10 is 32x32 RGB images, trained directly in pixel space

# Set these paths according to your setup
EXP_PATH=${1:-./experiments/cifar10_ddo}
DATA_PATH=${2:-./data}
FID_PATH=${3:-./fid_stats/cifar10}

python main.py --command_type=train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=50000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=cifar10 --train_img_height=32 --input_dim=3 --coord_dim=2 \
  --model=fnounet2d --use_pos --modes=16 --ch=64 --ch_mult=1,2,2 --num_res_blocks=2 --dropout=0.1 --norm=group_norm --use_pointwise_op \
  --ns_method=vp_cosine --timestep_sampler=low_discrepancy \
  --disp_method=sine --sigma_blur_min=0.05 --sigma_blur_max=0.25 \
  --gp_type=exponential --gp_exponent=2.0 --gp_length_scale=0.05 --gp_sigma=1.0 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --train_batch_size=128 --lr=0.0002 --weight_decay=0.0 --num_iterations=500000 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=32 --eval_batch_size=256 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=${FID_PATH} \
  --vis_batch_size=64
