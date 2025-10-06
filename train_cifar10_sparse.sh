#!/bin/bash

# Training script for DDO on Sparse CIFAR-10 reconstruction task
# Task: Given 10% observed pixels, reconstruct the full image
# Training uses another 10% as ground truth query points

# Set these paths according to your setup
EXP_PATH=${1:-./experiments/cifar10_sparse_ddo}
DATA_PATH=${2:-./data}
FID_PATH=${3:-./fid_stats/cifar10}

python main_sparse.py --command_type=train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=50000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=cifar10 --train_img_height=32 --input_dim=3 --coord_dim=2 \
  --context_ratio=0.1 --query_ratio=0.1 \
  --model=fnounet2d --use_pos --modes=32 --ch=128 --ch_mult=1,2,2,2 --num_res_blocks=4 --dropout=0.1 --norm=group_norm --use_pointwise_op \
  --ns_method=vp_cosine --timestep_sampler=low_discrepancy \
  --disp_method=sine --sigma_blur_min=0.05 --sigma_blur_max=0.25 \
  --gp_type=exponential --gp_exponent=2.0 --gp_length_scale=0.05 --gp_sigma=1.0 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --train_batch_size=128 --lr=0.0002 --weight_decay=0.0 --num_iterations=500000 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=32 --eval_batch_size=256 --eval_num_samples=5000 \
  --vis_batch_size=64
