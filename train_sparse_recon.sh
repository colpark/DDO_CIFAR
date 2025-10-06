#!/bin/bash

# Training script for Sparse Image Reconstruction with DDO
# Task: Reconstruct full 32x32 CIFAR-10 images from 10% randomly observed pixels

# Set paths
EXP_PATH=${1:-./experiments/sparse_reconstruction}
DATA_PATH=${2:-./data}

echo "====================================="
echo "Sparse Image Reconstruction with DDO"
echo "====================================="
echo "Experiment path: ${EXP_PATH}"
echo "Data path: ${DATA_PATH}"
echo "Context ratio: 10% (observed pixels)"
echo "Query ratio: 10% (training targets)"
echo "====================================="

python main_sparse_reconstruction.py \
  --command_type=train \
  --exp_path=${EXP_PATH} \
  --data=${DATA_PATH} \
  --seed=1 \
  --dataset=cifar10 \
  --train_img_height=32 \
  --input_dim=3 \
  --coord_dim=2 \
  --context_ratio=0.1 \
  --query_ratio=0.1 \
  --model=fnounet2d \
  --ch=64 \
  --ch_mult 1 2 2 \
  --num_res_blocks=2 \
  --modes=16 \
  --dropout=0.1 \
  --norm=group_norm \
  --use_pos \
  --use_pointwise_op \
  --ns_method=vp_cosine \
  --timestep_sampler=low_discrepancy \
  --disp_method=sine \
  --sigma_blur_min=0.05 \
  --sigma_blur_max=0.25 \
  --gp_type=exponential \
  --gp_exponent=2.0 \
  --gp_length_scale=0.05 \
  --gp_sigma=1.0 \
  --train_batch_size=128 \
  --lr=0.0002 \
  --weight_decay=0.0 \
  --num_iterations=200000 \
  --ema_decay=0.999 \
  --optimizer=adam \
  --beta1=0.9 \
  --beta2=0.999 \
  --print_every=100 \
  --save_every=5000 \
  --vis_every=1000 \
  --eval_every=10000 \
  --vis_batch_size=16 \
  --num_steps=250 \
  --sampler=denoise \
  --s_min=0.0001 \
  --resume

echo "Training completed!"
echo "Results saved to: ${EXP_PATH}"
echo "View samples: ${EXP_PATH}/samples/"
echo "View TensorBoard: tensorboard --logdir=${EXP_PATH}"
