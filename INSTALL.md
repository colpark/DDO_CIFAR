# Installation Guide

This guide will help you set up the environment to run DDO experiments on CIFAR-10.

## Prerequisites

- NVIDIA GPU with CUDA support (recommended for training)
- Anaconda or Miniconda installed
- CUDA 11.6 (for GPU support)

## Option 1: Using Conda (Recommended)

1. Create the conda environment from the environment file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate ddo-cifar10
```

3. Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Option 2: Using pip with virtualenv

1. Create a virtual environment:

```bash
python -m venv ddo-env
source ddo-env/bin/activate  # On Windows: ddo-env\Scripts\activate
```

2. Install PyTorch with CUDA support:

```bash
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. Install other dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start - CIFAR-10 Experiment

After setting up the environment:

1. **Train the model:**

```bash
bash train_cifar10.sh ./experiments/cifar10_ddo ./data ./fid_stats/cifar10
```

Or use Python directly:

```bash
python main.py --command_type=train \
  --exp_path=./experiments/cifar10_ddo \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=50000 --vis_every=10000 --resume \
  --data=./data --dataset=cifar10 --train_img_height=32 --input_dim=3 --coord_dim=2 \
  --model=fnounet2d --use_pos --modes=32 --ch=128 --ch_mult=1,2,2,2 --num_res_blocks=4 --dropout=0.1 --norm=group_norm --use_pointwise_op \
  --ns_method=vp_cosine --timestep_sampler=low_discrepancy \
  --disp_method=sine --sigma_blur_min=0.05 --sigma_blur_max=0.25 \
  --gp_type=exponential --gp_exponent=2.0 --gp_length_scale=0.05 --gp_sigma=1.0 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --train_batch_size=128 --lr=0.0002 --weight_decay=0.0 --num_iterations=500000 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=32 --eval_batch_size=256 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=./fid_stats/cifar10 \
  --vis_batch_size=64
```

2. **Generate samples using Jupyter notebook:**

```bash
jupyter notebook notebooks/cifar10_sample_ddo.ipynb
```

3. **Train interactively in Jupyter:**

```bash
jupyter notebook notebooks/cifar10_train_ddo.ipynb
```

4. **Evaluate the model:**

```bash
bash eval_cifar10.sh ./experiments/cifar10_ddo ./data ./fid_stats/cifar10
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

1. Check your CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install the appropriate PyTorch version for your CUDA version from [PyTorch website](https://pytorch.org/get-started/locally/)

### Memory Issues

If you run out of GPU memory during training:

- Reduce `--train_batch_size` (e.g., to 64 or 32)
- Reduce model size: `--ch=64` instead of `--ch=128`
- Use gradient accumulation (requires code modification)

### CPU-Only Mode

To run on CPU (not recommended for training):

```bash
# Install CPU-only PyTorch
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Notes

- CIFAR-10 will be automatically downloaded to the `--data` directory on first run
- Training checkpoints are saved in `--exp_path`
- TensorBoard logs are also saved in `--exp_path` and can be viewed with:
  ```bash
  tensorboard --logdir=./experiments/cifar10_ddo
  ```
