# Sparse Image Reconstruction with DDO

Complete experiment for conditional image reconstruction using diffusion models in function space.

## Overview

**Task**: Reconstruct full 32×32 RGB images from only 10% randomly observed pixels

**Method**: Conditional Denoising Diffusion in Operator space (DDO) with sparse context encoding

**Dataset**: CIFAR-10

## Key Features

- **Input**: 10% randomly sampled pixel values + coordinates
- **Output**: Full reconstructed 32×32×3 image
- **Training**: Uses additional 10% pixels as query targets for supervision
- **Inference**: Predicts all 90% unobserved pixels given sparse context

## Quick Start

### Option 1: Command Line Training

```bash
# Install environment
conda env create -f environment.yml
conda activate ddo-cifar10

# Run training
bash train_sparse_recon.sh ./experiments/sparse_recon ./data
```

### Option 2: Interactive Notebook

```bash
jupyter notebook notebooks/sparse_reconstruction_train.ipynb
```

## How It Works

### 1. Data Preparation

Each training sample consists of:
- **Full image**: Ground truth 32×32×3 CIFAR-10 image
- **Context**: 10% randomly sampled (coordinate, RGB value) pairs
- **Query**: Another 10% randomly sampled pairs for training targets

```python
from utils.sparse_datasets import SparseImageDatasetWrapper

sparse_dataset = SparseImageDatasetWrapper(
    dataset=cifar10_dataset,
    context_ratio=0.1,   # 10% observed
    query_ratio=0.1,     # 10% for training
    mode='train'
)
```

### 2. Model Architecture

The model consists of:

**Sparse Context Encoder**:
- Converts sparse (coord, value) pairs into dense feature map
- Unobserved pixels are filled with zeros or mean values

**Conditional DDO**:
- FNO-based U-Net with 64 base channels
- Processes noisy image conditioned on context features
- Learns to denoise while respecting observed constraints

**Structure**:
```
Context (10% pixels) → Dense Encoding → Context Features
                                              ↓
Noisy Full Image ────────────────────→ [Concatenate] → U-Net → Denoised Image
```

### 3. Training Procedure

1. Sample random 10% context pixels
2. Sample different random 10% query pixels
3. Create context encoding from observed pixels
4. Apply diffusion noise to full image
5. Model predicts noise conditioned on context
6. Compute loss on full image (standard DSM)

```python
# Training loop
context_image = create_context_conditioning(context_values, context_coords, 32)
v = get_mgrid(2, 32).repeat(batch_size, 1, 1, 1).cuda()
loss = gen_sde.dsm(full_images, v).mean()
```

### 4. Inference

Given sparse observations, reconstruct full image:

```python
# Create context from observed pixels
context_image = create_context_conditioning(observations, coords, 32)

# Run reverse diffusion
reconstructed = sample_image(
    gen_sde,
    context_image=context_image,
    num_steps=250,
    sampler='denoise'
)
```

## Model Configuration

### Default Settings

```python
# Model architecture
ch = 64                    # Base channels
ch_mult = [1, 2, 2]       # 3 resolution levels
num_res_blocks = 2         # Residual blocks per level
modes = 16                 # Fourier modes
dropout = 0.1

# Results in ~10-20M parameters
```

### Training Hyperparameters

```python
# Optimization
lr = 0.0002
batch_size = 128
num_iterations = 200000
ema_decay = 0.999

# Diffusion
num_steps = 250
ns_method = 'vp_cosine'
sampler = 'denoise'
```

## Expected Results

### Reconstruction Quality

With 10% observations:
- **PSNR**: ~20-25 dB (depends on training)
- **SSIM**: ~0.6-0.7
- **Visual**: Recognizable objects, slightly blurry

### Training Time

- **Per iteration**: ~0.5-1 sec (on V100)
- **Total training**: ~24-48 hours for 200K iterations
- **GPU memory**: ~8-12 GB

## File Structure

```
ddo/
├── main_sparse_reconstruction.py      # Main training script
├── train_sparse_recon.sh             # Shell script for training
├── utils/
│   └── sparse_datasets.py            # Sparse dataset utilities
├── notebooks/
│   ├── sparse_reconstruction_train.ipynb    # Training notebook
│   └── cifar10_sparse_reconstruction.ipynb  # Visualization demo
└── experiments/
    └── sparse_reconstruction/
        ├── checkpoint.pt             # Model checkpoints
        ├── samples/                  # Reconstruction visualizations
        │   └── iter_XXXXXX.png
        └── tensorboard/              # Training logs
```

## Outputs

### During Training

Every 1000 iterations:
- Saves reconstruction visualization to `samples/iter_XXXXXX.png`
- Format: Top row = sparse input, Bottom row = ground truth
- Logs loss to TensorBoard

### After Training

- **Checkpoint**: `checkpoint.pt` - full model state
- **Samples**: `samples/` - reconstruction examples
- **Logs**: View with `tensorboard --logdir=experiments/sparse_reconstruction`

## Evaluation

### Quantitative Metrics

```python
# TODO: Add evaluation script
# Metrics: PSNR, SSIM, LPIPS
```

### Qualitative Assessment

Check reconstruction quality:
1. Object recognition (can you identify the object?)
2. Color accuracy (correct hues?)
3. Edge sharpness (crisp or blurry?)
4. Artifacts (any unusual patterns?)

## Advanced Usage

### Different Observation Ratios

Try different sparsity levels:

```python
# 5% observations (harder)
context_ratio = 0.05

# 20% observations (easier)
context_ratio = 0.20
```

### Different Sampling Patterns

Instead of random sampling:

```python
from utils.sparse_datasets import GridMaskGenerator

# Regular grid
mask = GridMaskGenerator.grid_mask(32, 32, stride=3)

# Center-focused
mask = GridMaskGenerator.center_mask(32, 32, num_samples=102)
```

### Transfer to Other Datasets

The same pipeline works for other datasets:

```python
# ImageNet
base_dataset = torchvision.datasets.ImageNet(...)
sparse_dataset = SparseImageDatasetWrapper(base_dataset, ...)

# Custom dataset
base_dataset = YourCustomDataset(...)
sparse_dataset = SparseImageDatasetWrapper(base_dataset, ...)
```

## Troubleshooting

### Common Issues

**Problem**: Model reconstructs mean image

**Solution**:
- Ensure context encoding is working
- Check that context_image is being passed to model
- Verify random masks are different each iteration

**Problem**: Poor reconstruction quality

**Solution**:
- Increase model size (ch=128, more layers)
- Train longer (500K+ iterations)
- Reduce observation ratio to force better learning

**Problem**: Out of memory

**Solution**:
- Reduce batch_size (try 64 or 32)
- Reduce model size (ch=32)
- Use gradient accumulation

## References

- Paper: [Score-based Diffusion Models in Function Space](https://arxiv.org/abs/2302.07400)
- Code: Based on official DDO implementation
- Related: Image inpainting, super-resolution, compressed sensing

## Citation

If you use this sparse reconstruction experiment:

```bibtex
@article{lim2023score,
  title={Score-based diffusion models in function space},
  author={Lim*, Jae Hyun and Kovachki*, Nikola B and Baptista*, Ricardo and others},
  journal={arXiv preprint arXiv:2302.07400},
  year={2023}
}
```

## Next Steps

1. ✅ Train basic model with 10% observations
2. ⬜ Implement proper conditional model
3. ⬜ Add evaluation metrics (PSNR, SSIM)
4. ⬜ Experiment with different observation ratios
5. ⬜ Try structured sampling patterns
6. ⬜ Scale to higher resolution images

---

**Status**: Experimental - Training pipeline ready, conditional model implementation in progress

**Contact**: See main README for issues and questions
