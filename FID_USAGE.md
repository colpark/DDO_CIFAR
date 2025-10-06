# FID Evaluation - Quick Usage Guide

## What is FID?

**Fréchet Inception Distance** measures how similar generated images are to real images in InceptionV3 feature space.

- **Lower is better**: FID = 0 means identical distributions
- **Efficient**: Only needs ~500 samples (not thousands)
- **No ensemble needed**: Single sample per image
- **Industry standard**: Used by all major generative models

## Quick Start

### Option 1: Add to Existing Evaluation Notebook

After running the main CRPS evaluation in `conditional_evaluation.ipynb`, add this cell:

```python
# Load FID addon code
exec(open('notebooks/fid_evaluation_addon.py').read())
```

This will:
1. Generate 500 reconstructions from sparse context
2. Extract InceptionV3 features from real and generated images
3. Compute FID score
4. Add FID to results JSON

**Runtime:** ~5-10 minutes on GPU

### Option 2: Standalone Script

```python
from utils.fid_score import compute_fid

# Load your real and generated images
real_images = ...  # (N, 3, H, W) in [0, 1]
generated_images = ...  # (N, 3, H, W) in [0, 1]

# Compute FID
fid_score = compute_fid(
    real_images,
    generated_images,
    batch_size=50,
    device='cuda'
)

print(f"FID: {fid_score:.2f}")
```

## Implementation Details

### Feature Extraction

```python
from utils.fid_score import InceptionV3FeatureExtractor

extractor = InceptionV3FeatureExtractor()

# Extract 2048-dim features
features = extractor(images)  # (N, 2048)
```

**Key Points:**
- Uses InceptionV3 pretrained on ImageNet
- Extracts from pool3 layer (before final classification)
- Automatically resizes images to 299×299
- Normalizes to [-1, 1] range

### FID Computation

```python
# Compute mean and covariance
μ_real, Σ_real = compute_statistics_from_features(real_features)
μ_gen, Σ_gen = compute_statistics_from_features(gen_features)

# Fréchet distance
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real·Σ_gen))
```

## Interpretation

### FID Score Ranges

| FID Range | Quality | Description |
|-----------|---------|-------------|
| < 10 | Excellent | Nearly indistinguishable from real data |
| 10-30 | Good | High perceptual quality, minor artifacts |
| 30-50 | Fair | Noticeable quality gap, visible artifacts |
| > 50 | Poor | Significant distribution mismatch |

### Example Benchmarks

**Unconditional CIFAR-10 Generation:**
- DDPM: FID ~3.2
- StyleGAN2: FID ~2.4
- Diffusion (NCSN): FID ~10

**Conditional Generation (sparse → full):**
- Expected range: FID 20-40 (harder task)
- FID < 30: Good reconstruction quality
- FID 30-50: Reasonable, but room for improvement

## Why FID for Sparse Reconstruction?

### Advantages

1. **Distribution-level metric**
   - Not just pixel-wise accuracy
   - Captures overall perceptual quality

2. **Computationally efficient**
   - Only 500 samples needed
   - No ensemble required (unlike CRPS)
   - ~10 minutes vs hours for CRPS

3. **Perceptually meaningful**
   - Deep features correlate with human perception
   - Better than MSE/PSNR for quality assessment

4. **Industry standard**
   - Widely used and understood
   - Easy to compare with other models

### Complementary to Other Metrics

**FID + CRPS = Complete Picture:**

| Metric | What it measures | Use case |
|--------|------------------|----------|
| MSE/PSNR | Pixel accuracy | Individual image quality |
| CRPS | Forecast calibration | Ensemble uncertainty |
| FID | Distribution similarity | Overall perceptual quality |

**Example:**
```
Model A: Low MSE, Low CRPS, High FID
→ Accurate pixels, calibrated, but blurry (mode collapse)

Model B: Medium MSE, Low CRPS, Low FID
→ Less pixel-perfect, but perceptually better

Model C: Low MSE, High CRPS, Low FID
→ Good individual samples, but miscalibrated ensemble
```

## Sample Output

```
=========================================================
FID EVALUATION (Few Hundred Samples)
=========================================================
Computing FID on 500 samples...
This is computationally efficient (no ensemble needed)

Generating for FID: 100%|████████| 32/32 [03:24<00:00,  6.41s/it]

Collected 500 real and generated images
Computing FID...
  Real images: 500
  Generated images: 500
  Extracting features from real images...
  Extracting features from generated images...
  Computing statistics...
  Calculating Fréchet distance...
  FID: 28.34

=========================================================
FID RESULTS
=========================================================
FID Score: 28.34

Interpretation:
  - FID < 10:  Excellent (nearly indistinguishable)
  - FID 10-30: Good (high quality)
  - FID 30-50: Fair (noticeable artifacts)
  - FID > 50:  Poor (significant quality gap)
=========================================================

Updated results saved to experiments/conditional_sparse_recon/evaluation_results.json
```

## Technical Notes

### Memory Requirements

- **InceptionV3**: ~400MB
- **Feature storage (500 images)**: ~4MB (500 × 2048 × 4 bytes)
- **Total GPU memory**: ~4-6GB

### Batch Size Tuning

```python
# Adjust based on GPU memory
fid = compute_fid(real, gen, batch_size=50)  # 8GB GPU
fid = compute_fid(real, gen, batch_size=25)  # 4GB GPU
fid = compute_fid(real, gen, batch_size=100) # 16GB GPU
```

### Number of Samples

FID is stable with 500+ samples:
- 100 samples: High variance, unreliable
- 500 samples: Good balance (our choice)
- 1000+ samples: Marginal improvement
- 10000+ samples: Unnecessary (no benefit)

**Rule of thumb:** Use at least 500 samples, max 2000 for efficiency.

## Troubleshooting

### High FID (>50)

**Possible causes:**
1. Model undertrained
2. Reconstructions too blurry
3. Mode collapse (all similar outputs)
4. Context not being used properly

**Debug steps:**
1. Visualize generated images - do they look reasonable?
2. Check MSE/PSNR - if also bad, model needs more training
3. Check diversity - are all reconstructions similar?

### NaN or Inf in FID

**Cause:** Numerical instability in matrix square root

**Solution:** Already handled in implementation
```python
# Adds epsilon to diagonal for stability
covmean = linalg.sqrtm((sigma1 + eps*I) @ (sigma2 + eps*I))
```

### Slow Feature Extraction

**Problem:** Taking too long to extract features

**Solutions:**
1. Reduce batch size (less memory, more iterations)
2. Use fewer samples (500 is enough)
3. Use CPU if GPU out of memory (slower but works)

```python
fid = compute_fid(real, gen, device='cpu')  # Fallback to CPU
```

## Files

```
ddo/
├── utils/
│   └── fid_score.py                    # FID implementation
├── notebooks/
│   ├── conditional_evaluation.ipynb    # Main evaluation
│   └── fid_evaluation_addon.py         # FID addon code
└── FID_USAGE.md                        # This file
```

## References

1. **FID Paper**: Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
2. **Implementation**: Based on pytorch-fid (official implementation)
3. **InceptionV3**: Szegedy et al. (2016). "Rethinking the Inception Architecture"

## Citation

If you use FID in your evaluation:

```bibtex
@inproceedings{heusel2017gans,
  title={Gans trained by a two time-scale update rule converge to a local nash equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={NeurIPS},
  year={2017}
}
```

---

**Summary:** FID provides efficient, perceptually-meaningful evaluation of reconstruction quality in just ~10 minutes with 500 samples.
