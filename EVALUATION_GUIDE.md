# Conditional Sparse Reconstruction - Evaluation Guide

Complete evaluation framework with MSE, PSNR, MAE, and CRPS metrics.

## Overview

This evaluation measures how well the conditional diffusion model reconstructs full CIFAR-10 images from 10% sparse observations.

**Key Metrics:**
1. **MSE**: Mean Squared Error (standard reconstruction quality)
2. **PSNR**: Peak Signal-to-Noise Ratio (in dB)
3. **MAE**: Mean Absolute Error (deterministic baseline)
4. **CRPS**: Continuous Ranked Probability Score (probabilistic evaluation)
5. **FID**: Fréchet Inception Distance (distribution similarity)

## CRPS: Continuous Ranked Probability Score

### What is CRPS?

CRPS is the **gold standard** for evaluating probabilistic forecasts. It generalizes MAE to ensemble predictions.

**Mathematical Definition:**

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} [F(x) - H(x - y)]^2 dx$$

Where:
- $F(x)$ = CDF of your predictive distribution
- $y$ = Ground truth observation
- $H(x - y)$ = Heaviside step function (0 if x < y, 1 if x ≥ y)

**For M ensemble members:**

$$\text{CRPS} = \frac{1}{M} \sum_{i=1}^M |x_i - y| - \frac{1}{2M^2} \sum_{i=1}^M \sum_{j=1}^M |x_i - x_j|$$

**Interpretation:**
- **Term 1**: Average distance from ensemble to ground truth (accuracy)
- **Term 2**: Average pairwise distance between ensemble members (sharpness)
- Lower CRPS = better forecast

### Why CRPS?

**Advantages over MSE/MAE:**
1. **Rewards Sharpness**: Narrow distributions are better if correct
2. **Rewards Calibration**: Ground truth should be plausible from ensemble
3. **Handles Uncertainty**: Evaluates full predictive distribution
4. **Fair Comparison**: Can compare probabilistic vs deterministic models
5. **Collapses to MAE**: When ensemble size M=1, CRPS = MAE

**Example Scenarios:**

```
Scenario 1: Well-Calibrated Ensemble
  Ensemble: [0.48, 0.50, 0.52, 0.49, 0.51]
  Ground Truth: 0.50
  → Small Term 1 (accurate)
  → Small Term 2 (sharp)
  → Low CRPS ✓

Scenario 2: Over-Dispersed Ensemble
  Ensemble: [0.2, 0.4, 0.6, 0.8, 1.0]
  Ground Truth: 0.50
  → Medium Term 1 (somewhat accurate)
  → Large Term 2 (too spread out)
  → High CRPS ✗

Scenario 3: Biased but Sharp
  Ensemble: [0.72, 0.74, 0.76, 0.73, 0.75]
  Ground Truth: 0.50
  → Large Term 1 (inaccurate)
  → Small Term 2 (sharp but wrong)
  → High CRPS ✗
```

### Comparing Diffusion Model (CRPS) vs Deterministic (MAE)

**Setup:**
- **Diffusion Model**: Generate M=10 samples per image, compute CRPS
- **Deterministic Model**: Single prediction, compute MAE

**Fair Comparison:**
- Both metrics measure error in same units
- CRPS ≤ MAE for perfect deterministic forecast (both = 0)
- CRPS penalizes over-confidence (too narrow) or under-confidence (too wide)

**Interpretation:**
```
If CRPS < MAE:
  → Ensemble is well-calibrated AND sharp
  → Diffusion model provides better predictions

If CRPS ≈ MAE:
  → Ensemble behaves like deterministic
  → May need more diversity

If CRPS > MAE:
  → Ensemble is too spread out or miscalibrated
  → Need to improve model or sampling
```

## FID: Fréchet Inception Distance

### What is FID?

FID measures the **distributional distance** between real and generated images in InceptionV3 feature space.

**Mathematical Definition:**

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

Where:
- $\mu_r, \Sigma_r$ = Mean and covariance of real image features
- $\mu_g, \Sigma_g$ = Mean and covariance of generated image features
- Features extracted from InceptionV3 pool3 layer (2048-dim)

**Interpretation:**
- Measures how similar two distributions are
- Lower FID = more similar distributions = better generation quality
- FID < 10: Excellent (nearly indistinguishable)
- FID 10-30: Good (high quality)
- FID 30-50: Fair (noticeable artifacts)
- FID > 50: Poor (significant quality gap)

### Why FID?

**Advantages:**
1. **Distribution-level metric**: Captures overall quality, not just pixel-wise
2. **Perceptually meaningful**: Uses deep features, not raw pixels
3. **Industry standard**: Widely used for GANs and diffusion models
4. **Computationally efficient**: Only needs few hundred samples
5. **Robust**: Less sensitive to outliers than pixel metrics

**Comparison with other metrics:**
- **vs MSE/PSNR**: FID captures perceptual quality, not just pixel accuracy
- **vs CRPS**: FID measures distribution similarity, CRPS measures forecast calibration
- **vs IS (Inception Score)**: FID compares to real data, IS only measures generated diversity

### FID for Sparse Reconstruction

In our case, FID measures how well **reconstructed** images match the **distribution** of real images:

```
Real CIFAR-10 → InceptionV3 → Features → (μ_real, Σ_real)
                                              ↓
Reconstructions → InceptionV3 → Features → (μ_gen, Σ_gen)
                                              ↓
                                          FID = distance(real, gen)
```

**Key Point:** FID doesn't require paired comparisons. It measures whether your reconstructions "look like" real CIFAR-10 images, even if individual pixels don't match exactly.

### Efficient FID Computation

**Our Implementation:**
- **Sample size**: 500 images (computationally efficient)
- **No ensemble**: Single sample per image (unlike CRPS)
- **Runtime**: ~5-10 minutes on GPU
- **Memory**: ~4GB GPU memory

**Why few hundred samples?**
- FID is stable with 500+ samples
- Each sample requires one forward pass (no ensemble)
- Much faster than CRPS which needs 10× samples

## Evaluation Pipeline

### 1. Conditional Sampling

Generate reconstructions conditioned on sparse context:

```python
def sample_conditional(gen_sde, context_image, v, num_steps=250):
    # Start from noise
    x_T = torch.randn_like(context_image)

    # Reverse diffusion WITH context
    samples = diffuse(
        gen_sde,
        num_steps=num_steps,
        x_0=x_T,
        v=v,
        context_image=context_image  # Context flows through!
    )

    return samples[-1]
```

**Key Modification:**
- `context_image` is passed through entire sampling pipeline
- Modified `denoise_step`, `epsilon`, `pred` to accept `**kwargs`
- Context flows: `diffuse → denoise_step → epsilon → model`

### 2. Ensemble Generation

For CRPS, generate multiple samples:

```python
def sample_ensemble_conditional(gen_sde, context_image, v, num_ensemble=10):
    ensemble = []

    for i in range(num_ensemble):
        sample = sample_conditional(gen_sde, context_image, v)
        ensemble.append(sample)

    return torch.stack(ensemble)  # (M, B, C, H, W)
```

### 3. Metric Computation

**MSE (Per-Image):**
```python
mse = ((prediction - ground_truth) ** 2).mean(dim=(1,2,3))
```

**MAE (Deterministic Baseline):**
```python
mae = (prediction - ground_truth).abs().mean(dim=(1,2,3))
```

**PSNR:**
```python
psnr = 20 * log10(max_val / sqrt(mse))
```

**CRPS (Ensemble):**
```python
# Flatten: (M, B, C, H, W) → (M, B, D) where D = C*H*W
ensemble_flat = ensemble.reshape(M, B, -1)
gt_flat = ground_truth.reshape(B, -1)

# Term 1: Average distance to GT
term1 = |ensemble_flat - gt_flat|.mean(dim=0)  # (B, D)

# Term 2: Average pairwise distance (sharpness)
term2 = 0
for i in range(M):
    for j in range(M):
        term2 += |ensemble_flat[i] - ensemble_flat[j]|
term2 /= (2 * M * M)

# CRPS per pixel, then average
crps = (term1 - term2).mean(dim=1)  # (B,)
```

## Usage

### Quick Start

```bash
# Run evaluation notebook
jupyter notebook notebooks/conditional_evaluation.ipynb

# Execute all cells
# Results will be saved to experiments/conditional_sparse_recon/evaluation_results.json
```

### Configuration

```python
# Evaluation settings
args.num_eval_samples = 1000   # Number of test images
args.num_ensemble = 10         # Ensemble size for CRPS
args.eval_batch_size = 16      # Batch size

# Sampling settings
args.num_steps = 250           # Diffusion steps
args.sampler = 'denoise'       # Sampling method
```

### Expected Runtime

For 1000 images with ensemble size 10:
- Single sample: ~10 seconds (250 steps)
- Per image: ~10 samples × 10 sec = 100 seconds
- Total: ~100,000 seconds ≈ 28 hours on V100

**Recommendation:** Start with smaller subset (100 images) for testing.

## Expected Results

### Good Performance Indicators

**MSE:**
- < 0.01: Excellent
- 0.01-0.05: Good
- 0.05-0.10: Fair
- > 0.10: Poor

**PSNR:**
- > 25 dB: Excellent
- 20-25 dB: Good
- 15-20 dB: Fair
- < 15 dB: Poor

**CRPS vs MAE:**
- CRPS < MAE: Well-calibrated ensemble ✓
- CRPS ≈ MAE: Behaves deterministically
- CRPS > MAE: Over-dispersed or miscalibrated ✗

**FID:**
- < 10: Excellent
- 10-30: Good
- 30-50: Fair
- > 50: Poor

### Example Results

```json
{
  "num_samples": 1000,
  "ensemble_size": 10,
  "mse_mean": 0.0234,
  "mse_std": 0.0156,
  "mae_mean": 0.0876,
  "mae_std": 0.0234,
  "psnr_mean": 22.34,
  "psnr_std": 3.45,
  "crps_mean": 0.0812,
  "crps_std": 0.0221,
  "crps_to_mae_ratio": 0.927,
  "fid_score": 28.45,
  "num_fid_samples": 500
}
```

**Interpretation:**
- PSNR ~22 dB: Good reconstruction quality
- CRPS/MAE = 0.927: CRPS < MAE → Well-calibrated!
- FID = 28.45: Good perceptual quality (10-30 range)
- The ensemble provides better predictions than deterministic baseline
- Reconstructions match real image distribution well

## Output Files

After evaluation, the following files are generated:

```
experiments/conditional_sparse_recon/
├── evaluation_results.json       # Numerical results
├── evaluation_metrics.png        # Metric distributions
└── sample_reconstructions/       # Visual examples
```

### Results JSON

```json
{
  "num_samples": 1000,
  "ensemble_size": 10,
  "mse_mean": 0.0234,
  "mse_std": 0.0156,
  "mae_mean": 0.0876,
  "mae_std": 0.0234,
  "psnr_mean": 22.34,
  "psnr_std": 3.45,
  "crps_mean": 0.0812,
  "crps_std": 0.0221,
  "crps_to_mae_ratio": 0.927
}
```

## Code Modifications for Evaluation

### Modified Files

**1. `lib/diffusion.py`**

Added `**kwargs` support to enable `context_image` propagation:

```python
# Line 701: pred() method
def pred(self, y, s, v=None, level=None, **kwargs):
    epsilon = self.model(x=y, temp=s.view(-1), v=v, level=level, **kwargs)
    # ...

# Line 857-859: denoise_step() method
hat_eps = self.epsilon(y=z_t, s=t, v=v, **kwargs)

# Line 925-928: diffuse() function
func_step(x_t, t, v=v, num_steps=num_steps, ..., **kwargs)
```

**Result:** `context_image` now flows through entire sampling pipeline.

**2. `notebooks/conditional_evaluation.ipynb`**

Complete evaluation framework:
- Conditional sampling with context
- Ensemble generation
- MSE, MAE, PSNR, CRPS computation
- Visualization and result saving

## Theoretical Background

### Why CRPS is Better Than MSE for Ensembles

**MSE on Ensemble Mean:**
```python
mean_pred = ensemble.mean(dim=0)
mse = ((mean_pred - gt) ** 2).mean()
```

**Problem:** Ignores ensemble spread! Two very different ensembles can have same mean.

**CRPS on Ensemble:**
```python
crps = compute_crps_ensemble(ensemble, gt)
```

**Advantage:** Evaluates full distribution, rewards calibration.

### Connection to Proper Scoring Rules

CRPS is a **proper scoring rule**:
- Minimized when forecast distribution = true distribution
- Encourages honest, well-calibrated predictions
- Cannot be gamed by over/under-confidence

### Comparison Table

| Metric | Type | Ensemble | Calibration | Sharpness | Distribution | Deterministic |
|--------|------|----------|-------------|-----------|--------------|---------------|
| MSE | Point | ✗ | ✗ | ✗ | ✗ | ✓ |
| MAE | Point | ✗ | ✗ | ✗ | ✗ | ✓ |
| PSNR | Point | ✗ | ✗ | ✗ | ✗ | ✓ |
| CRPS | Probabilistic | ✓ | ✓ | ✓ | ✗ | ✓ (M=1) |
| FID | Distribution | ✗ | ✗ | ✗ | ✓ | ✓ |

## Troubleshooting

### High CRPS

**If CRPS >> MAE:**
- Ensemble too spread out
- Check: Are ensemble members very different?
- Solution: Tune diffusion sampling (fewer steps, different sampler)

### Low Diversity

**If all ensemble members identical:**
- CRPS ≈ MAE
- Check: Different noise seed each sample?
- Solution: Ensure proper noise sampling in `diffuse()`

### Poor Reconstruction

**If MSE high:**
- Model not trained enough
- Context not being used
- Check: Is `context_image` actually reaching model?

## References

1. **CRPS**: Gneiting & Raftery (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation"
2. **DDO**: Lim et al. (2023). "Score-based Diffusion Models in Function Space"
3. **Conditional Diffusion**: Various works on classifier-free guidance

## Citation

If you use this evaluation framework:

```bibtex
@article{lim2023score,
  title={Score-based diffusion models in function space},
  author={Lim*, Jae Hyun and Kovachki*, Nikola B and Baptista*, Ricardo and others},
  journal={arXiv preprint arXiv:2302.07400},
  year={2023}
}
```

---

**Status:** ✅ Complete evaluation framework implemented

**Key Achievement:** Comprehensive probabilistic evaluation with CRPS, enabling fair comparison between diffusion models and deterministic baselines.
