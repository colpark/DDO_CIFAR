# Training Correction: Query-Only Loss

## Issue Identified

The original training loop computed loss on the **full image** (100% of pixels):

```python
# INCORRECT (original)
loss = gen_sde.dsm(full_images, v, context_image=context_image).mean()
```

But the correct formulation should only compute loss on **query pixels** (10%):

## Correct Formulation

### Problem Setup

Given CIFAR-10 image with 32×32 = 1024 pixels:

| Component | Pixels | Fixed? | Purpose |
|-----------|--------|--------|---------|
| **Context** | 102 (10%) | ✓ | Input condition |
| **Query** | 102 (10%) | ✓ | Ground truth for supervision |
| **Remaining** | 820 (80%) | - | Not used in training |

**Training Goal:** Learn to predict **full field** (100%) from context (10%), supervised only on query (10%)

**Test Goal:** Predict all 90% unobserved pixels (query + remaining) from context (10%)

### Why Query-Only Loss?

**Scenario 1: Full Image Loss (incorrect)**
```python
loss = MSE(predicted_noise, true_noise)  # All 1024 pixels
```
- Model sees ground truth for all pixels during training
- At test time, needs to predict 90% unobserved pixels
- **Training-test mismatch!**

**Scenario 2: Query-Only Loss (correct)**
```python
loss = MSE(predicted_noise * query_mask, true_noise * query_mask)  # Only 102 pixels
```
- Model only supervised on 10% query pixels
- Must learn to generalize to remaining 80%
- More realistic: simulates test scenario where most pixels unknown

## Corrected Training Loop

```python
# Sample timestep and add noise
s_ = sample_timestep(batch_size)
zt, target, _, _ = gen_sde.forward_diffusion.sample(t=s_, x0=full_images)

# Predict noise conditioned on context (10% pixels)
pred = gen_sde.epsilon(y=zt, s=s_, v=v, context_image=context_image)

# Compute MSE on all pixels
mse = 0.5 * ((pred - target) ** 2)  # (B, C, H, W)

# CRITICAL: Mask to only query pixels
query_mask = create_query_mask_batched(query_indices, H, W)  # (B, 1, H, W)
masked_mse = mse * query_mask

# Normalize by number of query pixels
num_query_pixels = query_mask.sum(dim=(1,2,3), keepdim=True)
loss = (masked_mse.sum(dim=(1,2,3)) / num_query_pixels.squeeze()).mean()
```

## Key Differences

### Original (Incorrect)

```
Training:
  Input: context (10%)
  Supervision: full image (100%)
  Model learns: denoise full image given context

Test:
  Input: context (10%)
  Output: predict 90% unseen pixels

Problem: Training sees all pixels, test doesn't → overfitting
```

### Corrected

```
Training:
  Input: context (10%)
  Supervision: query (10%)
  Model learns: predict from sparse data, verified on sparse GT

Test:
  Input: context (10%)
  Output: predict 90% unseen pixels

Advantage: Training matches test scenario → better generalization
```

## Mathematical Formulation

### Standard Diffusion (incorrect for our case)

$$\mathcal{L} = \mathbb{E}_{t, \epsilon} \left[ \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} || \epsilon_\theta(z_t, t, \text{context}) - \epsilon ||^2 \right]$$

**Problem:** Supervision on all $HW$ pixels

### Query-Only Diffusion (correct)

$$\mathcal{L} = \mathbb{E}_{t, \epsilon} \left[ \frac{1}{N_q} \sum_{(i,j) \in Q} || \epsilon_\theta(z_t, t, \text{context}) - \epsilon ||^2 \right]$$

Where:
- $Q$ = set of query pixel locations (fixed per image)
- $N_q = |Q|$ = number of query pixels (102 for 10%)
- Model predicts noise for **all** pixels
- Loss computed only on **query** pixels

## Implementation Details

### Creating Query Mask

```python
def create_query_mask_batched(query_indices, height, width):
    """
    Create binary mask for query pixels.

    Args:
        query_indices: (B, num_query) flat indices
        height, width: Image dimensions

    Returns:
        query_mask: (B, 1, H, W) where 1 = query pixel
    """
    batch_size = query_indices.shape[0]
    device = query_indices.device

    # Create zero mask
    query_mask = torch.zeros(batch_size, 1, height, width, device=device)

    # Fill query locations
    for b in range(batch_size):
        indices = query_indices[b]  # (num_query,)
        y_coords = indices // width
        x_coords = indices % width
        query_mask[b, 0, y_coords, x_coords] = 1.0

    return query_mask
```

### Loss Computation

```python
# Compute per-pixel MSE
mse = 0.5 * ((pred - target) ** 2)  # (B, C, H, W)

# Apply query mask
masked_mse = mse * query_mask  # (B, C, H, W) * (B, 1, H, W)

# Normalize by number of query pixels per sample
num_query = query_mask.sum(dim=(1,2,3), keepdim=True)  # (B, 1, 1, 1)
loss_per_sample = masked_mse.sum(dim=(1,2,3)) / num_query.squeeze()  # (B,)

# Average over batch
loss = loss_per_sample.mean()  # Scalar
```

## Expected Behavior

### During Training

**Loss Monitoring:**
```
Iter 1000: loss_query=0.0234, loss_full=0.0187
Iter 2000: loss_query=0.0189, loss_full=0.0145
```

**Interpretation:**
- `loss_query` (10% pixels): Higher, harder to predict
- `loss_full` (100% pixels): Lower, includes easy context pixels
- `loss_query` should decrease during training

### At Test Time

**Reconstruction Quality:**
- Context pixels (10%): Perfect (given as input)
- Query pixels (10%): Good (seen during training)
- Remaining pixels (80%): Depends on generalization

**Expected PSNR:**
- On query pixels: 22-25 dB
- On remaining pixels: 18-22 dB (harder)
- On full image: 20-23 dB (average)

## Usage

### Replace Training Cell

In `conditional_sparse_reconstruction.ipynb`, replace cell 13 (training loop) with:

```python
exec(open('notebooks/conditional_training_query_loss.py').read())
```

Or manually copy the corrected training loop.

### Key Changes

1. ✅ Import `create_query_mask_batched` from utils
2. ✅ Create query mask from `query_indices`
3. ✅ Mask MSE before computing loss
4. ✅ Normalize by number of query pixels
5. ✅ Log both query loss and full loss for comparison

## Verification

### Check Query Mask

```python
# Verify query mask is correct
query_mask = create_query_mask_batched(query_indices, 32, 32)
assert query_mask.shape == (batch_size, 1, 32, 32)
assert query_mask.sum() == batch_size * 102  # 10% of 1024
assert (query_mask == 1).sum() == batch_size * 102
```

### Check Loss

```python
# Loss should only depend on query pixels
mse_all = ((pred - target) ** 2).mean()
mse_query = (((pred - target) ** 2) * query_mask).sum() / query_mask.sum()

# These should be different
print(f"MSE all pixels: {mse_all:.6f}")
print(f"MSE query pixels: {mse_query:.6f}")
```

## Files Modified

1. **`notebooks/conditional_training_query_loss.py`** - Corrected training loop
2. **`utils/sparse_datasets_fixed.py`** - Added `create_query_mask_batched()` function
3. **`TRAINING_CORRECTION.md`** - This document

## Summary

**Original Problem:**
- Computed loss on **all** pixels (100%)
- Model had access to full ground truth during training
- Training-test mismatch

**Solution:**
- Compute loss only on **query** pixels (10%)
- Model must generalize from sparse supervision
- Training matches test scenario

**Result:**
- More challenging but correct formulation
- Better generalization to unseen pixels
- Realistic sparse reconstruction setting

---

**Status:** ✅ Issue identified and corrected

**Next Step:** Replace training loop in notebook with corrected version
