# Fixes Applied to CIFAR-10 DDO Repository

This document summarizes all the fixes and improvements made to avoid common errors.

## 1. Missing Import: SpectralCutoff2d

**Error**: `ImportError: cannot import name 'SpectralCutoff2d' from 'lib.models.fourier'`

**Fix**: Added `SpectralCutoff2d` class to `/lib/models/fourier.py`

```python
class SpectralCutoff2d(nn.Module):
    """Spectral cutoff layer that zeros out high frequency modes."""
    def __init__(self, modes_height, modes_width):
        super().__init__()
        self.modes_height = modes_height
        self.modes_width = modes_width

    def forward(self, x):
        # Zeros out high frequencies beyond the specified modes
        ...
```

**Location**: `lib/models/fourier.py` line 221

---

## 2. Model Parameter Counting Error

**Error**: `AttributeError: 'function' object has no attribute 'named_parameters'`

**Issue**: Tried to access `gen_sde.model` but `model` is a method, not the actual model object.

**Fix**: Use `gen_sde._model` instead of `gen_sde.model`

```python
# WRONG
num_params = count_parameters_in_M(gen_sde.model)

# CORRECT
num_params = count_parameters_in_M(gen_sde._model)
```

**Files Updated**:
- `notebooks/cifar10_train_ddo.ipynb` (cell: init-model)

---

## 3. tqdm Import Conflict

**Error**: `TypeError: 'module' object is not callable`

**Issue**: `main.py` has `import tqdm` which makes tqdm a module. After `from main import *`, this overwrites the function import.

**Fix**: Re-import tqdm as a function after importing from main

```python
# At the top
from tqdm.auto import tqdm

# After main imports
from main import *

# Re-import before using
from tqdm.auto import tqdm  # Re-import since main.py overwrites it
```

**Files Updated**:
- `notebooks/cifar10_train_ddo.ipynb` (cells: imports, train-loop)
- `notebooks/cifar10_sparse_reconstruction.ipynb` (cell: imports)

---

## 4. Model Size Too Large

**Issue**: Original configuration created ~1000M parameter model (way too large for CIFAR-10)

**Fix**: Reduced model size significantly

### Before:
```python
args.ch = 128
args.ch_mult = (1,2,2,2)  # 4 levels
args.num_res_blocks = 4
args.modes = 32
```
**Result**: ~1000M parameters

### After:
```python
args.ch = 64              # 50% reduction
args.ch_mult = (1,2,2)    # 3 levels (removed 1)
args.num_res_blocks = 2   # 50% reduction
args.modes = 16           # 50% reduction
```
**Result**: ~10-50M parameters (appropriate for CIFAR-10)

**Files Updated**:
- `train_cifar10.sh`
- `notebooks/cifar10_train_ddo.ipynb` (cell: config)

---

## 5. Wrong Training Loss Function

**Error**: `TypeError: only integer tensors of a single element can be converted to an index`

**Issue**: Used `diffuse()` function (for sampling) instead of `dsm()` (for training)

**Fix**: Use correct training loss

```python
# WRONG - diffuse() is for sampling/generation
loss = diffuse(gen_sde, x, t)

# CORRECT - dsm() is for training (Denoising Score Matching)
v = get_mgrid(2, x.shape[-1]).repeat(x.shape[0], 1, 1, 1).cuda()
loss = gen_sde.dsm(x, v).mean()
```

**Files Updated**:
- `notebooks/cifar10_train_ddo.ipynb` (cell: train-loop)

---

## 6. Missing Image Saving

**Issue**: Generated samples only saved to TensorBoard, not as viewable PNG files

**Fix**: Added PNG file saving in addition to TensorBoard

```python
# Save to TensorBoard
writer.add_image('train/samples', ...)

# Also save as PNG file
sample_dir = os.path.join(args.exp_path, 'samples')
os.makedirs(sample_dir, exist_ok=True)
torchvision.utils.save_image(
    sample[:nrow**2],
    os.path.join(sample_dir, f'iter_{count:06d}.png'),
    nrow=nrow,
    padding=2,
    normalize=True,
    value_range=(0, 1)
)
```

**Files Updated**:
- `notebooks/cifar10_train_ddo.ipynb` (cell: train-loop)

---

## Summary of Key Improvements

1. ✅ Fixed missing `SpectralCutoff2d` class
2. ✅ Corrected model parameter counting
3. ✅ Resolved tqdm import conflicts
4. ✅ Reduced model size to appropriate scale for CIFAR-10
5. ✅ Fixed training loss function (dsm vs diffuse)
6. ✅ Added PNG image saving for easy visualization
7. ✅ Updated all notebooks and scripts with fixes

## Files Modified

### Core Library:
- `lib/models/fourier.py` - Added SpectralCutoff2d

### Training Scripts:
- `train_cifar10.sh` - Reduced model size
- `notebooks/cifar10_train_ddo.ipynb` - All fixes applied
- `notebooks/cifar10_sparse_reconstruction.ipynb` - Import fixes

### New Files:
- `utils/sparse_datasets.py` - Sparse reconstruction utilities
- `notebooks/cifar10_sample_ddo.ipynb` - Sampling notebook
- `environment.yml` - Conda environment
- `requirements.txt` - Pip requirements
- `INSTALL.md` - Installation guide

## How to Use

### Standard CIFAR-10 Training:
```bash
conda env create -f environment.yml
conda activate ddo-cifar10
bash train_cifar10.sh ./experiments/cifar10 ./data ./fid_stats
```

### Training in Jupyter:
```bash
jupyter notebook notebooks/cifar10_train_ddo.ipynb
```

All common errors have been addressed and the codebase is ready to use!
