# Score-based Diffusion Models in Function Space

This is a codebase for the following paper

**Score-based Diffusion Models in Function Space**

by Jae Hyun Lim\*, Nikola B Kovachki\*, Ricardo Baptista\*, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, Anima Anandkumar

[[arXiv](https://arxiv.org/abs/2302.07400)]

## Experiments on CIFAR-10 Dataset
CIFAR-10 experiments use DDO directly in pixel space (no SDF transform needed).

### Standard CIFAR-10 Generation

### Training
```bash
bash train_cifar10.sh ${EXP_PATH} ${DATA_PATH} ${FID_PATH}
```

Or use the Python command directly:
```
python main.py --command_type=train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=50000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=cifar10 --train_img_height=32 --input_dim=3 --coord_dim=2 \
  --model=fnounet2d --use_pos --modes=32 --ch=128 --ch_mult=1,2,2,2 --num_res_blocks=4 --dropout=0.1 --norm=group_norm --use_pointwise_op \
  --ns_method=vp_cosine --timestep_sampler=low_discrepancy \
  --disp_method=sine --sigma_blur_min=0.05 --sigma_blur_max=0.25 \
  --gp_type=exponential --gp_exponent=2.0 --gp_length_scale=0.05 --gp_sigma=1.0 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --train_batch_size=128 --lr=0.0002 --weight_decay=0.0 --num_iterations=500000 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=32 --eval_batch_size=256 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=${FID_PATH} \
  --vis_batch_size=64
```

### Sample generation
Run the notebook:
- `notebooks/cifar10_sample_ddo.ipynb`

### Evaluation
```bash
bash eval_cifar10.sh ${EXP_PATH} ${DATA_PATH} ${FID_PATH}
```

Or use the Python command:
```
python main.py --command_type=test \
  --exp_path=${EXP_PATH} \
  --data=${DATA_PATH} --dataset=cifar10 --train_img_height=32 --input_dim=3 --coord_dim=2 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=32 --eval_batch_size=512 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=${FID_PATH} \
  --checkpoint_file=checkpoint_fid.pt \
  --eval_fid
```

### Sparse CIFAR-10 Image Reconstruction

Reconstruct full images from 10% observed pixels (randomly sampled).

**Interactive Demo:**
```bash
jupyter notebook notebooks/cifar10_sparse_reconstruction.ipynb
```

This notebook demonstrates:
- Loading sparse CIFAR-10 dataset (10% context pixels, 10% query pixels)
- Visualizing different sampling patterns (random, grid, center)
- Data format for conditional reconstruction training

**Key Features:**
- **Context**: 10% randomly observed pixels (input)
- **Query**: 10% additional pixels for training supervision
- **Task**: Predict all unobserved pixels given context
- **Training**: Different random masks each iteration
- **Evaluation**: Fixed masks for consistent comparison

**Usage Example:**
```python
from utils.sparse_datasets import SparseImageDatasetWrapper

# Wrap any image dataset
sparse_dataset = SparseImageDatasetWrapper(
    dataset=cifar10_dataset,
    context_ratio=0.1,   # 10% observed
    query_ratio=0.1,     # 10% query for training
    mode='train'
)

# Each sample contains:
# - context_coords: (num_context, 2) - positions of observed pixels
# - context_values: (num_context, 3) - RGB values at observed positions
# - query_coords: (num_query, 2) - positions to predict
# - query_values: (num_query, 3) - ground truth for training
```

## Experiments on MNIST-SDF Dataset
Here's example command lines for training DDO and GANO models

### DDO
```
python main.py --command_type=train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=50000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=mnistsdf_32 --train_img_height=32 --input_dim=1 --coord_dim=2 --transform=sdf \
  --model=fnounet2d --use_pos --modes=32 --ch=64 --ch_mult=1,2,2 --num_res_blocks=4 --dropout=0.0 --norm=group_norm --use_pointwise_op \
  --ns_method=vp_cosine --timestep_sampler=low_discrepancy \
  --disp_method=sine --sigma_blur_min=0.05 --sigma_blur_max=0.25 \
  --gp_type=exponential --gp_exponent=2.0 --gp_length_scale=0.05 --gp_sigma=1.0 \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --train_batch_size=32 --lr=0.0001 --weight_decay=0.0 --num_iterations=2000000 \
  --upsample --upsample_resolution=64 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=64 --eval_batch_size=256 --eval_num_samples=5000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir={FID_PATH} \
  --vis_batch_size=36
```


### GANO

```
python gano.py --command_type=train \
  --exp_path=${EXP_PATH} \
  --seed=1 --print_every=1000 --save_every=5000 --ckpt_every=100000 --eval_every=20000 --vis_every=10000 --resume \
  --data=${DATA_PATH} --dataset=mnistsdf_32 --train_img_height=32 --input_dim=1 --coord_dim=2 \
  --model=gano-uno --modes=32 --d_co_domain=64 --lmbd_grad=10.0 --n_critic=10 \
  --train_batch_size=32 --lr=0.0001 --weight_decay=0.0 --num_iterations=1000000 \
  --upsample --upsample_resolution=64 \
  --eval_fid --eval_use_ema --ema_decay=0.999 --eval_img_height=64 --eval_batch_size=512 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir={FID_PATH} \
  --vis_batch_size=36
```

### Sample generation
Run the following notebook files
- `notebooks/mnistsdf_sample_ddo.ipynb`
- `notebooks/mnistsdf_sample_gano.ipynb`

### Evaluations
Save the below model as `${EXP_PATH}/checkpoint_fid.pt`
```
python main.py --command_type=test \
  --exp_path=${EXP_PATH} \
  --data=${DATA_PATH} --dataset=mnistsdf_32 --train_img_height=32 --input_dim=1 --coord_dim=2 --transform=sdf \
  --num_steps=250 --sampler=denoise --s_min=0.0001 \
  --upsample --upsample_resolution=64 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=64 --eval_batch_size=1024 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=${FID_PATH} \
  --checkpoint_file=checkpoint_fid.pt \
  --eval_fid
```
```
python gano.py --command_type=test \
  --exp_path=${EXP_PATH} \
  --data=${DATA_PATH} --dataset=mnistsdf_32 --train_img_height=32 --input_dim=1 --coord_dim=2 \
  --model=gano-uno --modes=32 --d_co_domain=64 --lmbd_grad=10.0 --n_critic=10 \
  --upsample --upsample_resolution=64 \
  --eval_use_ema --ema_decay=0.999 --eval_img_height=64 --eval_batch_size=1024 --eval_num_samples=50000 --eval_resize_mode=tensor --eval_interpolation=bilinear --fid_dir=${FID_PATH} \
  --checkpoint_file=checkpoint_fid.pt \
  --eval_fid
```

### Pre-trained models (Google Drive)
Save the model as `${EXP_PATH}/checkpoint_fid.pt`
- DDO  [[link](https://drive.google.com/file/d/1aMKEIEMI2sZKeK0TbFwHxDUNh6B-bP2l/view)]
- GANO [[link](https://drive.google.com/file/d/1aDa6sf5WFbW85kiTewvJbN55fhFZbx1M/view)]

## Experiments in Appendix K
Run the following notebook files
- `notebooks/afhq_fitting_ddo_big.ipynb`
- `notebooks/afhq_fitting_ddo.ipynb`
- `notebooks/afhq_fitting_uno.ipynb`
- `notebooks/afhq_fitting_fno.ipynb`
- `notebooks/afhq_fitting_sparse_unet_cc_det.ipynb`

## Experiments on Gaussian Mixture, Navier-Stokes, Volcano Dataset, and Darcyflow
We will update the repo.

## License
MIT License

## Citation
```
@article{lim2023score,
  title={Score-based diffusion models in function space},
  author={Lim\*, Jae Hyun and Kovachki\*, Nikola B and Baptista\*, Ricardo and Beckham, Christopher and Azizzadenesheli, Kamyar and Kossaifi, Jean and Voleti, Vikram and Song, Jiaming and Kreis, Karsten and Kautz, Jan and others},
  journal={arXiv preprint arXiv:2302.07400},
  year={2023}
}
```
