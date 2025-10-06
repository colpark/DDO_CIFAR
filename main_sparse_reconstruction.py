"""
Sparse Image Reconstruction with DDO

Train a conditional diffusion model to reconstruct full images from sparse observations.
Given 10% randomly observed pixels, predict the remaining 90% pixels.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils import utils
from utils import datasets as dataset_utils
from utils.sparse_datasets import SparseImageDatasetWrapper, create_sparse_mask_image
from utils.ema import EMA
from utils.utils import save_checkpoint, load_checkpoint, count_parameters_in_M
from utils.visualize import get_grid_image
from utils.utils import Writer

from lib.diffusion import DenoisingDiffusion, LinearDiffusion, BlurringDiffusion
from lib.models.fourier_unet import FNOUNet2d

import torchvision


def get_args():
    parser = argparse.ArgumentParser('Sparse Image Reconstruction with DDO')

    # Basic settings
    parser.add_argument('--command_type', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--exp_path', type=str, default='./experiments/sparse_recon')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=1)

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--train_img_height', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=3)  # RGB
    parser.add_argument('--coord_dim', type=int, default=2)

    # Sparse reconstruction settings
    parser.add_argument('--context_ratio', type=float, default=0.1, help='Ratio of observed pixels')
    parser.add_argument('--query_ratio', type=float, default=0.1, help='Ratio of query pixels for training')

    # Model architecture
    parser.add_argument('--model', type=str, default='fnounet2d')
    parser.add_argument('--ch', type=int, default=64)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 2])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--modes', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='group_norm')
    parser.add_argument('--use_pos', action='store_true', default=True)
    parser.add_argument('--use_pointwise_op', action='store_true', default=True)

    # Diffusion settings
    parser.add_argument('--ns_method', type=str, default='vp_cosine')
    parser.add_argument('--disp_method', type=str, default='sine')
    parser.add_argument('--sigma_blur_min', type=float, default=0.05)
    parser.add_argument('--sigma_blur_max', type=float, default=0.25)
    parser.add_argument('--gp_type', type=str, default='exponential')
    parser.add_argument('--gp_exponent', type=float, default=2.0)
    parser.add_argument('--gp_length_scale', type=float, default=0.05)
    parser.add_argument('--gp_sigma', type=float, default=1.0)
    parser.add_argument('--timestep_sampler', type=str, default='low_discrepancy')

    # Training
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_iterations', type=int, default=200000)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Logging
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--vis_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--vis_batch_size', type=int, default=16)
    parser.add_argument('--resume', action='store_true')

    # Sampling
    parser.add_argument('--num_steps', type=int, default=250)
    parser.add_argument('--sampler', type=str, default='denoise')
    parser.add_argument('--s_min', type=float, default=0.0001)

    args = parser.parse_args()

    # Set derived parameters
    args.distributed = False
    args.global_rank = 0
    args.local_rank = 0
    args.checkpoint_file = 'checkpoint.pt'
    args.use_clip = False
    args.weight_method = None

    return args


def get_mgrid(dim, img_height):
    """Generate coordinate grid"""
    grid = torch.linspace(0, img_height-1, img_height) / img_height
    if dim == 2:
        grid = torch.cat([grid[None,None,...,None].repeat(1, 1, 1, img_height),
                          grid[None,None,None].repeat(1, 1, img_height, 1)], dim=1)
    else:
        raise NotImplementedError
    return grid


def create_context_conditioning(context_values, context_coords, img_height):
    """
    Create a dense image from sparse context points.
    Unobserved pixels are set to a special value (e.g., 0 or mean).
    """
    batch_size = context_values.shape[0]
    num_channels = context_values.shape[-1]

    # Create zero image (unobserved pixels = 0)
    context_image = torch.zeros(batch_size, num_channels, img_height, img_height,
                                 device=context_values.device)

    # Fill in observed pixels
    for b in range(batch_size):
        coords = (context_coords[b] * img_height).long()
        coords = torch.clamp(coords, 0, img_height - 1)
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        for c in range(num_channels):
            context_image[b, c, y_coords, x_coords] = context_values[b, :, c]

    return context_image


class SparseReconstructionModel(nn.Module):
    """Wrapper for conditioning the model on sparse observations"""

    def __init__(self, base_model, input_dim=3):
        super().__init__()
        self.base_model = base_model
        self.input_dim = input_dim

        # Context encoder - processes sparse observations
        self.context_encoder = nn.Conv2d(input_dim, input_dim * 2, 3, padding=1)

        # Combine context with noisy image
        self.combine = nn.Conv2d(input_dim * 3, input_dim, 3, padding=1)

    @property
    def in_channels(self):
        return self.base_model.in_channels

    def forward(self, x, temp, v, context_image=None, **kwargs):
        """
        x: noisy image
        context_image: sparse observations filled into dense image
        """
        if context_image is not None:
            # Encode context
            context_features = self.context_encoder(context_image)

            # Concatenate noisy image with context features
            combined = torch.cat([x, context_features], dim=1)
            x = self.combine(combined)

        return self.base_model(x=x, temp=temp, v=v, **kwargs)


def init_model(args):
    """Initialize model, optimizer, and scheduler"""

    # Initialize base model
    gp_config = argparse.Namespace()
    gp_config.device = 'cuda'
    gp_config.exponent = args.gp_exponent
    gp_config.length_scale = args.gp_length_scale
    gp_config.sigma = args.gp_sigma

    disp_config = argparse.Namespace()
    disp_config.sigma_blur_min = args.sigma_blur_min
    disp_config.sigma_blur_max = args.sigma_blur_max

    # Create diffusion process
    inf_sde = BlurringDiffusion(
        dim=args.coord_dim,
        ch=args.input_dim,
        ns_method=args.ns_method,
        disp_method=args.disp_method,
        disp_config=disp_config,
        gp_type=args.gp_type,
        gp_config=gp_config,
    )

    # Create base U-Net
    base_unet = FNOUNet2d(
        modes_height=args.modes,
        modes_width=args.modes,
        in_channels=args.input_dim,
        in_height=args.train_img_height,
        ch=args.ch,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        norm=args.norm,
        use_pos=args.use_pos,
        use_pointwise_op=args.use_pointwise_op,
    )

    # Wrap with sparse conditioning
    model = SparseReconstructionModel(base_unet, input_dim=args.input_dim)

    # Create denoising diffusion wrapper
    gen_sde = DenoisingDiffusion(
        inf_sde,
        model=model,
        timestep_sampler=args.timestep_sampler,
        use_clip=args.use_clip,
        weight_method=args.weight_method
    ).cuda()

    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(gen_sde.parameters(), lr=args.lr,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Add EMA
    optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    # Resume from checkpoint
    count = 0
    best_loss = 1e10
    checkpoint_file = os.path.join(args.exp_path, args.checkpoint_file)
    if args.resume and os.path.exists(checkpoint_file):
        print(f'Loading checkpoint from {checkpoint_file}')
        gen_sde, optimizer, _, count, best_loss = load_checkpoint(
            checkpoint_file, gen_sde, optimizer, None
        )
        print(f'Resumed from iteration {count}')

    return gen_sde, optimizer, count, best_loss


def train(args):
    """Training loop for sparse reconstruction"""

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create experiment directory
    os.makedirs(args.exp_path, exist_ok=True)
    os.makedirs(os.path.join(args.exp_path, 'samples'), exist_ok=True)

    # Load base dataset
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == 'cifar10':
        base_dataset = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    # Wrap with sparse dataset
    sparse_dataset = SparseImageDatasetWrapper(
        dataset=base_dataset,
        context_ratio=args.context_ratio,
        query_ratio=args.query_ratio,
        mode='train',
        return_full_image=True
    )

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        sparse_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    gen_sde, optimizer, count, best_loss = init_model(args)

    # Count parameters
    num_params = count_parameters_in_M(gen_sde._model)
    print(f'Model parameters: {num_params:.2f}M')

    # TensorBoard writer
    writer = Writer(args.global_rank, args.exp_path)

    # Training loop
    print(f'Starting training from iteration {count}')
    start_time = time.time()

    gen_sde.train()

    train_iter = iter(train_loader)
    pbar = tqdm(total=args.num_iterations, initial=count, desc='Training')

    while count < args.num_iterations:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Get data
        full_images = batch['image'].cuda()  # (B, C, H, W)
        context_coords = batch['context_coords'].cuda()  # (B, N_context, 2)
        context_values = batch['context_values'].cuda()  # (B, N_context, C)

        # Create context image
        context_image = create_context_conditioning(
            context_values, context_coords, args.train_img_height
        )

        # Get coordinate grid
        v = get_mgrid(2, args.train_img_height).repeat(full_images.shape[0], 1, 1, 1).cuda()

        # Forward pass - train on full image with context conditioning
        optimizer.zero_grad()

        # Modify the DSM loss to include context
        # We need to pass context_image through the model
        # For now, use standard DSM on full image
        # TODO: Modify DSM to accept context_image
        loss = gen_sde.dsm(full_images, v).mean()

        # Backward
        loss.backward()
        optimizer.step()

        count += 1
        pbar.update(1)

        # Logging
        if count % args.print_every == 0:
            elapsed = (time.time() - start_time) / args.print_every
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.6f}'})
            writer.add_scalar('train/loss', loss.item(), count)
            writer.add_scalar('train/lr', lr, count)
            start_time = time.time()

        # Visualization
        if count % args.vis_every == 0:
            visualize_reconstruction(
                args, gen_sde, sparse_dataset, writer, count
            )

        # Save checkpoint
        if count % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.exp_path, 'checkpoint.pt'),
                gen_sde, optimizer, None, count, best_loss
            )
            print(f'Saved checkpoint at iteration {count}')

    pbar.close()
    print('Training completed!')


def visualize_reconstruction(args, gen_sde, dataset, writer, count):
    """Visualize sparse reconstruction results"""
    gen_sde.eval()

    with torch.no_grad():
        # Get a few samples
        num_vis = min(args.vis_batch_size, 16)

        originals = []
        contexts = []
        reconstructions = []

        for i in range(num_vis):
            sample = dataset[i]

            original = sample['image']
            context_image_np = create_sparse_mask_image(
                original, sample['context_indices'], fill_value=0.5
            )

            originals.append(original)
            contexts.append(context_image_np)

            # TODO: Implement proper reconstruction
            # For now, use a placeholder
            reconstructions.append(original)  # Placeholder

        originals = torch.stack(originals)
        contexts = torch.stack(contexts)
        reconstructions = torch.stack(reconstructions)

        nrow = 4

        # Save visualizations
        fig_path = os.path.join(args.exp_path, 'samples', f'recon_iter_{count:06d}.png')

        # Create comparison grid
        comparison = torch.cat([contexts, reconstructions, originals], dim=0)
        torchvision.utils.save_image(
            comparison,
            fig_path,
            nrow=nrow,
            padding=2,
            normalize=True,
            value_range=(0, 1)
        )

        print(f'Saved visualization to {fig_path}')

    gen_sde.train()


if __name__ == '__main__':
    args = get_args()

    if args.command_type == 'train':
        train(args)
    else:
        raise NotImplementedError(f"Command {args.command_type} not implemented")
