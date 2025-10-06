"""
Sparse dataset utilities for conditional image reconstruction.

Given a subset of observed pixel values, the task is to reconstruct the full image.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class SparseImageDatasetWrapper(Dataset):
    """
    Dataset wrapper for sparse image reconstruction task.

    During training:
    - Sample a random subset of pixels as observed points (context)
    - Sample another subset as query points (ground truth for training)

    During inference:
    - Use fixed observed points as context
    - Reconstruct all remaining pixels

    Args:
        dataset: Base dataset (e.g., CIFAR10)
        context_ratio: Fraction of pixels to use as observed context (e.g., 0.1 for 10%)
        query_ratio: Fraction of pixels to use as query/target during training (e.g., 0.1)
        mode: 'train' or 'test'
        fixed_mask: If provided, use this fixed mask for context points (for evaluation)
    """

    def __init__(self,
                 dataset,
                 context_ratio: float = 0.1,
                 query_ratio: float = 0.1,
                 mode: str = 'train',
                 fixed_mask: torch.Tensor = None,
                 return_full_image: bool = False):
        self.dataset = dataset
        self.context_ratio = context_ratio
        self.query_ratio = query_ratio
        self.mode = mode
        self.fixed_mask = fixed_mask
        self.return_full_image = return_full_image

        # Get image dimensions from first sample
        sample_img, _ = dataset[0]
        self.num_channels = sample_img.shape[0]
        self.height = sample_img.shape[1]
        self.width = sample_img.shape[2]
        self.num_pixels = self.height * self.width

        self.num_context = int(self.num_pixels * context_ratio)
        self.num_query = int(self.num_pixels * query_ratio)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get full image
        image, label = self.dataset[index]  # (C, H, W)

        if self.mode == 'train':
            # Random sampling for training
            # Sample context points (observed pixels)
            context_indices = torch.randperm(self.num_pixels)[:self.num_context]

            # Sample query points (different from context, for training target)
            remaining_indices = torch.randperm(self.num_pixels)
            # Make sure query doesn't overlap with context
            mask = ~torch.isin(remaining_indices, context_indices)
            query_indices = remaining_indices[mask][:self.num_query]

        else:
            # Fixed or deterministic sampling for evaluation
            if self.fixed_mask is not None:
                context_indices = self.fixed_mask
            else:
                # Use deterministic mask based on index
                torch.manual_seed(index)
                context_indices = torch.randperm(self.num_pixels)[:self.num_context]
                torch.manual_seed(torch.initial_seed())  # Reset to random

            # For evaluation, query is all non-context pixels
            all_indices = torch.arange(self.num_pixels)
            mask = ~torch.isin(all_indices, context_indices)
            query_indices = all_indices[mask]

        # Reshape image to (C, H*W)
        image_flat = image.reshape(self.num_channels, -1)  # (C, H*W)

        # Extract context and query values
        context_values = image_flat[:, context_indices]  # (C, num_context)
        query_values = image_flat[:, query_indices]      # (C, num_query)

        # Create coordinate grid [0, 1]
        y_coords = torch.arange(self.height).float() / self.height
        x_coords = torch.arange(self.width).float() / self.width
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=0)  # (2, H*W)

        context_coords = coords[:, context_indices]  # (2, num_context)
        query_coords = coords[:, query_indices]      # (2, num_query)

        if self.return_full_image:
            # Return everything including full image (useful for visualization)
            return {
                'image': image,  # (C, H, W)
                'context_coords': context_coords.T,  # (num_context, 2)
                'context_values': context_values.T,  # (num_context, C)
                'query_coords': query_coords.T,      # (num_query, 2)
                'query_values': query_values.T,      # (num_query, C)
                'context_indices': context_indices,
                'query_indices': query_indices,
                'label': label
            }
        else:
            # Return format compatible with training loop
            return {
                'context_coords': context_coords.T,  # (num_context, 2)
                'context_values': context_values.T,  # (num_context, C)
                'query_coords': query_coords.T,      # (num_query, 2)
                'query_values': query_values.T,      # (num_query, C)
                'label': label
            }

    def __repr__(self):
        return (f"SparseImageDatasetWrapper(\n"
                f"  base_dataset={self.dataset.__class__.__name__},\n"
                f"  mode={self.mode},\n"
                f"  image_size=({self.num_channels}, {self.height}, {self.width}),\n"
                f"  context_ratio={self.context_ratio} ({self.num_context} pixels),\n"
                f"  query_ratio={self.query_ratio} ({self.num_query} pixels)\n"
                f")")


class GridMaskGenerator:
    """Generate various mask patterns for sparse sampling."""

    @staticmethod
    def random_mask(num_pixels, num_samples):
        """Random uniform sampling."""
        return torch.randperm(num_pixels)[:num_samples]

    @staticmethod
    def grid_mask(height, width, stride=2):
        """Regular grid sampling."""
        indices = []
        for i in range(0, height, stride):
            for j in range(0, width, stride):
                indices.append(i * width + j)
        return torch.tensor(indices)

    @staticmethod
    def center_mask(height, width, num_samples):
        """Sample from center region."""
        center_h, center_w = height // 2, width // 2
        radius = int(np.sqrt(num_samples / np.pi))

        indices = []
        for i in range(max(0, center_h - radius), min(height, center_h + radius)):
            for j in range(max(0, center_w - radius), min(width, center_w + radius)):
                if (i - center_h)**2 + (j - center_w)**2 <= radius**2:
                    indices.append(i * width + j)

        return torch.tensor(indices[:num_samples])

    @staticmethod
    def corner_mask(height, width, num_samples):
        """Sample from corners."""
        indices = []
        samples_per_corner = num_samples // 4

        # Top-left
        for i in range(height // 4):
            for j in range(width // 4):
                indices.append(i * width + j)
                if len(indices) >= samples_per_corner:
                    break
            if len(indices) >= samples_per_corner:
                break

        # Add other corners similarly...
        return torch.tensor(indices[:num_samples])


def create_sparse_mask_image(image, mask_indices, fill_value=0.5):
    """
    Create visualization of sparse sampling.

    Args:
        image: (C, H, W) tensor
        mask_indices: Indices of observed pixels
        fill_value: Value to fill unobserved pixels

    Returns:
        masked_image: (C, H, W) tensor with only masked pixels visible
    """
    C, H, W = image.shape
    masked_image = torch.ones_like(image) * fill_value

    # Reshape to flat
    image_flat = image.reshape(C, -1)
    masked_flat = masked_image.reshape(C, -1)

    # Copy observed values
    masked_flat[:, mask_indices] = image_flat[:, mask_indices]

    return masked_flat.reshape(C, H, W)


def collate_sparse_batch(batch):
    """
    Custom collate function for sparse image batches.

    Handles variable-length context and query points by padding or
    organizing into lists.
    """
    if isinstance(batch[0], dict):
        # Stack each field separately
        keys = batch[0].keys()
        collated = {}

        for key in keys:
            if key in ['context_coords', 'context_values', 'query_coords', 'query_values']:
                # These can have different lengths, so we keep as list or pad
                collated[key] = torch.stack([item[key] for item in batch])
            elif key in ['image']:
                collated[key] = torch.stack([item[key] for item in batch])
            elif key == 'label':
                collated[key] = torch.tensor([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]

        return collated
    else:
        # Standard collation
        return torch.utils.data.dataloader.default_collate(batch)
