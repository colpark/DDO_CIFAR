"""
Fixed-mask sparse dataset for conditional image reconstruction.

Each instance has a FIXED sparse mask that never changes during training.
This enables proper conditional generation: context (10%) → model → query (10%)
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class FixedSparseMaskDataset(Dataset):
    """
    Dataset wrapper where each image has a FIXED sparse mask.

    Key features:
    - Each of N images has a deterministic, fixed mask (doesn't change across epochs)
    - Total allowance: 20% = 10% context (input) + 10% query (GT target)
    - Context pixels are ALWAYS the same for each image throughout training
    - Query pixels are ALWAYS the same for each image throughout training
    - Model learns: f(noisy_image, context) → denoise to match query

    Args:
        dataset: Base dataset (e.g., CIFAR10 with 60,000 images)
        context_ratio: Fraction for context (default 0.1 = 10%)
        query_ratio: Fraction for query/GT (default 0.1 = 10%)
        seed: Random seed for generating fixed masks
    """

    def __init__(self,
                 dataset,
                 context_ratio: float = 0.1,
                 query_ratio: float = 0.1,
                 seed: int = 42):
        self.dataset = dataset
        self.context_ratio = context_ratio
        self.query_ratio = query_ratio
        self.seed = seed

        # Get image dimensions
        sample_img, _ = dataset[0]
        self.num_channels = sample_img.shape[0]
        self.height = sample_img.shape[1]
        self.width = sample_img.shape[2]
        self.num_pixels = self.height * self.width

        self.num_context = int(self.num_pixels * context_ratio)
        self.num_query = int(self.num_pixels * query_ratio)

        # Pre-generate ALL fixed masks for entire dataset
        print(f"Generating fixed masks for {len(dataset)} images...")
        self.fixed_masks = self._generate_all_masks()
        print(f"Fixed masks generated: {self.num_context} context + {self.num_query} query per image")

    def _generate_all_masks(self):
        """Generate fixed masks for all images in dataset."""
        masks = []
        rng = np.random.RandomState(self.seed)

        for idx in range(len(self.dataset)):
            # Generate fixed mask for this specific image
            perm = torch.from_numpy(rng.permutation(self.num_pixels))

            # First 10% for context
            context_indices = perm[:self.num_context]

            # Next 10% for query (non-overlapping)
            query_indices = perm[self.num_context:self.num_context + self.num_query]

            masks.append({
                'context_indices': context_indices,
                'query_indices': query_indices
            })

        return masks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get full image
        image, label = self.dataset[index]  # (C, H, W)

        # Get FIXED mask for this image
        mask = self.fixed_masks[index]
        context_indices = mask['context_indices']
        query_indices = mask['query_indices']

        # Reshape image to (C, H*W)
        image_flat = image.reshape(self.num_channels, -1)

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

        return {
            'image': image,  # (C, H, W) - full image
            'context_coords': context_coords.T,  # (num_context, 2)
            'context_values': context_values.T,  # (num_context, C)
            'query_coords': query_coords.T,      # (num_query, 2)
            'query_values': query_values.T,      # (num_query, C)
            'context_indices': context_indices,  # (num_context,)
            'query_indices': query_indices,      # (num_query,)
            'label': label
        }

    def __repr__(self):
        return (f"FixedSparseMaskDataset(\n"
                f"  base_dataset={self.dataset.__class__.__name__},\n"
                f"  num_images={len(self.dataset)},\n"
                f"  image_size=({self.num_channels}, {self.height}, {self.width}),\n"
                f"  context_ratio={self.context_ratio} ({self.num_context} pixels),\n"
                f"  query_ratio={self.query_ratio} ({self.num_query} pixels),\n"
                f"  seed={self.seed}\n"
                f")")


def create_context_image_batched(context_values, context_indices, height, width, num_channels):
    """
    Create dense context images from sparse observations (batched version).

    Args:
        context_values: (B, num_context, C)
        context_indices: (B, num_context)
        height, width: Image dimensions
        num_channels: Number of channels

    Returns:
        context_images: (B, C, H, W) with observed pixels filled, rest = 0
    """
    batch_size = context_values.shape[0]
    device = context_values.device

    # Create zero images
    context_images = torch.zeros(batch_size, num_channels, height, width, device=device)

    # Fill in observed pixels for each batch
    for b in range(batch_size):
        indices = context_indices[b]  # (num_context,)
        values = context_values[b]    # (num_context, C)

        # Convert flat indices to 2D
        y_coords = indices // width
        x_coords = indices % width

        # Fill in values
        for c in range(num_channels):
            context_images[b, c, y_coords, x_coords] = values[:, c]

    return context_images


def create_query_mask_batched(query_indices, height, width):
    """
    Create query mask for loss computation.

    Args:
        query_indices: (B, num_query)
        height, width: Image dimensions

    Returns:
        query_mask: (B, 1, H, W) binary mask where 1 = query pixel
    """
    batch_size = query_indices.shape[0]
    device = query_indices.device

    # Create zero mask
    query_mask = torch.zeros(batch_size, 1, height, width, device=device)

    # Fill in query locations
    for b in range(batch_size):
        indices = query_indices[b]  # (num_query,)

        # Convert flat indices to 2D
        y_coords = indices // width
        x_coords = indices % width

        query_mask[b, 0, y_coords, x_coords] = 1.0

    return query_mask


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
