"""
Conditional model wrapper for sparse image reconstruction.

Wraps the base FNO-UNet to accept sparse context conditioning.
"""

import torch
import torch.nn as nn


class ConditionalDDOModel(nn.Module):
    """
    Conditional wrapper for DDO that accepts sparse observations.

    Architecture:
        Context (sparse) → Context Encoder → Dense Features
                                                   ↓
        Noisy Image ─────────────────→ [Concatenate] → Projection → Base Model

    The context encoder converts sparse observations into dense feature maps
    that are concatenated with the noisy image before passing to the base model.
    """

    def __init__(self, base_model, input_dim=3, context_feature_dim=32):
        """
        Args:
            base_model: Base FNO-UNet model
            input_dim: Number of input channels (e.g., 3 for RGB)
            context_feature_dim: Number of features to extract from context
        """
        super().__init__()
        self.base_model = base_model
        self.input_dim = input_dim
        self.context_feature_dim = context_feature_dim

        # Context encoder: processes sparse observations
        # Input: (B, input_dim, H, W) where unobserved = 0
        # Output: (B, context_feature_dim, H, W) dense features
        self.context_encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, context_feature_dim, kernel_size=3, padding=1),
        )

        # Projection layer: combines noisy image + context features
        # Input: (B, input_dim + context_feature_dim, H, W)
        # Output: (B, input_dim, H, W)
        self.combine = nn.Sequential(
            nn.Conv2d(input_dim + context_feature_dim, input_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, input_dim * 2),
            nn.SiLU(),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1),
        )

    @property
    def in_channels(self):
        return self.base_model.in_channels

    def forward(self, x, temp, v, context_image=None, **kwargs):
        """
        Forward pass with optional context conditioning.

        Args:
            x: Noisy image (B, C, H, W)
            temp: Time step (B,)
            v: Coordinate grid (B, coord_dim, H, W)
            context_image: Sparse observations as dense image (B, C, H, W)
                          Observed pixels have values, unobserved = 0

        Returns:
            Predicted noise (B, C, H, W)
        """
        if context_image is not None:
            # Encode sparse context into dense features
            # context_image: (B, 3, 32, 32) with 90% zeros
            # context_features: (B, 32, 32, 32)
            context_features = self.context_encoder(context_image)

            # Concatenate noisy image with context features
            # x: (B, 3, 32, 32)
            # combined: (B, 3+32, 32, 32) = (B, 35, 32, 32)
            combined = torch.cat([x, context_features], dim=1)

            # Project back to input dimensions
            # (B, 35, 32, 32) → (B, 3, 32, 32)
            x = self.combine(combined)

        # Pass through base model
        return self.base_model(x=x, temp=temp, v=v, **kwargs)


class ConditionalDDOModelSimple(nn.Module):
    """
    Simpler conditional wrapper - just concatenates context with noisy image.

    This is faster and may work just as well for smaller images.
    """

    def __init__(self, base_model, input_dim=3):
        super().__init__()
        self.base_model = base_model
        self.input_dim = input_dim

        # Simple projection: concatenate and reduce
        self.combine = nn.Conv2d(input_dim * 2, input_dim, kernel_size=1)

    @property
    def in_channels(self):
        return self.base_model.in_channels

    def forward(self, x, temp, v, context_image=None, **kwargs):
        """
        Args:
            x: Noisy image (B, C, H, W)
            context_image: Sparse observations (B, C, H, W), unobserved = 0
        """
        if context_image is not None:
            # Concatenate noisy image with context
            # (B, 3, 32, 32) + (B, 3, 32, 32) → (B, 6, 32, 32)
            combined = torch.cat([x, context_image], dim=1)

            # Project to original channels
            # (B, 6, 32, 32) → (B, 3, 32, 32)
            x = self.combine(combined)

        return self.base_model(x=x, temp=temp, v=v, **kwargs)
