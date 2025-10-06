"""
Fréchet Inception Distance (FID) computation for image quality assessment.

FID measures the distance between the distribution of generated images
and real images in the feature space of an InceptionV3 network.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
from scipy import linalg


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 network for extracting features for FID computation.

    Uses the pool3 layer (2048-dimensional features before final classification).
    """

    def __init__(self, resize_input=True, normalize_input=True):
        """
        Args:
            resize_input: If True, bilinearly resize input to 299x299
            normalize_input: If True, normalize input to [-1, 1]
        """
        super().__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input

        # Load pretrained InceptionV3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()

        # Remove final FC layer (we want features, not classification)
        self.inception.fc = nn.Identity()

    def forward(self, x):
        """
        Extract 2048-dimensional features.

        Args:
            x: (B, 3, H, W) images in [0, 1]

        Returns:
            features: (B, 2048)
        """
        # Resize to 299x299 if needed (InceptionV3 requirement)
        if self.resize_input:
            if x.shape[2] != 299 or x.shape[3] != 299:
                x = nn.functional.interpolate(
                    x, size=(299, 299), mode='bilinear', align_corners=False
                )

        # Normalize to [-1, 1] (InceptionV3 expects this range)
        if self.normalize_input:
            x = 2 * x - 1

        # Forward through InceptionV3
        # Output shape: (B, 2048, 1, 1) after avgpool
        features = self.inception(x)

        # If output is 4D (shouldn't be with Identity fc, but just in case)
        if features.dim() == 4:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
            features = features.squeeze(3).squeeze(2)

        return features


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet distance between two multivariate Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))

    Args:
        mu1: Mean of first distribution (D,)
        sigma1: Covariance of first distribution (D, D)
        mu2: Mean of second distribution (D,)
        sigma2: Covariance of second distribution (D, D)
        eps: Epsilon for numerical stability

    Returns:
        FID score (scalar)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Means have different shapes"
    assert sigma1.shape == sigma2.shape, "Covariances have different shapes"

    # Calculate ||mu1 - mu2||^2
    diff = mu1 - mu2

    # Product might be complex with inf/nan, so add eps to diagonal
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Numerical stability: if imaginary component is small, discard it
    if not np.isfinite(covmean).all():
        print(f"FID calculation produced inf/nan. Adding {eps} to diagonal of covariance.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Discard imaginary component if negligible
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large")
        covmean = covmean.real

    # Calculate FID
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return fid


def compute_statistics_from_features(features):
    """
    Compute mean and covariance from features.

    Args:
        features: (N, D) feature vectors

    Returns:
        mu: (D,) mean vector
        sigma: (D, D) covariance matrix
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


@torch.no_grad()
def extract_features_from_images(images, extractor, batch_size=50, device='cuda'):
    """
    Extract InceptionV3 features from a batch of images.

    Args:
        images: (N, 3, H, W) tensor in [0, 1]
        extractor: InceptionV3FeatureExtractor
        batch_size: Batch size for feature extraction
        device: Device to use

    Returns:
        features: (N, 2048) numpy array
    """
    extractor = extractor.to(device)
    extractor.eval()

    num_images = images.shape[0]
    features_list = []

    for i in range(0, num_images, batch_size):
        batch = images[i:i+batch_size].to(device)
        batch_features = extractor(batch)
        features_list.append(batch_features.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    return features


def compute_fid(real_images, generated_images, batch_size=50, device='cuda'):
    """
    Compute FID between real and generated images.

    Args:
        real_images: (N, 3, H, W) tensor in [0, 1]
        generated_images: (N, 3, H, W) tensor in [0, 1]
        batch_size: Batch size for feature extraction
        device: Device to use

    Returns:
        fid_score: Scalar FID value (lower is better)
    """
    print(f"Computing FID...")
    print(f"  Real images: {real_images.shape[0]}")
    print(f"  Generated images: {generated_images.shape[0]}")

    # Initialize feature extractor
    extractor = InceptionV3FeatureExtractor()

    # Extract features
    print("  Extracting features from real images...")
    real_features = extract_features_from_images(real_images, extractor, batch_size, device)

    print("  Extracting features from generated images...")
    gen_features = extract_features_from_images(generated_images, extractor, batch_size, device)

    # Compute statistics
    print("  Computing statistics...")
    mu_real, sigma_real = compute_statistics_from_features(real_features)
    mu_gen, sigma_gen = compute_statistics_from_features(gen_features)

    # Calculate FID
    print("  Calculating Fréchet distance...")
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    print(f"  FID: {fid_value:.2f}")

    return fid_value


def compute_fid_from_features(real_features, gen_features):
    """
    Compute FID directly from pre-extracted features.

    Useful when you already have features saved.

    Args:
        real_features: (N, D) numpy array
        gen_features: (M, D) numpy array

    Returns:
        fid_score: Scalar FID value
    """
    mu_real, sigma_real = compute_statistics_from_features(real_features)
    mu_gen, sigma_gen = compute_statistics_from_features(gen_features)

    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid_value


# Quick test
if __name__ == '__main__':
    # Test with random images
    real = torch.rand(100, 3, 32, 32)
    fake = torch.rand(100, 3, 32, 32)

    fid = compute_fid(real, fake, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test FID (should be high for random images): {fid:.2f}")
