"""
FID Evaluation Add-on for Conditional Sparse Reconstruction

This script adds FID (Fréchet Inception Distance) evaluation to the main
conditional evaluation notebook. Use this after running the main evaluation.

FID measures distribution distance between real and generated images in
InceptionV3 feature space. Lower is better.

Usage:
    Run this cell in the evaluation notebook after the CRPS evaluation.
"""

# Add this cell after CRPS evaluation in the notebook

print("="*60)
print("FID EVALUATION (Few Hundred Samples)")
print("="*60)

# Import FID utilities
from utils.fid_score import compute_fid, InceptionV3FeatureExtractor

# FID evaluation settings
args.num_fid_samples = 500  # Few hundred samples (computationally efficient)
args.fid_batch_size = 50     # Batch size for InceptionV3 feature extraction

print(f"Computing FID on {args.num_fid_samples} samples...")
print(f"This is computationally efficient (no ensemble needed)")
print()

# Collect real and generated images
real_images_fid = []
generated_images_fid = []

v_grid = get_mgrid(2, 32).cuda()

num_fid_eval = min(args.num_fid_samples, len(sparse_test_dataset))

for idx in tqdm(range(0, num_fid_eval, args.eval_batch_size), desc='Generating for FID'):
    batch_end = min(idx + args.eval_batch_size, num_fid_eval)
    batch_indices = range(idx, batch_end)

    # Gather batch
    batch_images = []
    batch_context_indices = []
    batch_context_values = []

    for i in batch_indices:
        sample = sparse_test_dataset[i]
        batch_images.append(sample['image'])
        batch_context_indices.append(sample['context_indices'])
        batch_context_values.append(sample['context_values'])

    batch_images = torch.stack(batch_images).cuda()
    batch_context_indices = torch.stack(batch_context_indices).cuda()
    batch_context_values = torch.stack(batch_context_values).cuda()

    batch_size = batch_images.shape[0]

    # Create context image
    context_image = create_context_image_batched(
        batch_context_values,
        batch_context_indices,
        32, 32, 3
    )

    v_batch = v_grid.repeat(batch_size, 1, 1, 1)

    # Generate single sample per image (no ensemble for FID)
    generated = sample_conditional(
        gen_sde,
        context_image,
        v_batch,
        num_steps=args.num_steps
    )

    # Collect images
    real_images_fid.append(batch_images.cpu())
    generated_images_fid.append(generated.cpu().clamp(0, 1))

# Stack all images
real_images_fid = torch.cat(real_images_fid, dim=0)
generated_images_fid = torch.cat(generated_images_fid, dim=0)

print(f"\nCollected {real_images_fid.shape[0]} real and generated images")

# Compute FID
fid_score = compute_fid(
    real_images_fid,
    generated_images_fid,
    batch_size=args.fid_batch_size,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("\n" + "="*60)
print("FID RESULTS")
print("="*60)
print(f"FID Score: {fid_score:.2f}")
print()
print("Interpretation:")
print("  - FID < 10:  Excellent (nearly indistinguishable)")
print("  - FID 10-30: Good (high quality)")
print("  - FID 30-50: Fair (noticeable artifacts)")
print("  - FID > 50:  Poor (significant quality gap)")
print("="*60)

# Update results dictionary with FID
results['fid_score'] = float(fid_score)
results['num_fid_samples'] = int(real_images_fid.shape[0])

# Re-save results with FID
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nUpdated results saved to {results_file}")
print("\nFinal Results:")
print(json.dumps(results, indent=2))
