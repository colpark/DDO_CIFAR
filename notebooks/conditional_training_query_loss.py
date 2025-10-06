"""
CORRECTED: Conditional Training with Query-Only Loss

Key Fix:
- OLD: Loss on full image (100%)
- NEW: Loss only on query pixels (10%)

This is the correct formulation:
1. Context (10%): Input condition (fixed per instance)
2. Query (10%): Ground truth for training (fixed per instance)
3. Remaining (80%): Not used in training, predicted at test time

Model learns to predict full field from sparse context,
but is only supervised on 10% query pixels.
"""

# Replace cell 12 (training loop) with this corrected version

print("="*60)
print("CORRECTED: Training with Query-Only Loss")
print("="*60)
print("Context (10%): Condition")
print("Query (10%): Supervision")
print("Target: Predict full field (100%)")
print("="*60)

# Training setup
torch.manual_seed(args.seed)
np.random.seed(args.seed)

writer = Writer(args.global_rank, args.exp_path)
start_time = time.time()

gen_sde.train()
train_iter = iter(train_loader)
pbar = tqdm(total=args.num_iterations, initial=count, desc='Conditional Training')

# Pre-compute coordinate grid
v_grid = get_mgrid(2, args.train_img_height).cuda()

while count < args.num_iterations:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    # Get data
    full_images = batch['image'].cuda()  # (B, C, H, W)
    context_indices = batch['context_indices'].cuda()  # (B, num_context)
    context_values = batch['context_values'].cuda()  # (B, num_context, C)
    query_indices = batch['query_indices'].cuda()  # (B, num_query)
    query_values = batch['query_values'].cuda()  # (B, num_query, C)

    batch_size = full_images.shape[0]

    # Create context image (sparse observations as dense image)
    context_image = create_context_image_batched(
        context_values,
        context_indices,
        height=args.train_img_height,
        width=args.train_img_height,
        num_channels=args.input_dim
    )

    # Create query mask for loss computation
    from utils.sparse_datasets_fixed import create_query_mask_batched
    query_mask = create_query_mask_batched(
        query_indices,
        height=args.train_img_height,
        width=args.train_img_height
    )  # (B, 1, H, W) - binary mask where 1 = query pixel

    # Coordinate grid
    v = v_grid.repeat(batch_size, 1, 1, 1)

    # Forward pass with conditioning
    optimizer.zero_grad()

    # DSM loss with context conditioning
    # Model predicts noise for full image, conditioned on context
    # But we only compute loss on QUERY pixels

    # Sample timestep
    if gen_sde.timestep_sampler == "uniform":
        s_ = torch.rand(batch_size, device=full_images.device) * gen_sde.T
    elif gen_sde.timestep_sampler == "low_discrepancy":
        from lib.diffusion import low_discrepancy_rand
        s_ = low_discrepancy_rand(batch_size, device=full_images.device) * gen_sde.T

    # Add noise to full image
    zt, target, _, _ = gen_sde.forward_diffusion.sample(t=s_, x0=full_images)

    # Predict noise conditioned on context
    pred = gen_sde.epsilon(y=zt, s=s_, v=v, context_image=context_image)

    # Compute MSE between predicted and true noise
    mse = 0.5 * ((pred - target) ** 2)  # (B, C, H, W)

    # CRITICAL: Only compute loss on QUERY pixels (10%)
    # Apply query mask: (B, C, H, W) * (B, 1, H, W) → (B, C, H, W)
    masked_mse = mse * query_mask

    # Sum over query pixels and normalize by number of query pixels
    num_query_pixels = query_mask.sum(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
    loss_per_sample = masked_mse.sum(dim=(1, 2, 3)) / num_query_pixels.squeeze()  # (B,)

    loss = loss_per_sample.mean()  # Scalar

    # Backward
    loss.backward()
    optimizer.step()

    count += 1
    pbar.update(1)

    # Logging
    if count % args.print_every == 0:
        elapsed = (time.time() - start_time) / args.print_every
        lr = optimizer.param_groups[0]['lr']

        # Compute what full-image loss would be for comparison
        full_loss = (mse.sum(dim=(1,2,3)) / (args.input_dim * args.train_img_height * args.train_img_height)).mean()

        pbar.set_postfix({
            'loss_query': f'{loss.item():.4f}',
            'loss_full': f'{full_loss.item():.4f}',
            'lr': f'{lr:.6f}'
        })
        writer.add_scalar('train/loss_query', loss.item(), count)
        writer.add_scalar('train/loss_full', full_loss.item(), count)
        writer.add_scalar('train/lr', lr, count)
        start_time = time.time()

    # Visualization
    if count % args.vis_every == 0:
        num_vis = min(args.vis_batch_size, 16)

        vis_samples = []
        for i in range(num_vis):
            sample = sparse_dataset[i]
            vis_samples.append({
                'original': sample['image'],
                'context': create_sparse_mask_image(
                    sample['image'], sample['context_indices'], fill_value=0.5
                ),
                'query': create_sparse_mask_image(
                    sample['image'], sample['query_indices'], fill_value=0.5
                )
            })

        contexts_vis = torch.stack([s['context'] for s in vis_samples])
        queries_vis = torch.stack([s['query'] for s in vis_samples])
        originals_vis = torch.stack([s['original'] for s in vis_samples])

        # Save comparison: [context | query | original]
        fig_path = os.path.join(args.exp_path, 'samples', f'iter_{count:06d}.png')
        comparison = torch.cat([contexts_vis, queries_vis, originals_vis], dim=0)
        torchvision.utils.save_image(
            comparison, fig_path, nrow=4, padding=2, normalize=True, value_range=(0, 1)
        )

        print(f'\n[Iter {count}] Saved visualization to {fig_path}')
        print(f'  Loss on query pixels (10%): {loss.item():.6f}')
        print(f'  Loss on full image (100%): {full_loss.item():.6f}')

    # Save checkpoint
    if count % args.save_every == 0:
        save_checkpoint(
            args, count, loss.item(), gen_sde, optimizer, None, 'checkpoint.pt'
        )
        print(f'\n[Iter {count}] Saved checkpoint')

pbar.close()
print('\n' + '='*60)
print('Training completed!')
print('='*60)
print(f'Final query loss: {loss.item():.6f}')
print(f'Model trained to predict full field from 10% context,')
print(f'supervised only on 10% query pixels.')
print('='*60)
