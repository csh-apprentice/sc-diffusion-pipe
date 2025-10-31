#!/usr/bin/env python3
"""
Direct analysis of k_fps(c) outputs for different FPS conditions.

This checks if the KEY adaptation (not value) is where FPS control happens.
"""

import torch
import sys
import os
import argparse
import logging
import toml
import json
import deepspeed
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')

from models.wan.wan import WanPipeline
from models.wan.model import compute_tau_rel
from utils.common import DTYPE_MAP

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def extract_kfps(pipeline, fps_value, device='cuda'):
    """Extract k_fps output for a given FPS value."""
    # Get FPS conditioning MLP
    fps_mlp = pipeline.transformer.fps_conditioning

    # Compute FPS embedding
    tau_rel = compute_tau_rel(
        torch.tensor([fps_value]),
        transform='raw',
        scale=1.0,
        reference_fps=240.0
    ).unsqueeze(0).to(device)

    # Convert to model dtype
    model_dtype = next(fps_mlp.parameters()).dtype
    tau_rel = tau_rel.to(model_dtype)

    # Get FPS embedding
    with torch.no_grad():
        fps_embedding = fps_mlp(tau_rel)  # [1, dim]

    # Extract k_fps for each block
    k_fps_outputs = {}

    for block_idx in range(len(pipeline.transformer.blocks)):
        block = pipeline.transformer.blocks[block_idx]

        # Check if this block has FPS adapter
        if not hasattr(block, 'fps_adapter') or block.fps_adapter is None:
            continue

        fps_adapter = block.fps_adapter

        # Generate k_fps
        with torch.no_grad():
            # LoRA projection for K
            k_down = fps_adapter.k_fps_down(fps_embedding)  # [1, rank]
            k_up = fps_adapter.k_fps_up(k_down)             # [1, num_tokens * dim]

            # Apply LoRA scaling
            k_proj = k_up * fps_adapter.lora_scale

            # Reshape to attention format
            num_tokens = fps_adapter.num_tokens
            num_heads = fps_adapter.num_heads
            head_dim = fps_adapter.head_dim

            k_fps = k_proj.view(1, num_tokens, num_heads, head_dim).squeeze(0)
            # Shape: [num_tokens, num_heads, head_dim]

            k_fps_outputs[block_idx] = k_fps.cpu()

    return k_fps_outputs


def initialize_deepspeed(port='29510'):
    """Initialize DeepSpeed distributed environment."""
    if not deepspeed.comm.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', port)
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        deepspeed.init_distributed()
    logging.info(f'DeepSpeed environment initialized on port {port}')


def load_pipeline_and_checkpoint(config_path, checkpoint_path):
    """Load WAN pipeline from TOML config and checkpoint."""
    logging.info(f'Loading TOML configuration: {config_path}')

    # Load TOML config
    with open(config_path) as f:
        config = json.loads(json.dumps(toml.load(f)))

    # Convert dtype strings to torch.dtype
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    logging.info(f"Model type: {config['model']['type']}")
    logging.info(f"Model dtype: {model_dtype_str} -> {config['model']['dtype']}")

    # Initialize DeepSpeed
    initialize_deepspeed(port='29510')

    # Initialize pipeline
    pipeline = WanPipeline(config)

    logging.info("Loading transformer model...")
    pipeline.load_diffusion_model()

    logging.info("Moving model to device: cuda")
    pipeline.transformer = pipeline.transformer.to('cuda')

    logging.info("Pipeline loaded successfully")

    # Load checkpoint
    logging.info(f'Loading checkpoint: {checkpoint_path}')
    pipeline.load_checkpoint(checkpoint_path)
    logging.info('Checkpoint loaded successfully')

    return pipeline


def analyze_kfps_magnitude(config_path, checkpoint_path, fps_values):
    """Analyze k_fps magnitudes across conditions."""

    logging.info(f'Loading pipeline from {config_path}')
    pipeline = load_pipeline_and_checkpoint(config_path, checkpoint_path)

    print('='*80)
    print('K_FPS MAGNITUDE ANALYSIS: ||k_fps(c)||')
    print('='*80)
    print()
    print('Expected: V-shape pattern')
    print('  - k_fps(0) should have LOW magnitude (minimal impact)')
    print('  - k_fps(±1) should have HIGH magnitude (strong control)')
    print()

    # Extract k_fps for all conditions
    all_k_fps = {}
    for fps_val in fps_values:
        logging.info(f'Extracting k_fps for FPS={fps_val}')
        all_k_fps[fps_val] = extract_kfps(pipeline, fps_val, device='cuda')

    # Get block indices
    block_indices = sorted(all_k_fps[fps_values[0]].keys())

    # Analyze each block
    for block_idx in [block_indices[0], block_indices[len(block_indices)//2], block_indices[-1]]:
        print(f'\nBlock {block_idx}:')
        print(f'{"Condition":>10} {"||k_fps||":>12} {"Relative":>10} {"Expected":>10}')
        print('-' * 45)

        norms = []
        for fps_val in fps_values:
            k_fps = all_k_fps[fps_val][block_idx]
            norm = torch.norm(k_fps).item()
            norms.append(norm)

        max_norm = max(norms)

        for fps_val, norm in zip(fps_values, norms):
            relative = norm / max_norm
            expected_rel = abs(fps_val)

            if abs(relative - expected_rel) < 0.2:
                status = '✓'
            else:
                status = '✗'

            print(f'{fps_val:>10.1f} {norm:>12.6f} {relative:>9.3f} {expected_rel:>9.3f} {status}')

    return all_k_fps, block_indices


def analyze_kfps_direction(all_k_fps, block_indices, fps_values):
    """Analyze k_fps directional relationships."""

    print()
    print('='*80)
    print('K_FPS DIRECTION ANALYSIS: cosine similarity between k_fps(c1) and k_fps(c2)')
    print('='*80)
    print()
    print('Expected patterns:')
    print('  - k_fps(-1) vs k_fps(+1): cos ≈ -1 (opposite directions)')
    print()

    def cosine_similarity(v1, v2):
        v1_flat = v1.flatten()
        v2_flat = v2.flatten()
        return torch.dot(v1_flat, v2_flat) / (torch.norm(v1_flat) * torch.norm(v2_flat))

    # Analyze representative blocks
    for block_idx in [block_indices[0], block_indices[len(block_indices)//2], block_indices[-1]]:
        print(f'\n{"="*80}')
        print(f'Block {block_idx}: Cosine Similarity Matrix')
        print(f'{"="*80}')

        # Print header
        print(f'{"":>8}', end='')
        for fps_val in fps_values:
            print(f'{fps_val:>8.1f}', end='')
        print()
        print('-' * (8 + 8 * len(fps_values)))

        # Print similarity matrix
        for fps1_val in fps_values:
            print(f'{fps1_val:>7.1f} ', end='')
            for fps2_val in fps_values:
                k_fps1 = all_k_fps[fps1_val][block_idx]
                k_fps2 = all_k_fps[fps2_val][block_idx]
                cos_sim = cosine_similarity(k_fps1, k_fps2).item()
                print(f'{cos_sim:>8.3f}', end='')
            print()

        # Key checks
        print()
        print('Key Checks:')
        k_neg1 = all_k_fps[-1.0][block_idx]
        k_zero = all_k_fps[0.0][block_idx]
        k_pos1 = all_k_fps[1.0][block_idx]

        cos_neg1_pos1 = cosine_similarity(k_neg1, k_pos1).item()
        cos_neg1_zero = cosine_similarity(k_neg1, k_zero).item()
        cos_pos1_zero = cosine_similarity(k_pos1, k_zero).item()

        print(f'  cos(k_fps(-1), k_fps(+1)) = {cos_neg1_pos1:>7.3f}  ', end='')
        if cos_neg1_pos1 < -0.8:
            print('✓ Opposite directions (expected!)')
        elif cos_neg1_pos1 > 0.8:
            print('✗ Same direction (unexpected!)')
        else:
            print('~ Partial opposition')

        print(f'  cos(k_fps(-1), k_fps(0))  = {cos_neg1_zero:>7.3f}')
        print(f'  cos(k_fps(+1), k_fps(0))  = {cos_pos1_zero:>7.3f}')


def main():
    parser = argparse.ArgumentParser(
        description='Direct analysis of k_fps(c) outputs'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config TOML'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--fps_values',
        type=float,
        nargs='+',
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
        help='FPS values to analyze'
    )

    args = parser.parse_args()

    # Extract and analyze
    all_k_fps, block_indices = analyze_kfps_magnitude(
        args.config,
        args.checkpoint,
        args.fps_values
    )

    analyze_kfps_direction(all_k_fps, block_indices, args.fps_values)

    print()
    print('='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()
