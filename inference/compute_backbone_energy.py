#!/usr/bin/env python3
"""
Compute Backbone Energy Change After Fine-tuning

This script computes and visualizes the normalized energy change in the backbone
after fine-tuning with LoRA. Energy is defined as the sum of squares of singular values.

For each block and weight matrix (q, k, v, o):
  - Extract pretrained weight matrix W_pre
  - Apply LoRA delta to get trained weight W_trained = W_pre + LoRA_B @ LoRA_A
  - Compute singular values: σ_pre and σ_trained
  - Compute energy: E = Σ(σ²)
  - Compute normalized energy: E_norm = E_trained / E_pre

Outputs:
  - 4 plots (q, k, v, o) showing normalized energy vs block index
  - Energy = 1.0 means no change (pretrained energy baseline)
  - Energy > 1.0 means increased energy after training
  - Energy < 1.0 means decreased energy after training
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import toml
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wan.wan import WanModel, WanPipeline
from utils.common import DTYPE_MAP


def setup_logging():
    """Configure logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def init_distributed_stub():
    """Initialize a stub distributed environment for single-process inference."""
    import torch.distributed as dist

    # Set environment variables for single-process
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')

    # Initialize process group if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)


def extract_pretrained_weights(model, blocks):
    """Extract pretrained weights from specified blocks."""
    logging.info("Extracting pretrained weights...")

    pretrained_weights = {}
    for block_idx in blocks:
        block = model.blocks[block_idx]
        cross_attn = block.cross_attn

        pretrained_weights[(block_idx, 'q')] = cross_attn.q.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'k')] = cross_attn.k.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'v')] = cross_attn.v.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'o')] = cross_attn.o.weight.data.cpu().clone()

    logging.info(f"  Extracted weights from {len(blocks)} blocks")
    return pretrained_weights


def load_model_and_weights(config_path, checkpoint_path, blocks, device='cuda'):
    """Load model once, extract pretrained weights, then load LoRA checkpoint."""
    logging.info(f"Loading model from config: {config_path}")

    # Load TOML configuration
    with open(config_path) as f:
        config = json.loads(json.dumps(toml.load(f)))

    # Convert dtype strings to torch.dtype
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    # Initialize and load base model
    wan_pipeline = WanPipeline(config)
    wan_pipeline.load_diffusion_model()
    wan_pipeline.transformer.to(device)

    logging.info("✅ Pretrained model loaded")

    # Extract pretrained weights BEFORE applying LoRA
    pretrained_weights = extract_pretrained_weights(wan_pipeline.transformer, blocks)

    # Load LoRA checkpoint
    logging.info(f"Loading LoRA checkpoint: {checkpoint_path}")
    import safetensors.torch
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        adapter_file = checkpoint_path / 'adapter_model.safetensors'
        if not adapter_file.exists():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {checkpoint_path}")
        lora_checkpoint = safetensors.torch.load_file(str(adapter_file))
    else:
        if str(checkpoint_path).endswith('.safetensors'):
            lora_checkpoint = safetensors.torch.load_file(str(checkpoint_path))
        else:
            lora_checkpoint = torch.load(checkpoint_path, map_location='cpu')

    logging.info(f"✅ LoRA checkpoint loaded ({len(lora_checkpoint)} keys)")

    return pretrained_weights, wan_pipeline.transformer, lora_checkpoint


def compute_energy(W):
    """
    Compute energy of a weight matrix as sum of squares of singular values.

    Energy = Σ(σ²) where σ are the singular values

    Args:
        W: Weight matrix [d_out, d_in]

    Returns:
        energy: Scalar energy value
        singular_values: Array of singular values
    """
    # Convert to float32 for numerical stability
    W_f32 = W.float()

    # Compute SVD
    _, S, _ = torch.svd(W_f32)

    # Compute energy: sum of squares of singular values
    energy = torch.sum(S ** 2).item()

    return energy, S.cpu().numpy()


def compute_block_energy(pretrained_weights, lora_checkpoint, block_idx):
    """
    Compute normalized energy for all weights in a single block.

    Args:
        pretrained_weights: Dict of pretrained weights
        lora_checkpoint: Dict of LoRA weights
        block_idx: Block index

    Returns:
        energy_dict: Dict with normalized energies for q, k, v, o
        metadata: Dict with additional info
    """
    logging.info(f"\nProcessing Block {block_idx}...")

    energy_dict = {}
    metadata = {
        'block_idx': block_idx,
        'energy_pre': {},
        'energy_trained': {},
        'energy_normalized': {},
        'lora_delta_norms': {},
        'has_lora': {}
    }

    for weight_name in ['q', 'k', 'v', 'o']:
        logging.info(f"  Computing energy for {weight_name.upper()}...")

        weight_pre = pretrained_weights[(block_idx, weight_name)]

        # Look for LoRA weights
        lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
        lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

        has_lora = lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint
        metadata['has_lora'][weight_name] = has_lora

        if has_lora:
            # Merge LoRA
            W_base = weight_pre.cpu().float()
            lora_A = lora_checkpoint[lora_a_key].cpu().float()
            lora_B = lora_checkpoint[lora_b_key].cpu().float()
            delta_W = lora_B @ lora_A
            weight_trained = W_base + delta_W

            # Store delta norm
            lora_delta_norm = torch.norm(delta_W).item()
            metadata['lora_delta_norms'][weight_name] = lora_delta_norm

            logging.info(f"    LoRA delta norm: {lora_delta_norm:.4f}")
        else:
            # No LoRA found
            weight_trained = weight_pre.cpu().float()
            metadata['lora_delta_norms'][weight_name] = 0.0
            logging.info(f"    No LoRA found - using pretrained weights")

        # Compute energies
        energy_pre, S_pre = compute_energy(weight_pre)
        energy_trained, S_trained = compute_energy(weight_trained)

        # Compute normalized energy (pretrained = 1.0 baseline)
        energy_normalized = energy_trained / energy_pre

        # Store results
        energy_dict[weight_name] = energy_normalized
        metadata['energy_pre'][weight_name] = energy_pre
        metadata['energy_trained'][weight_name] = energy_trained
        metadata['energy_normalized'][weight_name] = energy_normalized

        logging.info(f"    Energy (pretrained): {energy_pre:.2f}")
        logging.info(f"    Energy (trained): {energy_trained:.2f}")
        logging.info(f"    Normalized energy: {energy_normalized:.6f}")

    return energy_dict, metadata


def plot_energy_comparison(energy_data, blocks, output_dir):
    """
    Create 4 plots (q, k, v, o) showing normalized energy vs block index.

    Args:
        energy_data: List of energy dicts for each block
        blocks: List of block indices
        output_dir: Directory to save plots
    """
    logging.info("\nGenerating energy comparison plots...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for each weight type
    weight_names = ['q', 'k', 'v', 'o']
    colors = {'q': '#2E86AB', 'k': '#A23B72', 'v': '#F18F01', 'o': '#C73E1D'}

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Normalized Energy Change After Fine-tuning\n(Baseline = 1.0 = Pretrained Energy)',
                 fontsize=14, fontweight='bold')

    axes = axes.flatten()

    for idx, weight_name in enumerate(weight_names):
        ax = axes[idx]

        # Extract normalized energies for this weight type
        energies = [data[weight_name] for data in energy_data]

        # Plot
        ax.plot(blocks, energies, marker='o', linewidth=2, markersize=6,
                color=colors[weight_name], label=weight_name.upper())

        # Add horizontal line at y=1.0 (baseline)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Pretrained (baseline)')

        # Styling
        ax.set_xlabel('Block Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Energy', fontsize=11, fontweight='bold')
        ax.set_title(f'{weight_name.upper()} Matrix Energy', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Add statistics
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        min_energy = np.min(energies)
        max_energy = np.max(energies)

        stats_text = f'Mean: {mean_energy:.4f}\nStd: {std_energy:.4f}\nMin: {min_energy:.4f}\nMax: {max_energy:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'energy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"✅ Saved plot to: {output_file}")

    plt.close()

    # Also create individual plots for each weight type
    for weight_name in weight_names:
        fig, ax = plt.subplots(figsize=(10, 6))

        energies = [data[weight_name] for data in energy_data]

        ax.plot(blocks, energies, marker='o', linewidth=2, markersize=8,
                color=colors[weight_name], label=f'{weight_name.upper()} Energy')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Pretrained (baseline)')

        ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Energy', fontsize=12, fontweight='bold')
        ax.set_title(f'{weight_name.upper()} Matrix: Normalized Energy Change After Fine-tuning',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)

        # Add detailed statistics
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        min_energy = np.min(energies)
        max_energy = np.max(energies)

        stats_text = f'Statistics:\nMean: {mean_energy:.6f}\nStd Dev: {std_energy:.6f}\nMin: {min_energy:.6f}\nMax: {max_energy:.6f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save individual plot
        output_file = output_dir / f'energy_{weight_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"✅ Saved {weight_name.upper()} plot to: {output_file}")

        plt.close()


def save_results(energy_data, metadata_list, blocks, output_dir):
    """Save numerical results to files."""
    output_dir = Path(output_dir)

    # Save summary CSV
    csv_file = output_dir / 'energy_summary.csv'
    with open(csv_file, 'w') as f:
        f.write('block_idx,weight,energy_pre,energy_trained,energy_normalized,has_lora,lora_delta_norm\n')
        for block_idx, metadata in zip(blocks, metadata_list):
            for weight_name in ['q', 'k', 'v', 'o']:
                f.write(f'{block_idx},{weight_name},'
                       f'{metadata["energy_pre"][weight_name]:.6f},'
                       f'{metadata["energy_trained"][weight_name]:.6f},'
                       f'{metadata["energy_normalized"][weight_name]:.6f},'
                       f'{metadata["has_lora"][weight_name]},'
                       f'{metadata["lora_delta_norms"][weight_name]:.6f}\n')

    logging.info(f"✅ Saved summary to: {csv_file}")

    # Save detailed numpy arrays
    npz_file = output_dir / 'energy_data.npz'
    energy_arrays = {
        'blocks': np.array(blocks),
        'q': np.array([d['q'] for d in energy_data]),
        'k': np.array([d['k'] for d in energy_data]),
        'v': np.array([d['v'] for d in energy_data]),
        'o': np.array([d['o'] for d in energy_data]),
    }
    np.savez(npz_file, **energy_arrays)
    logging.info(f"✅ Saved numpy arrays to: {npz_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute normalized energy change in backbone after fine-tuning'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to LoRA checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots and data')
    parser.add_argument('--blocks', type=int, nargs='+', default=None,
                       help='Block indices to analyze (default: auto-detect from LoRA)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    setup_logging()

    logging.info("="*80)
    logging.info("COMPUTE BACKBONE ENERGY CHANGE")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("="*80)

    # Initialize distributed environment
    init_distributed_stub()

    # Auto-detect blocks if not specified
    if args.blocks is None:
        logging.info("\nAuto-detecting blocks with LoRA...")
        import safetensors.torch
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_dir():
            adapter_file = checkpoint_path / 'adapter_model.safetensors'
            lora_checkpoint = safetensors.torch.load_file(str(adapter_file))
        else:
            lora_checkpoint = safetensors.torch.load_file(str(checkpoint_path))

        # Extract unique block indices
        blocks_set = set()
        for key in lora_checkpoint.keys():
            if 'blocks.' in key and '.lora_' in key:
                # Extract block number
                parts = key.split('.')
                block_idx = int(parts[parts.index('blocks') + 1])
                blocks_set.add(block_idx)

        blocks = sorted(list(blocks_set))
        logging.info(f"  Found LoRA in {len(blocks)} blocks: {blocks}")
    else:
        blocks = sorted(args.blocks)
        logging.info(f"\nUsing specified blocks: {blocks}")

    # Load model and weights
    pretrained_weights, model, lora_checkpoint = load_model_and_weights(
        args.config, args.checkpoint, blocks, args.device
    )

    # Compute energy for all blocks
    logging.info("\n" + "="*80)
    logging.info("COMPUTING ENERGIES")
    logging.info("="*80)

    energy_data = []
    metadata_list = []

    for block_idx in blocks:
        energy_dict, metadata = compute_block_energy(
            pretrained_weights, lora_checkpoint, block_idx
        )
        energy_data.append(energy_dict)
        metadata_list.append(metadata)

    # Generate plots
    logging.info("\n" + "="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)

    plot_energy_comparison(energy_data, blocks, args.output_dir)

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    save_results(energy_data, metadata_list, blocks, args.output_dir)

    # Final summary
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info(f"Analyzed {len(blocks)} blocks")

    # Print summary statistics
    for weight_name in ['q', 'k', 'v', 'o']:
        energies = [d[weight_name] for d in energy_data]
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        logging.info(f"  {weight_name.upper()}: mean={mean_energy:.6f}, std={std_energy:.6f}")

    logging.info("="*80)


if __name__ == "__main__":
    main()
