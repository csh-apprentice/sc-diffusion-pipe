#!/usr/bin/env python3
"""
Analyze Y_text Change After LoRA Fine-tuning

This script compares the y_text output from clean backbone vs LoRA fine-tuned backbone:

1. Similarity Matrix Analysis:
   - For top-k most important y_text_lora vectors (from SVD)
   - Compute max cosine similarity to any original y_text vector
   - Generate heatmap showing similarity matrix across blocks

2. Singular Spectrum Bird's Eye View:
   - Compare singular value spectrums of y_text_clean vs y_text_lora
   - Show both spectrums on same plot for each block (not normalized)
   - Visualize how fine-tuning affects the spectral properties

The analysis is data-free, using random test queries to probe the attention mechanism.
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
    """Extract pretrained weights from specified blocks (before LoRA)."""
    logging.info("Extracting pretrained weights...")

    pretrained_weights = {}
    for block_idx in blocks:
        block = model.blocks[block_idx]
        cross_attn = block.cross_attn

        pretrained_weights[(block_idx, 'q')] = cross_attn.q.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'k')] = cross_attn.k.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'v')] = cross_attn.v.weight.data.cpu().clone()

    logging.info(f"  Extracted weights from {len(blocks)} blocks")
    return pretrained_weights


def load_model_and_weights(config_path, checkpoint_path, blocks, device='cuda'):
    """Load model, extract pretrained weights, then load LoRA checkpoint."""
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

    # Apply LoRA to model
    apply_lora_to_model(wan_pipeline.transformer, lora_checkpoint, blocks)

    return pretrained_weights, wan_pipeline.transformer, config


def apply_lora_to_model(model, lora_checkpoint, blocks):
    """Apply LoRA deltas to model weights in-place."""
    logging.info("Applying LoRA to model...")

    for block_idx in blocks:
        block = model.blocks[block_idx]
        cross_attn = block.cross_attn

        for weight_name, weight_module in [('q', cross_attn.q), ('k', cross_attn.k), ('v', cross_attn.v)]:
            lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
            lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

            if lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint:
                lora_A = lora_checkpoint[lora_a_key].to(weight_module.weight.device, dtype=weight_module.weight.dtype)
                lora_B = lora_checkpoint[lora_b_key].to(weight_module.weight.device, dtype=weight_module.weight.dtype)
                delta_W = lora_B @ lora_A

                # Apply LoRA delta
                weight_module.weight.data += delta_W

    logging.info("✅ LoRA applied to model")


def compute_ytext_outputs(model, pretrained_weights, block_idx, num_test_vectors, device):
    """
    Compute y_text outputs for both clean and LoRA-finetuned backbone.

    Args:
        model: Model with LoRA applied (finetuned)
        pretrained_weights: Dict of pretrained weights (clean)
        block_idx: Block index
        num_test_vectors: Number of random test vectors
        device: Device

    Returns:
        y_text_clean: Output from clean backbone [N, d_model]
        y_text_lora: Output from LoRA-finetuned backbone [N, d_model]
    """
    from models.wan.attention import flash_attention

    block = model.blocks[block_idx]
    cross_attn = block.cross_attn

    with torch.no_grad():
        N = num_test_vectors
        d_model = cross_attn.q.weight.shape[1]
        num_heads = cross_attn.num_heads
        head_dim = cross_attn.head_dim

        # Generate random test vectors
        q_test = torch.randn(N, d_model, device=device, dtype=model.dtype)
        k_test = torch.randn(N, d_model, device=device, dtype=model.dtype)
        v_test = torch.randn(N, d_model, device=device, dtype=model.dtype)

        q_test = torch.nn.functional.normalize(q_test, dim=-1)
        k_test = torch.nn.functional.normalize(k_test, dim=-1)
        v_test = torch.nn.functional.normalize(v_test, dim=-1)

        # ========== Compute y_text_lora (finetuned backbone) ==========
        q_lora = cross_attn.norm_q(cross_attn.q(q_test))
        k_lora = cross_attn.norm_k(cross_attn.k(k_test))
        v_lora = cross_attn.v(v_test)

        q_lora = q_lora.view(N, -1, num_heads, head_dim)
        k_lora = k_lora.view(N, -1, num_heads, head_dim)
        v_lora = v_lora.view(N, -1, num_heads, head_dim)

        y_text_lora = flash_attention(q_lora, k_lora, v_lora, k_lens=None)
        y_text_lora = y_text_lora.reshape(N, -1)

        # ========== Compute y_text_clean (clean backbone) ==========
        # Temporarily replace weights with pretrained ones
        W_q_lora = cross_attn.q.weight.data.clone()
        W_k_lora = cross_attn.k.weight.data.clone()
        W_v_lora = cross_attn.v.weight.data.clone()

        cross_attn.q.weight.data = pretrained_weights[(block_idx, 'q')].to(device, dtype=model.dtype)
        cross_attn.k.weight.data = pretrained_weights[(block_idx, 'k')].to(device, dtype=model.dtype)
        cross_attn.v.weight.data = pretrained_weights[(block_idx, 'v')].to(device, dtype=model.dtype)

        q_clean = cross_attn.norm_q(cross_attn.q(q_test))
        k_clean = cross_attn.norm_k(cross_attn.k(k_test))
        v_clean = cross_attn.v(v_test)

        q_clean = q_clean.view(N, -1, num_heads, head_dim)
        k_clean = k_clean.view(N, -1, num_heads, head_dim)
        v_clean = v_clean.view(N, -1, num_heads, head_dim)

        y_text_clean = flash_attention(q_clean, k_clean, v_clean, k_lens=None)
        y_text_clean = y_text_clean.reshape(N, -1)

        # Restore LoRA weights
        cross_attn.q.weight.data = W_q_lora
        cross_attn.k.weight.data = W_k_lora
        cross_attn.v.weight.data = W_v_lora

    return y_text_clean.cpu().float(), y_text_lora.cpu().float()


def compute_similarity_matrix(y_text_clean, y_text_lora, top_k):
    """
    Compute max cosine similarity for top-k y_text_lora vectors vs y_text_clean.

    For each of the top-k singular vectors of y_text_lora, compute:
        max_similarity[j] = max_i( |cos(y_text_lora_j, y_text_clean_i)| )

    Args:
        y_text_clean: Clean backbone output [N, d]
        y_text_lora: LoRA backbone output [N, d]
        top_k: Number of top singular vectors to analyze

    Returns:
        similarities: Array of max similarities [top_k]
        S_clean: Singular values of clean output
        S_lora: Singular values of LoRA output
    """
    # Compute SVD
    U_clean, S_clean, _ = torch.svd(y_text_clean)
    U_lora, S_lora, _ = torch.svd(y_text_lora)

    # Compute max similarities for top-k LoRA vectors
    similarities = []
    actual_k = min(top_k, U_lora.shape[1])

    for j in range(actual_k):
        u_lora_j = U_lora[:, j]
        # Compute cosine similarity with all clean vectors
        cos_sims = torch.abs(U_clean.T @ u_lora_j)
        max_sim = torch.max(cos_sims).item()
        similarities.append(max_sim)

    # Pad if needed
    while len(similarities) < top_k:
        similarities.append(1.0)

    return np.array(similarities), S_clean, S_lora


def analyze_block(model, pretrained_weights, block_idx, num_test_vectors, top_k, device):
    """Analyze a single block."""
    logging.info(f"\nAnalyzing Block {block_idx}...")

    # Compute y_text outputs
    y_text_clean, y_text_lora = compute_ytext_outputs(
        model, pretrained_weights, block_idx, num_test_vectors, device
    )

    logging.info(f"  ||y_text_clean||: {torch.norm(y_text_clean).item():.4f}")
    logging.info(f"  ||y_text_lora||:  {torch.norm(y_text_lora).item():.4f}")

    # Compute similarity matrix
    similarities, S_clean, S_lora = compute_similarity_matrix(
        y_text_clean, y_text_lora, top_k
    )

    logging.info(f"  Min similarity: {similarities.min():.4f}")
    logging.info(f"  Mean similarity: {similarities.mean():.4f}")
    logging.info(f"  Max similarity: {similarities.max():.4f}")

    return {
        'similarities': similarities,
        'S_clean': S_clean.numpy(),
        'S_lora': S_lora.numpy(),
        'block_idx': block_idx
    }


def plot_similarity_heatmap(results, blocks, top_k, output_dir):
    """Plot similarity heatmap across all blocks."""
    logging.info("\nGenerating similarity heatmap...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build similarity matrix [num_blocks, top_k]
    similarity_matrix = np.array([r['similarities'] for r in results])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(similarity_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)

    # Labels
    ax.set_xlabel('Singular Vector Index (top-k of y_text_lora)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_title('Y_text Change: Max Similarity of LoRA Vectors to Clean Vectors\n'
                 '(Higher = LoRA output aligns with clean backbone)',
                 fontsize=13, fontweight='bold')

    # Set ticks
    ax.set_yticks(np.arange(len(blocks)))
    ax.set_yticklabels(blocks)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Cosine Similarity', fontsize=11, fontweight='bold')

    # Add grid
    ax.set_xticks(np.arange(top_k)[::max(1, top_k//10)], minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / 'ytext_similarity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"✅ Saved heatmap to: {output_file}")

    plt.close()


def plot_singular_spectrum_comparison(results, blocks, output_dir):
    """Plot singular value spectrum comparison for each block."""
    logging.info("\nGenerating singular spectrum plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create combined plot with all blocks
    num_blocks = len(blocks)
    cols = 4
    rows = (num_blocks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle('Y_text Singular Value Spectrum: Clean vs LoRA Fine-tuned',
                 fontsize=14, fontweight='bold')

    if num_blocks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (result, block_idx) in enumerate(zip(results, blocks)):
        ax = axes[idx]

        S_clean = result['S_clean']
        S_lora = result['S_lora']

        # Plot both spectrums (not normalized)
        ax.plot(S_clean, 'b-', label='Clean Backbone', linewidth=2, alpha=0.7)
        ax.plot(S_lora, 'orange', label='LoRA Fine-tuned', linewidth=2, alpha=0.7)

        ax.set_xlabel('Singular Value Index', fontsize=9)
        ax.set_ylabel('Singular Value (not normalized)', fontsize=9)
        ax.set_title(f'Block {block_idx}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(num_blocks, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    output_file = output_dir / 'ytext_singular_spectrum_all.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"✅ Saved combined spectrum plot to: {output_file}")

    plt.close()

    # Also create individual plots for easier inspection
    for result, block_idx in zip(results, blocks):
        fig, ax = plt.subplots(figsize=(10, 6))

        S_clean = result['S_clean']
        S_lora = result['S_lora']

        ax.plot(S_clean, 'b-', label='Clean Backbone', linewidth=2, alpha=0.7, marker='o', markersize=3)
        ax.plot(S_lora, 'orange', label='LoRA Fine-tuned', linewidth=2, alpha=0.7, marker='s', markersize=3)

        ax.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Singular Value (not normalized)', fontsize=12, fontweight='bold')
        ax.set_title(f'Block {block_idx}: Y_text Singular Spectrum Comparison',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Add statistics
        mean_clean = np.mean(S_clean[:10])
        mean_lora = np.mean(S_lora[:10])
        stats_text = f'Top-10 Mean:\nClean: {mean_clean:.2f}\nLoRA: {mean_lora:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        output_file = output_dir / f'ytext_singular_spectrum_block{block_idx}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        plt.close()

    logging.info(f"✅ Saved {len(blocks)} individual spectrum plots")


def save_results(results, blocks, top_k, output_dir):
    """Save numerical results to files."""
    output_dir = Path(output_dir)

    # Save similarity matrix CSV
    csv_file = output_dir / 'ytext_similarity_matrix.csv'
    with open(csv_file, 'w') as f:
        # Header
        f.write('block_idx,' + ','.join([f'sv_{i}' for i in range(top_k)]) + '\n')
        # Data
        for result in results:
            block_idx = result['block_idx']
            similarities = result['similarities']
            f.write(f'{block_idx},' + ','.join([f'{s:.6f}' for s in similarities]) + '\n')

    logging.info(f"✅ Saved similarity matrix to: {csv_file}")

    # Save numpy arrays
    npz_file = output_dir / 'ytext_analysis_data.npz'
    data_dict = {
        'blocks': np.array(blocks),
        'similarity_matrix': np.array([r['similarities'] for r in results]),
    }
    # Add singular values
    for i, result in enumerate(results):
        data_dict[f'S_clean_block{result["block_idx"]}'] = result['S_clean']
        data_dict[f'S_lora_block{result["block_idx"]}'] = result['S_lora']

    np.savez(npz_file, **data_dict)
    logging.info(f"✅ Saved numpy arrays to: {npz_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze y_text change after LoRA fine-tuning'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to LoRA checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots and data')
    parser.add_argument('--blocks', type=int, nargs='+', default=None,
                       help='Block indices to analyze (default: auto-detect from LoRA)')
    parser.add_argument('--top_k', type=int, default=64,
                       help='Number of top singular vectors to analyze (default: 64)')
    parser.add_argument('--num_test_vectors', type=int, default=128,
                       help='Number of random test vectors (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    setup_logging()

    logging.info("="*80)
    logging.info("ANALYZE Y_TEXT CHANGE AFTER LORA FINE-TUNING")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Top-k vectors: {args.top_k}")
    logging.info(f"Test vectors: {args.num_test_vectors}")
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
            if 'blocks.' in key and '.lora_' in key and 'cross_attn' in key:
                parts = key.split('.')
                block_idx = int(parts[parts.index('blocks') + 1])
                blocks_set.add(block_idx)

        blocks = sorted(list(blocks_set))
        logging.info(f"  Found LoRA in {len(blocks)} blocks: {blocks}")
    else:
        blocks = sorted(args.blocks)
        logging.info(f"\nUsing specified blocks: {blocks}")

    # Load model and weights
    pretrained_weights, model, config = load_model_and_weights(
        args.config, args.checkpoint, blocks, args.device
    )

    # Analyze all blocks
    logging.info("\n" + "="*80)
    logging.info("ANALYZING BLOCKS")
    logging.info("="*80)

    results = []
    for block_idx in blocks:
        result = analyze_block(
            model, pretrained_weights, block_idx,
            args.num_test_vectors, args.top_k, args.device
        )
        results.append(result)

    # Generate visualizations
    logging.info("\n" + "="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)

    plot_similarity_heatmap(results, blocks, args.top_k, args.output_dir)
    plot_singular_spectrum_comparison(results, blocks, args.output_dir)

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    save_results(results, blocks, args.top_k, args.output_dir)

    # Final summary
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info(f"Analyzed {len(blocks)} blocks")

    # Print overall statistics
    all_similarities = np.array([r['similarities'] for r in results])
    mean_sim = np.mean(all_similarities)
    std_sim = np.std(all_similarities)
    logging.info(f"\nOverall similarity statistics:")
    logging.info(f"  Mean: {mean_sim:.4f}")
    logging.info(f"  Std:  {std_sim:.4f}")
    logging.info(f"  Min:  {all_similarities.min():.4f}")
    logging.info(f"  Max:  {all_similarities.max():.4f}")

    logging.info("="*80)


if __name__ == "__main__":
    main()
