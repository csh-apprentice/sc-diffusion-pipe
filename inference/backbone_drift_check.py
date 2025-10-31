#!/usr/bin/env python3
"""
Backbone Drift Check: Intruder Dimension Analysis

This script measures how much the final adapted backbone has spectrally deviated
from the original pretrained backbone by detecting "intruder dimensions" - new
high-ranking singular vectors that didn't exist in the pretrained model.

Methodology:
1. Load pretrained weights (W_pre) and trained weights (W_lora = W_pre + ΔW_lora)
2. Compute SVD for both: U_pre, S_pre and U_lora, S_lora
3. For each top-k singular vector in U_lora, find its max cosine similarity to U_pre
4. Count "intruders": vectors with max_similarity < threshold (e.g., 0.5)

Low intruder count (0-2) = clean, non-destructive fine-tuning
High intruder count = potential catastrophic forgetting
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import toml
import json

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
    # Suppress matplotlib debug spam
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def init_distributed_stub():
    """Initialize a stub distributed environment for single-process inference."""
    import torch.distributed as dist
    import os

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
    """
    Extract pretrained weights from specified blocks before applying LoRA.

    Args:
        model: Model with pretrained weights
        blocks: List of block indices to extract

    Returns:
        pretrained_weights: Dict mapping (block_idx, weight_name) -> weight tensor (on CPU)
    """
    logging.info("Extracting pretrained weights...")

    pretrained_weights = {}
    for block_idx in blocks:
        block = model.blocks[block_idx]
        cross_attn = block.cross_attn

        # Copy weights to CPU to save GPU memory
        pretrained_weights[(block_idx, 'q')] = cross_attn.q.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'k')] = cross_attn.k.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'v')] = cross_attn.v.weight.data.cpu().clone()

    logging.info(f"  Extracted weights from {len(blocks)} blocks")
    return pretrained_weights


def load_model_and_weights(config_path, checkpoint_path, blocks, device='cuda'):
    """
    Load model once, extract pretrained weights, then apply LoRA for trained weights.

    Args:
        config_path: Path to training config TOML
        checkpoint_path: Path to trained checkpoint with LoRA
        blocks: List of block indices to analyze
        device: Device to load model on

    Returns:
        pretrained_weights: Dict of pretrained weights (CPU)
        trained_model: Model with LoRA applied (GPU)
    """
    logging.info(f"Loading model from config: {config_path}")

    # Load TOML exactly like train.py does
    with open(config_path) as f:
        config = json.loads(json.dumps(toml.load(f)))

    # Convert dtype strings to torch.dtype objects
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    # Initialize pipeline with TOML configuration
    wan_pipeline = WanPipeline(config)
    wan_pipeline.load_diffusion_model()
    wan_pipeline.transformer.to(device)

    logging.info("✅ Pretrained model loaded")

    # Extract pretrained weights BEFORE applying LoRA
    pretrained_weights = extract_pretrained_weights(wan_pipeline.transformer, blocks)

    # Load LoRA checkpoint directly (don't configure adapter - too slow)
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


def compute_intruder_dimensions(W_pre, W_lora, top_k=64, threshold=0.5):
    """
    Detect intruder dimensions in the adapted weights.

    An intruder dimension is a high-ranking singular vector in W_lora that has
    low similarity to all singular vectors in W_pre, indicating a new feature
    that didn't exist in the pretrained model.

    Args:
        W_pre: Pretrained weight matrix [d_out, d_in]
        W_lora: Trained weight matrix (with LoRA applied) [d_out, d_in]
        top_k: Number of top singular vectors to check
        threshold: Cosine similarity threshold for detecting intruders

    Returns:
        intruders: List of intruder indices
        similarities: Max cosine similarities for each vector
        U_pre: Left singular vectors of pretrained weights
        U_lora: Left singular vectors of trained weights
    """
    # Convert to float32 for numerical stability
    W_pre_f32 = W_pre.float()
    W_lora_f32 = W_lora.float()

    # Compute SVD
    logging.info(f"  Computing SVD for W_pre (shape: {W_pre.shape})...")
    U_pre, S_pre, _ = torch.svd(W_pre_f32)

    logging.info(f"  Computing SVD for W_lora (shape: {W_lora.shape})...")
    U_lora, S_lora, _ = torch.svd(W_lora_f32)

    # Check top-k singular vectors from trained model
    intruders = []
    similarities = []

    logging.info(f"  Checking top-{top_k} singular vectors for intruders (threshold: {threshold})...")

    for j in range(min(top_k, U_lora.shape[1])):
        u_lora_j = U_lora[:, j]

        # Compute cosine similarity with all pretrained singular vectors
        # cos(u_lora_j, u_pre_i) = u_lora_j @ u_pre_i
        cos_sims = torch.abs(U_pre.T @ u_lora_j)
        max_sim = torch.max(cos_sims).item()

        similarities.append(max_sim)

        if max_sim < threshold:
            intruders.append(j)

    return intruders, similarities, U_pre, U_lora, S_pre, S_lora


def plot_similarity_distribution(similarities, threshold, block_idx, weight_name, output_dir):
    """
    Plot the distribution of max similarities.

    Args:
        similarities: List of max cosine similarities
        threshold: Intruder detection threshold
        block_idx: Block index for labeling
        weight_name: Weight name (q/k/v) for labeling
        output_dir: Directory to save plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        # Plot as a line with markers
        plt.plot(similarities, 'b-o', linewidth=2, markersize=4, label='Max Similarity to Pre-trained')

        # Add threshold line
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Intruder Threshold ({threshold})')

        # Highlight intruders
        intruders = [i for i, sim in enumerate(similarities) if sim < threshold]
        if intruders:
            plt.scatter(intruders, [similarities[i] for i in intruders],
                       color='red', s=100, zorder=5, label=f'Intruders ({len(intruders)})')

        plt.xlabel('Singular Vector Index (ranked by importance)', fontsize=12)
        plt.ylabel('Max Cosine Similarity to Pre-trained Vectors', fontsize=12)
        plt.title(f'Intruder Dimension Analysis - Block {block_idx} {weight_name.upper()}', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)

        # Save plot
        plot_path = os.path.join(output_dir, f'intruder_analysis_block{block_idx}_{weight_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"  Plot saved: {plot_path}")

    except ImportError:
        logging.warning("  matplotlib not available, skipping plot")


def analyze_block(pretrained_weights, trained_model, lora_checkpoint, block_idx, top_k=64, threshold=0.5, output_dir='output'):
    """
    Analyze a single block for intruder dimensions.

    Args:
        pretrained_weights: Dict of pretrained weights (on CPU)
        trained_model: Base model (without LoRA applied)
        lora_checkpoint: Dict of LoRA weights from checkpoint
        block_idx: Block index to analyze
        top_k: Number of top singular vectors to check
        threshold: Intruder detection threshold
        output_dir: Output directory for plots

    Returns:
        results: Dictionary with analysis results
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"ANALYZING BLOCK {block_idx}")
    logging.info(f"{'='*80}")

    results = {'block_idx': block_idx}

    # Analyze Q, K, V weight matrices
    for weight_name in ['q', 'k', 'v']:
        weight_pre = pretrained_weights[(block_idx, weight_name)]

        # Look for LoRA weights in checkpoint
        lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
        lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

        if lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint:
            # Merge LoRA: W_merged = W_base + B @ A
            W_base = weight_pre.cpu().float()
            lora_A = lora_checkpoint[lora_a_key].cpu().float()
            lora_B = lora_checkpoint[lora_b_key].cpu().float()

            # Compute LoRA delta: ΔW = B @ A
            delta_W = lora_B @ lora_A

            weight_trained = W_base + delta_W
            logging.info(f"\nAnalyzing {weight_name.upper()} weights (with LoRA merged)...")
            logging.info(f"  LoRA delta norm: {torch.norm(delta_W).item():.4f}")
            logging.info(f"  Base weight norm: {torch.norm(W_base).item():.4f}")
            logging.info(f"  Relative delta: {(torch.norm(delta_W) / torch.norm(W_base)).item():.4f}")
        else:
            # No LoRA found for this weight
            weight_trained = weight_pre.cpu().float()
            logging.info(f"\nAnalyzing {weight_name.upper()} weights (no LoRA found - comparing to self)...")
            logging.warning(f"  No LoRA weights found for blocks.{block_idx}.cross_attn.{weight_name}")

        intruders, similarities, U_pre, U_lora, S_pre, S_lora = compute_intruder_dimensions(
            weight_pre, weight_trained, top_k, threshold
        )

        # Compute spectral norm difference
        spectral_norm_pre = S_pre[0].item()
        spectral_norm_lora = S_lora[0].item()
        spectral_norm_diff = abs(spectral_norm_lora - spectral_norm_pre) / spectral_norm_pre

        logging.info(f"  Intruders detected: {len(intruders)}/{top_k}")
        logging.info(f"  Spectral norm (pre): {spectral_norm_pre:.4f}")
        logging.info(f"  Spectral norm (trained): {spectral_norm_lora:.4f}")
        logging.info(f"  Relative change: {spectral_norm_diff:.2%}")

        if intruders:
            logging.info(f"  Intruder indices: {intruders}")
            logging.info(f"  Intruder similarities: {[f'{similarities[i]:.3f}' for i in intruders]}")

        # Plot
        plot_similarity_distribution(similarities, threshold, block_idx, weight_name, output_dir)

        # Interpretation
        if len(intruders) == 0:
            logging.info(f"  ✅ EXCELLENT: No intruders - clean fine-tuning")
        elif len(intruders) <= 2:
            logging.info(f"  ✓  GOOD: Few intruders ({len(intruders)}) - minimal drift")
        elif len(intruders) <= 5:
            logging.info(f"  ⚠  WARNING: Moderate intruders ({len(intruders)}) - some drift detected")
        else:
            logging.info(f"  ❌ FAILURE: Many intruders ({len(intruders)}) - potential catastrophic forgetting")

        results[weight_name] = {
            'num_intruders': len(intruders),
            'intruder_indices': intruders,
            'similarities': similarities,
            'spectral_norm_diff': spectral_norm_diff,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Backbone Drift Check: Intruder Dimension Analysis')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--blocks', type=int, nargs='+', default=[27, 33, 39],
                       help='Block indices to analyze')
    parser.add_argument('--top_k', type=int, default=64,
                       help='Number of top singular vectors to check')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Cosine similarity threshold for detecting intruders')
    parser.add_argument('--output_dir', type=str, default='output/backbone_drift',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    setup_logging()
    init_distributed_stub()  # Initialize distributed stub for single-process

    logging.info("="*80)
    logging.info("BACKBONE DRIFT CHECK: INTRUDER DIMENSION ANALYSIS")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Trained checkpoint: {args.checkpoint}")
    logging.info(f"Blocks to analyze: {args.blocks}")
    logging.info(f"Top-k vectors: {args.top_k}")
    logging.info(f"Intruder threshold: {args.threshold}")
    logging.info(f"\nInterpretation:")
    logging.info(f"  Low intruder count (0-2): Clean, non-destructive fine-tuning")
    logging.info(f"  High intruder count: Potential catastrophic forgetting\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once, extract pretrained weights, load LoRA checkpoint
    pretrained_weights, trained_model, lora_checkpoint = load_model_and_weights(
        args.config, args.checkpoint, args.blocks, args.device
    )

    # Analyze each block
    results = []
    for block_idx in args.blocks:
        result = analyze_block(
            pretrained_weights, trained_model, lora_checkpoint, block_idx,
            args.top_k, args.threshold, args.output_dir
        )
        results.append(result)

    # Summary
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"{'Block':>6}  {'W_q':>8}  {'W_k':>8}  {'W_v':>8}  {'Total':>8}  {'Status':>20}")
    logging.info("-" * 80)

    total_intruders_all = 0
    for result in results:
        num_q = result['q']['num_intruders']
        num_k = result['k']['num_intruders']
        num_v = result['v']['num_intruders']
        total = num_q + num_k + num_v
        total_intruders_all += total

        if total == 0:
            status = "✅ Perfect"
        elif total <= 2:
            status = "✓  Excellent"
        elif total <= 5:
            status = "⚠  Warning"
        else:
            status = "❌ Drift"

        logging.info(f"{result['block_idx']:>6}  {num_q:>8}  {num_k:>8}  {num_v:>8}  {total:>8}  {status:>20}")

    avg_intruders = total_intruders_all / len(results)

    logging.info(f"\nOverall Statistics:")
    logging.info(f"  Total intruders across all blocks: {total_intruders_all}")
    logging.info(f"  Average intruders per block: {avg_intruders:.2f}")
    logging.info(f"  Max possible intruders per block: {args.top_k * 3} (3 weight matrices)")
    logging.info(f"  Intruder rate: {total_intruders_all / (len(results) * args.top_k * 3):.2%}")

    # Overall assessment
    if total_intruders_all == 0:
        logging.info("\n✅ OVERALL: Perfect - No catastrophic forgetting detected!")
        logging.info("   Base LoRA made only small perturbations to existing features.")
    elif avg_intruders <= 2:
        logging.info("\n✓  OVERALL: Excellent - Minimal backbone drift")
        logging.info("   Clean, non-destructive fine-tuning preserved pretrained knowledge.")
    elif avg_intruders <= 5:
        logging.info("\n⚠  OVERALL: Warning - Moderate backbone drift detected")
        logging.info("   Some new features learned, but may still preserve general capabilities.")
    else:
        logging.info("\n❌ OVERALL: Significant backbone drift - risk of catastrophic forgetting")
        logging.info("   LoRA introduced many new high-ranking features not in pretrained model.")

    logging.info(f"\n{'='*80}")
    logging.info("ANALYSIS COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"Results and plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
