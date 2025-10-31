#!/usr/bin/env python3
"""
Compute Backbone Similarity Matrix

This script computes and saves the max cosine similarity matrix S[block_idx][vector_idx]
for all blocks in the backbone. The matrix can be reused later for threshold analysis
without re-running expensive SVD computations.

For each block i and each of the top-k singular vectors j in the trained model:
  S[i][j] = max cosine similarity between U_trained[j] and all U_pretrained vectors

This allows us to:
1. Compute similarities once (expensive)
2. Experiment with different thresholds (cheap)
3. Visualize similarity patterns across all blocks
4. Identify which blocks changed most during training
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


def compute_similarity_for_weight(W_pre, W_trained, top_k):
    """
    Compute max cosine similarity for each trained singular vector.

    Args:
        W_pre: Pretrained weight matrix [d_out, d_in]
        W_trained: Trained weight matrix [d_out, d_in]
        top_k: Number of top singular vectors to compute

    Returns:
        similarities: Array of max similarities [top_k]
        S_pre: Singular values of pretrained weights
        S_trained: Singular values of trained weights
    """
    # Convert to float32 for numerical stability
    W_pre_f32 = W_pre.float()
    W_trained_f32 = W_trained.float()

    # Compute SVD
    U_pre, S_pre, _ = torch.svd(W_pre_f32)
    U_trained, S_trained, _ = torch.svd(W_trained_f32)

    # Compute max similarities
    similarities = []
    actual_k = min(top_k, U_trained.shape[1])

    for j in range(actual_k):
        u_trained_j = U_trained[:, j]
        # Compute cosine similarity with all pretrained vectors
        cos_sims = torch.abs(U_pre.T @ u_trained_j)
        max_sim = torch.max(cos_sims).item()
        similarities.append(max_sim)

    # Pad if needed
    while len(similarities) < top_k:
        similarities.append(1.0)  # Padding with 1.0 (perfect similarity)

    return np.array(similarities), S_pre, S_trained


def compute_block_similarity_matrix(pretrained_weights, lora_checkpoint, block_idx, top_k):
    """
    Compute similarity matrix for a single block.

    Args:
        pretrained_weights: Dict of pretrained weights
        lora_checkpoint: Dict of LoRA weights
        block_idx: Block index
        top_k: Number of top singular vectors

    Returns:
        similarity_matrix: [4, top_k] array (q, k, v, o)
        metadata: Dict with additional info
    """
    logging.info(f"\nProcessing Block {block_idx}...")

    similarity_matrix = np.zeros((4, top_k))
    metadata = {
        'block_idx': block_idx,
        'weight_names': ['q', 'k', 'v', 'o'],
        'spectral_norms_pre': [],
        'spectral_norms_trained': [],
        'lora_delta_norms': [],
        'relative_deltas': []
    }

    for i, weight_name in enumerate(['q', 'k', 'v', 'o']):
        logging.info(f"  Computing similarities for {weight_name.upper()}...")

        weight_pre = pretrained_weights[(block_idx, weight_name)]

        # Look for LoRA weights
        lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
        lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

        if lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint:
            # Merge LoRA
            W_base = weight_pre.cpu().float()
            lora_A = lora_checkpoint[lora_a_key].cpu().float()
            lora_B = lora_checkpoint[lora_b_key].cpu().float()
            delta_W = lora_B @ lora_A
            weight_trained = W_base + delta_W

            # Store metadata
            lora_delta_norm = torch.norm(delta_W).item()
            base_norm = torch.norm(W_base).item()
            relative_delta = lora_delta_norm / base_norm

            metadata['lora_delta_norms'].append(lora_delta_norm)
            metadata['relative_deltas'].append(relative_delta)

            logging.info(f"    LoRA delta norm: {lora_delta_norm:.4f}")
            logging.info(f"    Relative delta: {relative_delta:.4f}")
        else:
            # No LoRA found
            weight_trained = weight_pre.cpu().float()
            metadata['lora_delta_norms'].append(0.0)
            metadata['relative_deltas'].append(0.0)
            logging.info(f"    No LoRA found - using pretrained weights")

        # Compute similarities
        similarities, S_pre, S_trained = compute_similarity_for_weight(
            weight_pre, weight_trained, top_k
        )

        similarity_matrix[i, :] = similarities
        metadata['spectral_norms_pre'].append(S_pre[0].item())
        metadata['spectral_norms_trained'].append(S_trained[0].item())

        logging.info(f"    Spectral norm (pre): {S_pre[0].item():.4f}")
        logging.info(f"    Spectral norm (trained): {S_trained[0].item():.4f}")
        logging.info(f"    Min similarity: {similarities.min():.4f}")
        logging.info(f"    Mean similarity: {similarities.mean():.4f}")

    return similarity_matrix, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Compute Backbone Similarity Matrix for all blocks'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--blocks', type=int, nargs='+', default=None,
                       help='Block indices to analyze (default: all blocks with LoRA)')
    parser.add_argument('--top_k', type=int, default=64,
                       help='Number of top singular vectors to compute')
    parser.add_argument('--output_dir', type=str, default='output/similarity_matrix',
                       help='Output directory for similarity matrix')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    setup_logging()
    init_distributed_stub()

    logging.info("="*80)
    logging.info("COMPUTE BACKBONE SIMILARITY MATRIX")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Top-k vectors: {args.top_k}")
    logging.info(f"Output directory: {args.output_dir}")

    # Create output directory (no timestamp subdirectory)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {output_dir}")

    # Load model and extract weights
    # If blocks not specified, we'll determine them from the checkpoint
    initial_blocks = args.blocks if args.blocks else [0]  # Start with block 0 to load model

    pretrained_weights, trained_model, lora_checkpoint = load_model_and_weights(
        args.config, args.checkpoint, initial_blocks, args.device
    )

    # Determine which blocks have LoRA
    if args.blocks is None:
        logging.info("\nDetecting blocks with LoRA weights...")
        blocks_with_lora = set()
        for key in lora_checkpoint.keys():
            if 'blocks.' in key and 'lora_A.weight' in key:
                # Extract block index from key like 'diffusion_model.blocks.27.cross_attn.q.lora_A.weight'
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == 'blocks' and i + 1 < len(parts):
                        block_idx = int(parts[i + 1])
                        blocks_with_lora.add(block_idx)
                        break

        blocks = sorted(list(blocks_with_lora))
        logging.info(f"Found LoRA weights in {len(blocks)} blocks: {blocks}")

        # Extract weights for all blocks
        pretrained_weights = extract_pretrained_weights(trained_model, blocks)
    else:
        blocks = args.blocks

    logging.info(f"\nWill process {len(blocks)} blocks")

    # Compute similarity matrices for all blocks
    all_similarities = {}
    all_metadata = {}

    for block_idx in blocks:
        similarity_matrix, metadata = compute_block_similarity_matrix(
            pretrained_weights, lora_checkpoint, block_idx, args.top_k
        )
        all_similarities[block_idx] = similarity_matrix
        all_metadata[block_idx] = metadata

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    # Convert to numpy arrays for saving
    # Create a 3D array: [num_blocks, 4, top_k] (Q, K, V, O)
    num_blocks = len(blocks)
    similarity_tensor = np.zeros((num_blocks, 4, args.top_k))

    for i, block_idx in enumerate(blocks):
        similarity_tensor[i] = all_similarities[block_idx]

    # Save similarity matrix
    similarity_file = os.path.join(output_dir, 'similarity_matrix.npy')
    np.save(similarity_file, similarity_tensor)
    logging.info(f"✅ Similarity matrix saved: {similarity_file}")
    logging.info(f"   Shape: {similarity_tensor.shape} (blocks, weights, vectors)")

    # Save block indices
    blocks_file = os.path.join(output_dir, 'block_indices.npy')
    np.save(blocks_file, np.array(blocks))
    logging.info(f"✅ Block indices saved: {blocks_file}")

    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.npz')
    metadata_arrays = {}
    for block_idx in blocks:
        meta = all_metadata[block_idx]
        prefix = f'block_{block_idx}_'
        metadata_arrays[prefix + 'spectral_norms_pre'] = np.array(meta['spectral_norms_pre'])
        metadata_arrays[prefix + 'spectral_norms_trained'] = np.array(meta['spectral_norms_trained'])
        metadata_arrays[prefix + 'lora_delta_norms'] = np.array(meta['lora_delta_norms'])
        metadata_arrays[prefix + 'relative_deltas'] = np.array(meta['relative_deltas'])

    np.savez(metadata_file, **metadata_arrays)
    logging.info(f"✅ Metadata saved: {metadata_file}")

    # Save configuration
    config_file = os.path.join(output_dir, 'config.txt')
    with open(config_file, 'w') as f:
        f.write(f"Training config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Blocks analyzed: {blocks}\n")
        f.write(f"Top-k vectors: {args.top_k}\n")
        f.write(f"Computed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logging.info(f"✅ Configuration saved: {config_file}")

    # Print summary statistics
    logging.info("\n" + "="*80)
    logging.info("SUMMARY STATISTICS")
    logging.info("="*80)

    logging.info(f"{'Block':>6}  {'Min Sim':>8}  {'Mean Sim':>8}  {'Median Sim':>8}  {'Max Delta':>10}")
    logging.info("-" * 80)

    for i, block_idx in enumerate(blocks):
        block_sims = similarity_tensor[i].flatten()
        meta = all_metadata[block_idx]
        max_relative_delta = max(meta['relative_deltas'])

        logging.info(
            f"{block_idx:>6}  "
            f"{block_sims.min():>8.4f}  "
            f"{block_sims.mean():>8.4f}  "
            f"{np.median(block_sims):>8.4f}  "
            f"{max_relative_delta:>10.4f}"
        )

    logging.info("\n" + "="*80)
    logging.info("COMPUTATION COMPLETE")
    logging.info("="*80)
    logging.info(f"\nAll results saved to: {output_dir}")
    logging.info(f"\nNext steps:")
    logging.info(f"  1. Run analyze_similarity_matrix.py with different thresholds")
    logging.info(f"  2. Visualize similarity patterns across blocks")
    logging.info(f"  3. Compare different checkpoints")


if __name__ == "__main__":
    main()
