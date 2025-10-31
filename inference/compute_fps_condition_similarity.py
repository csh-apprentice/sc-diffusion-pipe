#!/usr/bin/env python3
"""
FPS Condition Similarity Analysis

This script analyzes how the FPS adapter's first (dominant) singular vector changes
across different FPS conditioning values. For each block and condition pair, we compute
the cosine similarity between their dominant y_fps singular vectors.

Methodology:
1. For each condition c in [-1.0, -0.75, ..., 0.75, 1.0]:
   - Extract y_fps = attention(q, k_fps(c), v_fps(c))
   - Compute SVD of y_fps
   - Extract the first (dominant) singular vector u_fps[0]
2. Build similarity matrix S[i][j] = cosine_similarity(u_fps[c_i], u_fps[c_j])
3. Visualize as heatmap for each block

Expected Results:
- High similarity (near 1.0): Conditions learn similar representations
- Low similarity (near 0.0): Each condition has unique signature
- Diagonal pattern: Adjacent conditions are more similar
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_trained_model(config_path, checkpoint_path, device='cuda'):
    """Load the trained model with FPS adapter."""
    logging.info(f"Loading trained model from config: {config_path}")

    # Load TOML configuration
    with open(config_path) as f:
        config = json.loads(json.dumps(toml.load(f)))

    # Convert dtype strings to torch.dtype
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    # Initialize and load model
    wan_pipeline = WanPipeline(config)
    wan_pipeline.load_diffusion_model()

    # Load checkpoint
    import safetensors.torch
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        adapter_file = checkpoint_path / 'adapter_model.safetensors'
        if not adapter_file.exists():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {checkpoint_path}")
        checkpoint = safetensors.torch.load_file(str(adapter_file))
    else:
        checkpoint = safetensors.torch.load_file(str(checkpoint_path))

    # Load adapter weights
    wan_pipeline.transformer.load_state_dict(checkpoint, strict=False)
    wan_pipeline.transformer.to(device)
    wan_pipeline.transformer.eval()

    logging.info("✅ Model loaded successfully")
    return wan_pipeline.transformer


def extract_fps_adapter_output(model, block_idx, fps_condition, num_test_vectors=128, device='cuda'):
    """
    Extract y_fps output from the FPS adapter for a given condition.

    This directly computes the FPS attention output without going through the full
    cross-attention block, matching the methodology in principal_orthogonality_check.py.

    Args:
        model: The trained transformer model
        block_idx: Block index to analyze
        fps_condition: FPS conditioning value (e.g., -1.0, 0.0, 1.0)
        num_test_vectors: Number of test query vectors to use
        device: Device for computation

    Returns:
        y_fps: [num_test_vectors, dim] tensor of FPS adapter outputs
    """
    from models.wan.attention import flash_attention

    block = model.blocks[block_idx]
    cross_attn = block.cross_attn
    fps_adapter = block.fps_adapter

    # Check if block has FPS adapter
    if fps_adapter is None:
        raise ValueError(f"Block {block_idx} does not have an FPS adapter!")

    dim = fps_adapter.dim
    num_heads = fps_adapter.num_heads
    head_dim = fps_adapter.head_dim

    with torch.no_grad():
        # Create random test query vectors [N, dim] - use model's dtype
        q_test = torch.randn(num_test_vectors, dim, device=device, dtype=model.dtype)
        q_test = torch.nn.functional.normalize(q_test, dim=-1)

        # Project through cross_attn Q projection (matching reference implementation)
        q = cross_attn.norm_q(cross_attn.q(q_test))
        # Reshape for multi-head attention: [N, 1, num_heads, head_dim]
        q = q.view(num_test_vectors, 1, num_heads, head_dim)

        # Encode FPS condition
        fps_tensor = torch.full((num_test_vectors, 1), fps_condition, dtype=model.dtype, device=device)
        fps_embed = model.fps_conditioning(fps_tensor)

        # Manually compute FPS adapter's K and V (matching reference)
        # This gives us direct access to y_fps without the gate or y_text
        k_fps_proj = fps_adapter.k_fps_up(fps_adapter.k_fps_down(fps_embed))
        v_fps_proj = fps_adapter.v_fps_up(fps_adapter.v_fps_down(fps_embed))

        # Normalize K (matching adapter's forward logic)
        B = num_test_vectors
        k_fps_proj = k_fps_proj.view(B * fps_adapter.num_tokens, fps_adapter.dim)
        k_fps_proj = fps_adapter.norm_k_fps(k_fps_proj)
        k_fps_proj = k_fps_proj.view(B, fps_adapter.num_tokens * fps_adapter.dim)

        # Apply LoRA scale
        k_fps_proj = k_fps_proj * fps_adapter.lora_scale
        v_fps_proj = v_fps_proj * fps_adapter.lora_scale

        # Reshape for attention: [B, num_tokens, num_heads, head_dim]
        k_fps = k_fps_proj.view(B, fps_adapter.num_tokens, num_heads, head_dim)
        v_fps = v_fps_proj.view(B, fps_adapter.num_tokens, num_heads, head_dim)

        # Compute FPS attention directly
        y_fps = flash_attention(q, k_fps, v_fps, k_lens=None)

        # Flatten: [N, num_heads, head_dim] -> [N, dim]
        y_fps = y_fps.reshape(num_test_vectors, -1)

    return y_fps.cpu().float()


def compute_dominant_singular_vector(y_fps):
    """
    Compute the dominant (first) singular vector of y_fps in the OUTPUT space.

    Args:
        y_fps: [num_test_vectors, dim] tensor

    Returns:
        v_dominant: [dim] tensor - the first RIGHT singular vector (in output space)
        s_dominant: float - the first singular value
    """
    # SVD: y_fps = U @ S @ V^T
    # U: [num_test_vectors, num_test_vectors] - left singular vectors (in query/test-vector space)
    # S: [min(num_test_vectors, dim)] - singular values
    # V: [dim, dim] - right singular vectors (in OUTPUT FEATURE SPACE)
    U, S, V = torch.svd(y_fps)

    # First RIGHT singular vector (most dominant direction in OUTPUT space)
    # This is what changes with FPS condition!
    v_dominant = V[:, 0]  # [dim]
    s_dominant = S[0].item()

    return v_dominant, s_dominant


def compute_condition_similarity_matrix(model, block_idx, conditions, num_test_vectors=128, device='cuda'):
    """
    Compute similarity matrix between FPS conditions.

    Args:
        model: Trained model
        block_idx: Block to analyze
        conditions: List of FPS condition values
        num_test_vectors: Number of test vectors
        device: Device for computation

    Returns:
        similarity_matrix: [n_conditions, n_conditions] numpy array
        singular_values: [n_conditions] numpy array - dominant singular values
    """
    n_conditions = len(conditions)
    logging.info(f"\nAnalyzing Block {block_idx} with {n_conditions} conditions...")

    # Extract dominant singular vectors for all conditions
    dominant_vectors = []
    singular_values = []

    for i, condition in enumerate(conditions):
        logging.info(f"  Processing condition {i+1}/{n_conditions}: {condition:.2f}")

        # Get y_fps for this condition
        y_fps = extract_fps_adapter_output(model, block_idx, condition, num_test_vectors, device)

        # Extract dominant singular vector
        u_dominant, s_dominant = compute_dominant_singular_vector(y_fps)

        dominant_vectors.append(u_dominant)
        singular_values.append(s_dominant)

        logging.info(f"    Dominant singular value: {s_dominant:.4f}")

    # Compute pairwise cosine similarities
    similarity_matrix = np.zeros((n_conditions, n_conditions))

    for i in range(n_conditions):
        for j in range(n_conditions):
            # Cosine similarity (vectors are already normalized via SVD)
            cos_sim = torch.abs(torch.dot(dominant_vectors[i], dominant_vectors[j])).item()
            similarity_matrix[i, j] = cos_sim

    logging.info(f"  ✅ Similarity matrix computed: shape {similarity_matrix.shape}")
    logging.info(f"     Min similarity: {similarity_matrix.min():.4f}")
    logging.info(f"     Max similarity: {similarity_matrix.max():.4f}")
    logging.info(f"     Mean off-diagonal similarity: {(similarity_matrix.sum() - n_conditions) / (n_conditions * (n_conditions - 1)):.4f}")

    return similarity_matrix, np.array(singular_values)


def plot_similarity_heatmap(similarity_matrix, conditions, block_idx, output_dir):
    """
    Plot similarity heatmap for a single block.

    Args:
        similarity_matrix: [n_conditions, n_conditions] array
        conditions: List of condition values
        block_idx: Block index
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Set ticks and labels
    condition_labels = [f"{c:.2f}" for c in conditions]
    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(conditions)))
    ax.set_xticklabels(condition_labels, rotation=45, ha='right')
    ax.set_yticklabels(condition_labels)

    ax.set_xlabel('FPS Condition', fontsize=12)
    ax.set_ylabel('FPS Condition', fontsize=12)
    ax.set_title(f'FPS Condition Similarity - Block {block_idx}\n(Dominant Singular Vector Cosine Similarity)',
                 fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)

    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            text_color = 'white' if similarity_matrix[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'fps_condition_similarity_block{block_idx}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"  Heatmap saved: {plot_path}")


def plot_singular_values(singular_values_dict, conditions, output_dir):
    """
    Plot dominant singular values across conditions for all blocks.

    Args:
        singular_values_dict: Dict mapping block_idx -> singular_values array
        conditions: List of condition values
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for block_idx, singular_values in singular_values_dict.items():
        ax.plot(conditions, singular_values, 'o-', label=f'Block {block_idx}',
                linewidth=2, markersize=6)

    ax.set_xlabel('FPS Condition', fontsize=12)
    ax.set_ylabel('Dominant Singular Value', fontsize=12)
    ax.set_title('Dominant Singular Values Across FPS Conditions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'fps_singular_values_across_conditions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Singular values plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute FPS condition similarity for adapter analysis'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--blocks', type=int, nargs='+', required=True,
                       help='Block indices to analyze (e.g., 27 33 39)')
    parser.add_argument('--conditions', type=float, nargs='+', required=True,
                       help='FPS conditions to test (e.g., -1.0 -0.5 0.0 0.5 1.0)')
    parser.add_argument('--num_test_vectors', type=int, default=128,
                       help='Number of test query vectors')
    parser.add_argument('--output_dir', type=str, default='output/fps_condition_similarity',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    setup_logging()

    logging.info("="*80)
    logging.info("FPS CONDITION SIMILARITY ANALYSIS")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Blocks: {args.blocks}")
    logging.info(f"Conditions: {args.conditions}")
    logging.info(f"Num test vectors: {args.num_test_vectors}")
    logging.info(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_trained_model(args.config, args.checkpoint, args.device)

    # Analyze each block
    all_similarity_matrices = {}
    all_singular_values = {}

    for block_idx in args.blocks:
        similarity_matrix, singular_values = compute_condition_similarity_matrix(
            model, block_idx, args.conditions, args.num_test_vectors, args.device
        )

        all_similarity_matrices[block_idx] = similarity_matrix
        all_singular_values[block_idx] = singular_values

        # Plot heatmap for this block
        plot_similarity_heatmap(similarity_matrix, args.conditions, block_idx, args.output_dir)

        # Save matrix to file
        matrix_file = os.path.join(args.output_dir, f'similarity_matrix_block{block_idx}.npy')
        np.save(matrix_file, similarity_matrix)
        logging.info(f"  Matrix saved: {matrix_file}")

    # Plot singular values across all blocks
    plot_singular_values(all_singular_values, args.conditions, args.output_dir)

    # Save summary
    summary_file = os.path.join(args.output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FPS CONDITION SIMILARITY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Blocks analyzed: {args.blocks}\n")
        f.write(f"Conditions: {args.conditions}\n")
        f.write(f"Num test vectors: {args.num_test_vectors}\n\n")

        for block_idx in args.blocks:
            similarity_matrix = all_similarity_matrices[block_idx]
            singular_values = all_singular_values[block_idx]

            n = len(args.conditions)
            off_diag_sum = similarity_matrix.sum() - n
            mean_off_diag = off_diag_sum / (n * (n - 1))

            f.write(f"\nBlock {block_idx}:\n")
            f.write(f"  Min similarity: {similarity_matrix.min():.4f}\n")
            f.write(f"  Max similarity: {similarity_matrix.max():.4f}\n")
            f.write(f"  Mean off-diagonal similarity: {mean_off_diag:.4f}\n")
            f.write(f"  Dominant singular value range: [{singular_values.min():.4f}, {singular_values.max():.4f}]\n")

    logging.info(f"\n✅ Summary saved: {summary_file}")

    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
