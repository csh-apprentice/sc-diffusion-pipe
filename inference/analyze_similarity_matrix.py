#!/usr/bin/env python3
"""
Analyze Precomputed Backbone Similarity Matrix

This script loads a precomputed similarity matrix and applies different thresholds
to detect intruder dimensions. This allows rapid experimentation with thresholds
without re-running expensive SVD computations.

The similarity matrix S[block_idx][weight_idx][vector_idx] contains the max cosine
similarity between each trained singular vector and all pretrained singular vectors.
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_logging():
    """Configure logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_similarity_matrix(matrix_dir):
    """
    Load precomputed similarity matrix and metadata.

    Args:
        matrix_dir: Directory containing similarity_matrix.npy and other files

    Returns:
        similarity_tensor: [num_blocks, 3, top_k] array
        block_indices: [num_blocks] array
        metadata: dict of metadata arrays
    """
    matrix_dir = Path(matrix_dir)

    # Load similarity matrix
    similarity_file = matrix_dir / 'similarity_matrix.npy'
    if not similarity_file.exists():
        raise FileNotFoundError(f"Similarity matrix not found: {similarity_file}")

    similarity_tensor = np.load(similarity_file)
    logging.info(f"✅ Loaded similarity matrix: {similarity_tensor.shape}")

    # Load block indices
    blocks_file = matrix_dir / 'block_indices.npy'
    if not blocks_file.exists():
        raise FileNotFoundError(f"Block indices not found: {blocks_file}")

    block_indices = np.load(blocks_file)
    logging.info(f"✅ Loaded block indices: {len(block_indices)} blocks")

    # Load metadata (optional)
    metadata_file = matrix_dir / 'metadata.npz'
    metadata = {}
    if metadata_file.exists():
        metadata = dict(np.load(metadata_file))
        logging.info(f"✅ Loaded metadata: {len(metadata)} entries")

    return similarity_tensor, block_indices, metadata


def detect_intruders(similarity_tensor, block_indices, threshold):
    """
    Detect intruder dimensions using a threshold.

    Args:
        similarity_tensor: [num_blocks, 4, top_k] array (Q, K, V, O)
        block_indices: [num_blocks] array
        threshold: Cosine similarity threshold

    Returns:
        results: List of dicts with per-block results
    """
    results = []
    weight_names = ['q', 'k', 'v', 'o']

    for i, block_idx in enumerate(block_indices):
        block_result = {
            'block_idx': int(block_idx),
            'intruders_per_weight': {},
            'total_intruders': 0
        }

        for j, weight_name in enumerate(weight_names):
            similarities = similarity_tensor[i, j, :]

            # Find intruders
            intruder_indices = np.where(similarities < threshold)[0].tolist()
            intruder_sims = similarities[intruder_indices].tolist()

            block_result['intruders_per_weight'][weight_name] = {
                'num_intruders': len(intruder_indices),
                'intruder_indices': intruder_indices,
                'intruder_similarities': intruder_sims,
                'min_similarity': float(similarities.min()),
                'mean_similarity': float(similarities.mean()),
                'median_similarity': float(np.median(similarities))
            }

            block_result['total_intruders'] += len(intruder_indices)

        results.append(block_result)

    return results


def plot_similarity_distribution(similarities, threshold, block_idx, weight_name, output_dir):
    """Plot similarity distribution for a single weight."""
    plt.figure(figsize=(10, 6))

    # Plot as a line with markers
    plt.plot(similarities, 'b-o', linewidth=2, markersize=4, label='Max Similarity to Pre-trained')

    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
                label=f'Intruder Threshold ({threshold})')

    # Highlight intruders
    intruders = np.where(similarities < threshold)[0]
    if len(intruders) > 0:
        plt.scatter(intruders, similarities[intruders],
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


def plot_heatmap(similarity_tensor, block_indices, threshold, output_dir):
    """
    Plot heatmap of similarities across all blocks and weights.
    Creates 5 heatmaps: combined (Q+K+V+O), and separate ones for Q, K, V, O.

    Args:
        similarity_tensor: [num_blocks, 4, top_k] array (Q, K, V, O)
        block_indices: [num_blocks] array
        threshold: Threshold for visualization
        output_dir: Output directory
    """
    num_blocks, _, top_k = similarity_tensor.shape

    # 1. Combined heatmap (all Q, K, V, O together)
    reshaped = similarity_tensor.reshape(num_blocks * 4, top_k)

    fig, ax = plt.subplots(figsize=(16, max(8, num_blocks * 0.6)))

    # Plot heatmap (removed contour)
    im = ax.imshow(reshaped, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    # Set ticks
    ax.set_xlabel('Singular Vector Index', fontsize=12)
    ax.set_ylabel('Block / Weight', fontsize=12)

    # Y-axis labels
    y_labels = []
    for block_idx in block_indices:
        y_labels.extend([f'Block {block_idx} Q', f'Block {block_idx} K',
                        f'Block {block_idx} V', f'Block {block_idx} O'])
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)

    # X-axis labels (show every 8th)
    x_ticks = range(0, top_k, 8)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Cosine Similarity', fontsize=12)

    plt.title(f'Backbone Similarity Heatmap - All Weights (Threshold: {threshold})', fontsize=14)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, f'similarity_heatmap_all_threshold_{threshold}.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"  Combined heatmap saved: {heatmap_path}")

    # 2. Separate heatmaps for Q, K, V, O
    weight_names = ['Q', 'K', 'V', 'O']
    weight_indices = [0, 1, 2, 3]  # Indices in similarity_tensor

    for weight_name, weight_idx in zip(weight_names, weight_indices):
        # Extract data for this weight only: [num_blocks, top_k]
        weight_data = similarity_tensor[:, weight_idx, :]

        fig, ax = plt.subplots(figsize=(16, max(6, num_blocks * 0.4)))

        # Plot heatmap
        im = ax.imshow(weight_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        # Set ticks
        ax.set_xlabel('Singular Vector Index', fontsize=12)
        ax.set_ylabel('Block Index', fontsize=12)

        # Y-axis labels (block indices)
        ax.set_yticks(range(len(block_indices)))
        ax.set_yticklabels([f'Block {idx}' for idx in block_indices], fontsize=9)

        # X-axis labels (show every 8th)
        x_ticks = range(0, top_k, 8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Max Cosine Similarity', fontsize=12)

        plt.title(f'Backbone Similarity Heatmap - {weight_name} Weights (Threshold: {threshold})',
                 fontsize=14)
        plt.tight_layout()

        weight_heatmap_path = os.path.join(output_dir,
                                          f'similarity_heatmap_{weight_name.lower()}_threshold_{threshold}.png')
        plt.savefig(weight_heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"  {weight_name} heatmap saved: {weight_heatmap_path}")


def plot_intruder_summary(results, output_dir, threshold):
    """
    Plot summary of intruders across all blocks.

    Args:
        results: List of result dicts
        output_dir: Output directory
        threshold: Threshold used
    """
    block_indices = [r['block_idx'] for r in results]
    intruders_q = [r['intruders_per_weight']['q']['num_intruders'] for r in results]
    intruders_k = [r['intruders_per_weight']['k']['num_intruders'] for r in results]
    intruders_v = [r['intruders_per_weight']['v']['num_intruders'] for r in results]
    intruders_o = [r['intruders_per_weight']['o']['num_intruders'] for r in results]
    total_intruders = [r['total_intruders'] for r in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Subplot 1: Bar chart (existing visualization)
    x = np.arange(len(block_indices))
    width = 0.2  # Reduced width to fit 4 bars

    ax1.bar(x - 1.5*width, intruders_q, width, label='Q', alpha=0.8, color='#1f77b4')
    ax1.bar(x - 0.5*width, intruders_k, width, label='K', alpha=0.8, color='#ff7f0e')
    ax1.bar(x + 0.5*width, intruders_v, width, label='V', alpha=0.8, color='#2ca02c')
    ax1.bar(x + 1.5*width, intruders_o, width, label='O', alpha=0.8, color='#9467bd')

    ax1.set_xlabel('Block Index', fontsize=12)
    ax1.set_ylabel('Number of Intruders', fontsize=12)
    ax1.set_title(f'Intruder Dimensions per Block - Bar Chart (Threshold: {threshold})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(block_indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Line chart with total intruders per block
    ax2.plot(block_indices, total_intruders, 'o-', linewidth=2.5, markersize=8,
             color='#d62728', label='Total Intruders')
    ax2.plot(block_indices, intruders_q, 's--', linewidth=1.5, markersize=6,
             alpha=0.7, color='#1f77b4', label='Q')
    ax2.plot(block_indices, intruders_k, '^--', linewidth=1.5, markersize=6,
             alpha=0.7, color='#ff7f0e', label='K')
    ax2.plot(block_indices, intruders_v, 'D--', linewidth=1.5, markersize=6,
             alpha=0.7, color='#2ca02c', label='V')
    ax2.plot(block_indices, intruders_o, 'p--', linewidth=1.5, markersize=6,
             alpha=0.7, color='#9467bd', label='O')

    ax2.set_xlabel('Block Index', fontsize=12)
    ax2.set_ylabel('Number of Intruders', fontsize=12)
    ax2.set_title(f'Intruder Dimensions per Block - Line Plot (Threshold: {threshold})', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(block_indices)

    # Highlight blocks with high intruder counts
    max_intruders = max(total_intruders) if total_intruders else 0
    if max_intruders > 0:
        for i, (block_idx, num) in enumerate(zip(block_indices, total_intruders)):
            if num > 5:  # Highlight blocks with more than 5 intruders
                ax2.axvspan(block_idx - 0.5, block_idx + 0.5, alpha=0.2, color='red')

    plt.tight_layout()

    summary_path = os.path.join(output_dir, f'intruder_summary_threshold_{threshold}.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"  Summary plot saved: {summary_path}")


def print_detailed_results(results, top_k, threshold):
    """Print detailed analysis results."""
    logging.info(f"\n{'='*80}")
    logging.info("DETAILED RESULTS")
    logging.info(f"{'='*80}")

    for result in results:
        block_idx = result['block_idx']
        logging.info(f"\nBlock {block_idx}:")

        for weight_name in ['q', 'k', 'v', 'o']:
            info = result['intruders_per_weight'][weight_name]
            num_intruders = info['num_intruders']

            logging.info(f"  {weight_name.upper()}: {num_intruders} intruders")
            logging.info(f"    Min similarity: {info['min_similarity']:.4f}")
            logging.info(f"    Mean similarity: {info['mean_similarity']:.4f}")
            logging.info(f"    Median similarity: {info['median_similarity']:.4f}")

            if num_intruders > 0:
                logging.info(f"    Intruder indices: {info['intruder_indices'][:10]}" +
                           (f" ... ({num_intruders} total)" if num_intruders > 10 else ""))
                logging.info(f"    Intruder similarities: " +
                           f"{[f'{s:.3f}' for s in info['intruder_similarities'][:5]]}" +
                           (f" ..." if num_intruders > 5 else ""))


def print_summary(results, top_k, threshold):
    """Print summary statistics."""
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"{'Block':>6}  {'W_q':>8}  {'W_k':>8}  {'W_v':>8}  {'W_o':>8}  {'Total':>8}  {'Status':>20}")
    logging.info("-" * 88)

    total_intruders_all = 0
    for result in results:
        num_q = result['intruders_per_weight']['q']['num_intruders']
        num_k = result['intruders_per_weight']['k']['num_intruders']
        num_v = result['intruders_per_weight']['v']['num_intruders']
        num_o = result['intruders_per_weight']['o']['num_intruders']
        total = num_q + num_k + num_v + num_o
        total_intruders_all += total

        if total == 0:
            status = "✅ Perfect"
        elif total <= 2:
            status = "✓  Excellent"
        elif total <= 5:
            status = "⚠  Warning"
        else:
            status = "❌ Drift"

        logging.info(f"{result['block_idx']:>6}  {num_q:>8}  {num_k:>8}  {num_v:>8}  {num_o:>8}  "
                    f"{total:>8}  {status:>20}")

    num_blocks = len(results)
    avg_intruders = total_intruders_all / num_blocks if num_blocks > 0 else 0

    logging.info(f"\nOverall Statistics:")
    logging.info(f"  Threshold: {threshold}")
    logging.info(f"  Total intruders across all blocks: {total_intruders_all}")
    logging.info(f"  Average intruders per block: {avg_intruders:.2f}")
    logging.info(f"  Max possible intruders per block: {top_k * 4} (4 weight matrices: Q, K, V, O)")
    logging.info(f"  Intruder rate: {total_intruders_all / (num_blocks * top_k * 4):.2%}")

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


def main():
    parser = argparse.ArgumentParser(
        description='Analyze precomputed backbone similarity matrix with different thresholds'
    )
    parser.add_argument('--matrix_dir', type=str, required=True,
                       help='Directory containing precomputed similarity matrix')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Cosine similarity threshold for detecting intruders')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results (default: matrix_dir/analysis_threshold_X)')
    parser.add_argument('--plot_individual', action='store_true',
                       help='Generate individual plots for each block/weight (slower)')
    parser.add_argument('--plot_heatmap', action='store_true', default=True,
                       help='Generate similarity heatmap')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed per-block results')

    args = parser.parse_args()

    setup_logging()

    logging.info("="*80)
    logging.info("ANALYZE BACKBONE SIMILARITY MATRIX")
    logging.info("="*80)
    logging.info(f"Matrix directory: {args.matrix_dir}")
    logging.info(f"Threshold: {args.threshold}")

    # Load precomputed similarity matrix
    similarity_tensor, block_indices, metadata = load_similarity_matrix(args.matrix_dir)

    num_blocks, num_weights, top_k = similarity_tensor.shape
    logging.info(f"\nMatrix shape: {similarity_tensor.shape}")
    logging.info(f"  {num_blocks} blocks")
    logging.info(f"  {num_weights} weights per block (Q, K, V)")
    logging.info(f"  {top_k} singular vectors per weight")

    # Create output directory
    if args.output_dir is None:
        threshold_str = str(args.threshold).replace('.', '_')
        output_dir = os.path.join(args.matrix_dir, f'analysis_threshold_{threshold_str}')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\nResults will be saved to: {output_dir}")

    # Detect intruders
    logging.info("\nDetecting intruder dimensions...")
    results = detect_intruders(similarity_tensor, block_indices, args.threshold)

    # Print results
    if args.verbose:
        print_detailed_results(results, top_k, args.threshold)

    print_summary(results, top_k, args.threshold)

    # Generate plots
    logging.info("\n" + "="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)

    if args.plot_heatmap:
        logging.info("Generating heatmap...")
        plot_heatmap(similarity_tensor, block_indices, args.threshold, output_dir)

    logging.info("Generating summary plot...")
    plot_intruder_summary(results, output_dir, args.threshold)

    if args.plot_individual:
        logging.info("Generating individual plots...")
        for i, block_idx in enumerate(block_indices):
            for j, weight_name in enumerate(['q', 'k', 'v']):
                similarities = similarity_tensor[i, j, :]
                plot_similarity_distribution(
                    similarities, args.threshold, int(block_idx), weight_name, output_dir
                )

    # Save results as text
    results_file = os.path.join(output_dir, 'analysis_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BACKBONE SIMILARITY MATRIX ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"Matrix directory: {args.matrix_dir}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Blocks analyzed: {list(block_indices)}\n")
        f.write(f"Top-k vectors: {top_k}\n\n")

        total_intruders = sum(r['total_intruders'] for r in results)
        f.write(f"Total intruders: {total_intruders}\n")
        f.write(f"Intruder rate: {total_intruders / (num_blocks * top_k * 4):.2%}\n\n")

        f.write(f"{'Block':>6}  {'W_q':>8}  {'W_k':>8}  {'W_v':>8}  {'W_o':>8}  {'Total':>8}\n")
        f.write("-" * 88 + "\n")

        for result in results:
            num_q = result['intruders_per_weight']['q']['num_intruders']
            num_k = result['intruders_per_weight']['k']['num_intruders']
            num_v = result['intruders_per_weight']['v']['num_intruders']
            num_o = result['intruders_per_weight']['o']['num_intruders']
            total = num_q + num_k + num_v + num_o

            f.write(f"{result['block_idx']:>6}  {num_q:>8}  {num_k:>8}  {num_v:>8}  {num_o:>8}  {total:>8}\n")

    logging.info(f"✅ Results saved: {results_file}")

    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
