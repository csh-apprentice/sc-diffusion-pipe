#!/usr/bin/env python3
"""
Compare Backbone Similarity Matrices (Realistic vs Synthetic)

This script loads two precomputed similarity matrices (from realistic and synthetic datasets),
analyzes intruder dimensions in the V matrix only, and generates comparison plots.
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


def configure_matplotlib():
    """Configure matplotlib to use serif fonts and larger sizes."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'serif']
    plt.rcParams['font.size'] = 40
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 20


def setup_logging():
    """Configure logging for the comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_similarity_matrix(matrix_dir):
    """
    Load precomputed similarity matrix and metadata.

    Args:
        matrix_dir: Directory containing similarity_matrix.npy

    Returns:
        similarity_tensor: [num_blocks, 4, top_k] array (Q, K, V, O)
        block_indices: [num_blocks] array
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

    return similarity_tensor, block_indices


def extract_v_matrix(similarity_tensor):
    """
    Extract V matrix from similarity tensor.

    Args:
        similarity_tensor: [num_blocks, 4, top_k] array (Q, K, V, O)

    Returns:
        v_matrix: [num_blocks, top_k] array (V only, index 2)
    """
    # V matrix is at index 2 (Q=0, K=1, V=2, O=3)
    v_matrix = similarity_tensor[:, 2, :]
    logging.info(f"✅ Extracted V matrix: {v_matrix.shape}")
    return v_matrix


def detect_intruders_v(v_matrix, block_indices, threshold):
    """
    Detect intruder dimensions in V matrix using a threshold.

    Args:
        v_matrix: [num_blocks, top_k] array
        block_indices: [num_blocks] array
        threshold: Cosine similarity threshold

    Returns:
        intruders_per_block: List of number of intruders per block
    """
    intruders_per_block = []

    for i, block_idx in enumerate(block_indices):
        similarities = v_matrix[i, :]
        intruder_indices = np.where(similarities < threshold)[0]
        num_intruders = len(intruder_indices)
        intruders_per_block.append(num_intruders)

    return intruders_per_block


def plot_comparison(block_indices, realistic_intruders, synthetic_intruders,
                    threshold, output_dir):
    """
    Plot comparison of intruder vectors between realistic and synthetic datasets.

    Args:
        block_indices: List of block indices
        realistic_intruders: List of intruder counts for realistic dataset (per block)
        synthetic_intruders: List of intruder counts for synthetic dataset (per block)
        threshold: Threshold used
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot realistic in red, synthetic in blue
    ax.plot(block_indices, realistic_intruders, 'o-', linewidth=3.5, markersize=10,
            color='red', label='Realistic Dataset', alpha=0.8)
    ax.plot(block_indices, synthetic_intruders, 's-', linewidth=3.5, markersize=10,
            color='blue', label='Synthetic Dataset', alpha=0.8)

    ax.set_xlabel('Block Index', fontsize=18)
    ax.set_ylabel('Number of Intruders (V Matrix)', fontsize=18)
    ax.set_title(f'Intruder Dimension Comparison - V Matrix (Threshold: {threshold})',
                 fontsize=20, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(block_indices)

    # Add subtle background shading to highlight differences
    for i, (block_idx, real, synth) in enumerate(zip(block_indices, realistic_intruders, synthetic_intruders)):
        if abs(real - synth) > 5:  # Highlight significant differences
            ax.axvspan(block_idx - 0.5, block_idx + 0.5, alpha=0.1, color='yellow')

    plt.tight_layout()

    # Save plot
    threshold_str = str(threshold).replace('.', '_')
    plot_path = os.path.join(output_dir, f'comparison_v_matrix_threshold_{threshold_str}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ Comparison plot saved: {plot_path}")


def plot_difference(block_indices, realistic_intruders, synthetic_intruders,
                    threshold, output_dir):
    """
    Plot the difference in intruder counts (realistic - synthetic).

    Args:
        block_indices: List of block indices
        realistic_intruders: List of intruder counts for realistic dataset
        synthetic_intruders: List of intruder counts for synthetic dataset
        threshold: Threshold used
        output_dir: Output directory
    """
    differences = [real - synth for real, synth in zip(realistic_intruders, synthetic_intruders)]

    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ['green' if d < 0 else 'red' if d > 0 else 'gray' for d in differences]
    ax.bar(block_indices, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Block Index', fontsize=18)
    ax.set_ylabel('Difference in Intruders (Realistic - Synthetic)', fontsize=18)
    ax.set_title(f'Intruder Difference per Block - V Matrix (Threshold: {threshold})',
                 fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(block_indices)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Realistic > Synthetic'),
        Patch(facecolor='green', alpha=0.7, label='Realistic < Synthetic'),
        Patch(facecolor='gray', alpha=0.7, label='Equal')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    # Save plot
    threshold_str = str(threshold).replace('.', '_')
    plot_path = os.path.join(output_dir, f'difference_v_matrix_threshold_{threshold_str}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ Difference plot saved: {plot_path}")


def print_comparison_summary(block_indices, realistic_intruders, synthetic_intruders, threshold):
    """Print comparison summary statistics."""
    logging.info(f"\n{'='*80}")
    logging.info("COMPARISON SUMMARY (V MATRIX ONLY)")
    logging.info(f"{'='*80}")
    logging.info(f"{'Block':>6}  {'Realistic':>12}  {'Synthetic':>12}  {'Difference':>12}  {'Status':>15}")
    logging.info("-" * 80)

    for block_idx, real, synth in zip(block_indices, realistic_intruders, synthetic_intruders):
        diff = real - synth

        if abs(diff) == 0:
            status = "Equal"
        elif abs(diff) <= 2:
            status = "Similar"
        elif diff > 0:
            status = "↑ More Real"
        else:
            status = "↓ More Synth"

        logging.info(f"{block_idx:>6}  {real:>12}  {synth:>12}  {diff:>12}  {status:>15}")

    # Overall statistics
    total_realistic = sum(realistic_intruders)
    total_synthetic = sum(synthetic_intruders)
    total_diff = total_realistic - total_synthetic

    avg_realistic = np.mean(realistic_intruders)
    avg_synthetic = np.mean(synthetic_intruders)

    logging.info(f"\nOverall Statistics:")
    logging.info(f"  Threshold: {threshold}")
    logging.info(f"  Total intruders (Realistic): {total_realistic}")
    logging.info(f"  Total intruders (Synthetic): {total_synthetic}")
    logging.info(f"  Total difference: {total_diff:+d}")
    logging.info(f"  Average per block (Realistic): {avg_realistic:.2f}")
    logging.info(f"  Average per block (Synthetic): {avg_synthetic:.2f}")

    # Interpretation
    if abs(total_diff) <= 5:
        logging.info(f"\n✅ RESULT: Very similar intruder patterns between datasets")
        logging.info("   Both datasets show comparable backbone drift in V matrix.")
    elif total_realistic > total_synthetic:
        logging.info(f"\n⚠️  RESULT: Realistic dataset shows MORE intruders (+{total_diff})")
        logging.info("   Realistic training may introduce more new V features.")
    else:
        logging.info(f"\n⚠️  RESULT: Synthetic dataset shows MORE intruders ({total_diff})")
        logging.info("   Synthetic training may introduce more new V features.")


def save_comparison_results(block_indices, realistic_intruders, synthetic_intruders,
                            threshold, output_dir, realistic_dir, synthetic_dir):
    """Save comparison results to text file."""
    results_file = os.path.join(output_dir, 'comparison_results.txt')

    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BACKBONE SIMILARITY MATRIX COMPARISON (V MATRIX ONLY)\n")
        f.write("="*80 + "\n")
        f.write(f"Realistic dataset matrix: {realistic_dir}\n")
        f.write(f"Synthetic dataset matrix: {synthetic_dir}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Blocks analyzed: {list(block_indices)}\n\n")

        total_realistic = sum(realistic_intruders)
        total_synthetic = sum(synthetic_intruders)

        f.write(f"Total intruders (Realistic): {total_realistic}\n")
        f.write(f"Total intruders (Synthetic): {total_synthetic}\n")
        f.write(f"Difference: {total_realistic - total_synthetic:+d}\n\n")

        f.write(f"{'Block':>6}  {'Realistic':>12}  {'Synthetic':>12}  {'Difference':>12}\n")
        f.write("-" * 80 + "\n")

        for block_idx, real, synth in zip(block_indices, realistic_intruders, synthetic_intruders):
            diff = real - synth
            f.write(f"{block_idx:>6}  {real:>12}  {synth:>12}  {diff:>12}\n")

    logging.info(f"✅ Results saved: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare backbone similarity matrices from realistic and synthetic datasets'
    )
    parser.add_argument('--realistic_dir', type=str, required=True,
                       help='Directory containing realistic dataset similarity matrix')
    parser.add_argument('--synthetic_dir', type=str, required=True,
                       help='Directory containing synthetic dataset similarity matrix')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Cosine similarity threshold for detecting intruders')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed analysis')

    args = parser.parse_args()

    # Configure matplotlib with serif fonts and larger sizes
    configure_matplotlib()

    setup_logging()

    logging.info("="*80)
    logging.info("COMPARE BACKBONE SIMILARITY MATRICES (V MATRIX ONLY)")
    logging.info("="*80)
    logging.info(f"Realistic dataset: {args.realistic_dir}")
    logging.info(f"Synthetic dataset: {args.synthetic_dir}")
    logging.info(f"Threshold: {args.threshold}")

    # Load both similarity matrices
    logging.info("\n" + "="*80)
    logging.info("LOADING REALISTIC DATASET MATRIX")
    logging.info("="*80)
    realistic_tensor, realistic_blocks = load_similarity_matrix(args.realistic_dir)

    logging.info("\n" + "="*80)
    logging.info("LOADING SYNTHETIC DATASET MATRIX")
    logging.info("="*80)
    synthetic_tensor, synthetic_blocks = load_similarity_matrix(args.synthetic_dir)

    # Verify blocks match
    if not np.array_equal(realistic_blocks, synthetic_blocks):
        logging.warning("⚠️  WARNING: Block indices differ between matrices!")
        logging.warning(f"   Realistic: {realistic_blocks}")
        logging.warning(f"   Synthetic: {synthetic_blocks}")
        logging.warning("   Using intersection of blocks for comparison.")

        # Use intersection
        common_blocks = np.intersect1d(realistic_blocks, synthetic_blocks)
        realistic_mask = np.isin(realistic_blocks, common_blocks)
        synthetic_mask = np.isin(synthetic_blocks, common_blocks)

        realistic_tensor = realistic_tensor[realistic_mask]
        synthetic_tensor = synthetic_tensor[synthetic_mask]
        block_indices = common_blocks
    else:
        block_indices = realistic_blocks

    # Extract V matrices
    logging.info("\n" + "="*80)
    logging.info("EXTRACTING V MATRICES")
    logging.info("="*80)
    realistic_v = extract_v_matrix(realistic_tensor)
    synthetic_v = extract_v_matrix(synthetic_tensor)

    # Detect intruders
    logging.info("\n" + "="*80)
    logging.info("DETECTING INTRUDERS")
    logging.info("="*80)
    realistic_intruders = detect_intruders_v(realistic_v, block_indices, args.threshold)
    synthetic_intruders = detect_intruders_v(synthetic_v, block_indices, args.threshold)

    logging.info(f"✅ Realistic dataset: {sum(realistic_intruders)} total intruders")
    logging.info(f"✅ Synthetic dataset: {sum(synthetic_intruders)} total intruders")

    # Print comparison
    print_comparison_summary(block_indices, realistic_intruders, synthetic_intruders, args.threshold)

    # Create output directory
    if args.output_dir is None:
        timestamp = Path(args.realistic_dir).name.split('_')[-1]
        threshold_str = str(args.threshold).replace('.', '_')
        output_dir = f"output/similarity_comparison/realistic_vs_synthetic_threshold_{threshold_str}_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\n✅ Results will be saved to: {output_dir}")

    # Generate plots
    logging.info("\n" + "="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)

    plot_comparison(block_indices, realistic_intruders, synthetic_intruders,
                   args.threshold, output_dir)
    plot_difference(block_indices, realistic_intruders, synthetic_intruders,
                   args.threshold, output_dir)

    # Save results
    save_comparison_results(block_indices, realistic_intruders, synthetic_intruders,
                          args.threshold, output_dir, args.realistic_dir, args.synthetic_dir)

    logging.info("\n" + "="*80)
    logging.info("COMPARISON COMPLETE")
    logging.info("="*80)
    logging.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
