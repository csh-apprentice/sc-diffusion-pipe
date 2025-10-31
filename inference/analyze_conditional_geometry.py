#!/usr/bin/env python3
"""
Geometric Analysis of Conditional Subspaces.

This script performs three geometric analyses on extracted conditional subspaces:
1. Effective Dimensionality: Measure conditioning compactness
2. Inter-Condition Orthogonality: Measure separation between conditions
3. Backbone Alignment: Measure entanglement with backbone outputs

Usage:
    python inference/analyze_conditional_geometry.py \
        --subspace_file output/subspace_analysis/subspace_data.pkl \
        --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
        --checkpoint outputs/checkpoint/step_10000 \
        --output_dir output/subspace_analysis
"""

import torch
import os
import sys
import logging
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_subspace_data(subspace_file):
    """
    Load pre-computed subspace data.

    Args:
        subspace_file: Path to subspace_data.pkl

    Returns:
        subspace_data dict and metadata
    """
    logging.info(f"Loading subspace data from {subspace_file}")

    with open(subspace_file, 'rb') as f:
        subspace_data = pickle.load(f)

    # Load metadata
    metadata_file = Path(subspace_file).parent / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    logging.info(f"Loaded {len(subspace_data)} conditions")
    logging.info(f"Blocks per condition: {len(next(iter(subspace_data.values())))}")

    return subspace_data, metadata


# ============================================================================
# Analysis 1: Effective Dimensionality
# ============================================================================

def analyze_dimensionality(subspace_data):
    """
    Analyze effective dimensionality across conditions and blocks.

    Args:
        subspace_data: Dict[fps_str][block_idx] -> subspace info

    Returns:
        pandas DataFrame with dimensionality statistics
    """
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 1: EFFECTIVE DIMENSIONALITY")
    logging.info("="*80)

    results = []

    for fps_str, subspace_list in subspace_data.items():
        for subspace_info in subspace_list:
            sv = subspace_info['singular_values'].numpy()

            # Normalize singular values
            sv_norm = sv / sv[0]

            # Compute entropy (measure of dimensionality spread)
            sv_prob = (sv ** 2) / (sv ** 2).sum()
            entropy = -np.sum(sv_prob * np.log(sv_prob + 1e-10))

            results.append({
                'fps': fps_str,
                'block_idx': subspace_info['block_idx'],
                'effective_rank': subspace_info['effective_rank'],
                'full_rank': subspace_info['full_rank'],
                'rank_ratio': subspace_info['effective_rank'] / subspace_info['full_rank'],
                'sv_entropy': entropy,
                'sv_max': sv[0],
                'sv_min': sv[-1],
                'sv_decay': np.log(sv[0] / (sv[-1] + 1e-10)),
                'sv_mean': sv.mean(),
                'sv_std': sv.std()
            })

    df = pd.DataFrame(results)

    # Print summary statistics
    logging.info("\nDimensionality Summary:")
    logging.info(f"  Mean effective rank: {df['effective_rank'].mean():.1f} ± {df['effective_rank'].std():.1f}")
    logging.info(f"  Mean rank ratio: {df['rank_ratio'].mean():.3f} ± {df['rank_ratio'].std():.3f}")
    logging.info(f"  Mean SV entropy: {df['sv_entropy'].mean():.3f} ± {df['sv_entropy'].std():.3f}")
    logging.info(f"  SV decay range: [{df['sv_decay'].min():.2f}, {df['sv_decay'].max():.2f}]")

    # Interpretation
    mean_ratio = df['rank_ratio'].mean()
    if mean_ratio < 0.3:
        logging.info("  ✅ Interpretation: COMPACT representation (efficient)")
    elif mean_ratio < 0.6:
        logging.info("  ⚠️  Interpretation: MODERATE compactness")
    else:
        logging.info("  ❌ Interpretation: HIGH dimensionality (potentially redundant)")

    return df


def plot_dimensionality_analysis(df, output_dir):
    """Generate dimensionality analysis plots."""
    logging.info("  Generating dimensionality plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Effective rank by block
    fps_values = df['fps'].unique()
    for fps in fps_values:
        subset = df[df['fps'] == fps]
        axes[0, 0].plot(subset['block_idx'], subset['effective_rank'],
                       marker='o', label=f'FPS={fps}', alpha=0.7)
    axes[0, 0].set_xlabel('Block Index')
    axes[0, 0].set_ylabel('Effective Rank')
    axes[0, 0].set_title('Effective Rank Across Blocks')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Rank ratio distribution
    sns.boxplot(data=df, x='fps', y='rank_ratio', ax=axes[0, 1])
    axes[0, 1].set_title('Rank Ratio Distribution')
    axes[0, 1].set_ylabel('Effective Rank / Full Rank')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Compact threshold')
    axes[0, 1].axhline(0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')

    # Plot 3: Singular value entropy
    for fps in fps_values:
        subset = df[df['fps'] == fps]
        axes[1, 0].plot(subset['block_idx'], subset['sv_entropy'],
                       marker='o', label=f'FPS={fps}', alpha=0.7)
    axes[1, 0].set_xlabel('Block Index')
    axes[1, 0].set_ylabel('SV Entropy')
    axes[1, 0].set_title('Singular Value Entropy Across Blocks')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: SV decay rate
    sns.violinplot(data=df, x='fps', y='sv_decay', ax=axes[1, 1])
    axes[1, 1].set_title('Singular Value Decay Rate')
    axes[1, 1].set_ylabel('log(σ_max / σ_min)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'dimensionality_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved to {output_file}")


# ============================================================================
# Analysis 2: Inter-Condition Orthogonality
# ============================================================================

def compute_principal_angles(basis1, basis2):
    """
    Compute principal angles between two subspaces.

    Uses the formula: cos(θ_i) = σ_i(B1^T @ B2)

    Args:
        basis1: [dim, rank1] orthonormal basis tensor
        basis2: [dim, rank2] orthonormal basis tensor

    Returns:
        angles in degrees [0, 90]
    """
    # Compute overlap matrix
    overlap = basis1.T @ basis2  # [rank1, rank2]

    # SVD gives principal angles via: cos(θ) = σ
    U, S, Vh = torch.linalg.svd(overlap, full_matrices=False)

    # Convert to angles (clamp for numerical stability)
    cos_angles = torch.clamp(S, 0.0, 1.0)
    angles_rad = torch.acos(cos_angles)
    angles_deg = angles_rad * 180.0 / math.pi

    return angles_deg.numpy()


def analyze_orthogonality(subspace_data):
    """
    Analyze orthogonality between different FPS conditions.

    Args:
        subspace_data: Dict[fps_str][block_idx] -> subspace info

    Returns:
        Dict with angle matrices and statistics
    """
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 2: INTER-CONDITION ORTHOGONALITY")
    logging.info("="*80)

    fps_strs = list(subspace_data.keys())
    n_fps = len(fps_strs)
    n_blocks = len(subspace_data[fps_strs[0]])

    logging.info(f"Computing pairwise angles for {n_fps} conditions across {n_blocks} blocks")

    # Storage for angles
    min_angles = np.zeros((n_fps, n_fps, n_blocks))
    mean_angles = np.zeros((n_fps, n_fps, n_blocks))
    max_angles = np.zeros((n_fps, n_fps, n_blocks))

    # Compute pairwise angles
    for block_idx in range(n_blocks):
        for i, fps1 in enumerate(fps_strs):
            for j, fps2 in enumerate(fps_strs):
                if i == j:
                    # Same condition = 0 angle
                    min_angles[i, j, block_idx] = 0.0
                    mean_angles[i, j, block_idx] = 0.0
                    max_angles[i, j, block_idx] = 0.0
                    continue

                basis1 = subspace_data[fps1][block_idx]['basis']
                basis2 = subspace_data[fps2][block_idx]['basis']

                angles = compute_principal_angles(basis1, basis2)

                min_angles[i, j, block_idx] = angles[0]
                mean_angles[i, j, block_idx] = angles.mean()
                max_angles[i, j, block_idx] = angles[-1]

    # Compute summary statistics (excluding diagonal)
    mask = np.ones_like(min_angles[:, :, 0], dtype=bool)
    np.fill_diagonal(mask, False)
    valid_min_angles = min_angles[mask].flatten()
    valid_mean_angles = mean_angles[mask].flatten()

    logging.info("\nOrthogonality Summary:")
    logging.info(f"  Mean pairwise angle (min): {valid_min_angles.mean():.1f}° ± {valid_min_angles.std():.1f}°")
    logging.info(f"  Range (min): [{valid_min_angles.min():.1f}°, {valid_min_angles.max():.1f}°]")
    logging.info(f"  Mean pairwise angle (mean): {valid_mean_angles.mean():.1f}° ± {valid_mean_angles.std():.1f}°")

    # Interpretation
    mean_angle = valid_min_angles.mean()
    if mean_angle > 60:
        logging.info("  ✅ Interpretation: STRONG orthogonality (distinct effects)")
    elif mean_angle > 30:
        logging.info("  ⚠️  Interpretation: MODERATE orthogonality")
    else:
        logging.info("  ❌ Interpretation: WEAK orthogonality (potential interference)")

    return {
        'min_angles': min_angles,
        'mean_angles': mean_angles,
        'max_angles': max_angles,
        'fps_strs': fps_strs
    }


def plot_orthogonality_analysis(orthogonality_data, output_dir):
    """Generate orthogonality heatmaps."""
    logging.info("  Generating orthogonality plots...")

    min_angles = orthogonality_data['min_angles']
    fps_strs = orthogonality_data['fps_strs']
    n_blocks = min_angles.shape[2]

    # Select representative blocks to visualize
    if n_blocks <= 3:
        blocks_to_plot = list(range(n_blocks))
    else:
        blocks_to_plot = [0, n_blocks // 2, n_blocks - 1]

    fig, axes = plt.subplots(1, len(blocks_to_plot), figsize=(6*len(blocks_to_plot), 5))
    if len(blocks_to_plot) == 1:
        axes = [axes]

    for idx, block_idx in enumerate(blocks_to_plot):
        sns.heatmap(
            min_angles[:, :, block_idx],
            annot=True, fmt='.1f',
            xticklabels=[f'{fps[:10]}' for fps in fps_strs],
            yticklabels=[f'{fps[:10]}' for fps in fps_strs],
            cmap='RdYlGn', vmin=0, vmax=90,
            ax=axes[idx],
            cbar_kws={'label': 'Min Principal Angle (°)'},
            square=True
        )
        axes[idx].set_title(f'Block {block_idx}')

    plt.suptitle('Inter-Condition Orthogonality (Minimum Principal Angles)', y=1.02)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'orthogonality_heatmaps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved to {output_file}")

    # Also plot angle evolution across blocks
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean angle across blocks for each condition pair
    n_fps = len(fps_strs)
    for i in range(n_fps):
        for j in range(i+1, n_fps):  # Only upper triangle
            angles_across_blocks = min_angles[i, j, :]
            label = f'{fps_strs[i][:8]} vs {fps_strs[j][:8]}'
            ax.plot(range(n_blocks), angles_across_blocks, marker='o', label=label, alpha=0.7)

    ax.set_xlabel('Block Index')
    ax.set_ylabel('Minimum Principal Angle (°)')
    ax.set_title('Inter-Condition Orthogonality Across Blocks')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(60, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'orthogonality_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved to {output_file}")


# ============================================================================
# Analysis 3: Backbone Alignment
# ============================================================================

def extract_backbone_subspace(pipeline, block_idx, energy_threshold=0.90):
    """
    Extract dominant output subspace from backbone's cross-attention value projection.

    NOTE: We extract from the value projection (not output projection) because
    FPS adapters modify values at the per-head level, so we need to compare
    in the same dimensional space (head_dim).

    Args:
        pipeline: WanPipeline instance
        block_idx: Which block to analyze
        energy_threshold: Cumulative energy to retain (default 90%)

    Returns:
        Dict with backbone subspace info
    """
    block = pipeline.transformer.blocks[block_idx]

    # Get value projection matrix (projects from dim -> num_heads * head_dim)
    v_weight = block.cross_attn.v.weight.data  # [num_heads * head_dim, dim]

    # Reshape to separate heads
    num_heads = block.cross_attn.num_heads
    head_dim = v_weight.shape[0] // num_heads

    # Reshape: [num_heads * head_dim, dim] -> [num_heads, head_dim, dim]
    v_weight_heads = v_weight.view(num_heads, head_dim, -1)

    # Average across heads to get representative head-space basis
    # This gives us the "typical" value space per head
    v_weight_avg = v_weight_heads.mean(dim=0)  # [head_dim, dim]

    # SVD to find dominant directions in head_dim space
    # v_weight_avg: [head_dim, dim]
    # After SVD: v_weight_avg ≈ U @ diag(S) @ Vh
    # U columns span the output space (head_dim), which is what we want
    v_weight_avg_fp32 = v_weight_avg.float()
    U, S, Vh = torch.linalg.svd(v_weight_avg_fp32, full_matrices=False)

    # U columns are basis vectors in head_dim space
    # Select top-k by energy
    energy = S ** 2
    cumulative_energy = torch.cumsum(energy, dim=0) / energy.sum()
    k = (cumulative_energy < energy_threshold).sum().item() + 1

    return {
        'basis': U[:, :k].cpu(),  # [head_dim, k]
        'singular_values': S.cpu(),
        'effective_rank': k,
        'energy_threshold': energy_threshold
    }


def analyze_backbone_alignment(subspace_data, pipeline):
    """
    Analyze alignment between FPS conditioning and backbone outputs.

    Requires loading the pipeline to extract backbone subspaces.

    Args:
        subspace_data: Dict[fps_str][block_idx] -> subspace info
        pipeline: WanPipeline instance (or None to skip this analysis)

    Returns:
        DataFrame with alignment statistics (or None if pipeline not provided)
    """
    if pipeline is None:
        logging.warning("\n⚠️  Skipping backbone alignment analysis (pipeline not provided)")
        return None

    logging.info("\n" + "="*80)
    logging.info("ANALYSIS 3: BACKBONE ALIGNMENT")
    logging.info("="*80)

    results = []
    fps_strs = list(subspace_data.keys())

    # Extract backbone subspaces (cached by block_idx)
    backbone_subspaces = {}

    logging.info("Extracting backbone output subspaces...")
    for fps_str in fps_strs:
        for subspace_info in subspace_data[fps_str]:
            block_idx = subspace_info['block_idx']

            if block_idx not in backbone_subspaces:
                backbone_subspaces[block_idx] = extract_backbone_subspace(
                    pipeline, block_idx
                )
                logging.info(f"  Block {block_idx}: backbone rank = {backbone_subspaces[block_idx]['effective_rank']}")

    logging.info("Computing alignment angles...")

    # Compute alignment for each (condition, block) pair
    for fps_str in fps_strs:
        for subspace_info in subspace_data[fps_str]:
            block_idx = subspace_info['block_idx']

            fps_basis = subspace_info['basis']
            backbone_basis = backbone_subspaces[block_idx]['basis']

            # Compute principal angles
            angles = compute_principal_angles(fps_basis, backbone_basis)

            results.append({
                'fps': fps_str,
                'block_idx': block_idx,
                'min_angle': angles[0],
                'mean_angle': angles.mean(),
                'max_angle': angles[-1],
                'median_angle': np.median(angles),
                'alignment_score': np.cos(np.deg2rad(angles[0]))  # 1=aligned, 0=orthogonal
            })

    df = pd.DataFrame(results)

    # Print summary statistics
    logging.info("\nBackbone Alignment Summary:")
    logging.info(f"  Mean min angle: {df['min_angle'].mean():.1f}° ± {df['min_angle'].std():.1f}°")
    logging.info(f"  Mean alignment score: {df['alignment_score'].mean():.3f} ± {df['alignment_score'].std():.3f}")
    logging.info(f"  Range (min angle): [{df['min_angle'].min():.1f}°, {df['min_angle'].max():.1f}°]")

    # Interpretation
    mean_min_angle = df['min_angle'].mean()
    if mean_min_angle > 60:
        logging.info("  ✅ Interpretation: STRONG disentanglement")
    elif mean_min_angle > 30:
        logging.info("  ⚠️  Interpretation: MODERATE entanglement risk")
    else:
        logging.info("  ❌ Interpretation: HIGH entanglement risk")

    return df


def plot_backbone_alignment(df, output_dir):
    """Generate backbone alignment plots."""
    if df is None:
        return

    logging.info("  Generating backbone alignment plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fps_values = df['fps'].unique()

    # Plot 1: Min angle by block
    for fps in fps_values:
        subset = df[df['fps'] == fps]
        axes[0, 0].plot(subset['block_idx'], subset['min_angle'],
                       marker='o', label=f'FPS={fps}', alpha=0.7)
    axes[0, 0].set_xlabel('Block Index')
    axes[0, 0].set_ylabel('Min Principal Angle (°)')
    axes[0, 0].set_title('Backbone Alignment: Minimum Angles')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].axhline(60, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    axes[0, 0].axhline(30, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean angle by block
    for fps in fps_values:
        subset = df[df['fps'] == fps]
        axes[0, 1].plot(subset['block_idx'], subset['mean_angle'],
                       marker='o', label=f'FPS={fps}', alpha=0.7)
    axes[0, 1].set_xlabel('Block Index')
    axes[0, 1].set_ylabel('Mean Principal Angle (°)')
    axes[0, 1].set_title('Backbone Alignment: Mean Angles')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Alignment score distribution
    sns.boxplot(data=df, x='fps', y='alignment_score', ax=axes[1, 0])
    axes[1, 0].set_title('Alignment Score Distribution')
    axes[1, 0].set_ylabel('Alignment Score (0=orthogonal, 1=aligned)')
    axes[1, 0].axhline(0.5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 4: Angle range (min to max) per block
    for fps in fps_values:
        subset = df[df['fps'] == fps]
        axes[1, 1].fill_between(subset['block_idx'],
                                subset['min_angle'],
                                subset['max_angle'],
                                alpha=0.3, label=f'FPS={fps}')
    axes[1, 1].set_xlabel('Block Index')
    axes[1, 1].set_ylabel('Principal Angle Range (°)')
    axes[1, 1].set_title('Backbone Alignment: Angle Range')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'backbone_alignment.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved to {output_file}")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def generate_summary_report(dim_df, ortho_data, backbone_df, output_dir):
    """Generate text summary report."""
    logging.info("\nGenerating summary report...")

    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("CONDITIONAL SUBSPACE CHARACTERIZATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1: Effective Dimensionality
    report_lines.append("1. EFFECTIVE DIMENSIONALITY ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Mean effective rank: {dim_df['effective_rank'].mean():.1f} ± {dim_df['effective_rank'].std():.1f}")
    report_lines.append(f"Mean rank ratio: {dim_df['rank_ratio'].mean():.3f} ± {dim_df['rank_ratio'].std():.3f}")
    report_lines.append(f"SV entropy range: [{dim_df['sv_entropy'].min():.2f}, {dim_df['sv_entropy'].max():.2f}]")

    mean_ratio = dim_df['rank_ratio'].mean()
    if mean_ratio < 0.3:
        report_lines.append("✅ COMPACT representation (efficient conditioning)")
    elif mean_ratio < 0.6:
        report_lines.append("⚠️  MODERATE compactness")
    else:
        report_lines.append("❌ HIGH dimensionality (potentially redundant)")
    report_lines.append("")

    # Section 2: Inter-Condition Orthogonality
    report_lines.append("2. INTER-CONDITION ORTHOGONALITY ANALYSIS")
    report_lines.append("-" * 80)
    min_angles = ortho_data['min_angles']
    mask = np.ones_like(min_angles[:, :, 0], dtype=bool)
    np.fill_diagonal(mask, False)
    valid_angles = min_angles[mask].flatten()
    report_lines.append(f"Mean pairwise angle: {valid_angles.mean():.1f}° ± {valid_angles.std():.1f}°")
    report_lines.append(f"Min pairwise angle: {valid_angles.min():.1f}°")
    report_lines.append(f"Max pairwise angle: {valid_angles.max():.1f}°")

    mean_angle = valid_angles.mean()
    if mean_angle > 60:
        report_lines.append("✅ STRONG orthogonality (distinct effects)")
    elif mean_angle > 30:
        report_lines.append("⚠️  MODERATE orthogonality")
    else:
        report_lines.append("❌ WEAK orthogonality (potential interference)")
    report_lines.append("")

    # Section 3: Backbone Alignment
    if backbone_df is not None:
        report_lines.append("3. BACKBONE ALIGNMENT ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Mean minimum angle: {backbone_df['min_angle'].mean():.1f}° ± {backbone_df['min_angle'].std():.1f}°")
        report_lines.append(f"Mean alignment score: {backbone_df['alignment_score'].mean():.3f} ± {backbone_df['alignment_score'].std():.3f}")

        mean_min_angle = backbone_df['min_angle'].mean()
        if mean_min_angle > 60:
            report_lines.append("✅ STRONG disentanglement")
        elif mean_min_angle > 30:
            report_lines.append("⚠️  MODERATE entanglement risk")
        else:
            report_lines.append("❌ HIGH entanglement risk")
    else:
        report_lines.append("3. BACKBONE ALIGNMENT ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append("⚠️  Skipped (pipeline not provided)")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write to file
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, 'summary_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    logging.info(f"Summary report saved to {report_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Geometric analysis of conditional subspaces'
    )

    parser.add_argument('--subspace_file', required=True,
                       help='Path to subspace_data.pkl')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (defaults to subspace_file parent dir)')
    parser.add_argument('--config', default=None,
                       help='TOML config (required for backbone alignment)')
    parser.add_argument('--checkpoint', default=None,
                       help='Checkpoint path (required for backbone alignment)')
    parser.add_argument('--skip_backbone', action='store_true',
                       help='Skip backbone alignment analysis')
    parser.add_argument('--port', default='29503',
                       help='DeepSpeed port')

    args = parser.parse_args()

    try:
        # Setup
        setup_logging()

        # Determine output directory
        if args.output_dir is None:
            args.output_dir = str(Path(args.subspace_file).parent)

        os.makedirs(args.output_dir, exist_ok=True)

        # Load subspace data
        subspace_data, metadata = load_subspace_data(args.subspace_file)

        # Analysis 1: Dimensionality
        dim_df = analyze_dimensionality(subspace_data)
        plot_dimensionality_analysis(dim_df, args.output_dir)

        # Save dimensionality results
        dim_df.to_csv(os.path.join(args.output_dir, 'dimensionality_results.csv'), index=False)

        # Analysis 2: Orthogonality
        ortho_data = analyze_orthogonality(subspace_data)
        plot_orthogonality_analysis(ortho_data, args.output_dir)

        # Save orthogonality results
        np.savez(os.path.join(args.output_dir, 'orthogonality_results.npz'),
                min_angles=ortho_data['min_angles'],
                mean_angles=ortho_data['mean_angles'],
                max_angles=ortho_data['max_angles'],
                fps_strs=ortho_data['fps_strs'])

        # Analysis 3: Backbone alignment (optional)
        backbone_df = None
        if not args.skip_backbone and args.config and args.checkpoint:
            # Load pipeline for backbone analysis
            from inference.analyze_conditional_subspace import (
                setup_environment, load_pipeline_from_toml, load_checkpoint
            )

            setup_environment(args.port)
            pipeline, config = load_pipeline_from_toml(args.config)
            load_checkpoint(pipeline, args.checkpoint)

            backbone_df = analyze_backbone_alignment(subspace_data, pipeline)
            plot_backbone_alignment(backbone_df, args.output_dir)

            # Save backbone results
            if backbone_df is not None:
                backbone_df.to_csv(os.path.join(args.output_dir, 'backbone_alignment_results.csv'), index=False)

        # Generate summary report
        generate_summary_report(dim_df, ortho_data, backbone_df, args.output_dir)

        logging.info("\n✅ Geometric analysis completed successfully!")
        logging.info(f"Results saved to {args.output_dir}")

        return 0

    except Exception as e:
        logging.error(f"❌ Geometric analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
