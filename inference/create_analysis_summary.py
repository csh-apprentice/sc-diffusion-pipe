#!/usr/bin/env python3
"""
Create a single-page visual summary of subspace analysis results.

Usage:
    python inference/create_analysis_summary.py \
        --analysis_dir output/subspace_analysis \
        --output output/subspace_analysis/summary_visualization.png
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )


def create_summary_visualization(analysis_dir, output_file):
    """
    Create a single-page summary combining all analysis results.

    Args:
        analysis_dir: Directory containing analysis results
        output_file: Output path for summary image
    """
    logging.info(f"Creating summary visualization from {analysis_dir}")

    # Load data
    dim_df = pd.read_csv(os.path.join(analysis_dir, 'dimensionality_results.csv'))
    ortho_npz = np.load(os.path.join(analysis_dir, 'orthogonality_results.npz'))
    min_angles = ortho_npz['min_angles']
    fps_strs = ortho_npz['fps_strs']

    # Check if backbone alignment exists
    backbone_file = os.path.join(analysis_dir, 'backbone_alignment_results.csv')
    if os.path.exists(backbone_file):
        backbone_df = pd.read_csv(backbone_file)
        has_backbone = True
    else:
        has_backbone = False
        logging.info("No backbone alignment data found, skipping that section")

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Conditional Subspace Characterization Summary', fontsize=20, fontweight='bold')

    # === ROW 1: DIMENSIONALITY ===

    # 1.1: Effective rank by block
    ax1 = fig.add_subplot(gs[0, 0])
    for fps in dim_df['fps'].unique():
        subset = dim_df[dim_df['fps'] == fps]
        ax1.plot(subset['block_idx'], subset['effective_rank'],
                marker='o', label=f'{fps[:10]}', alpha=0.7)
    ax1.set_xlabel('Block Index')
    ax1.set_ylabel('Effective Rank')
    ax1.set_title('Effective Rank Across Blocks')
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3)

    # 1.2: Rank ratio histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(dim_df['rank_ratio'], bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(0.3, color='green', linestyle='--', label='Compact', alpha=0.7)
    ax2.axvline(0.6, color='orange', linestyle='--', label='Moderate', alpha=0.7)
    ax2.set_xlabel('Rank Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Rank Ratio Distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 1.3: SV entropy
    ax3 = fig.add_subplot(gs[0, 2])
    for fps in dim_df['fps'].unique():
        subset = dim_df[dim_df['fps'] == fps]
        ax3.plot(subset['block_idx'], subset['sv_entropy'],
                marker='o', label=f'{fps[:10]}', alpha=0.7)
    ax3.set_xlabel('Block Index')
    ax3.set_ylabel('SV Entropy')
    ax3.set_title('Singular Value Entropy')
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)

    # 1.4: Summary stats text
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    mean_ratio = dim_df['rank_ratio'].mean()
    std_ratio = dim_df['rank_ratio'].std()
    mean_rank = dim_df['effective_rank'].mean()

    if mean_ratio < 0.3:
        status = "✅ COMPACT"
        color = 'green'
    elif mean_ratio < 0.6:
        status = "⚠️ MODERATE"
        color = 'orange'
    else:
        status = "❌ REDUNDANT"
        color = 'red'

    summary_text = f"""
DIMENSIONALITY ANALYSIS

Mean Rank Ratio:
  {mean_ratio:.3f} ± {std_ratio:.3f}

Mean Effective Rank:
  {mean_rank:.1f}

Status: {status}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    # === ROW 2: ORTHOGONALITY ===

    # 2.1: Heatmap (last block)
    ax5 = fig.add_subplot(gs[1, 0])
    last_block = min_angles.shape[2] - 1
    im = ax5.imshow(min_angles[:, :, last_block], cmap='RdYlGn', vmin=0, vmax=90)
    ax5.set_xticks(range(len(fps_strs)))
    ax5.set_yticks(range(len(fps_strs)))
    ax5.set_xticklabels([f'{fps[:8]}' for fps in fps_strs], rotation=45, ha='right', fontsize=8)
    ax5.set_yticklabels([f'{fps[:8]}' for fps in fps_strs], fontsize=8)
    ax5.set_title(f'Inter-Condition Angles (Block {last_block})')
    plt.colorbar(im, ax=ax5, label='Angle (°)')

    # 2.2: Evolution across blocks
    ax6 = fig.add_subplot(gs[1, 1:3])
    n_fps = len(fps_strs)
    for i in range(n_fps):
        for j in range(i+1, n_fps):
            angles = min_angles[i, j, :]
            label = f'{fps_strs[i][:6]} vs {fps_strs[j][:6]}'
            ax6.plot(angles, marker='o', label=label, alpha=0.7)
    ax6.set_xlabel('Block Index')
    ax6.set_ylabel('Min Principal Angle (°)')
    ax6.set_title('Inter-Condition Orthogonality Evolution')
    ax6.axhline(60, color='green', linestyle='--', alpha=0.5)
    ax6.axhline(30, color='orange', linestyle='--', alpha=0.5)
    ax6.legend(fontsize=6, ncol=2)
    ax6.grid(True, alpha=0.3)

    # 2.3: Summary stats text
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')

    mask = np.ones_like(min_angles[:, :, 0], dtype=bool)
    np.fill_diagonal(mask, False)
    valid_angles = min_angles[mask].flatten()
    mean_angle = valid_angles.mean()
    std_angle = valid_angles.std()
    min_angle = valid_angles.min()

    if mean_angle > 60:
        status = "✅ STRONG"
        color = 'green'
    elif mean_angle > 30:
        status = "⚠️ MODERATE"
        color = 'orange'
    else:
        status = "❌ WEAK"
        color = 'red'

    ortho_text = f"""
ORTHOGONALITY ANALYSIS

Mean Pairwise Angle:
  {mean_angle:.1f}° ± {std_angle:.1f}°

Min Pairwise Angle:
  {min_angle:.1f}°

Status: {status}
    """
    ax7.text(0.1, 0.5, ortho_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    # === ROW 3: BACKBONE ALIGNMENT ===

    if has_backbone:
        # 3.1: Min angle by block
        ax8 = fig.add_subplot(gs[2, 0])
        for fps in backbone_df['fps'].unique():
            subset = backbone_df[backbone_df['fps'] == fps]
            ax8.plot(subset['block_idx'], subset['min_angle'],
                    marker='o', label=f'{fps[:10]}', alpha=0.7)
        ax8.set_xlabel('Block Index')
        ax8.set_ylabel('Min Angle (°)')
        ax8.set_title('Backbone Alignment: Min Angles')
        ax8.axhline(60, color='green', linestyle='--', alpha=0.5)
        ax8.axhline(30, color='red', linestyle='--', alpha=0.5)
        ax8.legend(fontsize=6)
        ax8.grid(True, alpha=0.3)

        # 3.2: Alignment score distribution
        ax9 = fig.add_subplot(gs[2, 1])
        ax9.hist(backbone_df['alignment_score'], bins=20, alpha=0.7, edgecolor='black')
        ax9.axvline(0.5, color='orange', linestyle='--', alpha=0.7)
        ax9.set_xlabel('Alignment Score')
        ax9.set_ylabel('Count')
        ax9.set_title('Alignment Score Distribution')
        ax9.grid(True, alpha=0.3)

        # 3.3: Mean angle evolution
        ax10 = fig.add_subplot(gs[2, 2])
        for fps in backbone_df['fps'].unique():
            subset = backbone_df[backbone_df['fps'] == fps]
            ax10.plot(subset['block_idx'], subset['mean_angle'],
                    marker='o', label=f'{fps[:10]}', alpha=0.7)
        ax10.set_xlabel('Block Index')
        ax10.set_ylabel('Mean Angle (°)')
        ax10.set_title('Backbone Alignment: Mean Angles')
        ax10.legend(fontsize=6)
        ax10.grid(True, alpha=0.3)

        # 3.4: Summary stats text
        ax11 = fig.add_subplot(gs[2, 3])
        ax11.axis('off')

        mean_min_angle = backbone_df['min_angle'].mean()
        std_min_angle = backbone_df['min_angle'].std()
        mean_align = backbone_df['alignment_score'].mean()

        if mean_min_angle > 60:
            status = "✅ STRONG DISENTANGLEMENT"
            color = 'green'
        elif mean_min_angle > 30:
            status = "⚠️ MODERATE RISK"
            color = 'orange'
        else:
            status = "❌ HIGH RISK"
            color = 'red'

        backbone_text = f"""
BACKBONE ALIGNMENT

Mean Min Angle:
  {mean_min_angle:.1f}° ± {std_min_angle:.1f}°

Mean Alignment Score:
  {mean_align:.3f}

Status: {status}
        """
        ax11.text(0.1, 0.5, backbone_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    else:
        # Show placeholder for backbone analysis
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')
        ax8.text(0.5, 0.5, 'Backbone Alignment Analysis: Not Available\n\n'
                          'Run with --config and --checkpoint to enable backbone analysis',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.2))

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ Summary visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create visual summary of subspace analysis'
    )
    parser.add_argument('--analysis_dir', required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--output', default=None,
                       help='Output file path (default: analysis_dir/summary_visualization.png)')

    args = parser.parse_args()

    setup_logging()

    if args.output is None:
        args.output = os.path.join(args.analysis_dir, 'summary_visualization.png')

    create_summary_visualization(args.analysis_dir, args.output)

    logging.info("✅ Done!")


if __name__ == "__main__":
    main()
