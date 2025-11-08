#!/usr/bin/env python3
"""
Visualize Score Progression Across Epochs

This script plots how different metrics (SSF, SS-FD, DVS) change across training epochs.
Supports comparing multiple checkpoint directories and score types.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import re


def extract_epoch_number(epoch_folder_name):
    """
    Extract epoch number from folder name like 'epoch100' -> 100
    """
    match = re.search(r'epoch(\d+)', epoch_folder_name)
    if match:
        return int(match.group(1))
    return None


def load_scores_from_checkpoint_dir(checkpoint_dir, score_types):
    """
    Load scores from all epoch subdirectories in a checkpoint directory.

    Args:
        checkpoint_dir: Path to directory containing epoch subdirectories
        score_types: List of score types to extract (e.g., ['ssf', 'ss_fd', 'dvs'])

    Returns:
        dict mapping epoch_num -> {score_type: value}
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Find all epoch subdirectories
    epoch_dirs = sorted([d for d in checkpoint_path.iterdir() if d.is_dir() and 'epoch' in d.name])

    if not epoch_dirs:
        raise ValueError(f"No epoch subdirectories found in: {checkpoint_dir}")

    scores_by_epoch = {}

    for epoch_dir in epoch_dirs:
        epoch_num = extract_epoch_number(epoch_dir.name)
        if epoch_num is None:
            print(f"Warning: Could not extract epoch number from {epoch_dir.name}, skipping")
            continue

        # Look for scores.json in this epoch directory
        score_file = epoch_dir / 'scores.json'
        if not score_file.exists():
            print(f"Warning: No scores.json found in {epoch_dir.name}, skipping")
            continue

        # Load scores
        with open(score_file, 'r') as f:
            data = json.load(f)

        # Extract total scores for requested types
        if 'total' not in data:
            print(f"Warning: No 'total' scores in {score_file}, skipping")
            continue

        epoch_scores = {}
        for score_type in score_types:
            if score_type in data['total']:
                epoch_scores[score_type] = data['total'][score_type]
            else:
                print(f"Warning: Score type '{score_type}' not found in {score_file}")

        if epoch_scores:
            scores_by_epoch[epoch_num] = epoch_scores

    if not scores_by_epoch:
        raise ValueError(f"No valid scores found in any epoch subdirectory of {checkpoint_dir}")

    return scores_by_epoch


def load_baseline_scores(baseline_path, score_types):
    """
    Load baseline scores from a single scores.json file.

    Args:
        baseline_path: Path to directory containing scores.json
        score_types: List of score types to extract

    Returns:
        dict mapping score_type -> value
    """
    baseline_dir = Path(baseline_path)
    score_file = baseline_dir / 'scores.json'

    if not score_file.exists():
        raise ValueError(f"Baseline scores.json not found at: {score_file}")

    with open(score_file, 'r') as f:
        data = json.load(f)

    if 'total' not in data:
        raise ValueError(f"No 'total' scores in baseline file: {score_file}")

    baseline_scores = {}
    for score_type in score_types:
        if score_type in data['total']:
            baseline_scores[score_type] = data['total'][score_type]
        else:
            print(f"Warning: Score type '{score_type}' not found in baseline file")

    return baseline_scores


def plot_score_progression(checkpoint_dirs, labels, score_types, title_prefix="", baseline_path=None, output_dir="plots", figsize=(10, 6)):
    """
    Plot score progression across epochs for multiple checkpoints.

    Args:
        checkpoint_dirs: List of checkpoint directories (each containing epoch subdirs)
        labels: List of labels for each checkpoint directory
        score_types: List of score types to plot (e.g., ['ssf', 'ss_fd', 'dvs'])
        title_prefix: Prefix for plot titles (e.g., "One Shot Results")
        baseline_path: Optional path to baseline scores.json for reference line
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    if len(checkpoint_dirs) != len(labels):
        raise ValueError(f"Number of checkpoint dirs ({len(checkpoint_dirs)}) must match number of labels ({len(labels)})")

    # Load scores from all checkpoint directories
    all_scores = []
    for checkpoint_dir in checkpoint_dirs:
        scores = load_scores_from_checkpoint_dir(checkpoint_dir, score_types)
        all_scores.append(scores)

    # Load baseline scores if provided
    baseline_scores = None
    if baseline_path:
        baseline_scores = load_baseline_scores(baseline_path, score_types)
        print(f"📊 Loaded baseline scores from: {baseline_path}")
        for score_type in score_types:
            if score_type in baseline_scores:
                print(f"  {score_type}: {baseline_scores[score_type]:.6f}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create one plot for each score type
    for score_type in score_types:
        plt.figure(figsize=figsize)

        # Plot baseline reference line if available
        if baseline_scores and score_type in baseline_scores:
            baseline_value = baseline_scores[score_type]
            # Get epoch range for baseline line
            all_epochs = []
            for scores_by_epoch in all_scores:
                all_epochs.extend(scores_by_epoch.keys())
            if all_epochs:
                epoch_min = min(all_epochs)
                epoch_max = max(all_epochs)
                plt.axhline(y=baseline_value, color='red', linestyle='--', linewidth=2,
                           label='Clean Backbone Baseline', alpha=0.7, zorder=1)

        # Plot each checkpoint's scores
        for checkpoint_idx, (scores_by_epoch, label) in enumerate(zip(all_scores, labels)):
            # Extract epochs and scores for this checkpoint
            epochs = sorted(scores_by_epoch.keys())
            scores = [scores_by_epoch[epoch].get(score_type, np.nan) for epoch in epochs]

            # Filter out any NaN values
            valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
            valid_epochs = [epochs[i] for i in valid_indices]
            valid_scores = [scores[i] for i in valid_indices]

            if not valid_scores:
                print(f"Warning: No valid scores for {label} - {score_type}")
                continue

            # Add epoch 0 at the beginning
            # SSF starts at 1.0 (perfect similarity to itself)
            # SS-FD and DVS start at 0.0 (no distance/same variance)
            if score_type == 'ssf':
                epoch0_score = 1.0
            else:  # ss_fd, dvs
                epoch0_score = 0.0

            valid_epochs = [0] + valid_epochs
            valid_scores = [epoch0_score] + valid_scores

            # Plot with markers and lines (zorder=2 to appear above baseline)
            plt.plot(valid_epochs, valid_scores, marker='o', linewidth=2, markersize=6, label=label, zorder=2)

        # Formatting
        if title_prefix:
            plt.title(f"{title_prefix} of {score_type.upper()}", fontsize=14, fontweight='bold')
        else:
            plt.title(f"Score Progression: {score_type.upper()}", fontsize=14, fontweight='bold')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'{score_type.upper()} Score', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)

        # Add score interpretation hints
        if score_type == 'ssf':
            plt.text(0.02, 0.98, 'Higher is better (1.0 is perfect)',
                    transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        elif score_type == 'ss_fd':
            plt.text(0.02, 0.98, 'Lower is better (0.0 is perfect)',
                    transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        elif score_type == 'dvs':
            plt.text(0.02, 0.98, '1.0 is same variance\n>1.0 more spread, <1.0 more concentrated',
                    transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        # Save plot
        plot_filename = f"{score_type}_progression.png"
        if title_prefix:
            # Sanitize title prefix for filename
            sanitized_prefix = re.sub(r'[^\w\s-]', '', title_prefix).strip().replace(' ', '_')
            plot_filename = f"{sanitized_prefix}_{score_type}_progression.png"

        plot_path = output_path / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize score progression across training epochs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single checkpoint directory with baseline reference
  python visualize_score_progression.py \\
    --checkpoint_dirs scores/20251031_08-25-18 \\
    --labels "FPS Adapter Training" \\
    --score_types ssf ss_fd dvs \\
    --title_prefix "One Shot Results" \\
    --baseline scores/clean_category_99 \\
    --output_dir plots/with_baseline

  # Compare multiple checkpoint directories with baseline
  python visualize_score_progression.py \\
    --checkpoint_dirs scores/run1 scores/run2 scores/run3 \\
    --labels "Run 1" "Run 2" "Run 3" \\
    --score_types ssf ss_fd \\
    --title_prefix "Comparison Study" \\
    --baseline scores/clean_baseline \\
    --output_dir plots/comparison

  # Without baseline (just training progression)
  python visualize_score_progression.py \\
    --checkpoint_dirs scores/20251031_08-25-18 \\
    --labels "Training Run" \\
    --score_types ssf ss_fd dvs \\
    --output_dir plots/no_baseline
        """
    )

    parser.add_argument(
        '--checkpoint_dirs',
        nargs='+',
        required=True,
        help='List of checkpoint directories (each containing epoch subdirectories with scores.json)'
    )

    parser.add_argument(
        '--labels',
        nargs='+',
        required=True,
        help='Labels for each checkpoint directory (must match number of checkpoint_dirs)'
    )

    parser.add_argument(
        '--score_types',
        nargs='+',
        required=True,
        choices=['ssf', 'ss_fd', 'dvs'],
        help='Score types to plot (ssf, ss_fd, dvs)'
    )

    parser.add_argument(
        '--title_prefix',
        type=str,
        default='',
        help='Prefix for plot titles (e.g., "One Shot Results")'
    )

    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='Path to baseline scores directory (containing scores.json) for reference line'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots)'
    )

    parser.add_argument(
        '--figsize',
        nargs=2,
        type=float,
        default=[10, 6],
        help='Figure size as width height (default: 10 6)'
    )

    args = parser.parse_args()

    # Validate arguments
    if len(args.checkpoint_dirs) != len(args.labels):
        parser.error(f"Number of checkpoint_dirs ({len(args.checkpoint_dirs)}) must match number of labels ({len(args.labels)})")

    print("="*80)
    print("VISUALIZING SCORE PROGRESSION")
    print("="*80)
    print(f"Checkpoint directories: {len(args.checkpoint_dirs)}")
    for checkpoint_dir, label in zip(args.checkpoint_dirs, args.labels):
        print(f"  - {label}: {checkpoint_dir}")
    print(f"Score types: {', '.join(args.score_types)}")
    if args.title_prefix:
        print(f"Title prefix: {args.title_prefix}")
    if args.baseline:
        print(f"Baseline: {args.baseline}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    try:
        plot_score_progression(
            checkpoint_dirs=args.checkpoint_dirs,
            labels=args.labels,
            score_types=args.score_types,
            title_prefix=args.title_prefix,
            baseline_path=args.baseline,
            output_dir=args.output_dir,
            figsize=tuple(args.figsize)
        )

        print("\n" + "="*80)
        print(f"✅ Successfully created {len(args.score_types)} plot(s)")
        print(f"📁 Plots saved to: {args.output_dir}")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
