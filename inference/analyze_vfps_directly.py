#!/usr/bin/env python3
"""
Direct analysis of v_fps(c) outputs for different FPS conditions.

This script answers the key question: Does the FPS adapter produce outputs
with the expected magnitude and directional properties?

Expected behavior (given conditions normalized to [-1, 1]):
1. v_fps(0) should have minimal magnitude (near zero impact)
2. v_fps(-1) and v_fps(+1) should have high magnitude
3. v_fps(-1) and v_fps(+1) should point in opposite directions
4. Intermediate values should have intermediate magnitudes
"""

import torch
import pickle
import numpy as np
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def analyze_vfps_magnitudes(subspace_data_path):
    """
    Analyze v_fps magnitudes across conditions.

    Expected pattern: V-shape (low at c=0, high at c=±1)
    """
    with open(subspace_data_path, 'rb') as f:
        data = pickle.load(f)

    fps_values = sorted([float(k) for k in data.keys()])
    fps_strs = [str(v) for v in fps_values]

    print('='*80)
    print('MAGNITUDE ANALYSIS: ||v_fps(c)||')
    print('='*80)
    print()
    print('Expected: V-shape pattern')
    print('  - v_fps(0) should have LOW magnitude (minimal impact)')
    print('  - v_fps(±1) should have HIGH magnitude (strong control)')
    print('  - v_fps(±0.5) should have MEDIUM magnitude')
    print()

    # Analyze each block
    for block_idx in [27, 30, 33, 36, 39]:
        print(f'\nBlock {block_idx}:')
        print(f'{"Condition":>10} {"||v_fps||":>12} {"Relative":>10} {"Expected":>10}')
        print('-' * 45)

        norms = []
        for fps_str in fps_strs:
            for s in data[fps_str]:
                if s['block_idx'] == block_idx:
                    v_fps = s['v_fps']  # [num_tokens, num_heads, head_dim]
                    norm = torch.norm(v_fps).item()
                    norms.append(norm)
                    break

        # Compute relative to maximum
        max_norm = max(norms)

        for fps_val, norm in zip(fps_values, norms):
            relative = norm / max_norm

            # Expected relative magnitude (V-shape)
            expected_rel = abs(fps_val)  # Simple V-shape: |c|

            # Determine if matches expectation
            if abs(relative - expected_rel) < 0.2:
                status = '✓'
            else:
                status = '✗'

            print(f'{fps_val:>10.1f} {norm:>12.6f} {relative:>9.3f} {expected_rel:>9.3f} {status}')

    print()
    print('Legend:')
    print('  ||v_fps||  = Frobenius norm of adapter output')
    print('  Relative   = Normalized to max (1.0 = strongest)')
    print('  Expected   = |c| for V-shape pattern')
    print('  ✓/✗        = Match/mismatch with V-shape expectation')


def analyze_vfps_directions(subspace_data_path):
    """
    Analyze v_fps directional relationships.

    Expected patterns:
    1. v_fps(-1) and v_fps(+1) should be opposite (cos ≈ -1)
    2. v_fps(0) should be different from both extremes
    3. Smooth transition in directions
    """
    with open(subspace_data_path, 'rb') as f:
        data = pickle.load(f)

    fps_values = sorted([float(k) for k in data.keys()])
    fps_strs = [str(v) for v in fps_values]

    print('='*80)
    print('DIRECTION ANALYSIS: cosine similarity between v_fps(c1) and v_fps(c2)')
    print('='*80)
    print()
    print('Expected patterns:')
    print('  - v_fps(-1) ⊥ v_fps(+1): cos ≈ -1 (opposite directions)')
    print('  - v_fps(0) different from v_fps(±1)')
    print('  - Smooth transition: cos increases monotonically')
    print()

    def cosine_similarity(v1, v2):
        """Compute cosine similarity between two tensors."""
        v1_flat = v1.flatten()
        v2_flat = v2.flatten()
        return torch.dot(v1_flat, v2_flat) / (torch.norm(v1_flat) * torch.norm(v2_flat))

    # Analyze representative blocks
    for block_idx in [27, 33, 39]:
        print(f'\n{"="*80}')
        print(f'Block {block_idx}: Cosine Similarity Matrix')
        print(f'{"="*80}')

        # Extract v_fps for all conditions
        v_fps_dict = {}
        for fps_str in fps_strs:
            for s in data[fps_str]:
                if s['block_idx'] == block_idx:
                    v_fps_dict[fps_str] = s['v_fps']
                    break

        # Print header
        print(f'{"":>8}', end='')
        for fps_val in fps_values:
            print(f'{fps_val:>8.1f}', end='')
        print()
        print('-' * (8 + 8 * len(fps_values)))

        # Print similarity matrix
        for fps1_str, fps1_val in zip(fps_strs, fps_values):
            print(f'{fps1_val:>7.1f} ', end='')
            for fps2_str in fps_strs:
                cos_sim = cosine_similarity(v_fps_dict[fps1_str], v_fps_dict[fps2_str]).item()
                print(f'{cos_sim:>8.3f}', end='')
            print()

        # Key checks
        print()
        print('Key Checks:')
        cos_neg1_pos1 = cosine_similarity(v_fps_dict['-1.0'], v_fps_dict['1.0']).item()
        cos_neg1_zero = cosine_similarity(v_fps_dict['-1.0'], v_fps_dict['0.0']).item()
        cos_pos1_zero = cosine_similarity(v_fps_dict['1.0'], v_fps_dict['0.0']).item()

        print(f'  cos(v_fps(-1), v_fps(+1)) = {cos_neg1_pos1:>7.3f}  ', end='')
        if cos_neg1_pos1 < -0.8:
            print('✓ Opposite directions (expected!)')
        elif cos_neg1_pos1 > 0.8:
            print('✗ Same direction (unexpected!)')
        else:
            print('~ Partial opposition')

        print(f'  cos(v_fps(-1), v_fps(0))  = {cos_neg1_zero:>7.3f}')
        print(f'  cos(v_fps(+1), v_fps(0))  = {cos_pos1_zero:>7.3f}')


def analyze_vfps_combined(subspace_data_path):
    """
    Combined analysis showing both magnitude and direction.
    """
    with open(subspace_data_path, 'rb') as f:
        data = pickle.load(f)

    fps_values = sorted([float(k) for k in data.keys()])
    fps_strs = [str(v) for v in fps_values]

    print('='*80)
    print('COMBINED ANALYSIS: Magnitude + Direction')
    print('='*80)
    print()

    for block_idx in [27, 33, 39]:
        print(f'\nBlock {block_idx}:')
        print(f'{"Condition":>10} {"||v_fps||":>12} {"Relative":>10} {"cos(-1,c)":>10} {"cos(+1,c)":>10}')
        print('-' * 55)

        # Extract all v_fps
        v_fps_dict = {}
        norms = []
        for fps_str in fps_strs:
            for s in data[fps_str]:
                if s['block_idx'] == block_idx:
                    v_fps = s['v_fps']
                    v_fps_dict[fps_str] = v_fps
                    norms.append(torch.norm(v_fps).item())
                    break

        max_norm = max(norms)

        def cos_sim(v1, v2):
            v1_flat = v1.flatten()
            v2_flat = v2.flatten()
            return torch.dot(v1_flat, v2_flat) / (torch.norm(v1_flat) * torch.norm(v2_flat))

        for fps_str, fps_val, norm in zip(fps_strs, fps_values, norms):
            relative = norm / max_norm

            # Cosine with extremes
            cos_neg1 = cos_sim(v_fps_dict[fps_str], v_fps_dict['-1.0']).item()
            cos_pos1 = cos_sim(v_fps_dict[fps_str], v_fps_dict['1.0']).item()

            print(f'{fps_val:>10.1f} {norm:>12.6f} {relative:>9.3f} {cos_neg1:>9.3f} {cos_pos1:>9.3f}')

        print()
        print('Expected pattern:')
        print('  - Relative magnitude: 1.0 at c=±1, low at c=0 (V-shape)')
        print('  - cos(-1,c): +1 at c=-1, negative at c=+1 (smooth transition)')
        print('  - cos(+1,c): negative at c=-1, +1 at c=+1 (smooth transition)')


def main():
    parser = argparse.ArgumentParser(
        description='Direct analysis of v_fps(c) outputs'
    )
    parser.add_argument(
        '--subspace_file',
        type=str,
        required=True,
        help='Path to subspace_data.pkl'
    )
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['magnitude', 'direction', 'combined', 'all'],
        default='all',
        help='Which analysis to run'
    )

    args = parser.parse_args()

    logging.info(f'Loading subspace data from {args.subspace_file}')

    if args.analysis in ['magnitude', 'all']:
        analyze_vfps_magnitudes(args.subspace_file)
        print()

    if args.analysis in ['direction', 'all']:
        analyze_vfps_directions(args.subspace_file)
        print()

    if args.analysis in ['combined', 'all']:
        analyze_vfps_combined(args.subspace_file)
        print()

    print('='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)


if __name__ == '__main__':
    main()
