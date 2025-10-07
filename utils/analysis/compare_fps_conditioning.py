#!/usr/bin/env python3
"""
Compare FPS conditioning strength between two checkpoints.
Analyzes FPS MLP output variation and FPS adapter strengths.
"""

import torch
import safetensors.torch
from pathlib import Path
import sys

def analyze_fps_mlp_strength(checkpoint_path):
    """Analyze how much the FPS MLP output varies across different FPS values."""
    ckpt = safetensors.torch.load_file(checkpoint_path)

    # Extract FPS MLP weights (handle both with and without diffusion_model prefix)
    prefix = 'diffusion_model.' if 'diffusion_model.fps_conditioning.lin1.weight' in ckpt else ''
    lin1_weight = ckpt[f'{prefix}fps_conditioning.lin1.weight']
    lin1_bias = ckpt[f'{prefix}fps_conditioning.lin1.bias']
    lin2_weight = ckpt[f'{prefix}fps_conditioning.lin2.weight']
    lin2_bias = ckpt[f'{prefix}fps_conditioning.lin2.bias']

    print(f"\n{'='*80}")
    print(f"FPS MLP Analysis: {checkpoint_path.parent.name}")
    print(f"{'='*80}")
    print(f"Lin1 weight: {lin1_weight.shape}, norm={torch.norm(lin1_weight).item():.4f}")
    print(f"Lin1 bias: {lin1_bias.shape}, norm={torch.norm(lin1_bias).item():.4f}")
    print(f"Lin2 weight: {lin2_weight.shape}, norm={torch.norm(lin2_weight).item():.4f}")
    print(f"Lin2 bias: {lin2_bias.shape}, norm={torch.norm(lin2_bias).item():.4f}")

    # Test FPS MLP with different tau values (convert to bfloat16 to match weights)
    test_taus = torch.tensor([
        [-2.0],  # Very low FPS (relative to reference)
        [-1.0],  # Low FPS
        [0.0],   # Reference FPS
        [1.0],   # High FPS
        [2.0]    # Very high FPS
    ], dtype=lin1_weight.dtype)

    # Forward pass through FPS MLP: lin2(silu(lin1(tau)))
    h = torch.nn.functional.silu(test_taus @ lin1_weight.t() + lin1_bias)
    fps_embeds = h @ lin2_weight.t() + lin2_bias

    # Analyze output variation
    print(f"\nFPS MLP Output for different tau values:")
    print(f"  tau=-2.0: mean={fps_embeds[0].mean().item():.6f}, std={fps_embeds[0].std().item():.6f}, norm={torch.norm(fps_embeds[0]).item():.4f}")
    print(f"  tau=-1.0: mean={fps_embeds[1].mean().item():.6f}, std={fps_embeds[1].std().item():.6f}, norm={torch.norm(fps_embeds[1]).item():.4f}")
    print(f"  tau= 0.0: mean={fps_embeds[2].mean().item():.6f}, std={fps_embeds[2].std().item():.6f}, norm={torch.norm(fps_embeds[2]).item():.4f}")
    print(f"  tau= 1.0: mean={fps_embeds[3].mean().item():.6f}, std={fps_embeds[3].std().item():.6f}, norm={torch.norm(fps_embeds[3]).item():.4f}")
    print(f"  tau= 2.0: mean={fps_embeds[4].mean().item():.6f}, std={fps_embeds[4].std().item():.6f}, norm={torch.norm(fps_embeds[4]).item():.4f}")

    # Compute variation metrics
    output_norms = torch.tensor([torch.norm(fps_embeds[i]).item() for i in range(5)])
    norm_range = output_norms.max().item() - output_norms.min().item()
    norm_std = output_norms.std().item()

    print(f"\nFPS MLP Output Variation Metrics:")
    print(f"  Norm range (max-min): {norm_range:.4f}")
    print(f"  Norm std: {norm_std:.4f}")
    print(f"  Norm mean: {output_norms.mean().item():.4f}")

    # Larger variation means stronger FPS control
    print(f"\n  => FPS Sensitivity Score: {norm_std:.4f} (higher = more responsive to FPS changes)")

    return {
        'norm_range': norm_range,
        'norm_std': norm_std,
        'norm_mean': output_norms.mean().item(),
        'fps_embeds': fps_embeds,
        'output_norms': output_norms
    }


def analyze_fps_adapter_strength(checkpoint_path):
    """Analyze FPS adapter LoRA strengths."""
    ckpt = safetensors.torch.load_file(checkpoint_path)

    print(f"\n{'='*80}")
    print(f"FPS Adapter Analysis: {checkpoint_path.parent.name}")
    print(f"{'='*80}")

    # Find all FPS adapter parameters (strip diffusion_model prefix if present)
    fps_adapter_params = {}
    for k, v in ckpt.items():
        if 'fps_adapter' in k:
            # Strip prefix for consistent naming
            clean_key = k.replace('diffusion_model.', '')
            fps_adapter_params[clean_key] = v

    print(f"Found {len(fps_adapter_params)} FPS adapter parameters")

    # Group by block
    blocks = {}
    for name, param in fps_adapter_params.items():
        # Extract block number from name like "blocks.27.fps_adapter.k_fps_down.weight"
        # or "single_blocks.37.fps_adapter.to_k_fps.lora_A.weight"
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part in ['single_blocks', 'double_blocks', 'blocks']:
                block_type = part
                try:
                    block_num = int(parts[i+1])
                    block_key = f"{block_type}.{block_num}"
                    if block_key not in blocks:
                        blocks[block_key] = {}
                    blocks[block_key][name] = param
                    break
                except (IndexError, ValueError):
                    continue

    print(f"FPS adapters in {len(blocks)} blocks")

    # Analyze each block's adapter strength
    block_strengths = {}
    for block_key in sorted(blocks.keys()):
        block_params = blocks[block_key]

        # Compute total norm of all FPS adapter params in this block
        total_norm = sum(torch.norm(param).item()**2 for param in block_params.values())**0.5
        block_strengths[block_key] = total_norm

    # Print top 10 strongest blocks
    print(f"\nTop 10 Strongest FPS Adapter Blocks:")
    sorted_blocks = sorted(block_strengths.items(), key=lambda x: x[1], reverse=True)
    for i, (block_key, strength) in enumerate(sorted_blocks[:10], 1):
        print(f"  {i:2d}. {block_key:20s}: norm={strength:.4f}")

    # Overall statistics
    if len(block_strengths) > 0:
        strengths_tensor = torch.tensor(list(block_strengths.values()))
        print(f"\nFPS Adapter Overall Statistics:")
        print(f"  Mean block strength: {strengths_tensor.mean().item():.4f}")
        if len(block_strengths) > 1:
            print(f"  Std block strength: {strengths_tensor.std().item():.4f}")
        print(f"  Min block strength: {strengths_tensor.min().item():.4f}")
        print(f"  Max block strength: {strengths_tensor.max().item():.4f}")
        mean_strength = strengths_tensor.mean().item()
        std_strength = strengths_tensor.std().item() if len(block_strengths) > 1 else 0.0
    else:
        print(f"\nWarning: No FPS adapter blocks found!")
        mean_strength = 0.0
        std_strength = 0.0

    return {
        'block_strengths': block_strengths,
        'mean_strength': mean_strength,
        'std_strength': std_strength
    }


def compare_checkpoints(ckpt1_path, ckpt2_path):
    """Compare FPS conditioning strength between two checkpoints."""
    print(f"\n{'#'*80}")
    print(f"# Comparing FPS Conditioning Strength")
    print(f"#   Checkpoint 1: {ckpt1_path.parent.name}")
    print(f"#   Checkpoint 2: {ckpt2_path.parent.name}")
    print(f"{'#'*80}")

    # Analyze FPS MLP
    mlp1 = analyze_fps_mlp_strength(ckpt1_path)
    mlp2 = analyze_fps_mlp_strength(ckpt2_path)

    print(f"\n{'='*80}")
    print(f"FPS MLP Comparison")
    print(f"{'='*80}")
    print(f"Checkpoint 1 ({ckpt1_path.parent.name}):")
    print(f"  FPS Sensitivity Score: {mlp1['norm_std']:.4f}")
    print(f"Checkpoint 2 ({ckpt2_path.parent.name}):")
    print(f"  FPS Sensitivity Score: {mlp2['norm_std']:.4f}")
    print(f"\nDifference: {abs(mlp1['norm_std'] - mlp2['norm_std']):.4f}")
    if mlp1['norm_std'] > mlp2['norm_std']:
        print(f"  => Checkpoint 1 has {(mlp1['norm_std']/mlp2['norm_std']-1)*100:.1f}% STRONGER FPS control")
    else:
        print(f"  => Checkpoint 2 has {(mlp2['norm_std']/mlp1['norm_std']-1)*100:.1f}% STRONGER FPS control")

    # Analyze FPS adapters
    adapter1 = analyze_fps_adapter_strength(ckpt1_path)
    adapter2 = analyze_fps_adapter_strength(ckpt2_path)

    print(f"\n{'='*80}")
    print(f"FPS Adapter Comparison")
    print(f"{'='*80}")
    print(f"Checkpoint 1 ({ckpt1_path.parent.name}):")
    print(f"  Mean adapter strength: {adapter1['mean_strength']:.4f}")
    print(f"Checkpoint 2 ({ckpt2_path.parent.name}):")
    print(f"  Mean adapter strength: {adapter2['mean_strength']:.4f}")
    print(f"\nDifference: {abs(adapter1['mean_strength'] - adapter2['mean_strength']):.4f}")
    if adapter1['mean_strength'] > adapter2['mean_strength']:
        print(f"  => Checkpoint 1 has {(adapter1['mean_strength']/adapter2['mean_strength']-1)*100:.1f}% STRONGER adapters")
    else:
        print(f"  => Checkpoint 2 has {(adapter2['mean_strength']/adapter1['mean_strength']-1)*100:.1f}% STRONGER adapters")

    # Overall conclusion
    print(f"\n{'#'*80}")
    print(f"# OVERALL CONCLUSION")
    print(f"{'#'*80}")

    ckpt1_better = 0
    ckpt2_better = 0

    if mlp1['norm_std'] > mlp2['norm_std']:
        print(f"✓ Checkpoint 1 has stronger FPS MLP (better FPS sensitivity)")
        ckpt1_better += 1
    else:
        print(f"✓ Checkpoint 2 has stronger FPS MLP (better FPS sensitivity)")
        ckpt2_better += 1

    if adapter1['mean_strength'] > adapter2['mean_strength']:
        print(f"✓ Checkpoint 1 has stronger FPS adapters")
        ckpt1_better += 1
    else:
        print(f"✓ Checkpoint 2 has stronger FPS adapters")
        ckpt2_better += 1

    print(f"\n{'='*80}")
    if ckpt1_better > ckpt2_better:
        print(f"=> Checkpoint 1 ({ckpt1_path.parent.name}) has OVERALL STRONGER FPS conditioning")
        print(f"   This likely explains why it produces better inference results.")
    elif ckpt2_better > ckpt1_better:
        print(f"=> Checkpoint 2 ({ckpt2_path.parent.name}) has OVERALL STRONGER FPS conditioning")
        print(f"   This likely explains why it produces better inference results.")
    else:
        print(f"=> Both checkpoints have similar FPS conditioning strength")
        print(f"   Quality difference may be due to other factors (initialization, training dynamics)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    ckpt1 = Path("/root/workspace/sc-diffusion-pipe/checkpoints/20250928_22-18-11/epoch200/adapter_model.safetensors")
    ckpt2 = Path("/root/workspace/sc-diffusion-pipe/checkpoints/20251003_23-07-22/epoch200/adapter_model.safetensors")

    if not ckpt1.exists():
        print(f"Error: Checkpoint 1 not found: {ckpt1}")
        sys.exit(1)
    if not ckpt2.exists():
        print(f"Error: Checkpoint 2 not found: {ckpt2}")
        sys.exit(1)

    compare_checkpoints(ckpt1, ckpt2)
