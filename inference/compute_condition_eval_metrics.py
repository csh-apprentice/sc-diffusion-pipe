#!/usr/bin/env python3
"""
Data-Free Evaluation Metrics for Joint LoRA & Adapter Finetuning

This script computes quantitative, data-free metrics to find the "sweet point" checkpoint
where the conditional adapter is fully trained AND the backbone is not corrupted.

Metrics Computed:
1. CPD (Content Primitive Drift) Score: Measures backbone health (should stay ~1.0)
2. Conditional Disparity Score: Measures adapter learning (should rise to ~1.0)
3. Supporting metrics: Intruder Count, Effective Rank, Magnitude Ratio

Usage:
  python compute_condition_eval_metrics.py \\
    --config path/to/config.toml \\
    --checkpoint path/to/checkpoint \\
    --output_dir output/eval_metrics

The "Sweet Point" is where Disparity Score plateaus AND CPD Score is still high.
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
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wan.wan import WanModel, WanPipeline
from models.wan.model import compute_tau_rel
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


def extract_principal_components(weight, num_components):
    """
    Extract top-N principal components from a weight matrix using SVD.

    Args:
        weight: Weight matrix [d_out, d_in]
        num_components: Number of principal components to extract

    Returns:
        principal_components: Top-N principal vectors [d_in, num_components]
    """
    U, S, V = torch.svd(weight.float())
    # V contains right singular vectors (in input space)
    return V[:, :num_components]


def load_pretrained_weights(model, blocks):
    """Extract pretrained weights BEFORE LoRA is applied."""
    logging.info("Extracting pretrained weights...")

    pretrained_weights = {}
    for block_idx in blocks:
        block = model.blocks[block_idx]
        cross_attn = block.cross_attn

        pretrained_weights[(block_idx, 'q')] = cross_attn.q.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'k')] = cross_attn.k.weight.data.cpu().clone()
        pretrained_weights[(block_idx, 'v')] = cross_attn.v.weight.data.cpu().clone()

    logging.info(f"  Extracted weights from {len(blocks)} blocks")
    return pretrained_weights


def apply_lora_to_weights(pretrained_weights, lora_checkpoint, blocks):
    """Apply LoRA deltas to get adapted weights."""
    logging.info("Applying LoRA to weights...")

    adapted_weights = {}
    for block_idx in blocks:
        for weight_name in ['q', 'k', 'v']:
            lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
            lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

            W_pre = pretrained_weights[(block_idx, weight_name)]

            if lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint:
                lora_A = lora_checkpoint[lora_a_key].cpu().float()
                lora_B = lora_checkpoint[lora_b_key].cpu().float()
                delta_W = lora_B @ lora_A
                adapted_weights[(block_idx, weight_name)] = W_pre + delta_W
            else:
                # No LoRA, use pretrained
                adapted_weights[(block_idx, weight_name)] = W_pre.clone()

    logging.info("  ✅ LoRA applied")
    return adapted_weights


def compute_attention_output(q_prin, k_prin, v_prin, cross_attn, device, dtype):
    """
    Simulate attention output using principal components.

    Args:
        q_prin, k_prin, v_prin: Principal component matrices [d_model, N]
        cross_attn: CrossAttention module
        device, dtype: Device and dtype for computation

    Returns:
        y_text: Attention output [N, d_model]
    """
    from models.wan.attention import flash_attention

    N = q_prin.shape[1]
    num_heads = cross_attn.num_heads
    head_dim = cross_attn.head_dim

    # Transpose to [N, d_model]
    q_test = q_prin.T.to(device, dtype=dtype)
    k_test = k_prin.T.to(device, dtype=dtype)
    v_test = v_prin.T.to(device, dtype=dtype)

    with torch.no_grad():
        # Project through attention layers
        q = cross_attn.norm_q(cross_attn.q(q_test))
        k = cross_attn.norm_k(cross_attn.k(k_test))
        v = cross_attn.v(v_test)

        # Reshape for multi-head attention
        q = q.view(N, -1, num_heads, head_dim)
        k = k.view(N, -1, num_heads, head_dim)
        v = v.view(N, -1, num_heads, head_dim)

        # Compute attention
        y_text = flash_attention(q, k, v, k_lens=None)
        y_text = y_text.reshape(N, -1)

    return y_text.cpu().float()


def compute_cpd_score(pretrained_weights, adapted_weights, model, block_idx, num_components, device):
    """
    Compute Content Primitive Drift (CPD) Score for a single block.

    Measures how much the backbone has drifted from pretrained.
    Score ~1.0 = good (no drift), ~0.0 = bad (catastrophic forgetting)

    Args:
        pretrained_weights: Dict of pretrained weights
        adapted_weights: Dict of adapted weights (with LoRA)
        model: WanModel (for access to modules)
        block_idx: Block index
        num_components: Number of principal components
        device: Device

    Returns:
        cpd_score: Cosine similarity between pretrained and adapted outputs
    """
    # Extract principal components from pretrained weights
    q_pre_prin = extract_principal_components(pretrained_weights[(block_idx, 'q')], num_components)
    k_pre_prin = extract_principal_components(pretrained_weights[(block_idx, 'k')], num_components)
    v_pre_prin = extract_principal_components(pretrained_weights[(block_idx, 'v')], num_components)

    # Extract principal components from adapted weights
    q_ada_prin = extract_principal_components(adapted_weights[(block_idx, 'q')], num_components)
    k_ada_prin = extract_principal_components(adapted_weights[(block_idx, 'k')], num_components)
    v_ada_prin = extract_principal_components(adapted_weights[(block_idx, 'v')], num_components)

    # Get cross attention module
    block = model.blocks[block_idx]
    cross_attn = block.cross_attn

    # Compute y_text for pretrained (temporarily swap weights)
    original_q = cross_attn.q.weight.data.clone()
    original_k = cross_attn.k.weight.data.clone()
    original_v = cross_attn.v.weight.data.clone()

    # Use pretrained weights
    cross_attn.q.weight.data = pretrained_weights[(block_idx, 'q')].to(device, dtype=model.dtype)
    cross_attn.k.weight.data = pretrained_weights[(block_idx, 'k')].to(device, dtype=model.dtype)
    cross_attn.v.weight.data = pretrained_weights[(block_idx, 'v')].to(device, dtype=model.dtype)

    y_text_pretrained = compute_attention_output(
        q_pre_prin, k_pre_prin, v_pre_prin, cross_attn, device, model.dtype
    )

    # Use adapted weights
    cross_attn.q.weight.data = adapted_weights[(block_idx, 'q')].to(device, dtype=model.dtype)
    cross_attn.k.weight.data = adapted_weights[(block_idx, 'k')].to(device, dtype=model.dtype)
    cross_attn.v.weight.data = adapted_weights[(block_idx, 'v')].to(device, dtype=model.dtype)

    y_text_adapted = compute_attention_output(
        q_ada_prin, k_ada_prin, v_ada_prin, cross_attn, device, model.dtype
    )

    # Restore original weights
    cross_attn.q.weight.data = original_q
    cross_attn.k.weight.data = original_k
    cross_attn.v.weight.data = original_v

    # Compute cosine similarity
    y_pre_flat = y_text_pretrained.flatten()
    y_ada_flat = y_text_adapted.flatten()

    cpd_score = torch.nn.functional.cosine_similarity(
        y_pre_flat.unsqueeze(0), y_ada_flat.unsqueeze(0), dim=1
    ).item()

    return cpd_score, y_text_pretrained, y_text_adapted


def compute_disparity_score(model, block_idx, num_components, low_cond, high_cond, config, device):
    """
    Compute Conditional Disparity Score for a single block.

    Measures if the adapter has learned to distinguish between conditions.
    Score ~1.0 = good (distinct outputs), ~0.0 = bad (collapsed/untrained)

    Args:
        model: WanModel with FPS adapter
        block_idx: Block index
        num_components: Number of principal components for test queries
        low_cond: Low condition value (e.g., sharp)
        high_cond: High condition value (e.g., bokeh)
        config: Model config dict
        device: Device

    Returns:
        disparity_score: 1.0 - cosine_similarity(y_sharp, y_bokeh)
        y_cond_sharp, y_cond_bokeh: Adapter outputs
    """
    from models.wan.attention import flash_attention

    block = model.blocks[block_idx]
    cross_attn = block.cross_attn
    fps_adapter = block.fps_adapter

    if fps_adapter is None:
        logging.warning(f"  Block {block_idx} has no FPS adapter, returning 0.0")
        return 0.0, None, None

    # Extract principal components from adapted q weights for test queries
    q_weight = cross_attn.q.weight.data.cpu()
    q_prin = extract_principal_components(q_weight, num_components)
    q_test = q_prin.T.to(device, dtype=model.dtype)  # [N, d_model]

    N = q_test.shape[0]
    num_heads = cross_attn.num_heads
    head_dim = cross_attn.head_dim

    with torch.no_grad():
        # Project queries
        q = cross_attn.norm_q(cross_attn.q(q_test))
        q = q.view(N, -1, num_heads, head_dim)

        # === Compute y_cond_sharp (low condition) ===
        # Apply tau transform as specified in config
        fps_tau_transform = config['model'].get('fps_tau_transform', 'log1p')
        fps_tau_scale = config['model'].get('fps_tau_scale', 0.33333334)
        fps_reference_fps = config['model'].get('fps_reference_fps', 240.0)

        cond_sharp = compute_tau_rel(
            low_cond,
            reference_fps=fps_reference_fps,
            transform=fps_tau_transform,
            scale=fps_tau_scale
        )

        fps_tensor_sharp = torch.full((N, 1), cond_sharp, dtype=model.dtype, device=device)
        fps_embed_sharp = model.fps_conditioning(fps_tensor_sharp)

        # FPS adapter forward
        k_fps_proj_sharp = fps_adapter.k_fps_up(fps_adapter.k_fps_down(fps_embed_sharp))
        v_fps_proj_sharp = fps_adapter.v_fps_up(fps_adapter.v_fps_down(fps_embed_sharp))

        # Normalize and scale
        B = N
        k_fps_proj_sharp = k_fps_proj_sharp.view(B * fps_adapter.num_tokens, fps_adapter.dim)
        k_fps_proj_sharp = fps_adapter.norm_k_fps(k_fps_proj_sharp)
        k_fps_proj_sharp = k_fps_proj_sharp.view(B, fps_adapter.num_tokens * fps_adapter.dim)
        k_fps_proj_sharp = k_fps_proj_sharp * fps_adapter.lora_scale
        v_fps_proj_sharp = v_fps_proj_sharp * fps_adapter.lora_scale

        k_fps_sharp = k_fps_proj_sharp.view(B, fps_adapter.num_tokens, num_heads, head_dim)
        v_fps_sharp = v_fps_proj_sharp.view(B, fps_adapter.num_tokens, num_heads, head_dim)

        y_cond_sharp = flash_attention(q, k_fps_sharp, v_fps_sharp, k_lens=None)
        y_cond_sharp = y_cond_sharp.reshape(N, -1)

        # === Compute y_cond_bokeh (high condition) ===
        cond_bokeh = compute_tau_rel(
            high_cond,
            reference_fps=fps_reference_fps,
            transform=fps_tau_transform,
            scale=fps_tau_scale
        )

        fps_tensor_bokeh = torch.full((N, 1), cond_bokeh, dtype=model.dtype, device=device)
        fps_embed_bokeh = model.fps_conditioning(fps_tensor_bokeh)

        k_fps_proj_bokeh = fps_adapter.k_fps_up(fps_adapter.k_fps_down(fps_embed_bokeh))
        v_fps_proj_bokeh = fps_adapter.v_fps_up(fps_adapter.v_fps_down(fps_embed_bokeh))

        k_fps_proj_bokeh = k_fps_proj_bokeh.view(B * fps_adapter.num_tokens, fps_adapter.dim)
        k_fps_proj_bokeh = fps_adapter.norm_k_fps(k_fps_proj_bokeh)
        k_fps_proj_bokeh = k_fps_proj_bokeh.view(B, fps_adapter.num_tokens * fps_adapter.dim)
        k_fps_proj_bokeh = k_fps_proj_bokeh * fps_adapter.lora_scale
        v_fps_proj_bokeh = v_fps_proj_bokeh * fps_adapter.lora_scale

        k_fps_bokeh = k_fps_proj_bokeh.view(B, fps_adapter.num_tokens, num_heads, head_dim)
        v_fps_bokeh = v_fps_proj_bokeh.view(B, fps_adapter.num_tokens, num_heads, head_dim)

        y_cond_bokeh = flash_attention(q, k_fps_bokeh, v_fps_bokeh, k_lens=None)
        y_cond_bokeh = y_cond_bokeh.reshape(N, -1)

    # Compute disparity score
    y_sharp_flat = y_cond_sharp.cpu().float().flatten()
    y_bokeh_flat = y_cond_bokeh.cpu().float().flatten()

    cosine_sim = torch.nn.functional.cosine_similarity(
        y_sharp_flat.unsqueeze(0), y_bokeh_flat.unsqueeze(0), dim=1
    ).item()

    disparity_score = 1.0 - cosine_sim

    return disparity_score, y_cond_sharp.cpu().float(), y_cond_bokeh.cpu().float()


def compute_supporting_metrics(y_text_pretrained, y_text_adapted, y_cond_sharp, y_cond_bokeh):
    """
    Compute supporting metrics for additional analysis.

    Returns:
        metrics: Dict with intruder_count, effective_rank, magnitude_ratio
    """
    metrics = {}

    # 1. Intruder Count (simplified version)
    # Count how many adapted singular vectors have low similarity to pretrained
    U_pre, _, _ = torch.svd(y_text_pretrained)
    U_ada, _, _ = torch.svd(y_text_adapted)

    similarities = []
    for i in range(min(64, U_ada.shape[1])):
        u_ada_i = U_ada[:, i]
        max_sim = torch.max(torch.abs(U_pre.T @ u_ada_i)).item()
        similarities.append(max_sim)

    # Count vectors with similarity < 0.9 as "intruders"
    intruder_count = sum(1 for sim in similarities if sim < 0.9)
    metrics['intruder_count'] = intruder_count
    metrics['min_similarity'] = min(similarities) if similarities else 1.0

    # 2. Effective Rank of y_cond
    if y_cond_sharp is not None and y_cond_bokeh is not None:
        _, S_sharp, _ = torch.svd(y_cond_sharp)
        _, S_bokeh, _ = torch.svd(y_cond_bokeh)

        # Normalize by largest singular value
        S_sharp_norm = S_sharp / S_sharp[0]
        S_bokeh_norm = S_bokeh / S_bokeh[0]

        # Count effective rank (values > 0.01)
        eff_rank_sharp = (S_sharp_norm > 0.01).sum().item()
        eff_rank_bokeh = (S_bokeh_norm > 0.01).sum().item()

        metrics['effective_rank_sharp'] = eff_rank_sharp
        metrics['effective_rank_bokeh'] = eff_rank_bokeh
        metrics['effective_rank_avg'] = (eff_rank_sharp + eff_rank_bokeh) / 2.0

        # 3. Magnitude Ratio
        mag_text = torch.norm(y_text_adapted).item()
        mag_cond_sharp = torch.norm(y_cond_sharp).item()
        mag_cond_bokeh = torch.norm(y_cond_bokeh).item()
        mag_cond_avg = (mag_cond_sharp + mag_cond_bokeh) / 2.0

        metrics['magnitude_ratio'] = mag_cond_avg / mag_text if mag_text > 0 else 0.0
        metrics['mag_text'] = mag_text
        metrics['mag_cond_avg'] = mag_cond_avg
    else:
        metrics['effective_rank_sharp'] = 0
        metrics['effective_rank_bokeh'] = 0
        metrics['effective_rank_avg'] = 0
        metrics['magnitude_ratio'] = 0.0
        metrics['mag_text'] = 0.0
        metrics['mag_cond_avg'] = 0.0

    return metrics


def load_model_and_checkpoint(config_path, checkpoint_path, blocks, device):
    """Load model and checkpoint."""
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
    pretrained_weights = load_pretrained_weights(wan_pipeline.transformer, blocks)

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

    # Apply LoRA to get adapted weights
    adapted_weights = apply_lora_to_weights(pretrained_weights, lora_checkpoint, blocks)

    # Also apply LoRA to the model itself for FPS adapter usage
    apply_lora_to_model(wan_pipeline.transformer, lora_checkpoint, blocks)

    return pretrained_weights, adapted_weights, wan_pipeline.transformer, config


def apply_lora_to_model(model, lora_checkpoint, blocks):
    """Apply LoRA deltas to model weights in-place."""
    for block_idx in blocks:
        block = model.blocks[block_idx]
        cross_attn = block.cross_attn

        for weight_name, weight_module in [('q', cross_attn.q), ('k', cross_attn.k), ('v', cross_attn.v)]:
            lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
            lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

            if lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint:
                lora_A = lora_checkpoint[lora_a_key].to(weight_module.weight.device, dtype=weight_module.weight.dtype)
                lora_B = lora_checkpoint[lora_b_key].to(weight_module.weight.device, dtype=weight_module.weight.dtype)
                delta_W = lora_B @ lora_A
                weight_module.weight.data += delta_W


def analyze_block(pretrained_weights, adapted_weights, model, block_idx, num_components,
                  low_cond, high_cond, config, device):
    """Analyze a single block."""
    logging.info(f"\n{'='*80}")
    logging.info(f"ANALYZING BLOCK {block_idx}")
    logging.info(f"{'='*80}")

    # 1. Compute CPD Score
    logging.info("\n1. Computing CPD (Content Primitive Drift) Score...")
    cpd_score, y_text_pre, y_text_ada = compute_cpd_score(
        pretrained_weights, adapted_weights, model, block_idx, num_components, device
    )
    logging.info(f"   CPD Score: {cpd_score:.6f} {'✓ GOOD' if cpd_score > 0.95 else '⚠ WARNING' if cpd_score > 0.8 else '✗ BAD'}")

    # 2. Compute Disparity Score
    logging.info("\n2. Computing Conditional Disparity Score...")
    disparity_score, y_cond_sharp, y_cond_bokeh = compute_disparity_score(
        model, block_idx, num_components, low_cond, high_cond, config, device
    )
    logging.info(f"   Disparity Score: {disparity_score:.6f} {'✓ GOOD' if disparity_score > 0.5 else '⚠ LEARNING' if disparity_score > 0.1 else '✗ UNTRAINED'}")

    # 3. Compute Supporting Metrics
    logging.info("\n3. Computing Supporting Metrics...")
    supporting_metrics = compute_supporting_metrics(y_text_pre, y_text_ada, y_cond_sharp, y_cond_bokeh)

    logging.info(f"   Intruder Count: {supporting_metrics['intruder_count']}")
    logging.info(f"   Effective Rank (avg): {supporting_metrics['effective_rank_avg']:.2f}")
    logging.info(f"   Magnitude Ratio: {supporting_metrics['magnitude_ratio']:.6f}")

    return {
        'block_idx': block_idx,
        'cpd_score': cpd_score,
        'disparity_score': disparity_score,
        **supporting_metrics
    }


def plot_results(results, blocks, output_dir):
    """Plot evaluation metrics."""
    logging.info("\nGenerating plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metrics
    cpd_scores = [r['cpd_score'] for r in results]
    disparity_scores = [r['disparity_score'] for r in results]
    intruder_counts = [r['intruder_count'] for r in results]
    eff_ranks = [r['effective_rank_avg'] for r in results]
    mag_ratios = [r['magnitude_ratio'] for r in results]

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data-Free Evaluation Metrics for Finding "Sweet Point"',
                 fontsize=14, fontweight='bold')

    # Plot 1: CPD Score (Backbone Health)
    ax1 = axes[0, 0]
    ax1.plot(blocks, cpd_scores, marker='o', linewidth=2, markersize=6, color='#2E86AB', label='CPD Score')
    ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good threshold')
    ax1.axhline(y=0.8, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning threshold')
    ax1.set_xlabel('Block Index', fontweight='bold')
    ax1.set_ylabel('CPD Score', fontweight='bold')
    ax1.set_title('Content Primitive Drift (Backbone Health)\nShould stay HIGH (~1.0)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1.05])

    # Plot 2: Disparity Score (Adapter Learning)
    ax2 = axes[0, 1]
    ax2.plot(blocks, disparity_scores, marker='s', linewidth=2, markersize=6, color='#F18F01', label='Disparity Score')
    ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good threshold')
    ax2.set_xlabel('Block Index', fontweight='bold')
    ax2.set_ylabel('Disparity Score', fontweight='bold')
    ax2.set_title('Conditional Disparity (Adapter Learning)\nShould RISE and plateau', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.05])

    # Plot 3: Intruder Count
    ax3 = axes[1, 0]
    ax3.bar(blocks, intruder_counts, color='#A23B72', alpha=0.7)
    ax3.set_xlabel('Block Index', fontweight='bold')
    ax3.set_ylabel('Intruder Count', fontweight='bold')
    ax3.set_title('Intruder Count (Destructive Forgetting)\nShould stay LOW', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Effective Rank & Magnitude Ratio
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(blocks, eff_ranks, marker='D', linewidth=2, markersize=6,
                     color='#C73E1D', label='Effective Rank')
    line2 = ax4_twin.plot(blocks, mag_ratios, marker='^', linewidth=2, markersize=6,
                          color='#06A77D', label='Magnitude Ratio')

    ax4.set_xlabel('Block Index', fontweight='bold')
    ax4.set_ylabel('Effective Rank', fontweight='bold', color='#C73E1D')
    ax4_twin.set_ylabel('Magnitude Ratio', fontweight='bold', color='#06A77D')
    ax4.set_title('Supporting Metrics\n(Eff. Rank should be LOW, Mag. Ratio should be SMALL)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='#C73E1D')
    ax4_twin.tick_params(axis='y', labelcolor='#06A77D')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    output_file = output_dir / 'evaluation_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"✅ Saved plot to: {output_file}")

    plt.close()


def save_results(results, blocks, output_dir):
    """Save numerical results."""
    output_dir = Path(output_dir)

    # Save CSV
    csv_file = output_dir / 'evaluation_metrics.csv'
    with open(csv_file, 'w') as f:
        f.write('block_idx,cpd_score,disparity_score,intruder_count,effective_rank_avg,magnitude_ratio\n')
        for r in results:
            f.write(f"{r['block_idx']},{r['cpd_score']:.6f},{r['disparity_score']:.6f},"
                   f"{r['intruder_count']},{r['effective_rank_avg']:.2f},{r['magnitude_ratio']:.6f}\n")

    logging.info(f"✅ Saved CSV to: {csv_file}")

    # Save numpy arrays
    npz_file = output_dir / 'evaluation_metrics.npz'
    np.savez(npz_file,
             blocks=np.array(blocks),
             cpd_scores=np.array([r['cpd_score'] for r in results]),
             disparity_scores=np.array([r['disparity_score'] for r in results]),
             intruder_counts=np.array([r['intruder_count'] for r in results]),
             effective_ranks=np.array([r['effective_rank_avg'] for r in results]),
             magnitude_ratios=np.array([r['magnitude_ratio'] for r in results]))

    logging.info(f"✅ Saved numpy arrays to: {npz_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute data-free evaluation metrics for joint LoRA & adapter finetuning'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--blocks', type=int, nargs='+', default=None,
                       help='Block indices to analyze (default: auto-detect with FPS adapter)')
    parser.add_argument('--num_components', type=int, default=64,
                       help='Number of principal components (default: 64)')
    parser.add_argument('--low_cond', type=float, default=0.025,
                       help='Low condition value for disparity (default: 0.025)')
    parser.add_argument('--high_cond', type=float, default=1.4,
                       help='High condition value for disparity (default: 1.4)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    setup_logging()

    logging.info("="*80)
    logging.info("DATA-FREE EVALUATION METRICS")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Condition range: [{args.low_cond}, {args.high_cond}]")
    logging.info("="*80)

    # Initialize distributed environment
    init_distributed_stub()

    # Auto-detect blocks if not specified
    if args.blocks is None:
        logging.info("\nAuto-detecting blocks with FPS adapter...")
        import safetensors.torch
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_dir():
            adapter_file = checkpoint_path / 'adapter_model.safetensors'
            lora_checkpoint = safetensors.torch.load_file(str(adapter_file))
        else:
            lora_checkpoint = safetensors.torch.load_file(str(checkpoint_path))

        # Extract blocks with FPS adapter
        blocks_set = set()
        for key in lora_checkpoint.keys():
            if 'fps_adapter' in key:
                parts = key.split('.')
                block_idx = int(parts[parts.index('blocks') + 1])
                blocks_set.add(block_idx)

        blocks = sorted(list(blocks_set))
        logging.info(f"  Found FPS adapter in {len(blocks)} blocks: {blocks}")
    else:
        blocks = sorted(args.blocks)
        logging.info(f"\nUsing specified blocks: {blocks}")

    # Load model and weights
    pretrained_weights, adapted_weights, model, config = load_model_and_checkpoint(
        args.config, args.checkpoint, blocks, args.device
    )

    # Analyze all blocks
    logging.info("\n" + "="*80)
    logging.info("ANALYZING BLOCKS")
    logging.info("="*80)

    results = []
    for block_idx in blocks:
        result = analyze_block(
            pretrained_weights, adapted_weights, model, block_idx,
            args.num_components, args.low_cond, args.high_cond, config, args.device
        )
        results.append(result)

    # Generate visualizations
    logging.info("\n" + "="*80)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("="*80)

    plot_results(results, blocks, args.output_dir)

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    save_results(results, blocks, args.output_dir)

    # Final summary
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {args.output_dir}")

    # Print summary statistics
    avg_cpd = np.mean([r['cpd_score'] for r in results])
    avg_disparity = np.mean([r['disparity_score'] for r in results])

    logging.info(f"\nSummary Statistics:")
    logging.info(f"  Average CPD Score: {avg_cpd:.6f} {'✓' if avg_cpd > 0.95 else '⚠' if avg_cpd > 0.8 else '✗'}")
    logging.info(f"  Average Disparity Score: {avg_disparity:.6f} {'✓' if avg_disparity > 0.5 else '⚠' if avg_disparity > 0.1 else '✗'}")

    if avg_cpd > 0.95 and avg_disparity > 0.5:
        logging.info(f"\n🎉 SWEET POINT ACHIEVED! Backbone healthy + Adapter trained")
    elif avg_cpd > 0.95:
        logging.info(f"\n⚠ Backbone healthy but adapter needs more training")
    elif avg_disparity > 0.5:
        logging.info(f"\n⚠ Adapter trained but backbone shows drift")
    else:
        logging.info(f"\n⚠ Continue training or adjust hyperparameters")

    logging.info("="*80)


if __name__ == "__main__":
    main()
