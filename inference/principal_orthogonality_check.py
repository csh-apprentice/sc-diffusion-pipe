#!/usr/bin/env python3
"""
Principal Orthogonality Check for FPS Cross-Attention Adapter

This script performs a data-free analysis to verify functional orthogonality
between the FPS adapter and the pretrained backbone model. It uses principal
components from the pretrained model's attention weights as "content primitives"
and measures how orthogonally the adapter's output relates to the backbone's output.

Methodology:
1. Extract principal components (SVD) from pretrained W_q, W_k, W_v
2. Simulate attention with these principal components:
   - y_text_principal = attention(q_test, k_text_test, v_text_test)
   - y_fps_principal = attention(q_test, k_fps(c), v_fps(c))
3. Measure similarity using Centered Kernel Alignment (CKA)

Expected Result:
- CKA score near 0: Good functional orthogonality (adapter independent of content)
- CKA score near 1: Feature collision (risk of context drift)
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wan.wan import WanModel
from utils.common import DTYPE_MAP


def setup_logging():
    """Configure logging for the analysis."""
    logging.basicConfig(
        level=logging.INFO,  # INFO level for clean output
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress matplotlib debug spam
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


# REMOVED: No longer loading separate pretrained model
# We extract principal components from the trained model's backbone weights,
# which are the same as pretrained (since we only trained adapters with LoRA)


def load_trained_model(config_path, checkpoint_path, device='cuda', use_base_lora=True):
    """
    Load the trained model (with LoRA + FPS adapter).

    Args:
        config_path: Path to training config TOML
        checkpoint_path: Path to trained checkpoint
        device: Device to load model on
        use_base_lora: If True, merge base LoRA into backbone weights

    Returns:
        trained_model: The trained WanModel with adapters
        lora_checkpoint: Dict of LoRA weights (if use_base_lora=True), else None
    """
    logging.info(f"Loading trained model from config: {config_path}")

    import json
    from models.wan.wan import WanPipeline

    # Load TOML exactly like train.py does
    with open(config_path) as f:
        config = json.loads(json.dumps(toml.load(f)))

    # Convert dtype strings to torch.dtype objects
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    # Initialize pipeline with TOML configuration
    logging.info("Initializing WanPipeline with trained configuration...")
    wan_pipeline = WanPipeline(config)

    # Load the diffusion model (transformer)
    logging.info("Loading diffusion model...")
    wan_pipeline.load_diffusion_model()

    # Load trained checkpoint with adapters
    import safetensors.torch
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        # It's a directory, look for adapter_model.safetensors
        adapter_file = checkpoint_path / 'adapter_model.safetensors'
        if not adapter_file.exists():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {checkpoint_path}")
        logging.info(f"Loading trained checkpoint: {adapter_file}")
        checkpoint = safetensors.torch.load_file(str(adapter_file))
    else:
        # It's a file
        logging.info(f"Loading trained checkpoint: {checkpoint_path}")
        if str(checkpoint_path).endswith('.safetensors'):
            checkpoint = safetensors.torch.load_file(str(checkpoint_path))
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load FPS adapter and FPS conditioning weights (always needed)
    wan_pipeline.transformer.load_state_dict(checkpoint, strict=False)

    # Move to device
    logging.info(f"Moving model to device: {device}")
    wan_pipeline.transformer.to(device)
    wan_pipeline.transformer.eval()

    if use_base_lora:
        logging.info("✅ Trained model loaded successfully (base LoRA will be merged during analysis)")
        return wan_pipeline.transformer, checkpoint
    else:
        logging.info("✅ Trained model loaded successfully (using pretrained backbone weights)")
        return wan_pipeline.transformer, None


def extract_principal_components(model, block_idx, num_components=64, lora_checkpoint=None):
    """
    Extract principal components from attention projection matrices via SVD.

    Args:
        model: WanModel instance
        block_idx: Index of transformer block
        num_components: Number of principal components to extract
        lora_checkpoint: Optional dict of LoRA weights to merge into backbone

    Returns:
        q_principal: Top-N principal query vectors [N, d_model]
        k_principal: Top-N principal key vectors [N, d_model]
        v_principal: Top-N principal value vectors [N, d_model]
    """
    if lora_checkpoint is not None:
        logging.info(f"Extracting {num_components} principal components from block {block_idx} (with base LoRA merged)...")
    else:
        logging.info(f"Extracting {num_components} principal components from block {block_idx} (pretrained weights only)...")

    block = model.blocks[block_idx]
    cross_attn = block.cross_attn

    # Get projection weight matrices from cross attention
    W_q = cross_attn.q.weight.data.cpu().float()
    W_k = cross_attn.k.weight.data.cpu().float()
    W_v = cross_attn.v.weight.data.cpu().float()

    # Merge LoRA if provided
    if lora_checkpoint is not None:
        for weight_name, W in [('q', W_q), ('k', W_k), ('v', W_v)]:
            lora_a_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_A.weight'
            lora_b_key = f'diffusion_model.blocks.{block_idx}.cross_attn.{weight_name}.lora_B.weight'

            if lora_a_key in lora_checkpoint and lora_b_key in lora_checkpoint:
                lora_A = lora_checkpoint[lora_a_key].cpu().float()
                lora_B = lora_checkpoint[lora_b_key].cpu().float()
                delta_W = lora_B @ lora_A

                if weight_name == 'q':
                    W_q = W_q + delta_W
                elif weight_name == 'k':
                    W_k = W_k + delta_W
                elif weight_name == 'v':
                    W_v = W_v + delta_W

        logging.info(f"  Base LoRA merged into backbone weights")

    # Perform SVD to get principal directions (already in float32)
    with torch.no_grad():
        U_q, S_q, _ = torch.svd(W_q.T)
        U_k, S_k, _ = torch.svd(W_k.T)
        U_v, S_v, _ = torch.svd(W_v.T)

    # Take top N components
    q_principal = U_q[:, :num_components].T  # [N, d_model]
    k_principal = U_k[:, :num_components].T
    v_principal = U_v[:, :num_components].T

    # Report explained variance
    total_var_q = (S_q ** 2).sum()
    explained_var_q = (S_q[:num_components] ** 2).sum() / total_var_q
    logging.info(f"  Explained variance: {explained_var_q:.2%}")

    return q_principal, k_principal, v_principal


def simulate_attention_outputs(model, block_idx, q_test, k_text_test, v_text_test,
                               fps_condition=1.0, device='cuda'):
    """
    Simulate attention outputs using principal components.

    Args:
        model: Trained WanModel with FPS adapter
        block_idx: Block index
        q_test: Test query vectors [N, d_model]
        k_text_test: Test key vectors [N, d_model]
        v_text_test: Test value vectors [N, d_model]
        fps_condition: FPS conditioning value
        device: Device

    Returns:
        y_text_principal: Text attention output [N, d_model]
        y_fps_principal: FPS attention output [N, d_model]
    """
    from models.wan.attention import flash_attention

    logging.info(f"Simulating attention outputs...")

    block = model.blocks[block_idx]
    cross_attn = block.cross_attn
    fps_adapter = block.fps_adapter

    if fps_adapter is None:
        raise ValueError(f"Block {block_idx} does not have FPS adapter!")

    with torch.no_grad():
        N = q_test.shape[0]
        num_heads = cross_attn.num_heads
        head_dim = cross_attn.head_dim

        # Project and normalize inputs
        q = cross_attn.norm_q(cross_attn.q(q_test))
        k_text = cross_attn.norm_k(cross_attn.k(k_text_test))
        v_text = cross_attn.v(v_text_test)

        # Reshape for multi-head attention
        q = q.view(N, -1, num_heads, head_dim)
        k_text = k_text.view(N, -1, num_heads, head_dim)
        v_text = v_text.view(N, -1, num_heads, head_dim)

        # 1. Compute y_text_principal (backbone attention)
        y_text_principal = flash_attention(q, k_text, v_text, k_lens=None)

        # 2. Compute y_fps_principal (FPS adapter attention)
        fps_tensor = torch.full((N, 1), fps_condition, dtype=model.dtype, device=device)
        fps_embed = model.fps_conditioning(fps_tensor)

        # FPS adapter forward
        k_fps_proj = fps_adapter.k_fps_up(fps_adapter.k_fps_down(fps_embed))
        v_fps_proj = fps_adapter.v_fps_up(fps_adapter.v_fps_down(fps_embed))

        # Normalize and reshape
        B = N
        k_fps_proj = k_fps_proj.view(B * fps_adapter.num_tokens, fps_adapter.dim)
        k_fps_proj = fps_adapter.norm_k_fps(k_fps_proj)
        k_fps_proj = k_fps_proj.view(B, fps_adapter.num_tokens * fps_adapter.dim)
        k_fps_proj = k_fps_proj * fps_adapter.lora_scale
        v_fps_proj = v_fps_proj * fps_adapter.lora_scale

        k_fps = k_fps_proj.view(B, fps_adapter.num_tokens, num_heads, head_dim)
        v_fps = v_fps_proj.view(B, fps_adapter.num_tokens, num_heads, head_dim)

        y_fps_principal = flash_attention(q, k_fps, v_fps, k_lens=None)

        logging.info(f"  ||y_text||: {torch.norm(y_text_principal).item():.4f}")
        logging.info(f"  ||y_fps||:  {torch.norm(y_fps_principal).item():.4f}")

    return y_text_principal, y_fps_principal


def compute_singular_value_spectrum(X, Y):
    """
    Compute singular value spectrums to verify signal rank properties.

    This is a sanity check to verify that:
    - y_text (from principal components) is high-rank with energy spread across dimensions
    - y_fps (from FPS adapter) is low-rank with energy concentrated in few dimensions

    Args:
        X: Representation matrix [N, d1] (typically y_text_principal)
        Y: Representation matrix [N, d2] (typically y_fps_principal)

    Returns:
        S_X_normalized: Normalized singular values for X
        S_Y_normalized: Normalized singular values for Y
    """
    # Flatten and convert to float32
    X_flat = X.reshape(X.shape[0], -1).float()
    Y_flat = Y.reshape(Y.shape[0], -1).float()

    # Compute SVD
    _, S_X, _ = torch.svd(X_flat)
    _, S_Y, _ = torch.svd(Y_flat)

    # Normalize by the largest singular value
    S_X_normalized = S_X / S_X[0]
    S_Y_normalized = S_Y / S_Y[0]

    # Count effective rank (number of SVs > 0.01)
    effective_rank_X = (S_X_normalized > 0.01).sum().item()
    effective_rank_Y = (S_Y_normalized > 0.01).sum().item()

    logging.info(f"Singular Value Spectrum:")
    logging.info(f"  y_text effective rank: {effective_rank_X} (energy spread across {effective_rank_X} dimensions)")
    logging.info(f"  y_fps effective rank:  {effective_rank_Y} (energy concentrated in {effective_rank_Y} dimension)")

    # Determine if signals have expected properties
    if effective_rank_X > effective_rank_Y * 2:
        logging.info(f"  ✓ Expected: y_text is high-rank, y_fps is low-rank (different signal types)")
    else:
        logging.info(f"  ⚠ Unexpected: Both signals have similar rank")

    return S_X_normalized, S_Y_normalized


def plot_singular_value_spectrum(S_X_normalized, S_Y_normalized, block_idx, output_dir):
    """
    Plot singular value spectra with a clean aesthetic and point markers,
    then save directly as a high-quality PDF.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend (safe for servers)
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # --- Style (clean scientific look) ---
        plt.rcParams.update({
            "figure.figsize": (9.5, 6.0),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        })

        # Convert to numpy for plotting
        sv_x = S_X_normalized.detach().cpu().numpy()
        sv_y = S_Y_normalized.detach().cpu().numpy()

        eps = 1e-12
        sv_x_plot = np.clip(sv_x, eps, None)
        sv_y_plot = np.clip(sv_y, eps, None)

        fig, ax = plt.subplots()

        # --- Main plots ---
        ax.plot(
            sv_x_plot,
            label='y_text (Principal Components)',
            linewidth=2.2,
            marker='o',
            markersize=5.5,
            markerfacecolor='white',
            markeredgewidth=1.0,
            alpha=0.95,
        )
        ax.plot(
            sv_y_plot,
            label='y_cond (Adapter)',
            linewidth=2.2,
            marker='s',
            markersize=5.5,
            markerfacecolor='white',
            markeredgewidth=1.0,
            alpha=0.95,
        )

        # Labels and title
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Normalized Singular Value')
        ax.set_title(f'Singular Value Spectrum — Block {block_idx}')

        # Log-scale on y-axis
        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='--', alpha=0.35)
        ax.grid(True, which='minor', linestyle=':', alpha=0.22)

        # Polish axes and spines
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_alpha(0.8)

        ax.margins(x=0.02)
        ax.legend(frameon=False, loc='best')

        # Save to PDF
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f'singular_value_spectrum_block{block_idx}.pdf')
        plt.tight_layout()
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
        plt.close(fig)

        logging.info(f"  Spectrum plot (PDF): {pdf_path}")

    except ImportError:
        logging.warning("  matplotlib not available, skipping plot")


def compute_cka(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two representations.

    CKA measures similarity between two representational spaces.
    Score of 0 indicates orthogonality, 1 indicates perfect alignment.

    Args:
        X: Representation matrix [N, d1]
        Y: Representation matrix [N, d2]

    Returns:
        cka_score: CKA similarity score (0 to 1)
    """
    # Flatten if needed and convert to float32 for numerical stability
    X_flat = X.reshape(X.shape[0], -1).float()
    Y_flat = Y.reshape(Y.shape[0], -1).float()

    # Compute Gram matrices (linear kernel)
    K_X = X_flat @ X_flat.T
    K_Y = Y_flat @ Y_flat.T

    # Center the kernels
    n = K_X.shape[0]
    H = torch.eye(n, device=X.device, dtype=torch.float32) - torch.ones(n, n, device=X.device, dtype=torch.float32) / n
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H

    # Compute CKA
    numerator = torch.trace(K_X_centered @ K_Y_centered)
    trace_xx = torch.trace(K_X_centered @ K_X_centered)
    trace_yy = torch.trace(K_Y_centered @ K_Y_centered)
    denominator = torch.sqrt(trace_xx * trace_yy)

    cka_score = (numerator / (denominator + 1e-10)).item()

    # Check if CKA is reliable (both centered Gram matrices should have significant norm)
    k_y_norm = torch.norm(K_Y_centered).item()
    if k_y_norm < 1e-6:
        logging.warning(f"  ⚠️  CKA may be unreliable: K_Y_centered norm is nearly zero ({k_y_norm:.2e})")
        logging.warning(f"      This happens when Y is low-rank - cosine similarity is more reliable.")

    return cka_score


def analyze_block(trained_model, lora_checkpoint, block_idx, num_components=64,
                 fps_condition=1.0, device='cuda', output_dir='output'):
    """
    Perform principal orthogonality analysis for a single block.

    Args:
        trained_model: Trained model (with FPS adapter)
        lora_checkpoint: Optional dict of LoRA weights to merge (None for pretrained only)
        block_idx: Block index to analyze
        num_components: Number of principal components
        fps_condition: FPS conditioning value
        device: Device
        output_dir: Directory for saving plots

    Returns:
        results: Dictionary with analysis results
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"ANALYZING BLOCK {block_idx}")
    logging.info(f"{'='*80}")

    # 1. Extract principal components from backbone (with optional LoRA merging)
    q_principal, k_principal, v_principal = extract_principal_components(
        trained_model, block_idx, num_components, lora_checkpoint
    )

    # Move to device and convert to model's dtype (BFloat16)
    model_dtype = trained_model.dtype
    q_principal = q_principal.to(device=device, dtype=model_dtype)
    k_principal = k_principal.to(device=device, dtype=model_dtype)
    v_principal = v_principal.to(device=device, dtype=model_dtype)

    # 2. Simulate attention with trained model
    y_text_principal, y_fps_principal = simulate_attention_outputs(
        trained_model, block_idx, q_principal, k_principal, v_principal,
        fps_condition, device
    )

    # 3. Compute singular value spectrum
    S_text_normalized, S_fps_normalized = compute_singular_value_spectrum(
        y_text_principal, y_fps_principal
    )
    plot_singular_value_spectrum(S_text_normalized, S_fps_normalized, block_idx, output_dir)

    # 4. Compute orthogonality metrics
    logging.info(f"\nOrthogonality Metrics:")

    cosine_sim = torch.nn.functional.cosine_similarity(
        y_text_principal.flatten().unsqueeze(0),
        y_fps_principal.flatten().unsqueeze(0)
    ).item()
    logging.info(f"  Cosine Similarity: {cosine_sim:.6f} (target: ~0, indicates orthogonality)")

    cka_score = compute_cka(y_text_principal, y_fps_principal)
    logging.info(f"  CKA Score:         {cka_score:.6f}")

    norm_text = torch.norm(y_text_principal).item()
    norm_fps = torch.norm(y_fps_principal).item()
    norm_ratio = norm_fps / (norm_text + 1e-10)

    # Interpretation based on cosine similarity (more reliable)
    logging.info(f"\nInterpretation:")
    if abs(cosine_sim) < 0.01:
        logging.info(f"  ✅ EXCELLENT: Perfect orthogonality (|cos| < 0.01)")
        logging.info(f"     FPS adapter operates independently of text primitives")
    elif abs(cosine_sim) < 0.1:
        logging.info(f"  ✓  GOOD: Strong orthogonality (|cos| < 0.1)")
        logging.info(f"     Minimal interference between FPS and text")
    else:
        logging.info(f"  ⚠  WARNING: Weak orthogonality (|cos| >= 0.1)")
        logging.info(f"     Some alignment detected between FPS and text")

    return {
        'block_idx': block_idx,
        'cka_score': cka_score,
        'cosine_similarity': cosine_sim,
        'norm_text': norm_text,
        'norm_fps': norm_fps,
        'norm_ratio': norm_ratio,
        'effective_rank_text': (S_text_normalized > 0.01).sum().item(),
        'effective_rank_fps': (S_fps_normalized > 0.01).sum().item()
    }


def main():
    parser = argparse.ArgumentParser(description='Principal Orthogonality Check')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config TOML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--blocks', type=int, nargs='+', default=[27, 33, 39],
                       help='Block indices to analyze')
    parser.add_argument('--num_components', type=int, default=64,
                       help='Number of principal components to extract')
    parser.add_argument('--fps_condition', type=float, default=1.0,
                       help='FPS conditioning value for analysis')
    parser.add_argument('--output_dir', type=str, default='output/principal_orthogonality',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--use_base_lora', action='store_true', default=True,
                       help='Merge base LoRA into backbone weights (default: True)')
    parser.add_argument('--no_base_lora', action='store_false', dest='use_base_lora',
                       help='Do not merge base LoRA, use pretrained backbone only')

    args = parser.parse_args()

    setup_logging()

    logging.info("="*80)
    logging.info("PRINCIPAL ORTHOGONALITY CHECK")
    logging.info("="*80)
    logging.info(f"Config: {args.config}")
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Blocks to analyze: {args.blocks}")
    logging.info(f"Number of components: {args.num_components}")
    logging.info(f"FPS condition: {args.fps_condition}")
    logging.info(f"Use base LoRA: {args.use_base_lora}")

    if args.use_base_lora:
        logging.info(f"\nNOTE: Extracting principal components from trained backbone (pretrained + base LoRA)")
    else:
        logging.info(f"\nNOTE: Extracting principal components from pretrained backbone (LoRA not merged)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained model
    trained_model, lora_checkpoint = load_trained_model(
        args.config, args.checkpoint, args.device, args.use_base_lora
    )

    # Analyze each block
    results = []
    for block_idx in args.blocks:
        result = analyze_block(
            trained_model, lora_checkpoint, block_idx,
            args.num_components, args.fps_condition, args.device,
            args.output_dir
        )
        results.append(result)

    # Summary
    logging.info(f"\n{'='*80}")
    logging.info("SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"{'Block':>6}  {'CKA Score':>12}  {'Cos Sim':>10}  {'Rank_text':>12}  {'Rank_fps':>10}  {'Status':>20}")
    logging.info("-" * 90)

    for result in results:
        # Status based on cosine similarity (more reliable)
        status = "✅ Perfect" if abs(result['cosine_similarity']) < 0.01 else \
                 "✓  Strong" if abs(result['cosine_similarity']) < 0.1 else \
                 "⚠  Weak"

        logging.info(f"{result['block_idx']:>6}  {result['cka_score']:>12.6f}  "
                    f"{result['cosine_similarity']:>10.6f}  "
                    f"{result['effective_rank_text']:>12}  "
                    f"{result['effective_rank_fps']:>10}  {status:>20}")

    # Summary statistics
    avg_cka = np.mean([r['cka_score'] for r in results])
    avg_cos_sim = np.mean([abs(r['cosine_similarity']) for r in results])
    avg_rank_text = np.mean([r['effective_rank_text'] for r in results])
    avg_rank_fps = np.mean([r['effective_rank_fps'] for r in results])

    logging.info(f"\nSummary:")
    logging.info(f"  Average Cosine Similarity: {avg_cos_sim:.6f}")
    logging.info(f"  Average CKA Score: {avg_cka:.6f} (unreliable due to low-rank y_fps)")
    logging.info(f"  Average Effective Rank (text): {avg_rank_text:.1f}")
    logging.info(f"  Average Effective Rank (fps): {avg_rank_fps:.1f}")

    # Overall assessment based on cosine similarity
    if avg_cos_sim < 0.01:
        logging.info("\n✅ OVERALL: Excellent functional orthogonality!")
        logging.info("   FPS adapter operates independently of text content primitives.")
    elif avg_cos_sim < 0.1:
        logging.info("\n✓  OVERALL: Good functional orthogonality")
        logging.info("   Minimal interference between FPS and text conditioning.")
    else:
        logging.info("\n⚠  OVERALL: Moderate orthogonality")
        logging.info("   Some alignment detected - monitor for potential drift.")

    logging.info(f"\n{'='*80}")
    logging.info("ANALYSIS COMPLETE")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()
