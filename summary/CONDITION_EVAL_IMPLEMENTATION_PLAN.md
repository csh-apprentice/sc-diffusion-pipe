# Conditional Subspace Characterization - Implementation Plan

## Overview
This document provides a detailed implementation plan for **Experimental Plan C: Data-Free Analysis via Conditional Subspace Characterization** as specified in `/root/workspace/sc-diffusion-pipe/plans/CONDITION_EVAL.md`.

**Goal:** Empirically measure the geometric properties of the FPS-conditional pathway by analyzing the linear subspaces spanned by FPS adapter outputs, without requiring data sampling or full inference.

**Base Codebase:** `/root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align_old.py`

---

## Core Concept

### Key Insight
For any query `q`, the FPS adapter output `y_fps = Attention(q, k_fps, v_fps)` is a **linear combination** of value vectors in `v_fps`. Therefore:
- All possible outputs are confined to the **linear subspace** spanned by `v_fps`
- This subspace is **deterministic** given an FPS condition
- We can pre-calculate and analyze this subspace **without running inference**

### Mathematical Framework

For each FPS-adapted attention block `j` and condition `c`:

1. **Value Tensor Generation:**
   ```
   fps_embedding = FpsConditioning(tau_rel(c))  # Shape: [1, embed_dim]
   v_fps_raw = V_fps_proj(fps_embedding)        # Shape: [1, num_tokens * dim]
   v_fps = v_fps_raw.reshape([num_tokens, num_heads, head_dim])
   ```

2. **Subspace Basis:**
   ```
   V_c^(j) = orthonormal_basis(v_fps^(j)(c))   # Via SVD or QR decomposition
   ```

3. **Analyses:**
   - **Effective Dimensionality:** Singular value spectrum of `v_fps`
   - **Inter-Condition Orthogonality:** Principal angles between `V_c1^(j)` and `V_c2^(j)`
   - **Backbone Alignment:** Principal angles between `V_c^(j)` and `U_backbone^(j)` (backbone output subspace)

---

## Implementation Architecture

### Phase 1: Conditional Subspace Extraction
**Script:** `inference/analyze_conditional_subspace.py`

#### Component 1.1: Pipeline Loading
Reuse infrastructure from reference script:
```python
def load_pipeline_from_toml(toml_path):
    """Load WanPipeline using TOML configuration."""
    # Reuse from test_fps_multiple_experiments_align_old.py
    # Load TOML config, convert dtypes, initialize pipeline
    pass

def load_checkpoint(pipeline, checkpoint_path, fps_only=True):
    """Load trained checkpoint with FPS adapters."""
    # Load FPS-only parameters
    # Set model to eval mode
    pass
```

#### Component 1.2: FPS Adapter Extraction
```python
def get_fps_adapters(pipeline):
    """
    Extract FPS adapter modules from transformer blocks.

    Returns:
        List of dicts with adapter info for each adapted block
    """
    fps_adapters = []
    for block_idx, block in enumerate(pipeline.transformer.blocks):
        if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
            fps_adapters.append({
                'block_idx': block_idx,
                'adapter': block.fps_adapter,
                'dim': block.fps_adapter.dim,
                'num_heads': block.fps_adapter.num_heads,
                'head_dim': block.fps_adapter.head_dim,
                'num_tokens': block.fps_adapter.num_tokens,
                'rank': block.fps_adapter.rank
            })
    return fps_adapters
```

#### Component 1.3: FPS Conditioning Computation
```python
def compute_fps_embeddings(pipeline, fps_values):
    """
    Compute FPS conditioning embeddings for different conditions.

    Args:
        pipeline: WanPipeline
        fps_values: List of FPS values (scalar or list for multi-condition)

    Returns:
        Dict mapping fps_str -> embedding tensor [1, embed_dim]
    """
    from models.wan.model import compute_tau_rel

    fps_embeddings = {}
    device = next(pipeline.transformer.parameters()).device

    # Access FPS conditioning MLP through InitialLayer
    initial_layer = pipeline.to_layers()[0]
    fps_conditioning_mlp = initial_layer.fps_conditioning

    for fps in fps_values:
        # Handle single vs multi-condition
        if isinstance(fps, list):
            fps_tensor = torch.tensor([fps], dtype=torch.float32, device=device)
        else:
            fps_tensor = torch.tensor([fps], dtype=torch.float32, device=device)

        # Compute tau_rel transformation
        tau_rel = compute_tau_rel(
            fps_tensor,
            reference_fps=initial_layer.fps_reference_fps,
            transform=initial_layer.fps_tau_transform,
            scale=initial_layer.fps_tau_scale
        )

        # Handle shape
        if tau_rel.ndim == 1:
            tau_rel = tau_rel.unsqueeze(-1)

        # Pass through MLP
        with torch.no_grad():
            fps_embedding = fps_conditioning_mlp(tau_rel)

        fps_str = str(fps) if not isinstance(fps, list) else '_'.join(f"{v:.3f}" for v in fps)
        fps_embeddings[fps_str] = fps_embedding

    return fps_embeddings
```

#### Component 1.4: Subspace Generation
```python
def generate_conditional_subspace(fps_adapter, fps_embedding):
    """
    Generate value tensor and orthonormal basis for a condition.

    Args:
        fps_adapter: FPSCrossAttentionAdapter module
        fps_embedding: [1, embed_dim] tensor

    Returns:
        Dict with subspace information
    """
    with torch.no_grad():
        # 1) LoRA projection for V
        v_down = fps_adapter.v_fps_down(fps_embedding)  # [1, rank]
        v_up = fps_adapter.v_fps_up(v_down)             # [1, num_tokens * dim]

        # 2) Apply LoRA scaling
        v_proj = v_up * fps_adapter.lora_scale

        # 3) Reshape to attention format
        num_tokens = fps_adapter.num_tokens
        num_heads = fps_adapter.num_heads
        head_dim = fps_adapter.head_dim

        v_fps = v_proj.view(1, num_tokens, num_heads, head_dim).squeeze(0)
        # Shape: [num_tokens, num_heads, head_dim]

        # 4) Flatten to matrix for SVD
        v_matrix = v_fps.reshape(-1, head_dim)  # [num_tokens * num_heads, head_dim]

        # 5) Compute SVD
        U, S, Vh = torch.linalg.svd(v_matrix, full_matrices=False)

        # 6) Compute effective rank
        eff_rank = compute_effective_rank(S, threshold=0.01)

        return {
            'v_fps': v_fps.cpu(),
            'v_matrix': v_matrix.cpu(),
            'basis': U.cpu(),              # Orthonormal basis [num_tokens*num_heads, head_dim]
            'singular_values': S.cpu(),
            'effective_rank': eff_rank,
            'full_rank': len(S)
        }

def compute_effective_rank(singular_values, threshold=0.01):
    """Compute effective rank based on singular value threshold."""
    normalized_sv = singular_values / singular_values[0]
    return (normalized_sv > threshold).sum().item()
```

#### Component 1.5: Main Extraction Loop
```python
def extract_all_subspaces(pipeline, fps_values, output_dir):
    """
    Extract conditional subspaces for all FPS values and blocks.

    Args:
        pipeline: Loaded WanPipeline
        fps_values: List of FPS conditions to analyze
        output_dir: Where to save subspace data

    Returns:
        nested dict: subspace_data[fps_str][block_idx] -> subspace info
    """
    import os
    import pickle
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # Extract FPS adapters
    fps_adapters = get_fps_adapters(pipeline)
    print(f"Found {len(fps_adapters)} FPS-adapted blocks")

    # Compute FPS embeddings for all conditions
    fps_embeddings = compute_fps_embeddings(pipeline, fps_values)
    print(f"Computed embeddings for {len(fps_embeddings)} conditions")

    # Extract subspaces
    subspace_data = {}

    for fps in tqdm(fps_values, desc="Processing FPS conditions"):
        fps_str = str(fps) if not isinstance(fps, list) else '_'.join(f"{v:.3f}" for v in fps)
        fps_embedding = fps_embeddings[fps_str]

        subspace_data[fps_str] = []

        for adapter_info in tqdm(fps_adapters, desc=f"FPS={fps_str} blocks", leave=False):
            subspace_info = generate_conditional_subspace(
                adapter_info['adapter'],
                fps_embedding
            )
            subspace_info['block_idx'] = adapter_info['block_idx']
            subspace_data[fps_str].append(subspace_info)

    # Save to disk
    output_file = os.path.join(output_dir, 'subspace_data.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(subspace_data, f)
    print(f"Saved subspace data to {output_file}")

    return subspace_data
```

---

### Phase 2: Geometric Analyses
**Script:** `inference/analyze_conditional_geometry.py`

#### Analysis 1: Effective Dimensionality

```python
def analyze_dimensionality(subspace_data, fps_values):
    """
    Analyze effective dimensionality across conditions and blocks.

    Returns:
        pandas DataFrame with statistics
    """
    import pandas as pd

    results = []

    for fps in fps_values:
        fps_str = str(fps) if not isinstance(fps, list) else '_'.join(f"{v:.3f}" for v in fps)

        for subspace_info in subspace_data[fps_str]:
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
                'sv_decay': np.log(sv[0] / (sv[-1] + 1e-10))
            })

    df = pd.DataFrame(results)
    return df
```

#### Analysis 2: Inter-Condition Orthogonality

```python
def compute_principal_angles(basis1, basis2):
    """
    Compute principal angles between two subspaces.

    Args:
        basis1: [dim, rank1] orthonormal basis
        basis2: [dim, rank2] orthonormal basis

    Returns:
        angles in degrees [0, 90]
    """
    # Compute overlap matrix
    overlap = basis1.T @ basis2  # [rank1, rank2]

    # SVD gives principal angles via: cos(θ) = σ
    U, S, Vh = torch.linalg.svd(overlap, full_matrices=False)

    # Convert to angles
    cos_angles = torch.clamp(S, 0.0, 1.0)
    angles_rad = torch.acos(cos_angles)
    angles_deg = angles_rad * 180.0 / math.pi

    return angles_deg.numpy()

def analyze_orthogonality(subspace_data, fps_values):
    """
    Analyze orthogonality between different FPS conditions.

    Returns:
        angle_matrix: [n_fps, n_fps, n_blocks] with principal angles
    """
    fps_strs = [str(fps) if not isinstance(fps, list) else '_'.join(f"{v:.3f}" for v in fps)
                for fps in fps_values]

    n_fps = len(fps_strs)
    n_blocks = len(subspace_data[fps_strs[0]])

    # Storage for angles
    min_angles = np.zeros((n_fps, n_fps, n_blocks))
    mean_angles = np.zeros((n_fps, n_fps, n_blocks))

    for block_idx in range(n_blocks):
        for i, fps1 in enumerate(fps_strs):
            for j, fps2 in enumerate(fps_strs):
                if i == j:
                    continue

                basis1 = subspace_data[fps1][block_idx]['basis']
                basis2 = subspace_data[fps2][block_idx]['basis']

                angles = compute_principal_angles(basis1, basis2)

                min_angles[i, j, block_idx] = angles[0]
                mean_angles[i, j, block_idx] = angles.mean()

    return {
        'min_angles': min_angles,
        'mean_angles': mean_angles,
        'fps_values': fps_values,
        'fps_strs': fps_strs
    }
```

#### Analysis 3: Backbone Alignment

```python
def extract_backbone_subspace(pipeline, block_idx, energy_threshold=0.90):
    """
    Extract dominant output subspace from backbone's cross-attention.

    Args:
        pipeline: WanPipeline
        block_idx: Which block to analyze
        energy_threshold: Cumulative energy to retain

    Returns:
        Dict with backbone subspace info
    """
    block = pipeline.transformer.blocks[block_idx]

    # Get output projection matrix
    o_weight = block.cross_attn.o.weight.data  # [dim, dim]

    # SVD to find dominant directions
    U, S, Vh = torch.linalg.svd(o_weight, full_matrices=False)

    # Select top-k by energy
    energy = S ** 2
    cumulative_energy = torch.cumsum(energy, dim=0) / energy.sum()
    k = (cumulative_energy < energy_threshold).sum().item() + 1

    return {
        'basis': U[:, :k].cpu(),
        'singular_values': S.cpu(),
        'effective_rank': k,
        'energy_threshold': energy_threshold
    }

def analyze_backbone_alignment(subspace_data, pipeline, fps_values):
    """
    Analyze alignment between FPS conditioning and backbone outputs.

    Returns:
        DataFrame with alignment statistics
    """
    import pandas as pd

    results = []

    fps_strs = [str(fps) if not isinstance(fps, list) else '_'.join(f"{v:.3f}" for v in fps)
                for fps in fps_values]

    # Extract backbone subspaces (cached)
    backbone_subspaces = {}

    for fps_str in fps_strs:
        for subspace_info in subspace_data[fps_str]:
            block_idx = subspace_info['block_idx']

            if block_idx not in backbone_subspaces:
                backbone_subspaces[block_idx] = extract_backbone_subspace(
                    pipeline, block_idx
                )

            # Compute principal angles
            fps_basis = subspace_info['basis']
            backbone_basis = backbone_subspaces[block_idx]['basis']

            angles = compute_principal_angles(fps_basis, backbone_basis)

            results.append({
                'fps': fps_str,
                'block_idx': block_idx,
                'min_angle': angles[0],
                'mean_angle': angles.mean(),
                'max_angle': angles[-1],
                'median_angle': np.median(angles),
                'alignment_score': np.cos(np.deg2rad(angles[0]))
            })

    df = pd.DataFrame(results)
    return df
```

---

### Phase 3: Visualization and Reporting
**Script:** `inference/generate_subspace_report.py`

```python
def plot_dimensionality_analysis(df, output_dir):
    """Generate dimensionality analysis plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Effective rank by block
    for fps in df['fps'].unique():
        subset = df[df['fps'] == fps]
        axes[0, 0].plot(subset['block_idx'], subset['effective_rank'],
                       marker='o', label=f'FPS={fps}')
    axes[0, 0].set_xlabel('Block Index')
    axes[0, 0].set_ylabel('Effective Rank')
    axes[0, 0].set_title('Effective Rank Across Blocks')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Rank ratio distribution
    sns.boxplot(data=df, x='fps', y='rank_ratio', ax=axes[0, 1])
    axes[0, 1].set_title('Rank Ratio Distribution')
    axes[0, 1].set_ylabel('Effective Rank / Full Rank')

    # Plot 3: Singular value entropy
    for fps in df['fps'].unique():
        subset = df[df['fps'] == fps]
        axes[1, 0].plot(subset['block_idx'], subset['sv_entropy'],
                       marker='o', label=f'FPS={fps}')
    axes[1, 0].set_xlabel('Block Index')
    axes[1, 0].set_ylabel('SV Entropy')
    axes[1, 0].set_title('Singular Value Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: SV decay rate
    sns.violinplot(data=df, x='fps', y='sv_decay', ax=axes[1, 1])
    axes[1, 1].set_title('Singular Value Decay Rate')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/dimensionality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_orthogonality_analysis(orthogonality_data, output_dir):
    """Generate orthogonality heatmaps."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    min_angles = orthogonality_data['min_angles']
    fps_strs = orthogonality_data['fps_strs']
    n_blocks = min_angles.shape[2]

    # Select representative blocks
    blocks_to_plot = [0, n_blocks // 2, n_blocks - 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, block_idx in enumerate(blocks_to_plot):
        sns.heatmap(
            min_angles[:, :, block_idx],
            annot=True, fmt='.1f',
            xticklabels=fps_strs,
            yticklabels=fps_strs,
            cmap='RdYlGn', vmin=0, vmax=90,
            ax=axes[idx],
            cbar_kws={'label': 'Min Principal Angle (°)'}
        )
        axes[idx].set_title(f'Block {block_idx} - Inter-Condition Orthogonality')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/orthogonality_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_backbone_alignment(df, output_dir):
    """Generate backbone alignment plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Min angle by block
    for fps in df['fps'].unique():
        subset = df[df['fps'] == fps]
        axes[0, 0].plot(subset['block_idx'], subset['min_angle'],
                       marker='o', label=f'FPS={fps}')
    axes[0, 0].set_xlabel('Block Index')
    axes[0, 0].set_ylabel('Min Principal Angle (°)')
    axes[0, 0].set_title('Backbone Alignment: Minimum Angles')
    axes[0, 0].legend()
    axes[0, 0].axhline(60, color='green', linestyle='--', alpha=0.5, label='Threshold (60°)')
    axes[0, 0].axhline(30, color='red', linestyle='--', alpha=0.5, label='Warning (30°)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Mean angle by block
    for fps in df['fps'].unique():
        subset = df[df['fps'] == fps]
        axes[0, 1].plot(subset['block_idx'], subset['mean_angle'],
                       marker='o', label=f'FPS={fps}')
    axes[0, 1].set_xlabel('Block Index')
    axes[0, 1].set_ylabel('Mean Principal Angle (°)')
    axes[0, 1].set_title('Backbone Alignment: Mean Angles')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Alignment score distribution
    import seaborn as sns
    sns.boxplot(data=df, x='fps', y='alignment_score', ax=axes[1, 0])
    axes[1, 0].set_title('Alignment Score Distribution')
    axes[1, 0].set_ylabel('Alignment Score (0=orthogonal, 1=aligned)')
    axes[1, 0].axhline(0.5, color='orange', linestyle='--', alpha=0.5)

    # Plot 4: Angle range (min to max) per block
    for fps in df['fps'].unique():
        subset = df[df['fps'] == fps]
        axes[1, 1].fill_between(subset['block_idx'],
                                subset['min_angle'],
                                subset['max_angle'],
                                alpha=0.3, label=f'FPS={fps}')
    axes[1, 1].set_xlabel('Block Index')
    axes[1, 1].set_ylabel('Principal Angle Range (°)')
    axes[1, 1].set_title('Backbone Alignment: Angle Range')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/backbone_alignment.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(dim_df, ortho_data, backbone_df, output_dir):
    """Generate text summary report."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("CONDITIONAL SUBSPACE CHARACTERIZATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1: Effective Dimensionality
    report_lines.append("1. EFFECTIVE DIMENSIONALITY ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Mean effective rank: {dim_df['effective_rank'].mean():.1f}")
    report_lines.append(f"Mean rank ratio: {dim_df['rank_ratio'].mean():.3f}")
    report_lines.append(f"SV entropy range: [{dim_df['sv_entropy'].min():.2f}, {dim_df['sv_entropy'].max():.2f}]")
    report_lines.append("")

    # Section 2: Inter-Condition Orthogonality
    report_lines.append("2. INTER-CONDITION ORTHOGONALITY ANALYSIS")
    report_lines.append("-" * 80)
    min_angles = ortho_data['min_angles']
    # Mask diagonal (self-comparison)
    mask = np.ones_like(min_angles[:, :, 0], dtype=bool)
    np.fill_diagonal(mask, False)
    valid_angles = min_angles[mask].flatten()
    report_lines.append(f"Mean pairwise angle: {valid_angles.mean():.1f}°")
    report_lines.append(f"Min pairwise angle: {valid_angles.min():.1f}°")
    report_lines.append(f"Max pairwise angle: {valid_angles.max():.1f}°")
    report_lines.append("")

    # Section 3: Backbone Alignment
    report_lines.append("3. BACKBONE ALIGNMENT ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"Mean minimum angle: {backbone_df['min_angle'].mean():.1f}°")
    report_lines.append(f"Mean alignment score: {backbone_df['alignment_score'].mean():.3f}")
    report_lines.append("")

    # Section 4: Interpretation
    report_lines.append("4. INTERPRETATION")
    report_lines.append("-" * 80)

    # Dimensionality interpretation
    mean_ratio = dim_df['rank_ratio'].mean()
    if mean_ratio < 0.3:
        report_lines.append("✅ Effective dimensionality: COMPACT (efficient conditioning)")
    elif mean_ratio < 0.6:
        report_lines.append("⚠️  Effective dimensionality: MODERATE")
    else:
        report_lines.append("❌ Effective dimensionality: HIGH (potentially redundant)")

    # Orthogonality interpretation
    mean_angle = valid_angles.mean()
    if mean_angle > 60:
        report_lines.append("✅ Inter-condition orthogonality: STRONG (distinct effects)")
    elif mean_angle > 30:
        report_lines.append("⚠️  Inter-condition orthogonality: MODERATE")
    else:
        report_lines.append("❌ Inter-condition orthogonality: WEAK (potential interference)")

    # Backbone alignment interpretation
    mean_min_angle = backbone_df['min_angle'].mean()
    if mean_min_angle > 60:
        report_lines.append("✅ Backbone alignment: STRONG DISENTANGLEMENT")
    elif mean_min_angle > 30:
        report_lines.append("⚠️  Backbone alignment: MODERATE ENTANGLEMENT RISK")
    else:
        report_lines.append("❌ Backbone alignment: HIGH ENTANGLEMENT RISK")

    report_lines.append("=" * 80)

    # Write to file
    report_text = "\n".join(report_lines)
    with open(f'{output_dir}/summary_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    return report_text
```

---

## Implementation Schedule

### Week 1: Core Infrastructure (Days 1-5)
- **Day 1-2:** Set up `analyze_conditional_subspace.py`
  - Pipeline loading utilities
  - FPS adapter extraction
  - Testing infrastructure

- **Day 3-4:** Implement subspace extraction
  - FPS embedding computation
  - Value tensor generation
  - SVD and basis computation

- **Day 5:** Data storage and testing
  - Save/load functionality
  - Unit tests
  - Small-scale validation

### Week 2: Geometric Analyses (Days 6-10)
- **Day 6-7:** Dimensionality analysis
  - Effective rank computation
  - Statistical analysis
  - Visualization functions

- **Day 8:** Orthogonality analysis
  - Principal angle computation
  - Pairwise analysis
  - Heatmap generation

- **Day 9-10:** Backbone alignment
  - Backbone subspace extraction
  - Alignment computation
  - Interpretation metrics

### Week 3: Integration & Validation (Days 11-15)
- **Day 11-12:** End-to-end pipeline
  - Connect all modules
  - Error handling
  - Performance optimization

- **Day 13-14:** Report generation
  - Visualization refinement
  - Summary statistics
  - Automated reporting

- **Day 15:** Validation
  - Run on trained checkpoints
  - Compare with inference results
  - Bug fixes and refinement

---

## Usage Examples

### Example 1: Basic Analysis
```bash
# Step 1: Extract subspaces
python inference/analyze_conditional_subspace.py \
    --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint outputs/checkpoint/step_10000 \
    --fps_values 12 24 60 120 240 \
    --output_dir output/subspace_analysis

# Step 2: Run analyses
python inference/analyze_conditional_geometry.py \
    --subspace_file output/subspace_analysis/subspace_data.pkl \
    --output_dir output/subspace_analysis

# Step 3: Generate report
python inference/generate_subspace_report.py \
    --analysis_dir output/subspace_analysis \
    --output_report output/subspace_analysis/report.pdf
```

### Example 2: Multi-Condition Model
```bash
python inference/analyze_conditional_subspace.py \
    --config shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH.toml \
    --checkpoint outputs/multi_cond_checkpoint/step_5000 \
    --fps_values 0.5 0.02  1.0 0.05  0.125 0.1 \
    --num_conditions 2 \
    --output_dir output/multi_cond_subspace
```

---

## Expected Outputs

### Quantitative Benchmarks

**Good Conditioning:**
- Effective rank ratio: 0.2-0.4 (compact)
- Inter-condition angles: 50-80° (distinct)
- Backbone alignment: > 60° (disentangled)

**Warning Signs:**
- Effective rank ratio: > 0.7 (redundant)
- Inter-condition angles: < 30° (interfering)
- Backbone alignment: < 30° (entangled)

---

## Next Steps

1. **Immediate:** Review and approve plan
2. **Week 1:** Implement Phase 1 (subspace extraction)
3. **Week 2:** Implement Phase 2 (geometric analyses)
4. **Week 3:** Integration and validation
5. **Future:** Extend to training-time monitoring

---

**Document Version:** 1.0
**Last Updated:** 2025-10-18
**Status:** Ready for Implementation
