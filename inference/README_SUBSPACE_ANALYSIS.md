# Conditional Subspace Characterization

**Data-free geometric analysis of FPS conditioning adapters**

## Overview

This tool analyzes the geometric properties of FPS-conditional pathways by directly examining the linear subspaces they generate, **without requiring data sampling or full inference**. It provides three key analyses:

1. **Effective Dimensionality**: How compact is the conditioning representation?
2. **Inter-Condition Orthogonality**: How well-separated are different FPS conditions?
3. **Backbone Alignment**: How entangled is conditioning with the backbone model?

## Quick Start

### Single Command

```bash
# Run complete analysis pipeline
bash bash/run_subspace_analysis.sh \
    fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    outputs/checkpoint/step_10000 \
    12 24 60 120 240
```

### Step-by-Step

```bash
# Step 1: Extract conditional subspaces
python inference/analyze_conditional_subspace.py \
    --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint outputs/checkpoint/step_10000 \
    --fps_values 12 24 60 120 240 \
    --output_dir output/subspace_analysis

# Step 2: Run geometric analyses
python inference/analyze_conditional_geometry.py \
    --subspace_file output/subspace_analysis/subspace_data.pkl \
    --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint outputs/checkpoint/step_10000 \
    --output_dir output/subspace_analysis
```

## Usage Examples

### Example 1: Single-Condition Model (FPS Only)

```bash
python inference/analyze_conditional_subspace.py \
    --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint outputs/fps_checkpoint/step_10000 \
    --fps_values 12 24 60 120 240 \
    --output_dir output/fps_subspace_analysis \
    --port 29502
```

### Example 2: Multi-Condition Model (FPS + Shutter + Bokeh)

```bash
# For 2 conditions, values are automatically grouped: [0.5, 0.02], [1.0, 0.05], [0.125, 0.1]
python inference/analyze_conditional_subspace.py \
    --config shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH.toml \
    --checkpoint outputs/multi_cond/step_5000 \
    --fps_values 0.5 0.02  1.0 0.05  0.125 0.1 \
    --output_dir output/multi_cond_subspace
```

### Example 3: Skip Backbone Alignment (Faster)

```bash
# Only run dimensionality and orthogonality analyses
python inference/analyze_conditional_geometry.py \
    --subspace_file output/subspace_analysis/subspace_data.pkl \
    --skip_backbone \
    --output_dir output/subspace_analysis
```

### Example 4: Compare Multiple Checkpoints

```bash
# Analyze checkpoint progression
for step in 1000 5000 10000 15000; do
    python inference/analyze_conditional_subspace.py \
        --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
        --checkpoint outputs/checkpoint/step_${step} \
        --fps_values 12 24 60 120 240 \
        --output_dir output/subspace_step_${step}

    python inference/analyze_conditional_geometry.py \
        --subspace_file output/subspace_step_${step}/subspace_data.pkl \
        --skip_backbone \
        --output_dir output/subspace_step_${step}
done
```

## Output Files

After running the analysis, you'll get:

### Data Files
- `subspace_data.pkl`: Raw subspace data (bases, singular values, etc.)
- `metadata.json`: Analysis metadata
- `dimensionality_results.csv`: Dimensionality statistics per condition/block
- `orthogonality_results.npz`: Pairwise angle matrices
- `backbone_alignment_results.csv`: Backbone alignment statistics

### Visualizations
- `dimensionality_analysis.png`: 4-panel plot showing:
  - Effective rank across blocks
  - Rank ratio distribution
  - Singular value entropy
  - SV decay rate

- `orthogonality_heatmaps.png`: Heatmaps showing inter-condition angles
- `orthogonality_evolution.png`: How orthogonality changes across blocks
- `backbone_alignment.png`: 4-panel plot showing backbone entanglement

### Reports
- `summary_report.txt`: Text summary with interpretations

## Interpreting Results

### 1. Effective Dimensionality

**What it measures:** How many dimensions are needed to represent FPS conditioning

**Good indicators:**
- ✅ Rank ratio < 0.3 → **Compact** representation (efficient)
- ⚠️ Rank ratio 0.3-0.6 → **Moderate** compactness
- ❌ Rank ratio > 0.6 → **Redundant** representation

**Example interpretation:**
```
Mean rank ratio: 0.25 ± 0.05
✅ COMPACT representation (efficient conditioning)
```
This means FPS conditioning uses only ~25% of available dimensions, indicating a focused, efficient representation.

### 2. Inter-Condition Orthogonality

**What it measures:** How geometrically separated are different FPS conditions

**Good indicators:**
- ✅ Mean angle > 60° → **Strong** separation (distinct effects)
- ⚠️ Mean angle 30-60° → **Moderate** separation
- ❌ Mean angle < 30° → **Weak** separation (potential interference)

**Example interpretation:**
```
Mean pairwise angle: 72.3° ± 8.5°
✅ STRONG orthogonality (distinct effects)
```
This means different FPS conditions occupy nearly orthogonal subspaces, so they won't interfere with each other.

### 3. Backbone Alignment

**What it measures:** How much FPS conditioning overlaps with backbone's output space

**Good indicators:**
- ✅ Mean min angle > 60° → **Strong** disentanglement
- ⚠️ Mean min angle 30-60° → **Moderate** entanglement risk
- ❌ Mean min angle < 30° → **High** entanglement risk

**Example interpretation:**
```
Mean minimum angle: 68.1° ± 12.3°
✅ STRONG disentanglement
```
This means FPS conditioning operates in a subspace nearly orthogonal to the backbone, minimizing interference.

## Mathematical Background

### Principal Angles

Given two orthonormal bases U₁ and U₂, principal angles θ_i are computed via:

```
1. Compute overlap: M = U₁ᵀ @ U₂
2. SVD: M = A @ Σ @ Bᵀ
3. Angles: θ_i = arccos(σ_i)
```

Properties:
- 0° ≤ θ_i ≤ 90° for all i
- θ₁ (smallest) measures **strongest alignment**
- θ₁ = 0° ⟺ subspaces intersect
- θ₁ = 90° ⟺ subspaces are orthogonal

### Effective Rank

Measures the "true" dimensionality of a subspace:

```
Given singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ:

Threshold-based:
  eff_rank = |{i : σᵢ / σ₁ > 0.01}|

Energy-based:
  eff_rank = min{k : Σᵢ₌₁ᵏ σᵢ² / Σⱼ σⱼ² > 0.95}
```

## Troubleshooting

### Issue: "No FPS adapters found in model!"

**Solution:** Make sure your checkpoint has FPS parameters. Check with:
```bash
python -c "import safetensors; ckpt = safetensors.torch.load_file('checkpoint.safetensors'); print([k for k in ckpt.keys() if 'fps' in k.lower()][:5])"
```

### Issue: Analysis is slow

**Solutions:**
1. Skip backbone alignment: `--skip_backbone`
2. Analyze fewer conditions: reduce `--fps_values`
3. Use smaller checkpoint for testing

### Issue: Out of memory

**Solutions:**
1. Use CPU: `--device cpu`
2. Skip backbone alignment
3. Analyze blocks sequentially (modify code to process one block at a time)

## Advanced Usage

### Analyze Specific Blocks Only

Modify `analyze_conditional_subspace.py` to filter blocks:

```python
# In get_fps_adapters(), add filtering
fps_adapters = []
for block_idx, block in enumerate(pipeline.transformer.blocks):
    if block_idx in [27, 28, 29, 30]:  # Only analyze last 4 blocks
        # ... rest of code
```

### Custom Singular Value Threshold

Modify `compute_effective_rank()` threshold:

```python
def compute_effective_rank(singular_values, threshold=0.05):  # Changed from 0.01
    normalized_sv = singular_values / singular_values[0]
    return (normalized_sv > threshold).sum().item()
```

### Export for External Analysis

Load and analyze data in Python:

```python
import pickle
import numpy as np

# Load subspace data
with open('output/subspace_analysis/subspace_data.pkl', 'rb') as f:
    subspace_data = pickle.load(f)

# Access subspace for FPS=60, block=27
subspace_info = subspace_data['60.0'][10]  # 11th block (0-indexed)
basis = subspace_info['basis']             # Orthonormal basis
sv = subspace_info['singular_values']      # Singular values

# Load orthogonality data
ortho = np.load('output/subspace_analysis/orthogonality_results.npz')
min_angles = ortho['min_angles']  # Shape: [n_fps, n_fps, n_blocks]
```

## Integration with Training

### Monitor During Training

Add to your training script:

```python
# Every N steps, run subspace analysis
if step % 1000 == 0:
    os.system(f"""
        python inference/analyze_conditional_subspace.py \
            --config {config_path} \
            --checkpoint {checkpoint_dir}/step_{step} \
            --fps_values 12 24 60 \
            --output_dir output/subspace_step_{step}
    """)
```

### Early Stopping Criterion

Use geometric metrics for early stopping:

```python
# Load latest analysis results
df = pd.read_csv(f'output/subspace_step_{step}/dimensionality_results.csv')
mean_ratio = df['rank_ratio'].mean()

if mean_ratio > 0.7:
    print("Warning: Conditioning becoming redundant")
```

## Theory and Motivation

### Why This Works

The key insight is that attention output is a **linear combination** of value vectors:

```
y_fps = Attention(Q, K_fps, V_fps)
      = Σᵢ αᵢ * V_fps[i]    # αᵢ are attention weights
```

Therefore, all possible outputs lie in `span(V_fps)`. By analyzing this subspace:
- We understand the **geometry** of FPS conditioning
- We can predict **interference** between conditions
- We can measure **entanglement** with backbone
- **No data sampling needed!**

### Advantages Over Empirical Testing

| Approach | Subspace Analysis | Empirical Inference |
|----------|-------------------|---------------------|
| **Speed** | Minutes | Hours |
| **Data needed** | None | Many prompts |
| **Interpretation** | Geometric, rigorous | Qualitative |
| **Completeness** | All possible outputs | Limited samples |
| **Reproducibility** | Perfect | Depends on prompts |

## References

- **Principal Angles**: Björck & Golub (1973), "Numerical Methods for Computing Angles Between Linear Subspaces"
- **Effective Rank**: Roy & Vetterli (2007), "The Effective Rank: A Measure of Effective Dimensionality"
- **LoRA Analysis**: Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"

## Citation

If you use this analysis in your research, please cite:

```bibtex
@misc{wan_subspace_analysis,
  title={Conditional Subspace Characterization for FPS Adapters},
  author={WAN Conditioning Team},
  year={2025},
  howpublished={\url{https://github.com/...}}
}
```

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review the implementation plan: `summary/CONDITION_EVAL_IMPLEMENTATION_PLAN.md`
3. Examine code comments in `inference/analyze_conditional_*.py`
4. Open an issue on GitHub

---

**Last Updated:** 2025-10-18
**Version:** 1.0
**Status:** Production Ready
