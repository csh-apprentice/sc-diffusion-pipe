# WAN Model Changes Summary

## Overview
This document provides a high-level summary of the modifications made to the WAN (Alibaba Video Generation) model in the `sc-wan-conditioning` branch. The primary goal was to add **FPS conditioning** capabilities to enable controllable frame rate and exposure control during video generation.

---

## Major Features Added

### 1. FPS Conditioning System
**Purpose:** Allow the model to learn and control video temporal characteristics (frame rate, shutter speed, motion blur) as conditionable parameters.

**Core Components:**
- **FPS Embedding MLP** (`FpsConditioning` class in `models/wan/model.py:383-416`)
  - Converts FPS scalar values into embeddings compatible with the transformer
  - Architecture: `Linear(num_conditions, hidden) → SiLU → Linear(hidden, out_dim)`
  - Special zero-initialization to avoid disturbing pretrained weights at start

- **FPS Cross-Attention Adapters** (`FPSCrossAttentionAdapter` class in `models/wan/model.py:419-671`)
  - Injected into the deepest third of transformer blocks
  - Uses LoRA-style projections to create FPS-conditioned K', V' matrices
  - Combines text and FPS attention: `y = Attn(Q, K_text, V_text) + gate × Attn(Q, K_fps, V_fps)`
  - ~10.7M additional parameters for 14B model

### 2. Multi-Condition Support
**Purpose:** Enable training with multiple scalar conditions simultaneously (e.g., FPS + shutter speed + bokeh).

**Key Features:**
- **Flexible condition count:** Supports 1, 2, 3+ conditions through list-based configuration
- **Per-condition transforms:** Each condition can have independent transform functions, scales, and reference values
- **Backward compatible:** Single-condition training unchanged from original implementation
- **Dataset format:** Folder names encode conditions (e.g., `12_0.01/` = [FPS=12, shutter=0.01])

---

## Architecture Changes

### Model Components Modified

#### 1. `WanModel` Class (`models/wan/model.py`)
**New Parameters:**
```python
fps_adapter_rank=8              # LoRA rank for FPS adapters
fps_adapter_gate_init=0.0       # Initial gate value
fps_condition_blocks="deepest_third"  # Which blocks get FPS adapters
fps_adapter_num_tokens=1        # Number of FPS conditioning tokens
fps_tau_transform="log1p"       # Transform function (or list for multi-cond)
fps_tau_scale=0.33333334        # Scaling factor (or list for multi-cond)
fps_reference_fps=240.0         # Reference FPS (or list for multi-cond)
fps_embed_dim=256               # FPS embedding dimension
fps_condition_hidden=64         # Hidden size in FPS MLP
fps_lora_alpha=16               # LoRA scaling factor
fps_gate_mode='fixed'           # Gate mode: 'fixed'|'sigmoid'|'relu'|'identity'|...
fps_gate_fixed_value=0.5        # Fixed gate value (if gate_mode='fixed')
fps_scale=False                 # Whether to scale FPS attention to match text attention
fps_warmup_steps=100            # Warmup steps for gradual FPS influence ramp-up
```

#### 2. `WanAttentionBlock` Class (`models/wan/model.py:682-798`)
- Added optional `fps_adapter` module
- Modified `forward()` to accept and propagate `fps_conditioning` tensor
- FPS-enhanced blocks use adapter's combined attention, others use standard cross-attention

#### 3. `WanPipeline` Class (`models/wan/wan.py`)
- Integrated FPS conditioning configuration into model loading
- Added FPS parameter handling in `prepare_inputs()` (line 525-573)
- FPS values passed through entire forward pipeline: `InitialLayer → TransformerLayers → FinalLayer`

---

## Conditioning Transform Functions

The `compute_tau_rel()` function (`models/wan/model.py:20-166`) converts raw FPS values into normalized conditioning signals:

| Transform | Formula | Use Case |
|-----------|---------|----------|
| `"log1p"` | `log(1 + reference_fps/fps)` | Default, smooth scale for FPS ratios |
| `"raw"` | `fps` (or `reference_fps/fps`) | Direct FPS values |
| `"log"` | `log(reference_fps/fps)` | Logarithmic scale |
| `"neglogfps"` | `-log(fps)` | Inverse log scale, no reference needed |
| `"centerlog1p"` | `sign(ref-fps) × log(1 + |tau-1|)` | Centered around reference FPS |

**Multi-Condition Example:**
```python
# Two conditions: FPS (raw) + shutter speed (centerlog1p)
tau = compute_tau_rel(
    fps=torch.tensor([[12.0, 0.01], [24.0, 0.02]]),  # [batch, 2]
    transform=["raw", "centerlog1p"],
    scale=[1.0, 1.0],
    reference_fps=[1.0, 0.1]
)
# Output: [batch, 2] transformed conditions
```

---

## Training Changes

### Dataset Handling (`utils/dataset.py`)
- **Single condition:** Folder names like `12/`, `24/`, `60/` → FPS values
- **Multi-condition:** Folder names like `12_0.01_0.5/` → [FPS, shutter, bokeh]
- Validation ensures all folders have consistent condition count

### Checkpoint Compatibility
- **Loading pretrained models:** New FPS parameters initialized separately (not loaded from checkpoint)
- **Saving adapters:** FPS adapter parameters saved alongside LoRA weights (`save_adapter()` in `models/wan/wan.py:354-418`)
- **Parameter tracking:** `requires_grad=True` explicitly set for all FPS parameters

### Training Workflow
```python
# 1. Dataset provides FPS values per batch
fps_values = [12.0, 24.0, 60.0, ...]  # or [[12, 0.01], [24, 0.02], ...] for multi-cond

# 2. Transform to tau_rel
tau_rel = compute_tau_rel(fps_values, transform="log1p", ...)

# 3. Embed via FPS MLP
fps_conditioning = self.fps_conditioning(tau_rel)  # [batch, embed_dim]

# 4. Pass through transformer blocks
for block in blocks:
    x = block(x, ..., fps_conditioning=fps_conditioning)
```

---

## Key Design Decisions

### 1. Checkpoint-Safe Implementation
- **Problem:** Gradient checkpointing requires deterministic tensor shapes
- **Solution:** FPS adapter ALWAYS executes both text and FPS attention paths, even when FPS values are identical
- Ensures consistent computation graph for gradient checkpointing

### 2. Zero-Disturbance Initialization
- **Goal:** Avoid corrupting pretrained weights at training start
- **Implementation:**
  - FPS MLP final layer initialized to zero
  - LoRA "B" matrices use very small random init (1e-4) instead of zero
  - Gate starts at 0.0 (or configurable value)
- **Result:** Model starts as pretrained model, gradually learns FPS conditioning

### 3. LoRA-Style Adapters
- **Motivation:** Parameter-efficient fine-tuning
- **Architecture:** Rank-8 bottleneck by default (~10.7M params for 14B model)
- **Benefits:**
  - Small memory footprint
  - Fast training
  - Easy to save/load separately from base model

### 4. Gating Mechanism
- **Purpose:** Control FPS influence strength
- **Modes:**
  - `'fixed'`: Non-trainable constant gate (e.g., 0.5)
  - `'sigmoid'`: Learnable gate ∈ (0, 1)
  - `'identity'`: Learnable gate ∈ (-∞, ∞)
  - `'relu'`/`'silu'`/`'softplus'`: Various learnable non-linear gates
- **Warmup:** Optional gradual ramp-up from 0 to full strength over K steps

---

## Parameter Count

### FPS Conditioning Components (14B Model)
| Component | Parameters | Details |
|-----------|------------|---------|
| FPS MLP | 16,576 | 1→64→256 (configurable) |
| FPS Adapters (×11 blocks) | ~10.7M | 11 blocks × 4 LoRA matrices × rank 32 |
| **Total** | **~10.72M** | **0.076% of 14B base model** |

---

## Configuration Examples

### Single Condition (FPS Only)
```toml
[model]
type = 'wan'
fps_tau_transform = "log1p"         # String = single condition
fps_tau_scale = 0.33333334
fps_reference_fps = 240.0
fps_adapter_rank = 32
fps_condition_blocks = "deepest_third"
fps_gate_mode = 'fixed'
fps_gate_fixed_value = 0.5
```

### Multi-Condition (FPS + Shutter + Bokeh)
```toml
[model]
type = 'wan'
fps_tau_transform = ["log1p", "centerlog1p", "raw"]  # List = multi-condition
fps_tau_scale = [0.33, 1.0, 1.0]
fps_reference_fps = [240.0, 0.1, 1.0]
fps_adapter_rank = 32
fps_condition_blocks = "deepest_third"
```

Dataset structure:
```
dataset/
├── 12_0.01_0.5/      # FPS=12, shutter=0.01, bokeh=0.5
├── 24_0.02_1.0/      # FPS=24, shutter=0.02, bokeh=1.0
└── ...
```

---

## Backward Compatibility

✅ **Fully backward compatible:**
- Single-condition training with `fps_tau_transform = "log1p"` (string) works unchanged
- Dataset folders like `12/`, `24/`, `60/` work as before
- No changes needed to existing single-condition configs or checkpoints

⚠️ **Multi-condition checkpoints NOT compatible with single-condition inference:**
- Number of conditions must match between training and inference
- Condition ordering must match

---

## Testing & Validation

### Tests Implemented (`test_multicondition.py`)
1. ✅ Dataset parsing for multi-condition folders
2. ✅ `compute_tau_rel` backward compatibility (single condition)
3. ✅ `compute_tau_rel` multi-condition with list transforms
4. ✅ `FpsConditioning` single condition
5. ✅ `FpsConditioning` multi-condition (2+ inputs)
6. ✅ End-to-end pipeline integration

### Checkpoint Compatibility (`test_compatibility_check.py`)
- Verifies FPS parameters can be saved and loaded correctly
- Ensures gradient flow through FPS adapters
- Validates initialization correctness

---

## Files Modified

### Core Model Files
- `models/wan/model.py` - Added FPS conditioning classes and multi-condition support
- `models/wan/wan.py` - Integrated FPS conditioning into pipeline
- `utils/dataset.py` - Multi-condition folder parsing

### Configuration Files
- `fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml` - Single condition config
- `fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_RANDOM.toml` - Single condition with random augmentation
- `bokeh_TOML/wan_SC_TARGET_14B_SHAPES_RANDOM.toml` - Bokeh conditioning config
- Various shutter/bokeh multi-condition configs

### Testing Files
- `test_multicondition.py` - Multi-condition feature tests
- `test_compatibility_check.py` - Checkpoint compatibility tests
- `test_pipeline_integration.py` - End-to-end pipeline tests

### Documentation
- `MULTICONDITION_IMPLEMENTATION.md` - Detailed multi-condition implementation guide
- `CODE_CHECK_SUMMARY.md` - Code verification summary
- `plans/DATASET.md` - Dataset structure documentation
- `plans/run.md` - Training run plans

---

## Future Work

### Potential Improvements
1. **Attention mechanism enhancements:**
   - Cross-attention between FPS and text tokens
   - Multi-head FPS adapters with per-head gating

2. **Conditioning flexibility:**
   - Dynamic condition masking for training robustness
   - Per-condition learned scaling in adapters
   - Condition-specific adapter ranks

3. **Training strategies:**
   - Curriculum learning: start with single condition, gradually add more
   - Adversarial training for smooth conditioning manifold
   - Conditional dropout for inference-time flexibility

4. **Multi-modal conditioning:**
   - Combine FPS with semantic text conditioning
   - Hierarchical conditioning (coarse + fine control)

---

## References

### Key Code Locations
- FPS conditioning MLP: `models/wan/model.py:383-416`
- FPS adapter implementation: `models/wan/model.py:419-671`
- Transform functions: `models/wan/model.py:20-166`
- Pipeline integration: `models/wan/wan.py:614-738`
- Dataset parsing: `utils/dataset.py:509-560`

### Related Documentation
- Multi-condition implementation: `MULTICONDITION_IMPLEMENTATION.md`
- Original WAN paper: Alibaba Wan Team (2024-2025)
- LoRA paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"

---

## Contact & Maintenance

**Branch:** `sc-wan-conditioning`
**Base Branch:** `main`
**Last Updated:** 2025-10-16

For questions or issues, refer to:
- Code comments in `models/wan/model.py` and `models/wan/wan.py`
- Test files for usage examples
- `MULTICONDITION_IMPLEMENTATION.md` for detailed implementation notes
