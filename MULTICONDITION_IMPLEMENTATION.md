# Multi-Condition Support Implementation

## Overview
Successfully implemented multi-condition support for the FPS conditioning system, allowing training with multiple scalar conditions (e.g., FPS + shutter speed + bokeh).

## Implementation Date
2025-10-13

## Changes Made

### 1. Dataset Parsing (`utils/dataset.py`, lines 509-560)
**What changed:**
- Extended folder name parsing to support multi-condition format
- Single condition: `12/` → parsed as `[12.0]`
- Multi-condition: `12_0.1/` → parsed as `[12.0, 0.1]`
- Validates all folders have consistent number of conditions

**Example:**
```python
# Old: /dataset/12/video.mp4           -> fps = [12.0]
# New: /dataset/12_0.01/video.mp4      -> conditions = [12.0, 0.01]
#                                         (shutter speed, bokeh)
```

### 2. Tau Transform Function (`models/wan/model.py`, lines 20-160)
**What changed:**
- `compute_tau_rel()` now accepts list parameters for `transform`, `scale`, `reference_fps`
- Detects single vs multi-condition based on input tensor shape
- Single condition: `fps` has shape `[batch]` → backward compatible
- Multi-condition: `fps` has shape `[batch, num_conditions]` → applies per-condition transforms

**Example:**
```python
# Single condition (backward compatible)
fps = torch.tensor([12.0, 24.0])  # Shape: [2]
tau = compute_tau_rel(fps, transform="log1p", scale=0.33, reference_fps=240.0)
# Output shape: [2]

# Multi-condition (new feature)
fps = torch.tensor([[12.0, 0.01], [24.0, 0.02]])  # Shape: [2, 2]
tau = compute_tau_rel(
    fps,
    transform=["raw", "centerlog1p"],  # Different transform per condition
    scale=[1.0, 1.0],
    reference_fps=[1.0, 0.1]
)
# Output shape: [2, 2]
```

### 3. FPS Conditioning MLP (`models/wan/model.py`, lines 377-410)
**What changed:**
- Added `num_conditions` parameter to `FpsConditioning.__init__()`
- Changed input layer from `nn.Linear(1, hidden)` to `nn.Linear(num_conditions, hidden)`
- Defaults to `num_conditions=1` for backward compatibility

**Example:**
```python
# Single condition (backward compatible)
fps_cond = FpsConditioning(out_dim=256, hidden=64, num_conditions=1)
# Input shape: [batch, 1] → Output shape: [batch, 256]

# Multi-condition (new feature)
fps_cond = FpsConditioning(out_dim=256, hidden=64, num_conditions=2)
# Input shape: [batch, 2] → Output shape: [batch, 256]
```

### 4. Model Initialization (`models/wan/model.py`, lines 950-975)
**What changed:**
- Auto-detects number of conditions from `fps_tau_transform` parameter
- If `fps_tau_transform` is a list, its length determines `num_conditions`
- Passes `num_conditions` to `FpsConditioning` constructor

**Example:**
```python
# Single condition (backward compatible)
fps_tau_transform = "log1p"  # String → num_conditions = 1

# Multi-condition (new feature)
fps_tau_transform = ["raw", "centerlog1p"]  # List of 2 → num_conditions = 2
```

## Backward Compatibility

✅ **Single-condition training remains unchanged:**
- All existing configs with `fps_tau_transform = "log1p"` work without modification
- Dataset folders like `12/`, `24/`, `60/` still work correctly
- No changes required to existing training scripts or checkpoints

## Configuration Example

### Multi-Condition Config (2 conditions: shutter + bokeh)
```toml
[model]
type = 'wan'
# Multi-condition transforms (list of 2)
fps_tau_transform = ["raw", "centerlog1p"]
fps_tau_scale = [1.0, 1.0]
fps_reference_fps = [1.0, 0.1]

# Other FPS settings remain the same
fps_adapter_rank = 32
fps_condition_blocks = "deepest_third"
```

### Dataset Structure
```
dataset/2shapes_shutter_bokeh/
├── 0_0.01/          # Condition 1=0.0, Condition 2=0.01
│   ├── video1.mp4
│   └── video1.txt
├── 0.125_0.02/      # Condition 1=0.125, Condition 2=0.02
│   ├── video2.mp4
│   └── video2.txt
└── ...
```

## Testing

All tests passed successfully (`test_multicondition.py`):

✅ **Test 1:** Dataset parsing for multi-condition folders
✅ **Test 2:** `compute_tau_rel` single condition (backward compatibility)
✅ **Test 3:** `compute_tau_rel` multi-condition with list transforms
✅ **Test 4:** `FpsConditioning` single condition (backward compatibility)
✅ **Test 5:** `FpsConditioning` multi-condition with 2 inputs
✅ **Test 6:** End-to-end pipeline (raw FPS → tau → embedding)

## Ready to Train

Configuration: `/root/workspace/sc-diffusion-pipe/shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH.toml`

Dataset: `/root/workspace/sc-diffusion-pipe/dataset/2shapes_shutter_bokeh`

Training command:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe
torchrun --nproc_per_node=1 train.py \
    --config shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH.toml
```

## Technical Details

### Tensor Flow
1. **Dataset** loads conditions from folder names: `[batch, num_conditions]`
2. **compute_tau_rel** transforms each condition separately: `[batch, num_conditions]`
3. **FpsConditioning** MLP embeds conditions: `[batch, num_conditions]` → `[batch, embed_dim]`
4. **FPS Adapters** receive embeddings: `[batch, embed_dim]` → modulate attention

### Memory Impact
- Minimal increase: only `num_conditions` input weights in `FpsConditioning.lin1`
- Single condition: `1 × hidden` weights
- Two conditions: `2 × hidden` weights
- Example: `64 hidden → 64 params (1 cond)` vs `128 params (2 cond)` (+64 params)

### Design Principles
1. **Backward compatibility:** Single-condition training unchanged
2. **Flexibility:** Each condition can have its own transform/scale/reference
3. **Robustness:** Validates consistent number of conditions across dataset
4. **Simplicity:** Auto-detection from config, no manual specification needed

## Future Extensions

Potential improvements:
- Support 3+ conditions (already supported, just needs config)
- Per-condition learned scaling in FPS adapters
- Condition-specific adapter ranks
- Dynamic condition masking for training flexibility

## Notes

- All existing single-condition checkpoints remain compatible
- Multi-condition checkpoints are NOT backward compatible with single-condition inference
- Condition ordering matters: must match training order during inference
- Folder name format: conditions separated by underscores, e.g., `cond1_cond2_cond3`
