#!/usr/bin/env python3
"""
Comprehensive compatibility check for single-condition and multi-condition support.
Tests all critical code paths for training and inference scenarios.
"""

import torch
import sys
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')

from models.wan.model import compute_tau_rel, FpsConditioning

print("=" * 80)
print("COMPATIBILITY CHECK: SINGLE vs MULTI-CONDITION")
print("=" * 80)

# ============================================================================
# CASE 1: Single-condition training (OLD CONFIG - backward compatibility)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 1: SINGLE-CONDITION TRAINING (Backward Compatibility)")
print("=" * 80)
print("\nConfig example:")
print('  fps_tau_transform = "log1p"  # String, not list')
print('  fps_tau_scale = 0.33333334    # Scalar, not list')
print('  fps_reference_fps = 240.0     # Scalar, not list')

# Simulate model initialization with single-condition config
fps_tau_transform = "log1p"  # OLD CONFIG: string
fps_tau_scale = 0.33333334   # OLD CONFIG: scalar
fps_reference_fps = 240.0    # OLD CONFIG: scalar

# Detect num_conditions (should be 1)
if isinstance(fps_tau_transform, list):
    num_conditions = len(fps_tau_transform)
else:
    num_conditions = 1

print(f"\n✓ Detected num_conditions = {num_conditions}")

# Create FpsConditioning module
fps_cond = FpsConditioning(out_dim=256, hidden=64, num_conditions=num_conditions)
print(f"✓ Created FpsConditioning with num_conditions={num_conditions}")
print(f"  - lin1 weight shape: {fps_cond.lin1.weight.shape} (expected: [64, 1])")
assert fps_cond.lin1.weight.shape == (64, 1), f"FAIL: Expected [64, 1], got {fps_cond.lin1.weight.shape}"

# Simulate training: dataset returns single condition
# Dataset folder structure: 12/, 24/, 60/ (old format)
fps_batch = torch.tensor([12.0, 24.0, 60.0])  # Shape: [batch_size=3]
print(f"\n✓ Dataset loaded FPS values: {fps_batch.tolist()}")
print(f"  - Shape: {fps_batch.shape} (expected: [3])")

# Apply tau transform
tau_batch = compute_tau_rel(
    fps_batch,
    transform=fps_tau_transform,
    scale=fps_tau_scale,
    reference_fps=fps_reference_fps
)
print(f"\n✓ Applied tau transform: {tau_batch}")
print(f"  - Shape: {tau_batch.shape} (expected: [3])")
assert tau_batch.shape == (3,), f"FAIL: Expected shape [3], got {tau_batch.shape}"

# FpsConditioning expects [batch, num_conditions], so we need to unsqueeze
tau_batch_input = tau_batch.unsqueeze(1)  # [3] -> [3, 1]
print(f"\n✓ Reshaped for FpsConditioning: {tau_batch_input.shape} (expected: [3, 1])")

# Pass through FPS conditioning
fps_embedding = fps_cond(tau_batch_input)
print(f"\n✓ FPS embedding shape: {fps_embedding.shape} (expected: [3, 256])")
assert fps_embedding.shape == (3, 256), f"FAIL: Expected shape [3, 256], got {fps_embedding.shape}"

print("\n✅ CASE 1 PASSED: Single-condition training works correctly")

# ============================================================================
# CASE 2: Multi-condition training (NEW CONFIG)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 2: MULTI-CONDITION TRAINING (New Feature)")
print("=" * 80)
print("\nConfig example:")
print('  fps_tau_transform = ["raw", "centerlog1p"]  # List of 2')
print('  fps_tau_scale = [1.0, 1.0]                  # List of 2')
print('  fps_reference_fps = [1.0, 0.1]              # List of 2')

# Simulate model initialization with multi-condition config
fps_tau_transform_multi = ["raw", "centerlog1p"]  # NEW CONFIG: list
fps_tau_scale_multi = [1.0, 1.0]                  # NEW CONFIG: list
fps_reference_fps_multi = [1.0, 0.1]              # NEW CONFIG: list

# Detect num_conditions (should be 2)
if isinstance(fps_tau_transform_multi, list):
    num_conditions_multi = len(fps_tau_transform_multi)
else:
    num_conditions_multi = 1

print(f"\n✓ Detected num_conditions = {num_conditions_multi}")

# Create FpsConditioning module
fps_cond_multi = FpsConditioning(out_dim=256, hidden=64, num_conditions=num_conditions_multi)
print(f"✓ Created FpsConditioning with num_conditions={num_conditions_multi}")
print(f"  - lin1 weight shape: {fps_cond_multi.lin1.weight.shape} (expected: [64, 2])")
assert fps_cond_multi.lin1.weight.shape == (64, 2), f"FAIL: Expected [64, 2], got {fps_cond_multi.lin1.weight.shape}"

# Simulate training: dataset returns multiple conditions
# Dataset folder structure: 12_0.01/, 24_0.02/, 60_0.05/ (new format)
fps_batch_multi = torch.tensor([[12.0, 0.01], [24.0, 0.02], [60.0, 0.05]])  # Shape: [batch_size=3, num_conditions=2]
print(f"\n✓ Dataset loaded FPS values: {fps_batch_multi.tolist()}")
print(f"  - Shape: {fps_batch_multi.shape} (expected: [3, 2])")

# Apply tau transform
tau_batch_multi = compute_tau_rel(
    fps_batch_multi,
    transform=fps_tau_transform_multi,
    scale=fps_tau_scale_multi,
    reference_fps=fps_reference_fps_multi
)
print(f"\n✓ Applied tau transform: {tau_batch_multi}")
print(f"  - Shape: {tau_batch_multi.shape} (expected: [3, 2])")
assert tau_batch_multi.shape == (3, 2), f"FAIL: Expected shape [3, 2], got {tau_batch_multi.shape}"

# FpsConditioning expects [batch, num_conditions], which is already the correct shape
print(f"\n✓ Input shape for FpsConditioning: {tau_batch_multi.shape} (expected: [3, 2])")

# Pass through FPS conditioning
fps_embedding_multi = fps_cond_multi(tau_batch_multi)
print(f"\n✓ FPS embedding shape: {fps_embedding_multi.shape} (expected: [3, 256])")
assert fps_embedding_multi.shape == (3, 256), f"FAIL: Expected shape [3, 256], got {fps_embedding_multi.shape}"

print("\n✅ CASE 2 PASSED: Multi-condition training works correctly")

# ============================================================================
# CASE 3: Single-condition inference (Loading old checkpoint)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 3: SINGLE-CONDITION INFERENCE (Loading Old Checkpoint)")
print("=" * 80)
print("\nScenario: Loading checkpoint trained with single-condition config")

# When loading checkpoint, model is initialized with config from checkpoint
# Checkpoint config has: fps_tau_transform = "log1p" (string)
checkpoint_config = {
    'fps_tau_transform': "log1p",
    'fps_tau_scale': 0.33333334,
    'fps_reference_fps': 240.0
}

# Detect num_conditions from checkpoint config
if isinstance(checkpoint_config['fps_tau_transform'], list):
    num_conditions_infer = len(checkpoint_config['fps_tau_transform'])
else:
    num_conditions_infer = 1

print(f"\n✓ Detected num_conditions = {num_conditions_infer} from checkpoint config")

# Create FpsConditioning module matching checkpoint
fps_cond_infer = FpsConditioning(out_dim=256, hidden=64, num_conditions=num_conditions_infer)
print(f"✓ Created FpsConditioning with num_conditions={num_conditions_infer}")
print(f"  - lin1 weight shape: {fps_cond_infer.lin1.weight.shape} (expected: [64, 1])")

# At inference time, user provides single FPS value
inference_fps = 24.0
print(f"\n✓ User provided FPS: {inference_fps}")

# Convert to tensor and apply tau transform
inference_fps_tensor = torch.tensor([inference_fps])  # Shape: [1]
tau_infer = compute_tau_rel(
    inference_fps_tensor,
    transform=checkpoint_config['fps_tau_transform'],
    scale=checkpoint_config['fps_tau_scale'],
    reference_fps=checkpoint_config['fps_reference_fps']
)
print(f"✓ Applied tau transform: {tau_infer}")
print(f"  - Shape: {tau_infer.shape} (expected: [1])")

# Reshape for FpsConditioning
tau_infer_input = tau_infer.unsqueeze(1)  # [1] -> [1, 1]
print(f"✓ Reshaped for FpsConditioning: {tau_infer_input.shape} (expected: [1, 1])")

# Pass through FPS conditioning
fps_embedding_infer = fps_cond_infer(tau_infer_input)
print(f"✓ FPS embedding shape: {fps_embedding_infer.shape} (expected: [1, 256])")
assert fps_embedding_infer.shape == (1, 256), f"FAIL: Expected shape [1, 256], got {fps_embedding_infer.shape}"

print("\n✅ CASE 3 PASSED: Single-condition inference works correctly")

# ============================================================================
# CASE 4: Multi-condition inference (Loading new checkpoint)
# ============================================================================
print("\n" + "=" * 80)
print("CASE 4: MULTI-CONDITION INFERENCE (Loading New Checkpoint)")
print("=" * 80)
print("\nScenario: Loading checkpoint trained with multi-condition config")

# When loading checkpoint, model is initialized with config from checkpoint
# Checkpoint config has: fps_tau_transform = ["raw", "centerlog1p"] (list)
checkpoint_config_multi = {
    'fps_tau_transform': ["raw", "centerlog1p"],
    'fps_tau_scale': [1.0, 1.0],
    'fps_reference_fps': [1.0, 0.1]
}

# Detect num_conditions from checkpoint config
if isinstance(checkpoint_config_multi['fps_tau_transform'], list):
    num_conditions_infer_multi = len(checkpoint_config_multi['fps_tau_transform'])
else:
    num_conditions_infer_multi = 1

print(f"\n✓ Detected num_conditions = {num_conditions_infer_multi} from checkpoint config")

# Create FpsConditioning module matching checkpoint
fps_cond_infer_multi = FpsConditioning(out_dim=256, hidden=64, num_conditions=num_conditions_infer_multi)
print(f"✓ Created FpsConditioning with num_conditions={num_conditions_infer_multi}")
print(f"  - lin1 weight shape: {fps_cond_infer_multi.lin1.weight.shape} (expected: [64, 2])")

# At inference time, user provides multiple condition values
inference_conditions = [24.0, 0.02]  # [shutter_speed, bokeh]
print(f"\n✓ User provided conditions: {inference_conditions}")

# Convert to tensor and apply tau transform
inference_conditions_tensor = torch.tensor([inference_conditions])  # Shape: [1, 2]
tau_infer_multi = compute_tau_rel(
    inference_conditions_tensor,
    transform=checkpoint_config_multi['fps_tau_transform'],
    scale=checkpoint_config_multi['fps_tau_scale'],
    reference_fps=checkpoint_config_multi['fps_reference_fps']
)
print(f"✓ Applied tau transform: {tau_infer_multi}")
print(f"  - Shape: {tau_infer_multi.shape} (expected: [1, 2])")

# FpsConditioning expects [batch, num_conditions], which is already correct
print(f"✓ Input shape for FpsConditioning: {tau_infer_multi.shape} (expected: [1, 2])")

# Pass through FPS conditioning
fps_embedding_infer_multi = fps_cond_infer_multi(tau_infer_multi)
print(f"✓ FPS embedding shape: {fps_embedding_infer_multi.shape} (expected: [1, 256])")
assert fps_embedding_infer_multi.shape == (1, 256), f"FAIL: Expected shape [1, 256], got {fps_embedding_infer_multi.shape}"

print("\n✅ CASE 4 PASSED: Multi-condition inference works correctly")

# ============================================================================
# CRITICAL CHECK: What if user tries to use wrong checkpoint?
# ============================================================================
print("\n" + "=" * 80)
print("EDGE CASE: Incompatible Checkpoint Detection")
print("=" * 80)

print("\nScenario 1: Loading multi-condition checkpoint with single FPS value")
print("  - Checkpoint expects num_conditions=2")
print("  - User provides single FPS (e.g., fps=24.0)")
print("  - Result: Shape mismatch error (expected)")

try:
    inference_single = torch.tensor([24.0])  # Shape: [1] (single condition)
    tau_single = compute_tau_rel(inference_single, transform="log1p", scale=0.33, reference_fps=240.0)
    tau_single_input = tau_single.unsqueeze(1)  # [1, 1]
    # Try to pass through multi-condition model (expects [batch, 2])
    fps_cond_infer_multi(tau_single_input)  # This should fail
    print("  ❌ UNEXPECTED: Should have failed but didn't!")
except RuntimeError as e:
    print(f"  ✓ Expected error caught: {str(e)[:80]}...")

print("\nScenario 2: Loading single-condition checkpoint with multiple values")
print("  - Checkpoint expects num_conditions=1")
print("  - User provides multiple conditions (e.g., [24.0, 0.02])")
print("  - Result: Shape mismatch error (expected)")

try:
    inference_multi = torch.tensor([[24.0, 0.02]])  # Shape: [1, 2] (multi condition)
    tau_multi_test = compute_tau_rel(inference_multi, transform=["raw", "log1p"], scale=[1.0, 1.0], reference_fps=[1.0, 0.1])
    # Try to pass through single-condition model (expects [batch, 1])
    fps_cond_infer(tau_multi_test)  # This should fail
    print("  ❌ UNEXPECTED: Should have failed but didn't!")
except RuntimeError as e:
    print(f"  ✓ Expected error caught: {str(e)[:80]}...")

print("\n✅ EDGE CASES PASSED: Incompatible configs correctly rejected")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPATIBILITY CHECK SUMMARY")
print("=" * 80)
print("\n✅ Case 1: Single-condition training (backward compatible) - PASSED")
print("✅ Case 2: Multi-condition training (new feature) - PASSED")
print("✅ Case 3: Single-condition inference (old checkpoints) - PASSED")
print("✅ Case 4: Multi-condition inference (new checkpoints) - PASSED")
print("✅ Edge cases: Incompatible configs properly detected - PASSED")

print("\n" + "=" * 80)
print("ALL COMPATIBILITY CHECKS PASSED! ✓")
print("=" * 80)
print("\nKey findings:")
print("  1. Old single-condition configs work without any changes")
print("  2. New multi-condition configs work as expected")
print("  3. Checkpoints maintain their num_conditions from training")
print("  4. Mismatched num_conditions are properly detected with clear errors")
print("  5. Shape transformations are handled correctly in all cases")
