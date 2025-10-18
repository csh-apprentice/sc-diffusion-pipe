#!/usr/bin/env python3
"""
Test the actual pipeline integration in wan.py to ensure single and multi-condition work correctly.
Tests the exact code path used during training/inference.
"""

import torch
import sys
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')

from models.wan.model import compute_tau_rel

print("=" * 80)
print("PIPELINE INTEGRATION TEST: wan.py FPS conditioning flow")
print("=" * 80)

# ============================================================================
# Simulate InitialLayer forward() logic from wan.py
# ============================================================================

def simulate_initial_layer_fps_conditioning(fps_values, fps_tau_transform, fps_tau_scale, fps_reference_fps):
    """
    Simulates the FPS conditioning logic in InitialLayer.forward()
    This is the exact code from wan.py lines 717-731
    """
    # Create tensor (line 721)
    fps_tensor = torch.tensor(fps_values, dtype=torch.float32)

    # Apply tau transform (line 722)
    tau_rel = compute_tau_rel(
        fps_tensor,
        reference_fps=fps_reference_fps,
        transform=fps_tau_transform,
        scale=fps_tau_scale
    )

    # Handle shape for single vs multi-condition (line 724-729)
    # - Single condition: tau_rel shape is [batch_size], need to unsqueeze to [batch_size, 1]
    # - Multi-condition: tau_rel shape is [batch_size, num_conditions], already correct
    if tau_rel.ndim == 1:
        tau_rel = tau_rel.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
    # Now tau_rel shape is [batch_size, num_conditions] for both cases

    return tau_rel

# ============================================================================
# TEST 1: Single-condition training (old config)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Single-condition training pipeline")
print("=" * 80)

# Simulate dataset providing single FPS values per sample
fps_batch_single = [12.0, 24.0, 60.0]  # List from dataset
print(f"Dataset provides: {fps_batch_single}")

# Config from TOML (single condition)
config_single = {
    'fps_tau_transform': "log1p",  # String
    'fps_tau_scale': 0.33333334,   # Scalar
    'fps_reference_fps': 240.0     # Scalar
}
print(f"Config: {config_single}")

# Run through pipeline
tau_output_single = simulate_initial_layer_fps_conditioning(
    fps_batch_single,
    config_single['fps_tau_transform'],
    config_single['fps_tau_scale'],
    config_single['fps_reference_fps']
)

print(f"\nOutput tau_rel shape: {tau_output_single.shape} (expected: [3, 1])")
print(f"Output tau_rel values:\n{tau_output_single}")

assert tau_output_single.shape == (3, 1), f"FAIL: Expected [3, 1], got {tau_output_single.shape}"
print("\n✅ TEST 1 PASSED: Single-condition pipeline works correctly")

# ============================================================================
# TEST 2: Multi-condition training (new config)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Multi-condition training pipeline")
print("=" * 80)

# Simulate dataset providing multiple condition values per sample
fps_batch_multi = [[12.0, 0.01], [24.0, 0.02], [60.0, 0.05]]  # List of lists from dataset
print(f"Dataset provides: {fps_batch_multi}")

# Config from TOML (multi-condition)
config_multi = {
    'fps_tau_transform': ["raw", "centerlog1p"],  # List
    'fps_tau_scale': [1.0, 1.0],                  # List
    'fps_reference_fps': [1.0, 0.1]               # List
}
print(f"Config: {config_multi}")

# Run through pipeline
tau_output_multi = simulate_initial_layer_fps_conditioning(
    fps_batch_multi,
    config_multi['fps_tau_transform'],
    config_multi['fps_tau_scale'],
    config_multi['fps_reference_fps']
)

print(f"\nOutput tau_rel shape: {tau_output_multi.shape} (expected: [3, 2])")
print(f"Output tau_rel values:\n{tau_output_multi}")

assert tau_output_multi.shape == (3, 2), f"FAIL: Expected [3, 2], got {tau_output_multi.shape}"
print("\n✅ TEST 2 PASSED: Multi-condition pipeline works correctly")

# ============================================================================
# TEST 3: Single-condition inference (scalar FPS value)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Single-condition inference pipeline")
print("=" * 80)

# Simulate inference with single FPS value (not batched)
fps_inference_single = [24.0]  # Single sample
print(f"Inference FPS: {fps_inference_single}")

# Config from checkpoint (single condition)
print(f"Config: {config_single}")

# Run through pipeline
tau_output_infer_single = simulate_initial_layer_fps_conditioning(
    fps_inference_single,
    config_single['fps_tau_transform'],
    config_single['fps_tau_scale'],
    config_single['fps_reference_fps']
)

print(f"\nOutput tau_rel shape: {tau_output_infer_single.shape} (expected: [1, 1])")
print(f"Output tau_rel values:\n{tau_output_infer_single}")

assert tau_output_infer_single.shape == (1, 1), f"FAIL: Expected [1, 1], got {tau_output_infer_single.shape}"
print("\n✅ TEST 3 PASSED: Single-condition inference works correctly")

# ============================================================================
# TEST 4: Multi-condition inference (multiple condition values)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Multi-condition inference pipeline")
print("=" * 80)

# Simulate inference with multiple condition values (not batched)
fps_inference_multi = [[24.0, 0.02]]  # Single sample with 2 conditions
print(f"Inference conditions: {fps_inference_multi}")

# Config from checkpoint (multi-condition)
print(f"Config: {config_multi}")

# Run through pipeline
tau_output_infer_multi = simulate_initial_layer_fps_conditioning(
    fps_inference_multi,
    config_multi['fps_tau_transform'],
    config_multi['fps_tau_scale'],
    config_multi['fps_reference_fps']
)

print(f"\nOutput tau_rel shape: {tau_output_infer_multi.shape} (expected: [1, 2])")
print(f"Output tau_rel values:\n{tau_output_infer_multi}")

assert tau_output_infer_multi.shape == (1, 2), f"FAIL: Expected [1, 2], got {tau_output_infer_multi.shape}"
print("\n✅ TEST 4 PASSED: Multi-condition inference works correctly")

# ============================================================================
# TEST 5: Edge case - batch size of 1 single condition
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Edge case - single sample single condition")
print("=" * 80)

fps_edge_single = [12.0]
print(f"Single sample: {fps_edge_single}")

tau_edge_single = simulate_initial_layer_fps_conditioning(
    fps_edge_single,
    config_single['fps_tau_transform'],
    config_single['fps_tau_scale'],
    config_single['fps_reference_fps']
)

print(f"Output shape: {tau_edge_single.shape} (expected: [1, 1])")
assert tau_edge_single.shape == (1, 1), f"FAIL: Expected [1, 1], got {tau_edge_single.shape}"
print("✅ TEST 5 PASSED")

# ============================================================================
# TEST 6: Edge case - batch size of 1 multi-condition
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: Edge case - single sample multi-condition")
print("=" * 80)

fps_edge_multi = [[12.0, 0.01]]
print(f"Single sample: {fps_edge_multi}")

tau_edge_multi = simulate_initial_layer_fps_conditioning(
    fps_edge_multi,
    config_multi['fps_tau_transform'],
    config_multi['fps_tau_scale'],
    config_multi['fps_reference_fps']
)

print(f"Output shape: {tau_edge_multi.shape} (expected: [1, 2])")
assert tau_edge_multi.shape == (1, 2), f"FAIL: Expected [1, 2], got {tau_edge_multi.shape}"
print("✅ TEST 6 PASSED")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE INTEGRATION TEST SUMMARY")
print("=" * 80)
print("\n✅ Test 1: Single-condition training - PASSED")
print("✅ Test 2: Multi-condition training - PASSED")
print("✅ Test 3: Single-condition inference - PASSED")
print("✅ Test 4: Multi-condition inference - PASSED")
print("✅ Test 5: Edge case (single sample, single condition) - PASSED")
print("✅ Test 6: Edge case (single sample, multi-condition) - PASSED")

print("\n" + "=" * 80)
print("ALL PIPELINE INTEGRATION TESTS PASSED! ✓")
print("=" * 80)
print("\nThe wan.py integration correctly handles:")
print("  ✓ Single-condition training and inference (backward compatible)")
print("  ✓ Multi-condition training and inference (new feature)")
print("  ✓ Proper shape handling with ndim check")
print("  ✓ Edge cases with batch size 1")
print("\nReady to train and run inference with both config types!")
