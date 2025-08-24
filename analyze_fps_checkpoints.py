#!/usr/bin/env python3

import sys
import safetensors.torch
import torch
from collections import defaultdict

def analyze_checkpoint(file_path, epoch_name):
    """Analyze FPS parameters in a checkpoint file"""
    print(f"\n=== ANALYZING {epoch_name} ===")
    
    # Load the checkpoint
    try:
        state_dict = safetensors.torch.load_file(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    print(f"Total parameters in checkpoint: {len(state_dict)}")
    
    # Categorize parameters
    fps_mlp_params = {}
    fps_adapter_params = {}
    regular_lora_params = {}
    
    for key, tensor in state_dict.items():
        if 'fps_conditioning' in key:
            fps_mlp_params[key] = tensor
        elif 'fps_adapter' in key:
            fps_adapter_params[key] = tensor
        else:
            regular_lora_params[key] = tensor
    
    # Count parameters by type
    fps_mlp_count = sum(p.numel() for p in fps_mlp_params.values())
    fps_adapter_count = sum(p.numel() for p in fps_adapter_params.values())
    regular_lora_count = sum(p.numel() for p in regular_lora_params.values())
    
    print(f"FPS MLP parameters: {len(fps_mlp_params)} tensors, {fps_mlp_count:,} total params")
    print(f"FPS Adapter parameters: {len(fps_adapter_params)} tensors, {fps_adapter_count:,} total params")
    print(f"Regular LoRA parameters: {len(regular_lora_params)} tensors, {regular_lora_count:,} total params")
    print(f"Total: {fps_mlp_count + fps_adapter_count + regular_lora_count:,} parameters")
    
    # Show FPS MLP parameter details
    if fps_mlp_params:
        print(f"\nFPS MLP Parameters:")
        for key, tensor in sorted(fps_mlp_params.items()):
            print(f"  {key}: {tensor.shape} ({tensor.numel():,} params)")
    
    # Show sample FPS adapter parameters (first few blocks)
    if fps_adapter_params:
        print(f"\nFPS Adapter Parameters (sample from first few blocks):")
        block_params = defaultdict(list)
        for key in sorted(fps_adapter_params.keys()):
            # Extract block number from key like 'diffusion_model.blocks.27.fps_adapter.gate_alpha'
            parts = key.split('.')
            if 'blocks' in parts:
                block_idx = int(parts[parts.index('blocks') + 1])
                param_name = parts[-1]  # gate_alpha, k_fps_down.weight, etc.
                block_params[block_idx].append((param_name, fps_adapter_params[key]))
        
        # Show first 3 blocks as examples
        sample_blocks = sorted(block_params.keys())[:3]
        for block_idx in sample_blocks:
            print(f"  Block {block_idx}:")
            for param_name, tensor in sorted(block_params[block_idx], key=lambda x: x[0]):
                print(f"    {param_name}: {tensor.shape} ({tensor.numel():,} params)")
        
        if len(block_params) > 3:
            print(f"  ... and {len(block_params) - 3} more blocks with FPS adapters")
    
    return {
        'fps_mlp': fps_mlp_params,
        'fps_adapter': fps_adapter_params,
        'regular_lora': regular_lora_params,
        'counts': (fps_mlp_count, fps_adapter_count, regular_lora_count)
    }

def compare_checkpoints(data1, data2, epoch1_name, epoch2_name):
    """Compare two checkpoint datasets to see if parameters changed"""
    print(f"\n=== COMPARING {epoch1_name} vs {epoch2_name} ===")
    
    # Compare FPS MLP parameters
    print("FPS MLP Parameter Changes:")
    for key in sorted(data1['fps_mlp'].keys()):
        if key in data2['fps_mlp']:
            tensor1 = data1['fps_mlp'][key]
            tensor2 = data2['fps_mlp'][key]
            if torch.allclose(tensor1, tensor2, atol=1e-6):
                print(f"  {key}: NO CHANGE")
            else:
                diff_norm = (tensor2 - tensor1).norm().item()
                tensor1_norm = tensor1.norm().item()
                relative_change = diff_norm / tensor1_norm if tensor1_norm > 0 else float('inf')
                print(f"  {key}: CHANGED (rel_change: {relative_change:.6f})")
    
    # Compare sample FPS adapter parameters (first 2 blocks)
    print("FPS Adapter Parameter Changes (sample):")
    sample_keys = [k for k in sorted(data1['fps_adapter'].keys()) if 'blocks.27' in k or 'blocks.28' in k]
    for key in sample_keys:
        if key in data2['fps_adapter']:
            tensor1 = data1['fps_adapter'][key]
            tensor2 = data2['fps_adapter'][key]
            if torch.allclose(tensor1, tensor2, atol=1e-6):
                print(f"  {key}: NO CHANGE")
            else:
                diff_norm = (tensor2 - tensor1).norm().item()
                tensor1_norm = tensor1.norm().item()
                relative_change = diff_norm / tensor1_norm if tensor1_norm > 0 else float('inf')
                print(f"  {key}: CHANGED (rel_change: {relative_change:.6f})")

if __name__ == "__main__":
    epoch1_file = "checkpoints/20250822_01-48-59/epoch1/adapter_model.safetensors"
    epoch2_file = "checkpoints/20250822_01-48-59/epoch2/adapter_model.safetensors"
    
    # Analyze both checkpoints
    data1 = analyze_checkpoint(epoch1_file, "EPOCH 1")
    data2 = analyze_checkpoint(epoch2_file, "EPOCH 2")
    
    if data1 and data2:
        compare_checkpoints(data1, data2, "EPOCH 1", "EPOCH 2")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Expected FPS MLP params: ~2560 (Linear(1,64) + LayerNorm(64) + Linear(64,5120) = 64+64+64*5120+5120 = 393,472)")
        print(f"Expected FPS Adapter params (rank 4): ~1,064,973 (81,921 per block × 13 blocks)")
        print(f"Found FPS MLP params: {data1['counts'][0]:,}")
        print(f"Found FPS Adapter params: {data1['counts'][1]:,}")
        print(f"Found Regular LoRA params: {data1['counts'][2]:,}")
        
        if data1['counts'][0] > 0:
            print("✅ FPS MLP parameters are being saved!")
        else:
            print("❌ FPS MLP parameters are missing!")
            
        if data1['counts'][1] > 1000000:  # Expect ~1M params
            print("✅ FPS Adapter parameters are being saved!")
        else:
            print("❌ FPS Adapter parameters are missing!")