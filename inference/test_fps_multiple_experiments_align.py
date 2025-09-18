#!/usr/bin/env python3
"""
Multi-experiment FPS conditioning test script - TOML-aligned version.
Tests different FPS values to verify FPS conditioning works correctly.
Uses TOML configuration files for proper training/inference alignment.
"""

import torch
import os
import sys
import logging
from datetime import datetime
import deepspeed
from tqdm import tqdm
import math
import argparse
import toml
import json

# Add the project root to Python path
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')

from models.wan.wan import WanPipeline
import peft
from inference_utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from inference_utils.utils import cache_video
from utils.common import DTYPE_MAP

def setup_environment(port='29501'):
    """Initializes the DeepSpeed distributed environment for standalone use."""
    if not deepspeed.comm.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', port)
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        deepspeed.init_distributed()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logging.info(f"DeepSpeed environment initialized on port {port}.")

def load_pipeline_from_toml(toml_path):
    """Loads WanPipeline using TOML configuration (training-aligned)."""
    logging.info(f"Loading TOML configuration: {toml_path}")
    
    # Load TOML exactly like train.py does
    with open(toml_path) as f:
        config = json.loads(json.dumps(toml.load(f)))
    
    # Convert dtype strings to torch.dtype objects (like train.py does)
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)
    
    logging.info("Initializing WanPipeline with TOML configuration...")
    logging.info(f"Model type: {config['model']['type']}")
    logging.info(f"Model checkpoint: {config['model']['ckpt_path']}")
    logging.info(f"Model dtype: {model_dtype_str} -> {config['model']['dtype']}")
    
    # Log ALL FPS configuration from TOML (must match wan.py exactly)
    model_config = config['model']
    fps_settings = {
        'fps_adapter_rank': model_config.get('fps_adapter_rank'),
        'fps_adapter_gate_init': model_config.get('fps_adapter_gate_init'),
        'fps_condition_blocks': model_config.get('fps_condition_blocks'),
        'fps_adapter_num_tokens': model_config.get('fps_adapter_num_tokens'),
        'fps_tau_transform': model_config.get('fps_tau_transform'),
        'fps_tau_scale': model_config.get('fps_tau_scale'),
        'fps_embed_dim': model_config.get('fps_embed_dim'),
        'fps_condition_hidden': model_config.get('fps_condition_hidden'),
        'fps_lora_alpha': model_config.get('fps_lora_alpha'),
        'fps_gate_mode': model_config.get('fps_gate_mode'),
        'fps_gate_fixed_value': model_config.get('fps_gate_fixed_value')
    }
    logging.info("Complete FPS configuration from TOML (aligned with wan.py):")
    for key, value in fps_settings.items():
        if value is not None:
            logging.info(f"  {key}: {value}")
        else:
            # Show which settings are missing and will use defaults
            logging.info(f"  {key}: <not set, will use wan.py default>")
    
    # Initialize pipeline using TOML config (same as train.py)
    wan_t2v = WanPipeline(config)
    
    logging.info("Loading main transformer model...")
    wan_t2v.load_diffusion_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Moving model components to device: {device}")
    
    wan_t2v.transformer.to(device)
    wan_t2v.vae.model.to(device)
    wan_t2v.text_encoder.model.to(device)
    
    # SUBTASK 3: Set models to eval mode for sampling (critical for proper inference)
    logging.info("Setting models to eval mode for inference...")
    wan_t2v.transformer.eval()
    wan_t2v.vae.model.eval()
    wan_t2v.text_encoder.model.eval()
    
    logging.info("Pipeline loaded successfully with TOML configuration.")
    return wan_t2v, config

def detect_checkpoint_type(lora_path):
    """Detects FPS-only vs mixed checkpoint type."""
    import safetensors
    try:
        checkpoint = safetensors.torch.load_file(lora_path)
        
        has_base_lora = False
        has_fps_params = False
        
        for key in checkpoint.keys():
            if 'fps_conditioning' in key or 'fps_adapter' in key:
                has_fps_params = True
            elif key.endswith('.weight') and ('lora_A' in key or 'lora_B' in key):
                if 'fps' not in key:
                    has_base_lora = True
        
        if has_fps_params and not has_base_lora:
            return 'fps_only'
        elif has_fps_params and has_base_lora:
            return 'lora_and_fps'
        else:
            return 'unknown'
    except Exception as e:
        logging.warning(f"Could not detect checkpoint type: {e}")
        return 'unknown'

def apply_checkpoint(wan_t2v_pipeline, checkpoint_path, rank=32, dtype=torch.bfloat16):
    """Applies checkpoint weights (FPS-only or mixed LoRA+FPS)."""
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    logging.info(f"Detected checkpoint type: {checkpoint_type}")
    
    # Debug: Show FPS parameter values BEFORE loading checkpoint
    logging.info("🔍 FPS parameter values BEFORE checkpoint loading:")
    params_before = {}
    
    # FPS MLP parameters
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            params_before[name] = (param_mean, param_std, param_max)
            logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f}")
    
    # FPS Adapter parameters (sample first few)
    adapter_count = 0
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_adapter' in name and adapter_count < 6:  # Show first 6 adapter params
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            params_before[name] = (param_mean, param_std, param_max)
            
            if 'gate_alpha' in name:
                gate_sigmoid = torch.sigmoid(param.data).item()
                logging.info(f"  {name}: alpha={param_mean:.6f} -> gate={gate_sigmoid:.6f}")
            else:
                logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f}")
    
    # Also check for fixed gate buffers
    for name, buffer in wan_t2v_pipeline.transformer.named_buffers():
        if 'fps_adapter' in name and 'gate_fixed' in name and adapter_count < 6:
            gate_value = buffer.data.item()
            logging.info(f"  {name}: gate={gate_value:.6f} (FIXED MODE)")
            adapter_count += 1
            adapter_count += 1
    
    if adapter_count >= 6:
        logging.info(f"  ... and {sum(1 for n, _ in wan_t2v_pipeline.transformer.named_parameters() if 'fps_adapter' in n) - 6} more adapter parameters")
    
    # Configure base LoRA adapter only if needed
    if checkpoint_type in ['lora_and_fps', 'unknown']:
        adapter_config = {"type": "lora", "rank": rank, "alpha": rank, "dropout": 0.0, "dtype": dtype}
        wan_t2v_pipeline.configure_adapter(adapter_config)
        logging.info(f"Configured base LoRA adapter with rank {rank}")
    elif checkpoint_type == 'fps_only':
        logging.info("FPS-only checkpoint - skipping base LoRA configuration")
    
    # Load weights
    wan_t2v_pipeline.load_adapter_weights(checkpoint_path)
    
    # Debug: Show FPS parameter values AFTER loading checkpoint
    logging.info("✅ FPS parameter values AFTER checkpoint loading:")
    
    # FPS MLP parameters
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            
            if name in params_before:
                before_mean, before_std, before_max = params_before[name]
                mean_changed = abs(param_mean - before_mean) > 1e-6
                change_indicator = "📝 CHANGED" if mean_changed else "🔒 UNCHANGED"
                logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} {change_indicator}")
                if mean_changed:
                    logging.info(f"    Mean change: {before_mean:.6f} -> {param_mean:.6f} (Δ={param_mean-before_mean:.6f})")
            else:
                logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} 🆕 NEW")
    
    # FPS Adapter parameters (sample first few)
    adapter_count = 0
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_adapter' in name and adapter_count < 6:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item() if param.numel() > 1 else 0.0
            param_max = param.data.abs().max().item()
            
            if name in params_before:
                before_mean, before_std, before_max = params_before[name]
                mean_changed = abs(param_mean - before_mean) > 1e-6
                change_indicator = "📝 CHANGED" if mean_changed else "🔒 UNCHANGED"
                
                if 'gate_alpha' in name:
                    gate_sigmoid = torch.sigmoid(param.data).item()
                    logging.info(f"  {name}: alpha={param_mean:.6f} -> gate={gate_sigmoid:.6f} {change_indicator}")
                else:
                    logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} {change_indicator}")
                
                if mean_changed:
                    logging.info(f"    Mean change: {before_mean:.6f} -> {param_mean:.6f} (Δ={param_mean-before_mean:.6f})")
            else:
                if 'gate_alpha' in name:
                    gate_sigmoid = torch.sigmoid(param.data).item()
                    logging.info(f"  {name}: alpha={param_mean:.6f} -> gate={gate_sigmoid:.6f} 🆕 NEW")
                else:
                    logging.info(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, max={param_max:.6f} 🆕 NEW")
            adapter_count += 1
    
    # Also check for fixed gate buffers after loading
    logging.info("✅ Fixed gate buffer values AFTER checkpoint loading:")
    for name, buffer in wan_t2v_pipeline.transformer.named_buffers():
        if 'fps_adapter' in name and 'gate_fixed' in name:
            gate_value = buffer.data.item()
            logging.info(f"  {name}: gate={gate_value:.6f} (FIXED MODE)")
    
    # Verify loaded parameters
    fps_mlp_params = 0
    fps_adapter_params = 0
    base_lora_params = 0
    
    for name, param in wan_t2v_pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            fps_mlp_params += 1
        elif 'fps_adapter' in name:
            fps_adapter_params += 1
        elif 'lora' in name.lower() and 'fps' not in name:
            base_lora_params += 1
    
    logging.info(f"✅ Checkpoint loaded successfully!")
    logging.info(f"  FPS MLP parameters: {fps_mlp_params}")
    logging.info(f"  FPS Adapter parameters: {fps_adapter_params}")
    logging.info(f"  Base LoRA parameters: {base_lora_params}")
    
    if fps_mlp_params == 0 and fps_adapter_params == 0:
        logging.warning("⚠️  No FPS parameters found!")
    
    return wan_t2v_pipeline

def generate_video_with_fps(pipeline, config, prompt, n_prompt, fps=60, seed=42, steps=25, scale=7.0, frames=49, size=(512, 320), shift=3.0, force_gate_one=False):
    """Generates a video with specific FPS conditioning."""
    logging.info(f"🎬 Generating video with FPS={fps}")
    if force_gate_one:
        logging.info("🔧 DIAGNOSTIC MODE: Will force gates to 1.0")
    logging.info(f"  Prompt: {prompt}")
    logging.info(f"  Steps: {steps}, Frames: {frames}, Size: {size}")
    
    device = pipeline.transformer.device
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Encode text
    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)
    
    # Prepare FPS tensor
    fps_values = torch.tensor([fps], dtype=torch.float32, device=device)
    logging.info(f"  FPS tensor: {fps_values.tolist()}")
    
    # 🔍 DEBUG: Check FPS parameters before generation
    logging.info("🔍 FPS parameter status before generation:")
    fps_param_count = 0
    fps_condition_count = 0
    fps_adapter_count = 0
    
    for name, param in pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name:
            fps_condition_count += 1
            if fps_condition_count <= 2:  # Show first 2 FPS conditioning params
                param_mean = param.data.mean().item()
                param_max = param.data.abs().max().item()
                logging.info(f"  {name}: mean={param_mean:.6f}, max={param_max:.6f}")
        elif 'fps_adapter' in name:
            fps_adapter_count += 1
            if 'gate_alpha' in name and fps_adapter_count <= 3:  # Show first 3 gate values
                gate_alpha = param.data.item()
                gate_sigmoid = torch.sigmoid(param.data).item()
                logging.info(f"  {name}: alpha={gate_alpha:.6f} -> gate={gate_sigmoid:.6f}")
    
    # Also check for fixed gate buffers
    fixed_gate_count = 0
    for name, buffer in pipeline.transformer.named_buffers():
        if 'fps_adapter' in name and 'gate_fixed' in name and fixed_gate_count <= 3:
            gate_value = buffer.data.item()
            logging.info(f"  {name}: gate={gate_value:.6f} (FIXED MODE)")
            fixed_gate_count += 1
    
    logging.info(f"  Found {fps_condition_count} FPS conditioning params, {fps_adapter_count} FPS adapter params")
    
    # 🔧 DIAGNOSTIC: Force all FPS adapter gates to 1.0 for maximum impact
    if force_gate_one:
        logging.info("🔧 DIAGNOSTIC MODE: Forcing all FPS adapter gates to 1.0 for maximum FPS impact")
        gates_forced = 0
        
        # Handle learned gates (gate_alpha parameters)
        for name, param in pipeline.transformer.named_parameters():
            if 'fps_adapter' in name and 'gate_alpha' in name:
                # Set gate_alpha to a very large positive value to force sigmoid -> 1.0
                # sigmoid(10) ≈ 0.99995, which is effectively 1.0
                with torch.no_grad():
                    param.data.fill_(10.0)
                gates_forced += 1
        
        # Handle fixed gates (gate_fixed buffers)
        for name, buffer in pipeline.transformer.named_buffers():
            if 'fps_adapter' in name and 'gate_fixed' in name:
                # Set fixed gate buffer to 1.0
                with torch.no_grad():
                    buffer.data.fill_(1.0)
                gates_forced += 1
        
        logging.info(f"  Forced {gates_forced} gate values to maximum (1.0)")
        
        # Verify the forced gate values
        logging.info("  Verification - Gate values after forcing:")
        # Check learned gates
        for name, param in pipeline.transformer.named_parameters():
            if 'fps_adapter' in name and 'gate_alpha' in name:
                gate_alpha = param.data.item()
                gate_sigmoid = torch.sigmoid(param.data).item()
                logging.info(f"    {name}: alpha={gate_alpha:.1f} -> gate={gate_sigmoid:.6f} (LEARNED)")
                break  # Just show one example
        
        # Check fixed gates
        for name, buffer in pipeline.transformer.named_buffers():
            if 'fps_adapter' in name and 'gate_fixed' in name:
                gate_value = buffer.data.item()
                logging.info(f"    {name}: gate={gate_value:.6f} (FIXED)")
                break  # Just show one example
    
    # Initialize latents
    vae_stride = [4, 8, 8]
    target_shape = (16, frames // vae_stride[0], size[1] // vae_stride[1], size[0] // vae_stride[2])
    latents = torch.randn(target_shape, device=device)
    
    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]
    
    # Setup scheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device, shift=shift)
    timesteps = scheduler.timesteps
    
    # Denoising loop
    for step_idx, t in enumerate(tqdm(timesteps, desc=f"FPS={fps}")):
        t_batch = torch.full((1,), t, device=device)
        
        # SUBTASK 4: Fix autocast dtype to match training dtype (already converted to torch.dtype)
        autocast_dtype = config['model']['dtype']  # Already converted by DTYPE_MAP
        
        with torch.no_grad(), torch.autocast('cuda', dtype=autocast_dtype):
            # Unconditional pass
            initial_input_uncond = (
                latents.unsqueeze(0), None, t_batch, 
                uncond_inputs['text_embeddings'].to(device), 
                uncond_inputs['seq_lens'].to(device), 
                None, fps_values
            )
            
            layer_outputs = initial_layer(initial_input_uncond)
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
            
            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
                
            noise_pred_uncond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
            
            # Conditional pass
            initial_input_cond = (
                latents.unsqueeze(0), None, t_batch,
                cond_inputs['text_embeddings'].to(device), 
                cond_inputs['seq_lens'].to(device), 
                None, fps_values
            )
            
            layer_outputs = initial_layer(initial_input_cond)
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
            
            # 🔍 DEBUG: Check fps_conditioning tensor values (only for first step)
            if step_idx == 0:
                fps_mean = fps_conditioning.mean().item()
                fps_std = fps_conditioning.std().item()
                fps_max = fps_conditioning.abs().max().item()
                logging.info(f"  [Step {step_idx}] FPS conditioning tensor: mean={fps_mean:.6f}, std={fps_std:.6f}, max={fps_max:.6f}")
            
            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
                
            noise_pred_cond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
            
            # CFG
            noise_pred = noise_pred_uncond.squeeze(0) + scale * (noise_pred_cond.squeeze(0) - noise_pred_uncond.squeeze(0))
        
        latents = scheduler.step(noise_pred, t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
    
    # Decode to video
    logging.info("Decoding latents...")
    video_tensor = pipeline.vae.decode([latents])[0]
    
    # 🔍 DEBUG: Final verification of FPS parameters after generation
    logging.info("🔍 FPS parameter status after generation:")
    for name, param in pipeline.transformer.named_parameters():
        if 'fps_conditioning' in name and 'lin1.weight' in name:  # Check one representative param
            param_mean = param.data.mean().item()
            param_max = param.data.abs().max().item()
            logging.info(f"  {name}: mean={param_mean:.6f}, max={param_max:.6f}")
            break
    
    logging.info(f"✅ Generation complete for FPS={fps}")
    
    return video_tensor

def save_video_result(tensor, fps, prompt_short, output_dir, size):
    """Saves video with descriptive filename."""
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_prompt = "".join(c for c in prompt_short if c.isalnum() or c in (' ', '_')).strip()[:30]
    filename = f"fps_{fps:03d}_{clean_prompt.replace(' ', '_')}_{size[0]}x{size[1]}_{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)
    
    logging.info(f"💾 Saving: {filename}")
    cache_video(tensor=tensor[None], save_file=filepath, fps=16, normalize=True, value_range=(-1, 1))
    
    return filepath

def run_fps_experiments(pipeline, config, base_prompt, n_prompt, fps_values, output_dir, force_gate_one=False, **generation_kwargs):
    """Runs multiple FPS conditioning experiments."""
    logging.info("🧪 Starting FPS conditioning experiments...")
    logging.info(f"  FPS values to test: {fps_values}")
    logging.info(f"  Base prompt: {base_prompt}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, fps in enumerate(fps_values, 1):
        logging.info(f"\n--- Experiment {i}/{len(fps_values)}: FPS = {fps} ---")
        
        try:
            # Generate video
            video_tensor = generate_video_with_fps(
                pipeline=pipeline,
                config=config,  # Pass config for proper dtype handling
                prompt=base_prompt,
                n_prompt=n_prompt,
                fps=fps,
                force_gate_one=force_gate_one,
                **generation_kwargs
            )
            
            # Save result
            output_file = save_video_result(
                tensor=video_tensor,
                fps=fps,
                prompt_short=base_prompt.split()[0:3],  # First few words
                output_dir=output_dir,
                size=generation_kwargs.get('size', (512, 320))
            )
            
            # Collect stats for comparison
            stats = {
                'fps': fps,
                'file': output_file,
                'mean': video_tensor.mean().item(),
                'std': video_tensor.std().item(),
                'min': video_tensor.min().item(),
                'max': video_tensor.max().item()
            }
            results.append(stats)
            
            logging.info(f"✅ Experiment {i} complete")
            
        except Exception as e:
            logging.error(f"❌ Experiment {i} failed: {e}")
            results.append({'fps': fps, 'error': str(e)})
    
    # Analysis
    logging.info(f"\n📊 FPS Conditioning Analysis:")
    logging.info(f"{'FPS':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'File':<50}")
    logging.info("-" * 100)
    
    successful_results = [r for r in results if 'error' not in r]
    for result in results:
        if 'error' in result:
            logging.info(f"{result['fps']:<6} ERROR: {result['error']}")
        else:
            logging.info(f"{result['fps']:<6} {result['mean']:<10.4f} {result['std']:<10.4f} "
                        f"{result['min']:<10.4f} {result['max']:<10.4f} {os.path.basename(result['file'])}")
    
    # Check if FPS conditioning is working
    if len(successful_results) >= 2:
        means = [r['mean'] for r in successful_results]
        mean_variation = max(means) - min(means)
        if mean_variation > 0.01:  # Threshold for meaningful difference
            logging.info(f"\n✅ SUCCESS: FPS conditioning is working! Mean variation: {mean_variation:.4f}")
        else:
            logging.info(f"\n⚠️  WARNING: Low variation between FPS values. Mean variation: {mean_variation:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Multi-FPS experiment script - TOML-aligned')
    parser.add_argument('--config', required=True, help='Path to TOML configuration file')
    parser.add_argument('--checkpoint', required=True, help='Path to FPS checkpoint')
    parser.add_argument('--output_dir', default='./fps_experiments_align', help='Output directory')
    parser.add_argument('--fps_values', nargs='+', type=int, default=[12, 24, 60], help='FPS values to test')
    parser.add_argument('--prompt', default='A cat walking through a beautiful garden', help='Generation prompt')
    parser.add_argument('--negative_prompt', default='', help='Negative prompt (optional)')
    parser.add_argument('--steps', type=int, default=15, help='Denoising steps')
    parser.add_argument('--frames', type=int, default=33, help='Number of frames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', default='29501', help='DeepSpeed port')
    parser.add_argument('--width', type=int, help='Video width (overrides TOML if specified)')
    parser.add_argument('--height', type=int, help='Video height (overrides TOML if specified)')
    parser.add_argument('--scale', type=float, help='CFG scale (overrides TOML if specified)')
    parser.add_argument('--force_gate_one', action='store_true', help='🔧 DIAGNOSTIC: Force all FPS adapter gates to 1.0 for maximum FPS impact')
    
    args = parser.parse_args()
    
    try:
        # Setup
        setup_environment(args.port)
        
        # Load pipeline using TOML configuration
        logging.info("Loading pipeline from TOML configuration...")
        pipeline, config = load_pipeline_from_toml(args.config)
        
        # Apply checkpoint
        logging.info("Applying checkpoint...")
        # Get rank from TOML config or default
        fps_rank = config['model'].get('fps_adapter_rank', 32)
        pipeline = apply_checkpoint(pipeline, args.checkpoint, rank=fps_rank)
        
        # SUBTASK 2: Use TOML parameters for inference settings (with CLI overrides)
        # Extract inference parameters from TOML config or use defaults
        width = args.width if args.width is not None else 832
        height = args.height if args.height is not None else 480
        scale = args.scale if args.scale is not None else 6.0
        
        logging.info("Inference parameters:")
        logging.info(f"  Resolution: {width}x{height}")
        logging.info(f"  CFG scale: {scale}")
        logging.info(f"  Steps: {args.steps}")
        logging.info(f"  Frames: {args.frames}")
        logging.info(f"  Prompt: '{args.prompt}'")
        logging.info(f"  Negative prompt: '{args.negative_prompt}'")
        
        # Run experiments
        results = run_fps_experiments(
            pipeline=pipeline,
            config=config,  # Pass TOML config for proper dtype handling
            base_prompt=args.prompt,
            n_prompt=args.negative_prompt,  # Use provided negative prompt or empty string
            fps_values=args.fps_values,
            output_dir=args.output_dir,
            seed=args.seed,
            steps=args.steps,
            frames=args.frames,
            size=(width, height),
            scale=scale,
            shift=3.0,
            force_gate_one=args.force_gate_one
        )
        
        logging.info(f"\n🎉 All experiments complete! Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())