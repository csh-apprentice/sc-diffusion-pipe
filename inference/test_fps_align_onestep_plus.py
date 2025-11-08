#!/usr/bin/env python3
"""
Multi-experiment FPS conditioning test script - TOML-aligned version with PLUS alignment.
Tests different FPS values to verify FPS conditioning works correctly.
Uses TOML configuration files for proper training/inference alignment.

PLUS ALIGNMENT MODE:
- Captures TWO ratios per FPS adapter block:
  - r_text_i = ||y_text_base_lora_fps|| / ||y_text_clean_fps||
  - r_cond_i = ||y_fps_base_lora_fps|| / ||y_fps_clean_fps||
- Applies: new_gate_i = old_gate_i * r_text_i * r_cond_i
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

    # Store the TOML path in config for later reload (needed for alignment mode)
    config['_toml_path'] = toml_path

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
        'fps_gate_fixed_value': model_config.get('fps_gate_fixed_value'),
        'fps_scale': model_config.get('fps_scale'),
        'fps_warmup_steps': model_config.get('fps_warmup_steps')
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

def load_adapter_weights_selective(pipeline, checkpoint_path, fps_only=False, base_only=False):
    """Load adapter weights with selective parameter filtering using temporary files."""
    if not fps_only and not base_only:
        # Normal mode - use original loading method
        logging.info(f"📦 NORMAL MODE: Loading all parameters")
        pipeline.load_adapter_weights(checkpoint_path)
        return
    
    import safetensors
    import tempfile
    from pathlib import Path
    
    # Use the same checkpoint detection logic as the original WAN pipeline
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        # Look for .safetensors files in the directory
        safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")
        checkpoint_file = safetensors_files[0]  # Use the first one found
        logging.info(f"Found checkpoint file: {checkpoint_file}")
    else:
        checkpoint_file = checkpoint_dir
    
    # Load the original checkpoint
    logging.info(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = safetensors.torch.load_file(str(checkpoint_file))
    
    # Filter parameters based on mode
    if fps_only:
        # Only keep FPS-related parameters
        filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                             if 'fps_conditioning' in k or 'fps_adapter' in k}
        mode_name = "FPS-ONLY"
        logging.info(f"🎯 {mode_name} MODE: Filtered {len(filtered_checkpoint)} FPS parameters out of {len(checkpoint)} total")
        
    elif base_only:
        # Only keep base LoRA parameters (exclude FPS parameters)
        filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                             if 'fps_conditioning' not in k and 'fps_adapter' not in k}
        mode_name = "BASE-ONLY"
        logging.info(f"🎯 {mode_name} MODE: Filtered {len(filtered_checkpoint)} base LoRA parameters out of {len(checkpoint)} total")
    
    # Show detailed breakdown of what's being loaded/skipped
    fps_conditioning_params = [k for k in checkpoint.keys() if 'fps_conditioning' in k]
    fps_adapter_params = [k for k in checkpoint.keys() if 'fps_adapter' in k]
    base_lora_params = [k for k in checkpoint.keys() if 'fps_conditioning' not in k and 'fps_adapter' not in k]
    
    logging.info(f"📊 Parameter breakdown in original checkpoint:")
    logging.info(f"  FPS conditioning (MLP): {len(fps_conditioning_params)} parameters")
    logging.info(f"  FPS adapters (LoRA): {len(fps_adapter_params)} parameters") 
    logging.info(f"  Base LoRA: {len(base_lora_params)} parameters")
    logging.info(f"  Total: {len(checkpoint)} parameters")
    
    if fps_only:
        logging.info(f"🎯 FPS-ONLY filtering results:")
        logging.info(f"  ✅ Loading FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  ✅ Loading FPS adapters: {len(fps_adapter_params)} parameters")
        logging.info(f"  ❌ Skipping base LoRA: {len(base_lora_params)} parameters")
        
        # Show examples of each type
        if fps_conditioning_params:
            logging.info(f"  📝 Example FPS conditioning params: {fps_conditioning_params[:2]}")
        if fps_adapter_params:
            logging.info(f"  📝 Example FPS adapter params: {fps_adapter_params[:2]}")
            
    elif base_only:
        logging.info(f"🎯 BASE-ONLY filtering results:")
        logging.info(f"  ✅ Loading base LoRA: {len(base_lora_params)} parameters") 
        logging.info(f"  ❌ Skipping FPS conditioning: {len(fps_conditioning_params)} parameters")
        logging.info(f"  ❌ Skipping FPS adapters: {len(fps_adapter_params)} parameters")
        
        # Show examples of base LoRA params
        if base_lora_params:
            logging.info(f"  📝 Example base LoRA params: {base_lora_params[:3]}")
    
    # Create temporary directory and filtered safetensors file
    temp_dir = tempfile.mkdtemp(prefix=f"selective_loading_{mode_name.lower()}_")
    temp_checkpoint_file = Path(temp_dir) / "filtered_adapter.safetensors"
    
    logging.info(f"Creating temporary filtered checkpoint: {temp_checkpoint_file}")
    safetensors.torch.save_file(filtered_checkpoint, str(temp_checkpoint_file))
    
    # Use the original load_adapter_weights method with the filtered temporary file
    try:
        pipeline.load_adapter_weights(temp_dir)
        logging.info(f"✅ Successfully loaded {len(filtered_checkpoint)} filtered parameters using temporary file")
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        logging.info(f"🧹 Cleaned up temporary directory: {temp_dir}")

def verify_fps_config_applied(pipeline, config):
    """Verify that FPS configuration from TOML was properly applied to the model."""
    model_config = config['model']
    logging.info("🔍 Verifying FPS configuration from TOML was applied correctly:")
    
    # Check FPS adapters exist in the model
    fps_adapter_count = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_adapter' in n)
    fps_conditioning_count = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_conditioning' in n)
    
    logging.info(f"  Found {fps_adapter_count} FPS adapter parameters in model")
    logging.info(f"  Found {fps_conditioning_count} FPS conditioning (MLP) parameters in model")
    
    # Check if the expected FPS settings were used
    expected_settings = {
        'fps_adapter_rank': model_config.get('fps_adapter_rank', 'default'),
        'fps_condition_blocks': model_config.get('fps_condition_blocks', 'default'),
        'fps_gate_mode': model_config.get('fps_gate_mode', 'default'),
        'fps_gate_fixed_value': model_config.get('fps_gate_fixed_value', 'default'),
        'fps_adapter_num_tokens': model_config.get('fps_adapter_num_tokens', 'default'),
        'fps_embed_dim': model_config.get('fps_embed_dim', 'default'),
        'fps_reference_fps': model_config.get('fps_reference_fps', 'default (240.0)'),
        'fps_scale': model_config.get('fps_scale', 'default (False)'),
        'fps_warmup_steps': model_config.get('fps_warmup_steps', 'default (100)')
    }
    
    logging.info("  Expected FPS configuration from TOML:")
    for key, value in expected_settings.items():
        logging.info(f"    {key}: {value}")
    
    # Check gate mode by looking at buffer vs parameter names
    has_gate_alpha = any('gate_alpha' in n for n, _ in pipeline.transformer.named_parameters())
    has_gate_fixed = any('gate_fixed' in n for n, _ in pipeline.transformer.named_buffers())

    # Learnable modes: sigmoid, identity, relu, silu, softplus (all use gate_alpha parameter)
    # Fixed mode: uses gate_fixed buffer
    if has_gate_alpha and not has_gate_fixed:
        detected_mode = "learnable (sigmoid/identity/relu/silu/softplus)"
    elif has_gate_fixed and not has_gate_alpha:
        detected_mode = "fixed"
    elif has_gate_alpha and has_gate_fixed:
        detected_mode = "mixed (both found)"
    else:
        detected_mode = "unknown"

    expected_mode = model_config.get('fps_gate_mode', 'sigmoid')  # Default changed from 'learned' to 'sigmoid'
    # Check if expected mode matches detected mode category
    learnable_modes = ['sigmoid', 'identity', 'relu', 'silu', 'softplus']
    expected_is_learnable = expected_mode in learnable_modes
    detected_is_learnable = has_gate_alpha and not has_gate_fixed
    mode_match = "✅" if (expected_is_learnable == detected_is_learnable) or (expected_mode == 'fixed' and detected_mode == 'fixed') else "⚠️"
    logging.info(f"  Gate mode check: Expected='{expected_mode}', Detected='{detected_mode}' {mode_match}")

    # Verify fps_reference_fps is being used correctly
    actual_reference_fps = getattr(pipeline, 'fps_reference_fps', 'not found')
    expected_reference_fps = model_config.get('fps_reference_fps', 240.0)
    reference_match = "✅" if actual_reference_fps == expected_reference_fps else "⚠️"
    logging.info(f"  Reference FPS check: Expected={expected_reference_fps}, Actual={actual_reference_fps} {reference_match}")

def apply_checkpoint(wan_t2v_pipeline, checkpoint_path, rank=32, dtype=torch.bfloat16, fps_only=False, base_only=False, config=None):
    """Applies checkpoint weights (FPS-only or mixed LoRA+FPS)."""
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint_type = detect_checkpoint_type(checkpoint_path)
    logging.info(f"Detected checkpoint type: {checkpoint_type}")
    
    # Verify FPS configuration for fps_only mode
    if fps_only and config:
        verify_fps_config_applied(wan_t2v_pipeline, config)
    
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
    
    # Configure base LoRA adapter based on selective loading mode
    configure_base_lora = False
    
    if fps_only:
        logging.info("🎯 FPS-ONLY MODE: Skipping base LoRA configuration")
        logging.info("   ✅ FPS adapters were already configured from TOML during pipeline initialization")
        logging.info("   ✅ FPS configuration from TOML will be used (rank, gate_mode, etc.)")
        configure_base_lora = False
    elif base_only:
        logging.info("🎯 BASE-ONLY MODE: Configuring only base LoRA adapter")
        configure_base_lora = True
    else:
        # Normal mode - configure based on checkpoint type
        if checkpoint_type in ['lora_and_fps', 'unknown']:
            configure_base_lora = True
        elif checkpoint_type == 'fps_only':
            configure_base_lora = False
    
    if configure_base_lora:
        adapter_config = {"type": "lora", "rank": rank, "alpha": rank, "dropout": 0.0, "dtype": dtype}
        wan_t2v_pipeline.configure_adapter(adapter_config)
        logging.info(f"Configured base LoRA adapter with rank {rank}")
    
    # Load weights with selective parameter filtering
    load_adapter_weights_selective(wan_t2v_pipeline, checkpoint_path, fps_only, base_only)
    
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

    # 🐛 DEBUG MODE VERIFICATION: Explicitly check for base LoRA exclusion
    if fps_only:
        logging.info("\n🐛 DEBUG MODE VERIFICATION: Checking parameter loading in FPS-ONLY mode")

        # List all parameter names to verify what's actually in the model
        all_param_names = [name for name, _ in wan_t2v_pipeline.transformer.named_parameters()]

        # Check for base LoRA parameters (should be ZERO in fps_only mode)
        base_lora_names = [name for name in all_param_names if 'lora' in name.lower() and 'fps' not in name]
        fps_param_names = [name for name in all_param_names if 'fps' in name.lower()]

        logging.info(f"  Total parameters in model: {len(all_param_names)}")
        logging.info(f"  FPS-related parameters: {len(fps_param_names)}")
        logging.info(f"  Base LoRA parameters: {len(base_lora_names)}")

        if len(base_lora_names) > 0:
            logging.error(f"  ❌ ERROR: Found {len(base_lora_names)} base LoRA parameters (should be 0!)")
            logging.error(f"     Example base LoRA params: {base_lora_names[:5]}")
        else:
            logging.info(f"  ✅ VERIFIED: No base LoRA parameters present (fps_only mode correct)")

        # Show sample of FPS parameters to confirm they're loaded
        if len(fps_param_names) > 0:
            logging.info(f"  ✅ Sample FPS parameters loaded:")
            for name in fps_param_names[:5]:
                logging.info(f"     - {name}")
            if len(fps_param_names) > 5:
                logging.info(f"     ... and {len(fps_param_names) - 5} more FPS parameters")

    return wan_t2v_pipeline

def run_one_denoising_step(pipeline, config, prompt, n_prompt, fps_condition, seed, steps, scale, frames, size, shift, capture_storage=None):
    """
    Run ONE denoising step to capture ||y_text|| and ||y_fps|| magnitudes at each FPS block.

    PLUS VERSION: Captures BOTH text and FPS conditioning magnitudes.

    Args:
        capture_storage: Dict to store {block_idx: {'y_text': ||y_text||, 'y_fps': ||y_fps||}}

    Note: The pipeline should already have the desired checkpoint loaded (clean, base_lora+fps, or clean+fps)
    """
    device = pipeline.transformer.device

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Encode text
    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)

    # Prepare FPS tensor
    fps_values = torch.tensor([fps_condition], dtype=torch.float32, device=device)

    # Initialize latents (match generate_video_with_fps format)
    vae_stride = [4, 8, 8]
    latent_shape = (16, frames // vae_stride[0], size[1] // vae_stride[1], size[0] // vae_stride[2])
    latents = torch.randn(latent_shape, device=device)

    # Setup scheduler for one step (match generate_video_with_fps format)
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device, shift=shift)
    timesteps = scheduler.timesteps

    # Take only the FIRST timestep for one-step denoising
    t = timesteps[0:1]  # Shape: [1]

    # Prepare latent input (no need to double for CFG in calibration)
    latent_model_input = latents.unsqueeze(0)  # Add batch dimension
    timestep_cond = t

    # Hook to capture BOTH ||y_text|| and ||y_fps|| from each FPS block (PLUS VERSION)
    #
    # Strategy: Since FPS adapter returns y_text + gate * y_fps, we'll:
    # 1. Capture y_text directly from the inputs using flash_attention
    # 2. Capture the combined output
    # 3. Compute y_fps = (output - y_text) / gate
    captured_magnitudes = {}

    def make_capture_hook(block_idx):
        def hook(module, args, kwargs, output):
            # The FPS adapter forward signature is:
            # forward(q, k_text, v_text, fps_conditioning, context_lens)
            # output is y_combined = y_text + gate * y_fps

            if len(args) >= 3:
                q, k_text, v_text = args[0], args[1], args[2]

                # Compute y_text using flash attention (same as FPS adapter does internally)
                from models.wan.attention import flash_attention

                # Get context_lens if available
                context_lens = args[4] if len(args) > 4 else (kwargs.get('context_lens', None) if kwargs else None)

                with torch.no_grad():
                    # Capture y_text magnitude
                    y_text = flash_attention(q, k_text, v_text, k_lens=context_lens)
                    y_text_norm = torch.norm(y_text).item()

                    # Get gate value
                    if module.gate_mode == 'sigmoid':
                        gate = torch.sigmoid(module.gate_alpha)
                    elif module.gate_mode == 'identity':
                        gate = module.gate_alpha
                    elif module.gate_mode == 'relu':
                        gate = torch.relu(module.gate_alpha)
                    elif module.gate_mode == 'silu':
                        gate = torch.nn.functional.silu(module.gate_alpha)
                    elif module.gate_mode == 'softplus':
                        gate = torch.nn.functional.softplus(module.gate_alpha)
                    elif module.gate_mode == 'fixed':
                        gate = module.gate_fixed
                    else:
                        gate = 1.0

                    # Compute y_fps from: output = y_text + gate * y_fps
                    # => y_fps = (output - y_text) / gate
                    y_fps = (output - y_text) / (gate + 1e-8)
                    y_fps_norm = torch.norm(y_fps).item()

                    # Store both magnitudes
                    captured_magnitudes[block_idx] = {
                        'y_text': y_text_norm,
                        'y_fps': y_fps_norm
                    }

            return output
        return hook

    # Run one forward pass using layer-based API (match generate_video_with_fps)
    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]

    # Register hooks on FPS adapter blocks
    # Find blocks that actually have FPS adapters instead of relying on config
    hook_handles = []
    fps_blocks = []

    # Access the blocks from the transformer within the layers
    transformer_model = pipeline.transformer
    for block_idx, block in enumerate(transformer_model.blocks):
        if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
            fps_blocks.append(block_idx)

            # Hook to capture y_text and y_fps magnitudes
            handle = block.fps_adapter.register_forward_hook(
                make_capture_hook(block_idx),
                with_kwargs=True
            )
            hook_handles.append(handle)

    logging.info(f"  Found {len(fps_blocks)} FPS adapter blocks: {fps_blocks}")

    with torch.no_grad():
        # Run initial layer
        initial_input = (
            latent_model_input, None, timestep_cond,
            cond_inputs['text_embeddings'].to(device),
            cond_inputs['seq_lens'].to(device),
            None, fps_values
        )

        layer_outputs = initial_layer(initial_input)
        x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs

        # Run transformer layers
        for transformer_layer in transformer_layers:
            layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs

        # Run final layer
        _ = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Store captured magnitudes
    if capture_storage is not None:
        capture_storage.update(captured_magnitudes)

    logging.info(f"  Captured {len(captured_magnitudes)} block magnitudes")


def calibrate_alignment_ratios_onestep(pipeline, config, checkpoint_path, prompt, n_prompt, fps_condition, seed, steps, scale, frames, size, shift):
    """
    ONE-STEP PLUS ALIGNMENT ALGORITHM (ALIGN_INFERNCE_TEXT_ONESTEP_PLUS.md):

    Subtask 1 Plus: Log TWO ratios for each FPS adapter-injected block i:
        r_text_i = ||y_text_base_lora_fps|| / ||y_text_clean_fps||
        r_cond_i = ||y_fps_base_lora_fps|| / ||y_fps_clean_fps||

    This is done by running ONE denoising step TWICE:
    1. Run one step with CLEAN + FPS adapter → capture ||y_text_clean_fps|| and ||y_fps_clean_fps||
    2. Run one step with BASE_LORA + FPS adapter → capture ||y_text_base_fps|| and ||y_fps_base_fps||
    3. Compute both r_text_i and r_cond_i for each block i

    Returns: ratio_dict mapping {block_idx: {'r_text': r_text_i, 'r_cond': r_cond_i}}
    """
    logging.info("🔧 Running ONE-STEP PLUS TEXT+COND MAGNITUDE alignment calibration...")
    logging.info("  Will run ONE denoising step TWICE with different checkpoint loading modes")
    logging.info("  Capturing BOTH ||y_text|| and ||y_fps|| magnitudes at each FPS-injected block")

    print(f"\n{'='*80}", flush=True)
    print(f"🔧 CALIBRATION PHASE (PLUS): One-step magnitude comparison", flush=True)
    print(f"{'='*80}", flush=True)

    # Storage for magnitudes
    clean_fps_magnitudes = {}  # {block_idx: {'y_text': ..., 'y_fps': ...}}
    base_fps_magnitudes = {}   # {block_idx: {'y_text': ..., 'y_fps': ...}}

    # Step 1: Run one step with CLEAN + FPS adapter
    print(f"\n  [1/2] Running one step with CLEAN backbone + FPS adapter...", flush=True)
    base_rank = config.get('adapter', {}).get('rank', 32)
    apply_checkpoint(pipeline, checkpoint_path, rank=base_rank, fps_only=True, config=config)

    run_one_denoising_step(
        pipeline, config, prompt, n_prompt, fps_condition, seed, steps, scale, frames, size, shift,
        capture_storage=clean_fps_magnitudes
    )

    # Step 2: Load BASE_LORA + FPS adapter from checkpoint (full checkpoint)
    print(f"\n  Loading BASE_LORA + FPS adapter from checkpoint...", flush=True)
    apply_checkpoint(pipeline, checkpoint_path, rank=base_rank, fps_only=False, base_only=False, config=config)

    # Run one step with BASE_LORA + FPS adapter
    print(f"\n  [2/2] Running one step with BASE_LORA + FPS adapter...", flush=True)
    run_one_denoising_step(
        pipeline, config, prompt, n_prompt, fps_condition, seed, steps, scale, frames, size, shift,
        capture_storage=base_fps_magnitudes
    )

    # Step 3: Compute TWO ratios per block
    ratio_dict = {}
    print(f"\n{'='*80}", flush=True)
    print(f"📊 CALIBRATION RESULTS (PLUS): Magnitude Ratios per Block", flush=True)
    print(f"   FPS Condition = {fps_condition}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Block':<8} {'||y_text_clean||':<18} {'||y_text_base||':<18} {'r_text':<12} {'||y_fps_clean||':<18} {'||y_fps_base||':<18} {'r_cond':<12} {'combined':<12}", flush=True)
    print(f"{'-'*116}", flush=True)

    for block_idx in sorted(clean_fps_magnitudes.keys()):
        clean_data = clean_fps_magnitudes[block_idx]
        base_data = base_fps_magnitudes.get(block_idx, {'y_text': 0.0, 'y_fps': 0.0})

        y_text_clean = clean_data['y_text']
        y_fps_clean = clean_data['y_fps']
        y_text_base = base_data['y_text']
        y_fps_base = base_data['y_fps']

        r_text = y_text_base / (y_text_clean + 1e-8)
        r_cond = y_fps_base / (y_fps_clean + 1e-8)
        combined = r_text * r_cond

        ratio_dict[block_idx] = {
            'r_text': r_text,
            'r_cond': r_cond
        }

        print(f"{block_idx:<8} {y_text_clean:<18.4f} {y_text_base:<18.4f} {r_text:<12.4f} {y_fps_clean:<18.4f} {y_fps_base:<18.4f} {r_cond:<12.4f} {combined:<12.4f}", flush=True)

    print(f"{'='*80}", flush=True)
    print(f"✅ Calibration complete for FPS={fps_condition}: Captured {len(ratio_dict)} blocks with TWO ratios each", flush=True)

    # Show summary statistics
    r_text_values = [v['r_text'] for v in ratio_dict.values()]
    r_cond_values = [v['r_cond'] for v in ratio_dict.values()]
    combined_values = [v['r_text'] * v['r_cond'] for v in ratio_dict.values()]

    print(f"\n📈 Summary Statistics:", flush=True)
    print(f"   r_text: mean={sum(r_text_values)/len(r_text_values):.4f}, min={min(r_text_values):.4f}, max={max(r_text_values):.4f}", flush=True)
    print(f"   r_cond: mean={sum(r_cond_values)/len(r_cond_values):.4f}, min={min(r_cond_values):.4f}, max={max(r_cond_values):.4f}", flush=True)
    print(f"   combined (r_text × r_cond): mean={sum(combined_values)/len(combined_values):.4f}, min={min(combined_values):.4f}, max={max(combined_values):.4f}", flush=True)

    return ratio_dict


def generate_video_with_fps(pipeline, config, prompt, n_prompt, fps=60, seed=42, steps=25, scale=7.0, frames=49, size=(512, 320), shift=3.0, force_gate_one=False, alignment_ratios=None, capture_rboth=False, rboth_storage=None, fps_idx=None, apply_onestep_alignment=False):
    """Generates a video with specific FPS conditioning.

    PARAMS for ONE-STEP PLUS alignment (ALIGN_INFERNCE_TEXT_ONESTEP_PLUS.md):
    - apply_onestep_alignment: If True, scale gates using alignment_ratios
    - alignment_ratios: Dict {block_idx: {'r_text': r_text_i, 'r_cond': r_cond_i}} where:
        - r_text_i = ||y_text_base_lora_fps|| / ||y_text_clean_fps||
        - r_cond_i = ||y_fps_base_lora_fps|| / ||y_fps_clean_fps||
      Applied as: new_gate_i = old_gate_i * r_text_i * r_cond_i

      Backward compatible: Also accepts Dict {block_idx: ratio_i} for standard mode

    LEGACY PARAMS (for multi-step alignment - not used in onestep mode):
    - capture_rboth: If True, capture rboth_ijk ratios during generation (calibration mode)
    - rboth_storage: Dict to store captured ratios
    - fps_idx: FPS index k for storage key
    """
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

    # === NEW ALIGNMENT: Wrap FPS adapters for capture or apply mode ===
    original_fps_forwards = {}
    alignment_stats = {'ratios_sum': {}, 'ratios_count': {}}  # For debugging (Subtask 3)

    if capture_rboth or alignment_ratios is not None:
        # Find all FPS adapter blocks (access via pipeline.transformer.blocks, not DeepSpeed layers)
        fps_adapter_blocks = []
        for block_idx, block in enumerate(pipeline.transformer.blocks):
            if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                fps_adapter_blocks.append((block_idx, block.fps_adapter))

        print(f"\n[ALIGN_DEBUG] Found {len(fps_adapter_blocks)} FPS adapter blocks to wrap", flush=True)
        if len(fps_adapter_blocks) == 0:
            print(f"[ALIGN_DEBUG] WARNING: No FPS adapters found in transformer blocks!", flush=True)
            print(f"[ALIGN_DEBUG] Total transformer blocks: {len(pipeline.transformer.blocks)}", flush=True)
            if len(pipeline.transformer.blocks) > 0:
                sample_block = pipeline.transformer.blocks[0]
                print(f"[ALIGN_DEBUG] Sample block has fps_adapter: {hasattr(sample_block, 'fps_adapter')}", flush=True)
        else:
            print(f"[ALIGN_DEBUG] Will wrap FPS adapters in blocks: {[idx for idx, _ in fps_adapter_blocks]}", flush=True)

        def make_fps_wrapper(original_forward, block_idx, step_idx_ref):
            """Wraps FPS adapter to capture or apply alignment ratios"""
            def wrapper(q, k_text, v_text, fps_conditioning, context_lens):
                from models.wan.attention import flash_attention

                # Compute y_text and y_fps separately
                y_text = flash_attention(q, k_text, v_text, k_lens=context_lens)

                # Call original forward to get y_fps (need to extract it)
                # The FPS adapter computes: y_combined = y_text + g * y_fps
                # We need to compute y_fps ourselves
                module = wrapper.__self__

                # Compute y_fps (same as in FPSCrossAttentionAdapter.forward)
                k_fps_proj = module.k_fps_up(module.k_fps_down(fps_conditioning))
                v_fps_proj = module.v_fps_up(module.v_fps_down(fps_conditioning))

                B = q.size(0)
                k_fps_proj = k_fps_proj.view(B * module.num_tokens, module.dim)
                k_fps_proj = module.norm_k_fps(k_fps_proj)
                k_fps_proj = k_fps_proj.view(B, module.num_tokens * module.dim)
                k_fps_proj = k_fps_proj * module.lora_scale
                v_fps_proj = v_fps_proj * module.lora_scale

                k_fps = k_fps_proj.view(B, module.num_tokens, module.num_heads, module.head_dim)
                v_fps = v_fps_proj.view(B, module.num_tokens, module.num_heads, module.head_dim)

                y_fps = flash_attention(q, k_fps, v_fps, k_lens=None)

                # Apply fps_scale if enabled
                if module.fps_scale:
                    y_text_norm = torch.norm(y_text)
                    y_fps_norm = torch.norm(y_fps)
                    scale_factor = y_text_norm / (y_fps_norm + 1e-8)
                    y_fps = y_fps * scale_factor

                # Compute gate
                if module.gate_mode == 'sigmoid':
                    gate = torch.sigmoid(module.gate_alpha)
                elif module.gate_mode == 'identity':
                    gate = module.gate_alpha
                elif module.gate_mode == 'relu':
                    gate = torch.relu(module.gate_alpha)
                elif module.gate_mode == 'silu':
                    gate = torch.nn.functional.silu(module.gate_alpha)
                elif module.gate_mode == 'softplus':
                    gate = torch.nn.functional.softplus(module.gate_alpha)
                elif module.gate_mode == 'fixed':
                    gate = module.gate_fixed
                else:
                    gate = 1.0

                warmup_factor = 1.0

                # MODE 1: ONE-STEP PLUS ALIGNMENT - Scale gate by TWO ratios (ALIGN_INFERNCE_TEXT_ONESTEP_PLUS.md)
                if apply_onestep_alignment and alignment_ratios is not None:
                    # Subtask 2 Plus: Apply alignment by scaling gate with BOTH ratios
                    # new_gate_i = old_gate_i * r_text_i * r_cond_i
                    if block_idx in alignment_ratios:
                        ratios = alignment_ratios[block_idx]

                        # PLUS version: ratios is a dict with {'r_text': ..., 'r_cond': ...}
                        # Standard version: ratios is a single float value
                        if isinstance(ratios, dict):
                            # PLUS mode: use both ratios
                            r_text_i = ratios['r_text']
                            r_cond_i = ratios['r_cond']
                            combined_ratio = r_text_i * r_cond_i
                        else:
                            # Standard mode: single ratio (backward compatibility)
                            combined_ratio = ratios
                            r_text_i = ratios
                            r_cond_i = 1.0

                        original_gate = gate.item() if isinstance(gate, torch.Tensor) else gate
                        gate = gate * combined_ratio

                        # Enhanced logging for debugging - show first step and first few blocks
                        step_i = step_idx_ref[0]
                        if step_i == 0:
                            # Show all blocks in first step for debug mode verification
                            sorted_blocks = sorted(alignment_ratios.keys())
                            if block_idx in sorted_blocks[:10]:  # Show first 10 blocks
                                scaled_gate = gate.item() if isinstance(gate, torch.Tensor) else gate
                                if isinstance(ratios, dict):
                                    logging.info(f"  🎯 [Block {block_idx}] PLUS ALIGNMENT: "
                                               f"gate {original_gate:.4f} × r_text {r_text_i:.4f} × r_cond {r_cond_i:.4f} = {scaled_gate:.4f}")
                                else:
                                    logging.info(f"  🎯 [Block {block_idx}] ONE-STEP ALIGNMENT: "
                                               f"gate {original_gate:.4f} × ratio {combined_ratio:.4f} = {scaled_gate:.4f}")
                            elif block_idx == sorted_blocks[10]:
                                logging.info(f"  ... (gate scaling applied to all {len(alignment_ratios)} FPS blocks)")

                # MODE 2: Capture rboth_ijk = ||y_text|| (LEGACY - TEXT MAGNITUDE ALIGNMENT)
                elif capture_rboth:
                    # NEW: Just capture the text magnitude (with base LoRA + FPS)
                    y_text_norm = torch.norm(y_text).item()
                    rboth = y_text_norm  # Store ||y_text|| directly

                    step_i = step_idx_ref[0]
                    rboth_storage[(step_i, block_idx, fps_idx)] = rboth

                # MODE 3: Apply alignment ratios (LEGACY - TEXT MAGNITUDE ALIGNMENT)
                elif alignment_ratios is not None:
                    step_i = step_idx_ref[0]
                    key = (step_i, block_idx, fps_idx)

                    if key in alignment_ratios:
                        rboth = alignment_ratios[key]  # ||y_text|| with base LoRA + FPS

                        # NEW: Compute rfps_ijk = ||y_text|| (current, FPS only, no base LoRA)
                        y_text_norm = torch.norm(y_text).item()
                        rfps = y_text_norm

                        # Alignment ratio: scale to match text magnitude from calibration
                        align_ratio = rboth / (rfps + 1e-8)

                        # Scale y_fps by this ratio
                        y_fps = y_fps * align_ratio

                        # Collect stats for Subtask 3
                        if step_i not in alignment_stats['ratios_sum']:
                            alignment_stats['ratios_sum'][step_i] = 0.0
                            alignment_stats['ratios_count'][step_i] = 0
                        alignment_stats['ratios_sum'][step_i] += align_ratio
                        alignment_stats['ratios_count'][step_i] += 1

                # Compute final output
                y_combined = y_text + (warmup_factor * gate) * y_fps
                return y_combined

            return wrapper

        # Wrap all FPS adapters
        step_idx_container = [0]  # Mutable container to share step_idx
        for block_idx, fps_adapter in fps_adapter_blocks:
            original_fps_forwards[block_idx] = fps_adapter.forward
            wrapper = make_fps_wrapper(fps_adapter.forward, block_idx, step_idx_container)
            wrapper.__self__ = fps_adapter  # Bind module reference
            fps_adapter.forward = wrapper

    # Denoising loop
    for step_idx, t in enumerate(tqdm(timesteps, desc=f"FPS={fps}")):
        # Update step index for FPS adapter wrappers
        if capture_rboth or alignment_ratios is not None:
            step_idx_container[0] = step_idx

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

    # Restore FPS adapter forwards
    if capture_rboth or alignment_ratios is not None:
        for block_idx, fps_adapter in fps_adapter_blocks:
            fps_adapter.forward = original_fps_forwards[block_idx]

    # Subtask 3: Print alignment stats (mean ratio per step)
    if alignment_ratios is not None and alignment_stats['ratios_count']:
        print(f"\n--- TEXT MAGNITUDE Alignment Stats for FPS={fps} (fps_idx={fps_idx}) ---", flush=True)
        for step_i in sorted(alignment_stats['ratios_sum'].keys()):
            mean_ratio = alignment_stats['ratios_sum'][step_i] / alignment_stats['ratios_count'][step_i]
            print(f"  Step {step_i:3d}: Mean ||y_text_both|| / ||y_text_fps|| ratio = {mean_ratio:.4f}", flush=True)

    # Decode to video
    logging.info("Decoding latents...")

    # Move latents to CPU temporarily to free GPU memory, then back to GPU for VAE
    # This ensures we have maximum free memory for VAE decoding
    device = next(pipeline.vae.model.parameters()).device
    latents_cpu = latents.cpu()
    torch.cuda.empty_cache()
    latents = latents_cpu.to(device)
    del latents_cpu

    video_tensor = pipeline.vae.decode([latents])[0]

    # Clear cache after VAE decoding
    del latents
    torch.cuda.empty_cache()

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

def save_video_result(tensor, fps, prompt_short, output_dir, size, suffix=""):
    """Saves video with descriptive filename.

    Args:
        suffix: Optional suffix to add before timestamp (e.g., "_calib", "_aligned")
    """
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_prompt = "".join(c for c in prompt_short if c.isalnum() or c in (' ', '_')).strip()[:30]
    fps_str = f"{fps:.2f}".replace('.', '_')
    filename = f"fps_{fps_str}_{clean_prompt.replace(' ', '_')}_{size[0]}x{size[1]}{suffix}_{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)

    logging.info(f"💾 Saving: {filename}")
    cache_video(tensor=tensor[None], save_file=filepath, fps=16, normalize=True, value_range=(-1, 1))

    return filepath

def run_fps_experiments(pipeline, config, base_prompt, n_prompt, fps_values, output_dir, checkpoint_path=None, force_gate_one=False, align=False, debug_gate_scale=None, **generation_kwargs):
    """Runs multiple FPS conditioning experiments."""
    logging.info("🧪 Starting FPS conditioning experiments...")
    logging.info(f"  FPS values to test: {fps_values}")
    logging.info(f"  Base prompt: {base_prompt}")
    logging.info(f"  Alignment mode: {align}")
    if debug_gate_scale is not None:
        logging.info(f"  🐛 DEBUG MODE: Hardcoded gate scale = {debug_gate_scale}")

    os.makedirs(output_dir, exist_ok=True)

    alignment_ratios = None
    calibration_videos = []

    # NEW ONE-STEP ALIGNMENT: If --align flag is set, run calibration first
    if align:
        if not checkpoint_path:
            raise ValueError("--align requires --checkpoint to be specified")

        # 🐛 DEBUG MODE: Skip calibration and use hardcoded gate scaling values
        if debug_gate_scale is not None:
            logging.info(f"\n🐛 DEBUG MODE ENABLED: Skipping calibration, using hardcoded gate scale = {debug_gate_scale}")
            logging.info("   This mode verifies the alignment mechanism is working correctly")
            logging.info("   and that base LoRA is excluded during inference.\n")

            # Create hardcoded alignment ratios for all FPS blocks
            # We need to identify which blocks have FPS adapters
            fps_blocks = []
            for block_idx, block in enumerate(pipeline.transformer.blocks):
                if hasattr(block, 'fps_adapter') and block.fps_adapter is not None:
                    fps_blocks.append(block_idx)

            alignment_ratios = {block_idx: debug_gate_scale for block_idx in fps_blocks}

            logging.info(f"🐛 DEBUG: Created hardcoded alignment ratios for {len(fps_blocks)} blocks:")
            for block_idx in sorted(fps_blocks)[:5]:  # Show first 5
                logging.info(f"    Block {block_idx}: ratio = {debug_gate_scale}")
            if len(fps_blocks) > 5:
                logging.info(f"    ... and {len(fps_blocks) - 5} more blocks")

            # Load checkpoint with FPS-only mode
            logging.info(f"\n🔧 Loading checkpoint in FPS-ONLY mode (debug mode inference)...")
            base_rank = config.get('adapter', {}).get('rank', 32)
            apply_checkpoint(pipeline, checkpoint_path, rank=base_rank, fps_only=True, config=config)
            logging.info(f"   ✅ Loaded FPS adapters only (base LoRA excluded)\n")

        else:
            # Normal calibration mode
            # Use the first FPS value as a representative condition for calibration
            calibration_fps = fps_values[0]
            logging.info(f"Using FPS={calibration_fps} as reference for one-step alignment calibration")

            alignment_ratios = calibrate_alignment_ratios_onestep(
                pipeline, config, checkpoint_path, base_prompt, n_prompt, calibration_fps,
                seed=generation_kwargs.get('seed', 42),
                steps=generation_kwargs.get('steps', 25),
                scale=generation_kwargs.get('scale', 7.0),
                frames=generation_kwargs.get('frames', 49),
                size=generation_kwargs.get('size', (512, 320)),
                shift=generation_kwargs.get('shift', 3.0)
            )

            if alignment_ratios:
                # ⚠️ CRITICAL FIX: During calibration, base_only=True configured base LoRA adapters on ALL blocks
                # Even though weights are filtered in fps_only mode, the PEFT adapter structure persists
                # We MUST reload the entire pipeline fresh to remove those adapter wrappers
                print(f"\n🔧 CRITICAL: Reloading pipeline FRESH after calibration...", flush=True)
                print(f"   Reason: Calibration added base LoRA wrappers to ALL blocks that must be removed", flush=True)
                print(f"   (FPS adapters are only on deepest third, but base LoRA was on all blocks)", flush=True)

                # Get the TOML path from config (we'll need this to reload)
                toml_path = config.get('_toml_path')
                if not toml_path:
                    raise ValueError("Config missing '_toml_path' - cannot reload pipeline after calibration")

                # Free GPU memory from old pipeline before reloading
                print(f"🧹 Freeing GPU memory from calibration pipeline...", flush=True)
                import gc
                del pipeline.transformer
                del pipeline.vae
                del pipeline.text_encoder
                del pipeline
                torch.cuda.empty_cache()
                gc.collect()
                print(f"   ✅ Old pipeline freed from GPU memory", flush=True)

                # Reload pipeline completely fresh from TOML
                print(f"🔧 Loading fresh pipeline from TOML...", flush=True)
                pipeline, config = load_pipeline_from_toml(toml_path)

                # Now load checkpoint with FPS-only mode (clean pipeline, no base LoRA wrappers)
                print(f"\n🔧 Loading checkpoint in FPS-ONLY mode for inference with alignment...", flush=True)
                base_rank = config.get('adapter', {}).get('rank', 32)
                apply_checkpoint(pipeline, checkpoint_path, rank=base_rank, fps_only=True, config=config)
                print(f"   ✅ Inference will use ONLY FPS adapters with alignment\n", flush=True)

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
                alignment_ratios=alignment_ratios,
                apply_onestep_alignment=align,  # Pass alignment flag
                fps_idx=i-1,  # FPS index for alignment lookup
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

def validate_fps_config(config):
    """Validates that the config has FPS parameter definitions for selective loading."""
    model_config = config.get('model', {})
    
    # Check for required FPS parameters
    fps_params = [
        'fps_adapter_rank', 'fps_adapter_num_tokens', 'fps_embed_dim', 
        'fps_lora_alpha', 'fps_condition_blocks'
    ]
    
    missing_params = [param for param in fps_params if param not in model_config]
    if missing_params:
        raise ValueError(
            f"--fps_only requires FPS parameters in config, but missing: {missing_params}. "
            f"Cannot load FPS-only parameters from checkpoint."
        )
    
    logging.info("✅ Config validation passed: FPS parameters found in configuration")

def validate_base_lora_config(config):
    """Validates that the config has base LoRA parameter definitions for selective loading."""
    if 'adapter' not in config:
        raise ValueError(
            "--base_only requires [adapter] section in config, but it's missing. "
            "Cannot load base LoRA parameters from checkpoint."
        )
    
    adapter_config = config['adapter']
    required_params = ['type', 'rank']
    missing_params = [param for param in required_params if param not in adapter_config]
    
    if missing_params:
        raise ValueError(
            f"--base_only requires adapter parameters in config, but missing: {missing_params}. "
            f"Cannot load base LoRA parameters from checkpoint."
        )
    
    if adapter_config['type'] != 'lora':
        raise ValueError(
            f"--base_only requires adapter type 'lora', but found '{adapter_config['type']}'. "
            f"Cannot load base LoRA parameters from checkpoint."
        )
    
    logging.info("✅ Config validation passed: Base LoRA parameters found in configuration")

def load_prompts_from_folder(folder_path):
    """
    Load prompts from text files in a folder.

    Args:
        folder_path: Path to folder containing .txt files with prompts

    Returns:
        List of (prompt_id, prompt_text) tuples
    """
    from pathlib import Path

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Prompt folder does not exist: {folder_path}")

    prompts = []
    txt_files = sorted(folder.glob('*.txt'))

    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found in prompt folder: {folder_path}")

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
            if prompt_text:
                prompts.append((txt_file.stem, prompt_text))
                logging.info(f"  ✓ Loaded '{txt_file.stem}': {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")

    return prompts


def main():
    parser = argparse.ArgumentParser(description='Multi-FPS experiment script - TOML-aligned with one-step alignment')
    parser.add_argument('--config', required=True, help='Path to TOML configuration file')
    parser.add_argument('--checkpoint', required=True, help='Path to FPS checkpoint')
    parser.add_argument('--output_dir', default='./fps_experiments_align', help='Output directory')
    parser.add_argument('--fps_values', nargs='+', type=float, default=[12, 24, 60], help='FPS values to test')
    parser.add_argument('--prompt', default=None, help='Generation prompt (single string)')
    parser.add_argument('--prompt_folder', default=None, help='Folder containing .txt files with prompts (alternative to --prompt)')
    parser.add_argument('--negative_prompt', default='', help='Negative prompt (optional)')
    parser.add_argument('--steps', type=int, default=15, help='Denoising steps')
    parser.add_argument('--frames', type=int, default=33, help='Number of frames')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', default='29501', help='DeepSpeed port')
    parser.add_argument('--width', type=int, help='Video width (overrides TOML if specified)')
    parser.add_argument('--height', type=int, help='Video height (overrides TOML if specified)')
    parser.add_argument('--scale', type=float, help='CFG scale (overrides TOML if specified)')
    parser.add_argument('--force_gate_one', action='store_true', help='🔧 DIAGNOSTIC: Force all FPS adapter gates to 1.0 for maximum FPS impact')
    parser.add_argument('--fps_only', action='store_true', help='Load only FPS-related parameters from checkpoint (ignore base LoRA)')
    parser.add_argument('--base_only', action='store_true', help='Load only base LoRA parameters from checkpoint (ignore FPS parameters)')
    parser.add_argument('--align', action='store_true', help='Enable NEW alignment mode: calibrate rboth_ijk ratios and apply during fps_only inference')
    parser.add_argument('--debug_gate_scale', type=float, default=None, help='🐛 DEBUG MODE: Hardcode gate scaling value for all blocks (skips calibration, verifies alignment mechanism)')

    args = parser.parse_args()

    # Validate prompt arguments
    if args.prompt is None and args.prompt_folder is None:
        parser.error("Either --prompt or --prompt_folder must be provided")
    if args.prompt is not None and args.prompt_folder is not None:
        parser.error("--prompt and --prompt_folder are mutually exclusive. Choose one.")

    # Validate selective loading arguments
    if args.fps_only and args.base_only:
        parser.error("--fps_only and --base_only are mutually exclusive. Choose one or neither.")
    
    try:
        # Setup
        setup_environment(args.port)
        
        # Load pipeline using TOML configuration
        logging.info("Loading pipeline from TOML configuration...")
        pipeline, config = load_pipeline_from_toml(args.config)
        
        # Validate selective loading requirements
        if args.fps_only:
            validate_fps_config(config)
            logging.info("🎯 FPS-ONLY MODE: Will load only FPS-related parameters")
        elif args.base_only:
            validate_base_lora_config(config)
            logging.info("🎯 BASE-ONLY MODE: Will load only base LoRA parameters")
        
        # Apply checkpoint (skip if using alignment mode, as it will load checkpoints during calibration)
        if not args.align:
            logging.info("Applying checkpoint...")
            # Get base LoRA rank from TOML config or default
            base_rank = config.get('adapter', {}).get('rank', 32)
            pipeline = apply_checkpoint(pipeline, args.checkpoint, rank=base_rank,
                                       fps_only=args.fps_only, base_only=args.base_only, config=config)
        else:
            logging.info("Skipping initial checkpoint load (alignment mode will load checkpoints during calibration)")
        
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
        logging.info(f"  Negative prompt: '{args.negative_prompt}'")

        # Prepare prompts list
        if args.prompt_folder:
            # Load multiple prompts from folder
            logging.info(f"Loading prompts from folder: {args.prompt_folder}")
            prompts = load_prompts_from_folder(args.prompt_folder)
            logging.info(f"Loaded {len(prompts)} prompts")
        else:
            # Single prompt from command line
            prompts = [("single", args.prompt)]
            logging.info(f"  Prompt: '{args.prompt}'")

        # Run experiments for each prompt
        all_results = []
        for prompt_idx, (prompt_id, prompt_text) in enumerate(prompts, 1):
            if len(prompts) > 1:
                logging.info(f"\n{'='*80}")
                logging.info(f"PROCESSING PROMPT {prompt_idx}/{len(prompts)}: {prompt_id}")
                logging.info(f"{'='*80}")
                logging.info(f"Prompt text: {prompt_text}")

            # IMPORTANT: For alignment mode with multiple prompts, reload pipeline fresh for each prompt
            # This prevents PEFT adapter configuration conflicts when calibrating per-prompt
            if args.align and len(prompts) > 1 and prompt_idx > 1:
                logging.info(f"🔄 Reloading pipeline fresh for prompt {prompt_idx} (alignment mode)")
                pipeline, config = load_pipeline_from_toml(args.config)

            # Create subdirectory for this prompt if using multiple prompts
            if len(prompts) > 1:
                prompt_output_dir = os.path.join(args.output_dir, prompt_id)
            else:
                prompt_output_dir = args.output_dir

            # Run experiments for this prompt
            # NOTE: For alignment mode, calibration is done PER PROMPT inside run_fps_experiments
            results = run_fps_experiments(
                pipeline=pipeline,
                config=config,  # Pass TOML config for proper dtype handling
                base_prompt=prompt_text,
                n_prompt=args.negative_prompt,  # Use provided negative prompt or empty string
                fps_values=args.fps_values,
                output_dir=prompt_output_dir,
                checkpoint_path=args.checkpoint,  # NEW: Pass checkpoint path for alignment
                seed=args.seed,
                steps=args.steps,
                frames=args.frames,
                size=(width, height),
                scale=scale,
                shift=3.0,
                force_gate_one=args.force_gate_one,
                align=args.align,  # NEW: Pass alignment flag
                debug_gate_scale=args.debug_gate_scale  # NEW: Pass debug mode gate scaling value
            )

            all_results.append({
                'prompt_id': prompt_id,
                'prompt_text': prompt_text,
                'results': results,
                'output_dir': prompt_output_dir
            })

        logging.info(f"\n🎉 All experiments complete! Results saved to: {args.output_dir}")
        if len(prompts) > 1:
            logging.info(f"  Processed {len(prompts)} prompts")
            for result in all_results:
                logging.info(f"    {result['prompt_id']}: {result['output_dir']}")
        
    except Exception as e:
        logging.error(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())