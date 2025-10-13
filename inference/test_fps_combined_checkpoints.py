#!/usr/bin/env python3
"""
Multi-checkpoint FPS conditioning test script.
Allows loading FPS adapters from one checkpoint and base LoRA from another.
This enables mixing and matching different trained components.
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

    # Initialize pipeline using TOML config (same as train.py)
    wan_t2v = WanPipeline(config)

    logging.info("Loading main transformer model...")
    wan_t2v.load_diffusion_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Moving model components to device: {device}")

    wan_t2v.transformer.to(device)
    wan_t2v.vae.model.to(device)
    wan_t2v.text_encoder.model.to(device)

    # Set models to eval mode for sampling
    logging.info("Setting models to eval mode for inference...")
    wan_t2v.transformer.eval()
    wan_t2v.vae.model.eval()
    wan_t2v.text_encoder.model.eval()

    logging.info("Pipeline loaded successfully with TOML configuration.")
    return wan_t2v, config

def load_combined_checkpoints(pipeline, checkpoint_fps=None, checkpoint_base=None, config=None):
    """
    Load FPS adapters and base LoRA from separate checkpoints.

    Args:
        pipeline: WanPipeline instance
        checkpoint_fps: Path to checkpoint containing FPS adapters (optional)
        checkpoint_base: Path to checkpoint containing base LoRA (optional)
        config: TOML config dict for proper adapter initialization
    """
    import safetensors
    from pathlib import Path

    if checkpoint_fps is None and checkpoint_base is None:
        logging.warning("No checkpoints specified. Using base model only.")
        return

    # Get adapter configuration for proper initialization
    if config and 'adapter' in config:
        adapter_config = config['adapter'].copy()  # Copy to avoid modifying original config
        rank = adapter_config.get('rank', 32)

        # Ensure required fields are present (required by configure_adapter)
        if 'alpha' not in adapter_config:
            adapter_config['alpha'] = rank  # Common default: alpha = rank
        if 'dropout' not in adapter_config:
            adapter_config['dropout'] = 0.0  # Default: no dropout for inference
        if 'type' not in adapter_config:
            adapter_config['type'] = 'lora'  # Default adapter type

        # Convert dtype string to torch dtype if needed
        if 'dtype' in adapter_config and isinstance(adapter_config['dtype'], str):
            import torch
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            adapter_config['dtype'] = dtype_map.get(adapter_config['dtype'], torch.bfloat16)

        logging.info(f"Using base LoRA rank: {rank}, alpha: {adapter_config['alpha']}, dropout: {adapter_config['dropout']}, dtype: {adapter_config.get('dtype', 'default')}")

        # Configure adapter on the model
        pipeline.configure_adapter(adapter_config)
    else:
        rank = 32
        logging.info(f"No adapter config found, using default rank: {rank}")

    # Track what we're loading
    fps_params_loaded = 0
    base_params_loaded = 0

    # === STEP 1: Load FPS parameters ===
    if checkpoint_fps:
        logging.info(f"\n{'='*80}")
        logging.info(f"Loading FPS parameters from: {checkpoint_fps}")
        logging.info(f"{'='*80}")

        checkpoint_dir = Path(checkpoint_fps)
        if checkpoint_dir.is_dir():
            safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
            if safetensors_files:
                checkpoint_file = safetensors_files[0]
                logging.info(f"Reading checkpoint: {checkpoint_file.name}")
                checkpoint = safetensors.torch.load_file(str(checkpoint_file))

                # Filter for FPS parameters only
                fps_params = {}
                for k, v in checkpoint.items():
                    # Strip 'diffusion_model.' prefix if present
                    key = k.replace('diffusion_model.', '', 1) if k.startswith('diffusion_model.') else k

                    if 'fps_conditioning' in key or 'fps_adapter' in key:
                        fps_params[key] = v

                if fps_params:
                    # Load FPS parameters
                    missing, unexpected = pipeline.transformer.load_state_dict(fps_params, strict=False)
                    fps_params_loaded = len(fps_params)

                    logging.info(f"✓ Loaded {fps_params_loaded} FPS parameters")
                    if missing:
                        logging.info(f"  Missing keys (expected): {len(missing)}")
                    if unexpected:
                        logging.info(f"  Unexpected keys: {len(unexpected)} (first 5: {list(unexpected)[:5]})")
                else:
                    logging.warning("No FPS parameters found in checkpoint_fps!")
            else:
                logging.error(f"No .safetensors files found in {checkpoint_dir}")
        else:
            logging.error(f"Checkpoint directory not found: {checkpoint_dir}")

    # === STEP 2: Load base LoRA parameters ===
    if checkpoint_base:
        logging.info(f"\n{'='*80}")
        logging.info(f"Loading base LoRA from: {checkpoint_base}")
        logging.info(f"{'='*80}")

        checkpoint_dir = Path(checkpoint_base)
        if checkpoint_dir.is_dir():
            safetensors_files = list(checkpoint_dir.glob('*.safetensors'))
            if safetensors_files:
                checkpoint_file = safetensors_files[0]
                logging.info(f"Reading checkpoint: {checkpoint_file.name}")
                checkpoint = safetensors.torch.load_file(str(checkpoint_file))

                # Filter for base LoRA parameters only (exclude FPS)
                base_params = {}
                for k, v in checkpoint.items():
                    # Strip 'diffusion_model.' prefix if present
                    key = k.replace('diffusion_model.', '', 1) if k.startswith('diffusion_model.') else k

                    # Include LoRA parameters that are NOT FPS-related
                    if ('.lora_A' in key or '.lora_B' in key) and 'fps' not in key:
                        base_params[key] = v

                if base_params:
                    # Load base LoRA parameters
                    missing, unexpected = pipeline.transformer.load_state_dict(base_params, strict=False)
                    base_params_loaded = len(base_params)

                    logging.info(f"✓ Loaded {base_params_loaded} base LoRA parameters")
                    if missing:
                        logging.info(f"  Missing keys (expected): {len(missing)}")
                    if unexpected:
                        logging.info(f"  Unexpected keys: {len(unexpected)} (first 5: {list(unexpected)[:5]})")
                else:
                    logging.warning("No base LoRA parameters found in checkpoint_base!")
            else:
                logging.error(f"No .safetensors files found in {checkpoint_dir}")
        else:
            logging.error(f"Checkpoint directory not found: {checkpoint_dir}")

    # === STEP 3: Summary ===
    logging.info(f"\n{'='*80}")
    logging.info(f"CHECKPOINT LOADING SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"FPS parameters loaded: {fps_params_loaded}")
    logging.info(f"Base LoRA parameters loaded: {base_params_loaded}")
    logging.info(f"Total parameters loaded: {fps_params_loaded + base_params_loaded}")
    logging.info(f"{'='*80}\n")

def generate_video_with_fps(pipeline, config, prompt, n_prompt, fps, seed=42, steps=25,
                            scale=7.0, frames=49, size=(512, 320), shift=3.0, force_gate_one=False):
    """Generate a single video with specified FPS conditioning."""
    logging.info(f"🎬 Generating video with FPS={fps}")
    logging.info(f"  Prompt: {prompt}")
    logging.info(f"  Steps: {steps}, Frames: {frames}, Size: {size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare latent shape (from pipeline logic)
    height, width = size
    num_frames = frames

    # Calculate latent dimensions (based on VAE downsampling)
    latent_height = height // 8
    latent_width = width // 8
    latent_frames = (num_frames - 1) // 4 + 1

    target_shape = (1, 16, latent_frames, latent_height, latent_width)
    latents = torch.randn(target_shape, device=device)

    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]

    # Setup scheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(steps, device=device, shift=shift)
    timesteps = scheduler.timesteps

    # Encode text
    text_encoder_device = next(pipeline.text_encoder.model.parameters()).device
    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(text_encoder_device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)

    # Concatenate for CFG
    context = torch.cat([uncond_inputs['encoder_hidden_states'], cond_inputs['encoder_hidden_states']], dim=0)
    context_mask = torch.cat([uncond_inputs['encoder_attention_mask'], cond_inputs['encoder_attention_mask']], dim=0)

    # Prepare for generation
    torch.manual_seed(seed)
    initial_layer.eval()
    for layer in transformer_layers:
        layer.eval()
    final_layer.eval()

    # Check FPS configuration
    logging.info(f"  FPS tensor: [{fps}]")

    # Log FPS parameter status
    logging.info(f"🔍 FPS parameter status before generation:")
    fps_mlp_params = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_conditioning' in n)
    fps_adapter_params = sum(1 for n, _ in pipeline.transformer.named_parameters() if 'fps_adapter' in n)
    logging.info(f"  fps_conditioning.lin1.weight: mean={pipeline.transformer.fps_conditioning.lin1.weight.mean().item():.6f}, max={pipeline.transformer.fps_conditioning.lin1.weight.abs().max().item():.6f}")
    logging.info(f"  fps_conditioning.lin1.bias: mean={pipeline.transformer.fps_conditioning.lin1.bias.mean().item():.6f}, max={pipeline.transformer.fps_conditioning.lin1.bias.abs().max().item():.6f}")

    # Check FPS adapter gates
    fps_adapter_count = 0
    for layer in transformer_layers:
        if hasattr(layer, 'block') and hasattr(layer.block, 'fps_adapter') and layer.block.fps_adapter is not None:
            adapter = layer.block.fps_adapter
            if hasattr(adapter, 'gate_mode'):
                if adapter.gate_mode == 'fixed':
                    logging.info(f"  {layer.__class__.__name__}.fps_adapter.gate_fixed: gate={adapter.gate_fixed.item():.6f} (FIXED MODE)")
                elif adapter.gate_mode == 'sigmoid':
                    gate_value = torch.sigmoid(adapter.gate_alpha).item()
                    logging.info(f"  {layer.__class__.__name__}.fps_adapter.gate_sigmoid: alpha={adapter.gate_alpha.item():.6f}, gate={gate_value:.6f}")
            fps_adapter_count += 1

    logging.info(f"  Found {fps_mlp_params} FPS conditioning params, {fps_adapter_params} FPS adapter params")

    # Denoising loop
    with torch.no_grad():
        x_t = latents

        for step_idx, t in enumerate(tqdm(timesteps, desc=f"FPS={fps}")):
            timestep_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # Initial layer (embed time + FPS)
            fps_values = [fps]  # Single FPS value for this generation
            time_emb, fps_emb = initial_layer(timestep_tensor, fps_values=fps_values)

            # Debug FPS conditioning tensor at first step
            if step_idx == 0:
                logging.info(f"  [Step {step_idx}] FPS conditioning tensor: mean={fps_emb.mean().item():.6f}, std={fps_emb.std().item():.6f}, max={fps_emb.abs().max().item():.6f}")

            # Apply force_gate_one if requested
            if force_gate_one:
                for layer in transformer_layers:
                    if hasattr(layer, 'block') and hasattr(layer.block, 'fps_adapter'):
                        adapter = layer.block.fps_adapter
                        if adapter is not None and hasattr(adapter, 'gate_alpha'):
                            # Force gate to 1.0 by setting alpha very high (for sigmoid mode)
                            if adapter.gate_mode == 'sigmoid':
                                adapter.gate_alpha.data.fill_(10.0)  # sigmoid(10) ≈ 1.0
                            elif adapter.gate_mode == 'fixed':
                                adapter.gate_fixed.data.fill_(1.0)

            # Transformer layers
            for layer in transformer_layers:
                x_t = layer(x_t, time_emb, context, context_mask, fps_conditioning=fps_emb)

            # Final layer
            pred = final_layer(x_t)

            # Scheduler step
            x_t = scheduler.step(pred, t, x_t, return_dict=False)[0]

    # Decode
    logging.info("Decoding latents...")
    vae_device = next(pipeline.vae.model.parameters()).device
    video_tensor = pipeline.vae.decode_video(x_t.to(vae_device), num_frames=num_frames)

    # Log FPS parameter status after generation
    logging.info(f"🔍 FPS parameter status after generation:")
    logging.info(f"  fps_conditioning.lin1.weight: mean={pipeline.transformer.fps_conditioning.lin1.weight.mean().item():.6f}, max={pipeline.transformer.fps_conditioning.lin1.weight.abs().max().item():.6f}")

    logging.info(f"✅ Generation complete for FPS={fps}")

    return video_tensor

def save_video_result(tensor, fps, prompt_short, output_dir, size):
    """Saves video with descriptive filename."""
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_prompt = "".join(c for c in prompt_short if c.isalnum() or c in (' ', '_')).strip()[:30]
    fps_str = f"{fps:.2f}".replace('.', '_')
    filename = f"fps_{fps_str}_{clean_prompt.replace(' ', '_')}_{size[0]}x{size[1]}_{timestamp}.mp4"
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
                config=config,
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
                prompt_short=base_prompt.split()[0:3],
                output_dir=output_dir,
                size=generation_kwargs.get('size', (512, 320))
            )

            results.append({
                'fps': fps,
                'file': output_file,
                'mean': video_tensor.mean().item(),
                'std': video_tensor.std().item(),
                'min': video_tensor.min().item(),
                'max': video_tensor.max().item()
            })

            logging.info(f"✅ Experiment {i} complete")

        except Exception as e:
            logging.error(f"❌ Experiment {i} failed: {e}")
            import traceback
            traceback.print_exc()

    # Print results summary
    logging.info("\n📊 FPS Conditioning Analysis:")
    logging.info(f"{'FPS':<7} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'File':<50}")
    logging.info("-" * 100)
    for r in results:
        logging.info(f"{r['fps']:<7.1f} {r['mean']:<10.4f} {r['std']:<10.4f} {r['min']:<10.4f} {r['max']:<10.4f} {os.path.basename(r['file']) if r['file'] else 'N/A':<50}")

    # Check variation
    if len(results) > 1:
        means = [r['mean'] for r in results]
        mean_variation = max(means) - min(means)
        logging.info(f"\n⚠️  WARNING: Low variation between FPS values. Mean variation: {mean_variation:.4f}")

    logging.info(f"\n🎉 All experiments complete! Results saved to: {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Combined Checkpoint FPS Conditioning Test")
    parser.add_argument('--config', type=str, required=True, help='Path to TOML config file')
    parser.add_argument('--checkpoint_fps', type=str, default=None, help='Path to checkpoint containing FPS adapters')
    parser.add_argument('--checkpoint_base', type=str, default=None, help='Path to checkpoint containing base LoRA')
    parser.add_argument('--fps_values', type=float, nargs='+', default=[1.0, 8.0, 16.0, 24.0, 30.0, 60.0],
                       help='List of FPS values to test')
    parser.add_argument('--steps', type=int, default=25, help='Number of denoising steps')
    parser.add_argument('--frames', type=int, default=49, help='Number of frames to generate')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for videos')
    parser.add_argument('--prompt', type=str, default="A serene mountain landscape with flowing water.",
                       help='Text prompt for generation')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', type=str, default='29501', help='DeepSpeed port')
    parser.add_argument('--width', type=int, default=512, help='Video width')
    parser.add_argument('--height', type=int, default=320, help='Video height')
    parser.add_argument('--scale', type=float, default=None, help='CFG scale (overrides config)')
    parser.add_argument('--force_gate_one', action='store_true', help='Force FPS adapter gates to 1.0')

    args = parser.parse_args()

    # Validate arguments
    if args.checkpoint_fps is None and args.checkpoint_base is None:
        parser.error("At least one of --checkpoint_fps or --checkpoint_base must be provided")

    # Setup
    setup_environment(port=args.port)

    # Load pipeline
    pipeline, config = load_pipeline_from_toml(args.config)

    # Load combined checkpoints
    load_combined_checkpoints(
        pipeline=pipeline,
        checkpoint_fps=args.checkpoint_fps,
        checkpoint_base=args.checkpoint_base,
        config=config
    )

    # Prepare generation parameters
    width, height = args.width, args.height
    scale = args.scale if args.scale is not None else config.get('inference', {}).get('cfg_scale', 7.0)

    # Run experiments
    results = run_fps_experiments(
        pipeline=pipeline,
        config=config,
        base_prompt=args.prompt,
        n_prompt=args.negative_prompt,
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

    logging.info("✨ Script complete!")

if __name__ == "__main__":
    main()
