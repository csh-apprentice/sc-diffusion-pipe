# FILE: lora_inference_sct2v_cross.py
# FPS-enabled inference script for WAN T2V with FPS conditioning support

import torch
import os
import sys
import logging
from datetime import datetime
import deepspeed
from tqdm import tqdm
import math

# --- Prerequisites ---
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')

from models.wan.wan import WanPipeline
import peft
from inference_utils.fm_solvers_unipc import FlowUniPCMultistepScheduler # Import the sampler
from inference_utils.utils import cache_video

# --- Toolkit Functions ---

def setup_environment():
    """Initializes the DeepSpeed distributed environment for standalone use."""
    if not deepspeed.comm.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        deepspeed.init_distributed()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logging.info("DeepSpeed environment initialized for single-GPU inference.")

def load_base_pipeline(ckpt_path, dtype=torch.bfloat16):
    """Loads the heavy base WanPipeline. THIS IS THE SLOW PART."""
    logging.info("Initializing WanPipeline with FPS conditioning support...")
    pipeline_config = {
        "model": {
            "ckpt_path": ckpt_path, 
            "dtype": dtype, 
            "transformer_dtype": dtype,
            # FPS adapter configuration to match training
            "fps_adapter_rank": 4,
            "fps_adapter_gate_init": 0.0,
            "fps_condition_blocks": "deepest_third"
        }
    }
    wan_t2v = WanPipeline(config=pipeline_config)
    
    logging.info("Loading main transformer model... (This may take a while)")
    wan_t2v.load_diffusion_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Moving model components to device: {device}")
    
    # Move ALL necessary model parts to the GPU
    wan_t2v.transformer.to(device)
    wan_t2v.vae.model.to(device)
    
    logging.info("Pipeline and transformer loaded successfully.")
    return wan_t2v

def detect_checkpoint_type(lora_path):
    """
    Detects whether a checkpoint contains base LoRA parameters or is FPS-only.
    Returns: ('fps_only', 'lora_and_fps', 'unknown')
    """
    import safetensors
    try:
        # Load the checkpoint to inspect parameter names
        checkpoint = safetensors.torch.load_file(lora_path)
        
        has_base_lora = False
        has_fps_params = False
        
        for key in checkpoint.keys():
            if 'fps_conditioning' in key or 'fps_adapter' in key:
                has_fps_params = True
            elif key.endswith('.weight') and ('lora_A' in key or 'lora_B' in key or '.weight' in key):
                # Check if this is a base LoRA parameter (not FPS-related)
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

def apply_lora_weights_with_fps(wan_t2v_pipeline, lora_path, rank=32, dtype=torch.bfloat16):
    """
    Applies LoRA weights including FPS adapter and MLP parameters.
    This function loads both regular LoRA weights AND FPS conditioning parameters.
    Now supports FPS-only checkpoints.
    """
    logging.info(f"Applying LoRA weights with FPS conditioning from: {lora_path}")
    
    # Detect checkpoint type
    checkpoint_type = detect_checkpoint_type(lora_path)
    logging.info(f"Detected checkpoint type: {checkpoint_type}")
    
    # Only configure base LoRA adapter if checkpoint contains base LoRA parameters
    if checkpoint_type in ['lora_and_fps', 'unknown']:
        # Configure LoRA adapter for base LoRA + FPS
        adapter_config = {"type": "lora", "rank": rank, "alpha": rank, "dropout": 0.0, "dtype": dtype}
        wan_t2v_pipeline.configure_adapter(adapter_config)
        logging.info(f"Configured base LoRA adapter with rank {rank}")
    elif checkpoint_type == 'fps_only':
        logging.info("FPS-only checkpoint detected - skipping base LoRA adapter configuration")
    
    # Load adapter weights (this includes FPS parameters regardless of checkpoint type)
    wan_t2v_pipeline.load_adapter_weights(lora_path)
    
    # Verify FPS parameters are loaded
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
    
    logging.info(f"--- SUCCESS! Checkpoint applied. ---")
    logging.info(f"FPS MLP parameters loaded: {fps_mlp_params}")
    logging.info(f"FPS Adapter parameters loaded: {fps_adapter_params}")
    logging.info(f"Base LoRA parameters loaded: {base_lora_params}")
    
    if fps_mlp_params == 0 and fps_adapter_params == 0:
        logging.warning("⚠️  No FPS parameters found! This checkpoint may not support FPS conditioning.")
    
    return wan_t2v_pipeline

def generate_video_with_fps(pipeline, prompt, n_prompt, fps=60, seed=69, steps=25, scale=7.0, frames=81, size=(832, 480), shift=5.0, sample_solver='unipc'):
    """
    Generates a video with FPS conditioning support.
    
    Args:
        pipeline: WanPipeline with FPS conditioning
        prompt: Text prompt for generation
        n_prompt: Negative prompt
        fps: Frame rate for FPS conditioning (scalar value)
        seed: Random seed
        steps: Denoising steps
        scale: CFG scale
        frames: Number of frames
        size: Video resolution (width, height)
        shift: Sampling shift parameter
        sample_solver: Sampling method
    
    Returns:
        Generated video tensor
    """
    logging.info(f"Starting FPS-conditioned video generation...")
    logging.info(f"  FPS: {fps}")
    logging.info(f"  Solver: {sample_solver}, Shift: {shift}")
    logging.info(f"  Prompt: {prompt}")
    
    device = pipeline.transformer.device
    
    # 1. PREPARE INPUTS
    torch.manual_seed(seed)
    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)
    
    # Prepare FPS values for conditioning
    # Create FPS tensor for batch (batch_size=1 for inference)
    fps_values = torch.tensor([fps], dtype=torch.float32, device=device)  # Convert to tensor
    logging.info(f"Using FPS conditioning: {fps_values.tolist()}")
    
    vae_stride = [4, 8, 8]
    target_shape = (16, frames // vae_stride[0], size[1] // vae_stride[1], size[0] // vae_stride[2])
    latents = torch.randn(target_shape, device=device)

    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]

    # 2. SETUP SCHEDULER & SAMPLING LOOP
    if sample_solver == 'unipc':
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
    elif sample_solver == 'dpm++':
        scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(steps, shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    else:
        raise NotImplementedError(f"Unsupported solver: {sample_solver}")
    
    for t in tqdm(timesteps, desc="FPS-Conditioned Sampling"):
        t_batch = torch.full((1,), t, device=device)

        with torch.no_grad(), torch.autocast('cuda'):
            # UNCONDITIONAL PASS (batch_size = 1)
            # Include FPS values in the input tuple
            initial_input_uncond = (
                latents.unsqueeze(0), 
                None, 
                t_batch, 
                uncond_inputs['text_embeddings'].to(device), 
                uncond_inputs['seq_lens'].to(device), 
                None,
                fps_values  # Add FPS values here
            )
            
            # The initial layer should handle fps_values and create fps_conditioning
            layer_outputs = initial_layer(initial_input_uncond)
            # Extract outputs: x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
            
            # Pass through transformer layers (fps_conditioning should be passed through)
            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
                
            noise_pred_uncond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))

            # CONDITIONAL PASS (batch_size = 1)
            initial_input_cond = (
                latents.unsqueeze(0), 
                None, 
                t_batch, 
                cond_inputs['text_embeddings'].to(device), 
                cond_inputs['seq_lens'].to(device), 
                None,
                fps_values  # Add FPS values here
            )
            
            layer_outputs = initial_layer(initial_input_cond)
            x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
            
            for transformer_layer in transformer_layers:
                layer_outputs = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
                x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning = layer_outputs
                
            noise_pred_cond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context, fps_conditioning))
            
            # Classifier-Free Guidance
            noise_pred = noise_pred_uncond.squeeze(0) + scale * (noise_pred_cond.squeeze(0) - noise_pred_uncond.squeeze(0))
        
        # SCHEDULER STEP
        latents = scheduler.step(noise_pred, t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)

    logging.info("FPS-conditioned denoising loop complete.")
    
    # 3. DECODE LATENTS TO VIDEO
    logging.info("Decoding latents with VAE...")
    video_tensor = pipeline.vae.decode([latents])[0]
    
    logging.info("FPS-conditioned generation complete.")
    return video_tensor

def save_video(tensor, size, fps=60, output_dir="."):
    """Saves a video tensor to a file with FPS information in filename."""
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"video_fps{fps}_{size[0]}x{size[1]}_{formatted_time}.mp4")
    logging.info(f"Saving FPS-conditioned video to {output_filename}")
    cache_video(tensor=tensor[None], save_file=output_filename, fps=16, normalize=True, value_range=(-1, 1))
    logging.info("Video saved successfully!")
    return output_filename

def test_fps_conditioning():
    """Test function to verify FPS conditioning works with different FPS values."""
    logging.info("=== FPS CONDITIONING TEST ===")
    
    # Test parameters
    test_fps_values = [24, 60, 120]  # Test different FPS values
    base_prompt = "A beautiful sunset over calm ocean waves"
    n_prompt = "blurry, low quality, distorted"
    
    results = []
    
    for fps in test_fps_values:
        logging.info(f"\n--- Testing FPS = {fps} ---")
        
        video_tensor = generate_video_with_fps(
            pipeline=lora_pipeline,
            prompt=f"{base_prompt} at {fps} fps",
            n_prompt=n_prompt,
            fps=fps,
            seed=42,  # Same seed for comparison
            steps=25,  # Fewer steps for faster testing
            scale=7.0,
            frames=49,  # Shorter for testing
            size=(512, 320),  # Smaller for testing
            shift=3.0,
            sample_solver='unipc'
        )
        
        output_file = save_video(video_tensor, size=(512, 320), fps=fps, output_dir="/root/workspace/sc-diffusion-pipe/output/fps_test")
        results.append((fps, output_file))
        logging.info(f"FPS {fps} test complete: {output_file}")
    
    logging.info("\n=== FPS CONDITIONING TEST COMPLETE ===")
    for fps, file in results:
        logging.info(f"  FPS {fps}: {file}")
    
    return results

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
    setup_environment()

    # Updated paths to use our FPS-enabled checkpoints
    BASE_CKPT = "/root/workspace/diffusion-pipe/cindy_ckpts/Wan2.1-T2V-14B"
    # Use our checkpoint that contains FPS parameters
    LORA_PATH = "/root/workspace/sc-diffusion-pipe/checkpoints/20250822_19-38-12/epoch3"  # Latest verified checkpoint with FPS params
    
    # Create output directory for FPS tests
    os.makedirs("/root/workspace/sc-diffusion-pipe/output/fps_test", exist_ok=True)
    
    try:
        logging.info("Loading base pipeline...")
        wan_pipeline = load_base_pipeline(BASE_CKPT)
        
        logging.info("Applying FPS-enabled LoRA weights...")
        lora_pipeline = apply_lora_weights_with_fps(wan_pipeline, LORA_PATH, rank=8)  # Use rank=8 to match training

        # Test 1: Single FPS generation
        logging.info("\n=== SINGLE FPS GENERATION TEST ===")
        video_tensor = generate_video_with_fps(
            pipeline=lora_pipeline,
            prompt="A man running on the beach",
            n_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            fps=60,  # Test with 60 FPS
            seed=42,
            steps=30,
            scale=7.0,
            frames=81,
            size=(832, 480),
            sample_solver='unipc',
            shift=3.0
        )
        
        output_file = save_video(video_tensor, size=(832, 480), fps=60, output_dir="/root/workspace/sc-diffusion-pipe/output/fps_test")
        logging.info(f"Single FPS test complete: {output_file}")
        
        # Test 2: Multi-FPS comparison (optional - uncomment to run)
        # test_fps_conditioning()
        
    except Exception as e:
        logging.error(f"Error during FPS-conditioned inference: {e}")
        import traceback
        traceback.print_exc()