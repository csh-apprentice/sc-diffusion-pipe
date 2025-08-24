# FILE: lora_inference_toolkit.py

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

# In lora_inference_toolkit.py -> Replace the load_base_pipeline function with this final version.

def load_base_pipeline(ckpt_path, dtype=torch.bfloat16):
    """Loads the heavy base WanPipeline. THIS IS THE SLOW PART."""
    logging.info("Initializing WanPipeline...")
    pipeline_config = {"model": {"ckpt_path": ckpt_path, "dtype": dtype, "transformer_dtype": dtype}}
    wan_t2v = WanPipeline(config=pipeline_config)
    
    logging.info("Loading main transformer model... (This may take a while)")
    wan_t2v.load_diffusion_model()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Moving model components to device: {device}")
    
    # --- (FIX) Move ALL necessary model parts to the GPU ---
    wan_t2v.transformer.to(device)
    wan_t2v.vae.model.to(device) # <--- 添加这一行来移动 VAE
    
    logging.info("Pipeline and transformer loaded successfully.")
    return wan_t2v

def apply_lora_weights(wan_t2v_pipeline, lora_path, rank=32, dtype=torch.bfloat16):
    """Applies a LoRA to an already-loaded pipeline. THIS IS THE FAST PART."""
    logging.info(f"Applying LoRA weights from: {lora_path}")
    adapter_config = {"type": "lora", "rank": rank, "alpha": rank, "dropout": 0.0, "dtype": dtype}
    wan_t2v_pipeline.configure_adapter(adapter_config)
    wan_t2v_pipeline.load_adapter_weights(lora_path)
    logging.info(f"--- SUCCESS! LoRA from {lora_path} applied. ---")
    return wan_t2v_pipeline

# In lora_inference_toolkit.py -> Replace the generate_video function with this verification version.

def generate_video(pipeline, prompt, n_prompt, seed=69, steps=25, scale=7.0, frames=81, size=(832, 480), shift=5.0, sample_solver='unipc'):
    """
    Generates a video by manually executing the layers of the WanPipeline object.
    Includes logic for shift and choice of sampler, mirroring text2video.py.
    """
    logging.info(f"Starting video generation with solver: {sample_solver}, shift: {shift}...")
    device = pipeline.transformer.device
    
    # 1. PREPARE INPUTS
    torch.manual_seed(seed)
    text_encoder_fn = pipeline.get_call_text_encoder_fn(pipeline.text_encoder.model.to(device))
    cond_inputs = text_encoder_fn([prompt], is_video=True)
    uncond_inputs = text_encoder_fn([n_prompt], is_video=True)
    
    vae_stride = [4, 8, 8]
    target_shape = (16, frames // vae_stride[0], size[1] // vae_stride[1], size[0] // vae_stride[2])
    latents = torch.randn(target_shape, device=device)

    layers = pipeline.to_layers()
    initial_layer, transformer_layers, final_layer = layers[0], layers[1:-1], layers[-1]

    # 2. SETUP SCHEDULER & SAMPLING LOOP (with solver choice)
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
    
    for t in tqdm(timesteps, desc="Sampling"):
        t_batch = torch.full((1,), t, device=device)

        with torch.no_grad(), torch.autocast('cuda'):
            # UNCONDITIONAL PASS (batch_size = 1)
            initial_input_uncond = (latents.unsqueeze(0), None, t_batch, uncond_inputs['text_embeddings'].to(device), uncond_inputs['seq_lens'].to(device), None)
            x, e, e0, seq_lens, grid_sizes, freqs, context = initial_layer(initial_input_uncond)
            for transformer_layer in transformer_layers:
                x, e, e0, seq_lens, grid_sizes, freqs, context = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context))
            noise_pred_uncond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context))

            # CONDITIONAL PASS (batch_size = 1)
            initial_input_cond = (latents.unsqueeze(0), None, t_batch, cond_inputs['text_embeddings'].to(device), cond_inputs['seq_lens'].to(device), None)
            x, e, e0, seq_lens, grid_sizes, freqs, context = initial_layer(initial_input_cond)
            for transformer_layer in transformer_layers:
                x, e, e0, seq_lens, grid_sizes, freqs, context = transformer_layer((x, e, e0, seq_lens, grid_sizes, freqs, context))
            noise_pred_cond = final_layer((x, e, e0, seq_lens, grid_sizes, freqs, context))
            
            # Classifier-Free Guidance
            noise_pred = noise_pred_uncond.squeeze(0) + scale * (noise_pred_cond.squeeze(0) - noise_pred_uncond.squeeze(0))
        
        # SCHEDULER STEP
        latents = scheduler.step(noise_pred, t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)

    logging.info("Denoising loop complete.")
    
    # 3. DECODE LATENTS TO VIDEO
    logging.info("Decoding latents with VAE...")
    video_tensor = pipeline.vae.decode([latents])[0]
    
    logging.info("Generation complete.")
    return video_tensor

def save_video(tensor, size, output_dir="."):
    """Saves a video tensor to a file."""
    if tensor is None:
        logging.warning("Video tensor is None, skipping save.")
        return
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"video_{size[0]}x{size[1]}_{formatted_time}.mp4")
    logging.info(f"Saving generated video to {output_filename}")
    cache_video(tensor=tensor[None], save_file=output_filename, fps=16, normalize=True, value_range=(-1, 1))
    logging.info("Video saved successfully!")

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
    setup_environment()

    BASE_CKPT = "/root/workspace/diffusion-pipe/cindy_ckpts/Wan2.1-T2V-14B"
    LORA_PATH = "/root/workspace/diffusion-pipe/logs/20250803_20-40-58/epoch500"
    
    wan_pipeline = load_base_pipeline(BASE_CKPT)
    lora_pipeline = apply_lora_weights(wan_pipeline, LORA_PATH)

    video_tensor = generate_video(
    pipeline=lora_pipeline,
    prompt="Monica Vitti in an elegant event holding a drink , skyscrapers in the background in a night scene",
    n_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=44,
    steps=50,
    scale=6.0,
    frames=81,
    size=(832, 480),
    sample_solver='unipc', # Default, but explicit here
    shift=3.0             # 3.0 for 480P, 5.0 for 720P
    )
    print(video_tensor.shape)
    save_video(video_tensor, size=(832, 480),output_dir="/root/workspace/sc-diffusion-pipe/output")