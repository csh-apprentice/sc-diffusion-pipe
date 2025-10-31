import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
import os

# --- 1. Configuration ---
BASE_MODEL_ID = '/root/workspace/diffusion-pipe/cindy_ckpts/Wan2.1-T2V-14B'

# SET YOUR LORA_PATH HERE
# To run with LoRA: "/path/to/your/finetuned_lora.safetensors"
# To run with Base Model: "" (an empty string)
# LORA_PATH = "/path/to/your/finetuned_lora.safetensors" 
LORA_PATH = "/path/to/your/finetuned_lora.safetensors" 

PROMPT_FILE = "validation_prompts.txt"

# Output directories
OUT_VIDEOS_DIR = "generated_data/videos"
OUT_FRAMES_DIR = "generated_data/frames"

# --- 2. Accelerate Initialization ---
accelerator = Accelerator()
distributed_state = PartialState()

# --- 3. Load Prompts (on main process only) ---
all_prompts = []
if accelerator.is_main_process:
    print(f"Loading prompts from {PROMPT_FILE}...")
    with open(PROMPT_FILE, 'r') as f:
        all_prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(all_prompts)} prompts in total.")
    
    # Create output directories
    os.makedirs(OUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(OUT_FRAMES_DIR, exist_ok=True)

# Broadcast the list of prompts to all processes
all_prompts = gather_object([all_prompts] * accelerator.num_processes)
all_prompts = all_prompts[0] 

# Wait for main process to create directories
accelerator.wait_for_everyone()

# --- 4. Load Pipeline ---
print(f"[Process {accelerator.process_index}] Loading base pipeline: {BASE_MODEL_ID}...")
pipeline = DiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
)

# --- THIS IS THE NEW PART ---
# Check if LORA_PATH is provided and is not an empty string
if LORA_PATH and LORA_PATH.strip():
    print(f"[Process {accelerator.process_index}] Loading LoRA from: {LORA_PATH}")
    pipeline.load_lora_weights(LORA_PATH)
    pipeline.fuse_lora()
    print(f"[Process {accelerator.process_index}] LoRA loaded and fused.")
else:
    print(f"[Process {accelerator.process_index}] No LoRA path provided, running with base model.")
# --- END OF NEW PART ---

# This is the simple way: just move the model to the correct GPU
pipeline.to(accelerator.device)

# --- 5. Distributed Generation Loop ---
print(f"[Process {accelerator.process_index}] Starting generation...")

# This will automatically split the 'all_prompts' list between your 2 GPUs
with accelerator.split_between_processes(all_prompts, apply_padding=True) as prompts_for_this_process:
    
    for i, prompt in enumerate(prompts_for_this_process):
        if prompt is None:
            continue
            
        global_index = all_prompts.index(prompt)

        print(f"[Process {accelerator.process_index}] Generating {i+1}/{len(prompts_for_this_process)} (Global #{global_index}): '{prompt}'")
        
        video_frames = pipeline(
            prompt, 
            num_inference_steps=25, 
            num_frames=16 
        ).frames[0]

        # --- 6. Save Video and Frames ---
        video_filename = f"video_{global_index:04d}.mp4"
        video_path = os.path.join(OUT_VIDEOS_DIR, video_filename)
        export_to_video(video_frames, video_path, fps=10)

        frame_subfolder = os.path.join(OUT_FRAMES_DIR, f"video_{global_index:04d}")
        os.makedirs(frame_subfolder, exist_ok=True)
        for j, frame in enumerate(video_frames):
            frame_path = os.path.join(frame_subfolder, f"frame_{j:04d}.png")
            frame.save(frame_path)

# --- 7. Synchronization ---
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("\n--- All processes finished generation! ---")
    print(f"Generated videos are in: {OUT_VIDEOS_DIR}")
    print(f"Generated frames are in: {OUT_FRAMES_DIR}")
    print("\nYou can now run the metric calculation scripts.")