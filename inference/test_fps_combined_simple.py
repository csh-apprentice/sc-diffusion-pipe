#!/usr/bin/env python3
"""
Simple combined checkpoint inference - merges FPS and base LoRA checkpoints into temp file.
"""

import torch
import os
import sys
import logging
from pathlib import Path
import argparse
import safetensors.torch

# Add the project root to Python path
sys.path.insert(0, '/root/workspace/sc-diffusion-pipe')

# Import the working inference script
from test_fps_multiple_experiments_align_old import (
    setup_environment,
    load_pipeline_from_toml,
    run_fps_experiments,
    apply_checkpoint
)


def merge_checkpoints(checkpoint_base, checkpoint_fps, output_path):
    """
    Merge base LoRA and FPS adapter checkpoints into a single checkpoint.

    Args:
        checkpoint_base: Path to base LoRA checkpoint directory
        checkpoint_fps: Path to FPS adapter checkpoint directory
        output_path: Path to save merged checkpoint
    """
    logging.info("\n" + "="*80)
    logging.info("MERGING CHECKPOINTS")
    logging.info("="*80)

    merged_state_dict = {}

    # Load FPS checkpoint
    if checkpoint_fps:
        fps_path = Path(checkpoint_fps)
        safetensors_files = list(fps_path.glob('*.safetensors'))
        if safetensors_files:
            logging.info(f"Loading FPS parameters from: {checkpoint_fps}")
            fps_checkpoint = safetensors.torch.load_file(str(safetensors_files[0]))

            # Add FPS parameters (fps_conditioning and fps_adapter)
            fps_count = 0
            for key, value in fps_checkpoint.items():
                if 'fps_conditioning' in key or 'fps_adapter' in key:
                    merged_state_dict[key] = value
                    fps_count += 1

            logging.info(f"✓ Added {fps_count} FPS parameters")

    # Load base LoRA checkpoint
    if checkpoint_base:
        base_path = Path(checkpoint_base)
        safetensors_files = list(base_path.glob('*.safetensors'))
        if safetensors_files:
            logging.info(f"Loading base LoRA from: {checkpoint_base}")
            base_checkpoint = safetensors.torch.load_file(str(safetensors_files[0]))

            # Add base LoRA parameters (everything with .lora_ but not fps)
            base_count = 0
            for key, value in base_checkpoint.items():
                if '.lora_' in key and 'fps' not in key:
                    merged_state_dict[key] = value
                    base_count += 1

            logging.info(f"✓ Added {base_count} base LoRA parameters")

    # Save merged checkpoint
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_file = output_path / "adapter_model.safetensors"

    logging.info(f"\nSaving merged checkpoint to: {merged_file}")
    logging.info(f"Total parameters: {len(merged_state_dict)}")
    safetensors.torch.save_file(merged_state_dict, str(merged_file))
    logging.info("✓ Checkpoint merge complete!")
    logging.info("="*80 + "\n")

    return str(output_path)


def main():
    import toml
    import json
    from utils.common import DTYPE_MAP

    parser = argparse.ArgumentParser(description="Combined checkpoint FPS inference (simple version)")
    parser.add_argument('--config', type=str, required=True, help='Path to TOML config file')
    parser.add_argument('--checkpoint_base', type=str, required=True, help='Base LoRA checkpoint directory')
    parser.add_argument('--checkpoint_fps', type=str, required=True, help='FPS adapter checkpoint directory')
    parser.add_argument('--fps_values', nargs='+', type=float, default=[0.025, 0.05, 0.1, 1.4],
                        help='FPS values to test')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps')
    parser.add_argument('--frames', type=int, default=49, help='Number of frames to generate')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--prompt', type=str, default=None, help='Generation prompt (single string)')
    parser.add_argument('--prompt_folder', type=str, default=None, help='Folder containing .txt files with prompts (alternative to --prompt)')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--port', type=str, default='29500', help='DeepSpeed port')
    parser.add_argument('--width', type=int, default=512, help='Video width')
    parser.add_argument('--height', type=int, default=512, help='Video height')

    args = parser.parse_args()

    # Validate prompt arguments
    if args.prompt is None and args.prompt_folder is None:
        parser.error("Either --prompt or --prompt_folder must be provided")
    if args.prompt is not None and args.prompt_folder is not None:
        parser.error("--prompt and --prompt_folder are mutually exclusive. Choose one.")

    # Setup environment
    setup_environment(port=args.port)

    # Load config (needed by run_fps_experiments)
    logging.info(f"Loading TOML configuration: {args.config}")
    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))

    # Convert dtype strings to torch.dtype objects (like train.py does)
    model_dtype_str = config['model']['dtype']
    config['model']['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := config['model'].get('transformer_dtype', None):
        config['model']['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    # Create temporary merged checkpoint
    temp_checkpoint_dir = "/tmp/merged_checkpoint"
    merged_checkpoint = merge_checkpoints(
        checkpoint_base=args.checkpoint_base,
        checkpoint_fps=args.checkpoint_fps,
        output_path=temp_checkpoint_dir
    )

    # Load pipeline with TOML config
    pipeline, _ = load_pipeline_from_toml(args.config)  # Returns (pipeline, config) tuple

    # Apply the merged checkpoint using the working function from the old script
    logging.info(f"\nApplying merged checkpoint from: {merged_checkpoint}")
    apply_checkpoint(
        wan_t2v_pipeline=pipeline,
        checkpoint_path=merged_checkpoint,
        rank=32,
        dtype=torch.bfloat16,
        fps_only=False,  # Load both FPS and base LoRA
        base_only=False,
        config=config
    )
    logging.info(f"✓ Checkpoint applied successfully")

    # Import load_prompts_from_folder if needed
    if args.prompt_folder:
        from test_fps_multiple_experiments_align_old import load_prompts_from_folder
        prompts = load_prompts_from_folder(args.prompt_folder)
    else:
        prompts = [("single", args.prompt)]

    # Run FPS experiments for each prompt
    for prompt_idx, (prompt_id, prompt_text) in enumerate(prompts, 1):
        if len(prompts) > 1:
            logging.info(f"\n{'='*80}")
            logging.info(f"PROCESSING PROMPT {prompt_idx}/{len(prompts)}: {prompt_id}")
            logging.info(f"{'='*80}")
            logging.info(f"Prompt text: {prompt_text}")

            # Create subdirectory for this prompt
            prompt_output_dir = os.path.join(args.output_dir, prompt_id)
        else:
            prompt_output_dir = args.output_dir

        # Run FPS experiments using the working inference code
        run_fps_experiments(
            pipeline=pipeline,
            config=config,  # Pass config as required
            base_prompt=prompt_text,
            n_prompt=args.negative_prompt,
            fps_values=args.fps_values,
            output_dir=prompt_output_dir,
            steps=args.steps,
            frames=args.frames,
            size=(args.width, args.height),
            seed=args.seed
        )

        logging.info(f"✅ Prompt '{prompt_id}' complete!")

    # Final summary
    if len(prompts) > 1:
        logging.info(f"\n{'='*80}")
        logging.info(f"🎉 ALL PROMPTS COMPLETE!")
        logging.info(f"{'='*80}")
        logging.info(f"Total prompts processed: {len(prompts)}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"{'='*80}\n")

    logging.info("✨ Script complete!")


if __name__ == "__main__":
    main()
