#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference


# =============================================================================
# PLUS ALIGNMENT MODE - Testing with TWO ratios (r_text and r_cond)
# =============================================================================
# This script tests the PLUS alignment approach from:
# /root/workspace/sc-diffusion-pipe/plans/ALIGN_INFERNCE_TEXT_ONESTEP_PLUS.md
#
# Key differences from standard alignment:
# - Captures TWO ratios per block:
#   r_text_i = ||y_text_base_lora_fps|| / ||y_text_clean_fps||
#   r_cond_i = ||y_fps_base_lora_fps|| / ||y_fps_clean_fps||
# - Applies: new_gate_i = old_gate_i * r_text_i * r_cond_i
# =============================================================================


# Test with temperature/motion blur checkpoint
python test_fps_align_onestep_plus.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 0.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_PLUS_align \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/span1 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_PLUS_align.out 2>&1


# Test with bokeh checkpoint
python test_fps_align_onestep_plus.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000 \
    --fps_values -1.0 0.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_PLUS_align \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1\
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_PLUS_align.out 2>&1


# Test with shutter speed checkpoint
python test_fps_align_onestep_plus.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 0.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_PLUS_align \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span1 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --align \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_PLUS_align.out 2>&1



python test_fps_align_onestep_plus.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_align_debug_ratio \
    --prompt "A bear swiping at a fish in a river." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_align_plus.out 2>&1 
