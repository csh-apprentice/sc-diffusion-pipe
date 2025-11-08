#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference


# =============================================================================
# 🐛 DEBUG MODE EXAMPLE: Verify alignment mechanism with hardcoded gate scaling
# =============================================================================
# Uncomment the section below to test with hardcoded gate scaling value
# This mode:
# - Skips calibration entirely
# - Uses a fixed gate scaling ratio for all FPS adapter blocks
# - Verifies that base LoRA is excluded during inference
# - Shows detailed logging of gate scaling and parameter loading
#
# Usage: Add --debug_gate_scale 1.5 to use a hardcoded ratio of 1.5x for all gates
#
# python test_fps_align_onestep.py  \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
#     --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/temp_20251023_00-19-38_epoch1000_DEBUG_1.5x \
#     --prompt "A person walking in a park" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --align \
#     --debug_gate_scale 1.5 \
#     > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_DEBUG_1.5x.out 2>&1
# =============================================================================


python test_fps_align_onestep.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 0.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_align \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/span1 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_align_span1.out 2>&1 


python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 0.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_align_debug \
    --prompt "A bear swiping at a fish in a river." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_align_span1.out 2>&1 


python test_fps_align_new.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 0.0 1.0 \
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
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_align_new.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000 \
    --fps_values -3.0 -2.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_align_debug \
    --prompt  "A painting on an easel in a meadow of wildflowers."  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_align_debug.out 2>&1 



python test_fps_align_onestep.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251031_08-25-18/wan_SC_TARGET_14B_HUMAN.toml  \
    --checkpoint ../checkpoints/20251031_08-25-18/epoch1000 \
    --fps_values -1.0 0.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251031_08-25-18_epoch1000_align_debug \
    --prompt "A painting on an easel in a meadow of wildflowers." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/bokeh_20251031_08-25-18_epoch1000_align_debug.out 2>&1 


python test_fps_align_onestep.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_align \
    --prompt "A ceiling fan spinning under warm yellow light." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_light.out 2>&1 



python test_fps_align_onestep.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 0.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_align_debug \
    --prompt "A ceiling fan spinning under warm yellow light."\
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_align_debug.out 2>&1 


python test_fps_align_onestep.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_align \
    --prompt "A ceiling fan spinning under warm yellow light."\
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --fps_only \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_align_fps_only.out 2>&1 



python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_align \
    --prompt "A ceiling fan spinning under warm yellow light."\
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --force_gate_one \
    --fps_only \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_align_fps_only_correct_inference_light_force_gateone.out 2>&1 


python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_align \
    --prompt "A ceiling fan spinning under warm yellow light."\
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --clean \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_align_clean_light.out 2>&1 



python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_fps_only \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span3 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --fps_only\
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span3_fpsonly.out 2>&1 



python test_fps_align_onestep.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 0.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_align \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span1 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --align \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_align_span1.out 2>&1 