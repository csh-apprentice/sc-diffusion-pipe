#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

echo "$(date) starting inferencing epoch 1000"
cd inference
python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20250917_23-08-48/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml \
    --checkpoint ../checkpoints/20250917_23-08-48/epoch1000\
    --fps_values 12 24 40 60 120 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250917_23-08-48_epoch1000_allscales \
    --prompt "A man running on the beach." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/shutter_20250917_23-08-48_epoch1000_allscales.out 2>&1 


# echo "$(date) starting inferencing epoch 1000"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250918_22-35-16/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_VID_NEW.toml \
#     --checkpoint ../checkpoints/20250918_22-35-16/epoch1000\
#     --fps_values 12 24 40 60 120 240 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20250918_22-35-16_epoch1000_allscales \
#     --prompt "A man running on the beach."  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20250918_22-35-16_epoch1000_shapes.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250918_22-35-16/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_VID_NEW.toml \
#     --checkpoint ../checkpoints/20250918_22-35-16/epoch1000\
#     --fps_values 240 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20250918_22-35-16_epoch1000_shapes \
#     --prompt "A child draw a orange triangle on the paper."  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20250918_22-35-16_epoch1000_shapes.out 2>&1 

# echo "$(date) starting inferencing epoch 200"
# # cd inference
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints_archive/20250917_23-08-48/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml \
#     --checkpoint ../checkpoints/20250917_23-08-48/epoch200 \
#     --fps_values 12 24 40 60 120 240 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20250917_23-08-48_epoch200_shutter_fps_only \
#     --prompt "A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/shutter_20250917_23-08-48_epoch200_run_fps_only.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-48-34/wan_SC_TARGET_14B_HUMAN_HOUSE.toml \
#     --checkpoint ../checkpoints/20251008_06-48-34/epoch1000 \
#     --fps_values 1 8 64 \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251008_06-48-34_epoch1000_bokeh_fpsonly\
#     --prompt "Smiling man in blue polo on a lawn, suburban house behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_06-48-34_epoch1000_debug.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-48-34/wan_SC_TARGET_14B_HUMAN_HOUSE.toml \
#     --checkpoint ../checkpoints/20251008_06-48-34/epoch1000 \
#     --fps_values 1 8 64 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_06-48-34_epoch1000_bokeh_fpsonly_sameprompt\
#     --prompt "Smiling man in blue polo on a lawn, suburban house behind." \
#     --negative_prompt "" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251008_06-48-34_epoch1000_sameprompt_fps.out 2>&1 

