#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

# echo "$(date) starting inferencing epoch 1000"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251002_03-30-22/wan_SC_TARGET_14B_HUMAN_EMPTY.toml \
#     --checkpoint ../checkpoints/20251002_03-30-22/epoch1000 \
#     --fps_values 1 4 64 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251002_03-30-22_epoch1000_bokeh_empty_fps_only\
#     --prompt "A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251002_03-30-22_epoch1000_original_prompt_empty.out 2>&1 

echo "$(date) starting inferencing epoch 1000"
cd inference
python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251009_18-32-24/wan_SC_TARGET_14B_HUMAN_HOUSE_EMPTY.toml \
    --checkpoint ../checkpoints/20251009_18-32-24/epoch1000\
    --fps_values 1 4 64 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251009_18-32-24_epoch1000_bokeh_empty_fps_only \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/original_prompts \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --fps_only \
    > ../output/nohup_log/bokeh_20251009_18-32-24_epoch1000_original_prompt_empty.out 2>&1 