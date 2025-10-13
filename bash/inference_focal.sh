#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_00-31-25/wan_SC_TARGET_14B_3SHAPES_FOCAL.toml \
#     --checkpoint ../checkpoints/20251008_00-31-25/epoch500 \
#     --fps_values  40 50 65 \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251008_00-31-25_epoch500 \
#     --prompt "A red sphere is positioned in the foreground, a blue cube sits behind it at the center, and a green cylinder is placed further back in the background, all aligned diagonally on a gray surface under soft directional lighting." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_00-31-25_epoch500_debug.out 2>&1 


echo "$(date) starting inferencing"
cd inference
python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251010_06-35-50/wan_SC_TARGET_14B_3SHAPES_FOCAL_ALL.toml \
    --checkpoint ../checkpoints/20251010_17-40-12/epoch1000 \
    --fps_values  40 50 65 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251010_17-40-12_epoch1000 \
    --prompt "A boat on the river." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/focal_20251010_17-40-12_epoch1000_debug.out 2>&1 