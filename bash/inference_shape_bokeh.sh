#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251002_23-52-43/wan_SC_TARGET_14B_SHAPE_SINGLE_CENTER.toml \
#     --checkpoint ../checkpoints/20251002_23-52-43/epoch1000 \
#     --fps_values 1.4 2 2.8 4 5.6 8 11 \
#     --steps 30 \
#     --frames 49 \
#     --output_dir ../output/20251002_23-52-43_epoch1000_bokeh \
#     --prompt "A dog jumping over a wooden hurdle." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251002_23-52-43_epoch1000_dog.out 2>&1 


echo "$(date) starting inferencing"
cd inference
python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251007_00-35-06/wan_SC_TARGET_14B_BALL_KITCHEN.toml \
    --checkpoint ../checkpoints/20251007_00-35-06/epoch500 \
    --fps_values 0.1 0.2 0.3 0.4 0.8 1.4 \
    --steps 50 \
    --frames 16 \
    --output_dir ../output/20251007_00-35-06_epoch500_bokeh_fpsonly \
    --prompt "A dog jumping over a wooden hurdle." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --fps_only \
    > ../output/nohup_log/bokeh_20251007_00-35-06_epoch500_dog.out 2>&1 