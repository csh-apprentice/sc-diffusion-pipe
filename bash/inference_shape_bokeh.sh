#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference
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


# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml \
#     --checkpoint ../checkpoints/20250930_05-53-32/epoch1000 \
#     --fps_values 1 64 \
#     --steps 50 \
#     --frames 16 \
#     --output_dir ../output/20250930_05-53-32_epoch1000_negative \
#     --prompt "A dog jumping over a wooden hurdle." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20250930_05-53-32_epoch1000_dog_negative.out 2>&1 


# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-47-24/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_06-47-24/epoch1000 \
#     --fps_values 0.025  0.4 1.4 \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251008_06-47-24_epoch1000_debug \
#     --prompt "A red sphere is positioned in the foreground, a blue cube sits behind it at the center, and a green cylinder is placed further back in the background, all aligned diagonally on a gray surface under soft directional lighting." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_06-47-24_epoch1000_debug.out 2>&1 

# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-47-24/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_06-47-24/epoch1000 \
#     --fps_values 0.025  0.4 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_06-47-24_epoch1000_debug \
#     --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_06-47-24_epoch1000_run.out 2>&1 

echo "$(date) starting inferencing 1"
python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251010_22-08-59/wan_SC_TARGET_14B_SYN3SHAPES_ALL.toml \
    --checkpoint ../checkpoints/20251010_22-08-59/epoch2000 \
    --fps_values 2  16 64 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251010_22-08-59_epoch2000_debug \
    --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251010_22-08-59_epoch2000_debug.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.025 0.1 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000 \
#     --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_dog_50.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.025 0.1 1.4 \
#     --steps 75 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000 \
#     --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_dog_75.out 2>&1 

# echo "$(date) starting inferencing"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_00-31-01/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_00-31-01/epoch400 \
#     --fps_values 0.1  0.4 2.8 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_00-31-01_epoch400_fpsonly \
#     --prompt "A dog jumping over a wooden hurdle in the foreground, behind is the grass field." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251008_00-31-01_epoch400_dog_fpsonly.out 2>&1 

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