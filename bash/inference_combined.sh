#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference


# echo "$(date) starting inferencing with simple combined checkpoint approach"
# python test_fps_combined_simple.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint_base ../checkpoints/20251008_20-21-45/epoch1000 \
#     --checkpoint_fps ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.025 0.05 0.1 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_allprompts_compare \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_compare.out 2>&1 


# echo "$(date) starting inferencing with simple combined checkpoint approach"
# python test_fps_combined_simple.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint_base ../checkpoints/20251008_20-21-45/epoch1000 \
#     --checkpoint_fps ../checkpoints/20251009_07-17-52/epoch1000 \
#     --fps_values 0.2 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251009_07-17-52_epoch1000_bounce \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/bouncing  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251009_07-17-52_epoch1000_bounce.out 2>&1 


echo "$(date) starting inferencing with simple combined checkpoint approach"
python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.025 0.05 0.1 1.4 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251009_07-17-52_epoch1000_bounce \
    --prompt "A person in the foreground under a tree, with a city full of colorful signs in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251009_07-17-52_epoch1000_tree.out 2>&1 

# echo "$(date) starting inferencing with simple combined checkpoint approach"
# python test_fps_combined_simple.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_00-31-25/wan_SC_TARGET_14B_3SHAPES_FOCAL.toml \
#     --checkpoint_base ../checkpoints/20251008_00-31-25/epoch500 \
#     --checkpoint_fps ../checkpoints/20251009_07-23-46/epoch1000 \
#     --fps_values 40 50 65 \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251009_07-23-46_epoch1000 \
#     --prompt "A red sphere is positioned in the foreground, a blue cube sits behind it at the center, and a green cylinder is placed further back in the background, all aligned diagonally on a gray surface under soft directional lighting." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/focal_20251009_07-23-46_epoch1000_debug.out 2>&1 