#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference



# echo "$(date) starting inferencing with simple combined checkpoint approach"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/wan_SC_TARGET_14B_SHAPES_RANDOM.toml \
#     --checkpoint ../checkpoints/20251015_14-02-14/epoch600 \
#     --fps_values -1.0 -0.5 0.0 0.5 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251015_14-02-14_epoch600 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/original_prompts  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251015_14-02-14_epoch600.out 2>&1 



python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml\
    --checkpoint ../checkpoints/20251028_21-19-57/epoch750 \
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251028_21-19-57_epoch750 \
    --prompt "A man in the foreground standing on the bridge, with a city full of colorful signs in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251028_21-19-57_epoch750_quick_test.out 2>&1 