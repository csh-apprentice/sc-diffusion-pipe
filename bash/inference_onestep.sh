#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference

# echo "$(date) starting inferencing"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml \
#     --checkpoint ../checkpoints/20250930_05-53-32/epoch1000 \
#     --fps_values 64 \
#     --steps 1 \
#     --frames 16 \
#     --output_dir ../output/onestep/20250930_05-53-32_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/bokeh_20250930_05-53-32_epoch1000_onestep.out 2>&1 




# echo "$(date) starting inferencing"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml \
#     --checkpoint ../checkpoints/20250930_05-53-32/epoch1000 \
#     --fps_values 64 \
#     --steps 1 \
#     --frames 16 \
#     --output_dir ../output/onestep/20250930_05-53-32_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/bokeh_20250930_05-53-32_epoch1000_onestep.out 2>&1 

echo "$(date) starting inferencing 2"
python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.2 \
    --steps 1 \
    --frames 16 \
    --output_dir ../output/onestep/20251008_20-21-45_epoch1000 \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_dog_50.out 2>&1 