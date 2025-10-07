#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251004_18-07-42/wan_SC_TARGET_14B_SHAPE_TEMP_all.toml \
#     --checkpoint ../checkpoints/20251004_18-07-42/epoch1000 \
#     --fps_values 2000 4000 6000 8000 10000 12000 \
#     --steps 50 \
#     --frames 16 \
#     --output_dir ../output/temp_20251004_18-07-42_epoch1000_temp_all \
#     --prompt "Image of a red circle at the top-left, a green star at the bottom-right, a blue square at the top-right. (All shapes sharp)" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_220251004_18-07-42_epoch1000_debug.out 2>&1 


# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251003_23-07-22/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml \
#     --checkpoint ../checkpoints/20251003_23-07-22/epoch200 \
#     --fps_values 2000 4000 6000 8000 10000 12000 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251003_23-07-22_epoch200_temp \
#     --prompt "A man running on the beach in front of the moutain." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/20251003_23-07-22_epoch200_debug_run.out 2>&1 


# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250928_22-18-11/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml \
#     --checkpoint ../checkpoints/20250928_22-18-11/epoch200 \
#     --fps_values 2000 4000 6000 8000 10000 12000 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20250928_22-18-11_epoch200_temp_newdebug \
#     --prompt "A man running on the beach in front of the moutains." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20250928_22-18-11_epoch200_run_newdebug.out 2>&1 


# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_07-13-38/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUE.toml \
#     --checkpoint ../checkpoints/20251005_07-13-38/epoch500 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_07-13-38_epoch500_temp_debug \
#     --prompt "Red roses blossom among green leaves, lying on the pink table with cute texture." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20251005_07-13-38_epoch500_rose_new.out 2>&1 


# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_08-32-53/wan_SC_TARGET_14B_SHAPE_TEMP_FOURCOLOR.toml \
#     --checkpoint ../checkpoints/20251005_08-32-53/epoch800 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_08-32-53_epoch800_temp_debug \
#     --prompt "A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_0251005_08-32-53_epoch800_run.out 2>&1 


cd inference
# echo "$(date) starting inferencing 1"
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_20-28-40/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUE.toml \
#     --checkpoint ../checkpoints/20251005_20-28-40/epoch500 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_20-28-40_epoch500_temp_debug \
#     --prompt "A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20251005_20-28-40_epoch500_run.out 2>&1 

# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_20-28-40/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUE.toml \
#     --checkpoint ../checkpoints/20251005_20-28-40/epoch500 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_20-28-40_epoch500_temp_debug_fpsonly \
#     --prompt "A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/temp_20251005_20-28-40_epoch500_run_fpsonly.out 2>&1 

echo "$(date) starting inferencing 0"
python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251007_07-53-48/wan_SC_TARGET_14B_MOUTAIN_FLOWER_SOFTPLUS.toml \
    --checkpoint ../checkpoints/20251007_07-53-48/epoch1000 \
    --fps_values 2000 6000 12000  \
    --steps 30 \
    --frames 16 \
    --output_dir ../output/20251007_07-53-48_epoch1000_temp_debug \
    --prompt "A man running on the beach." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --fps_only \
    > ../output/nohup_log/temp_20251007_07-53-48_epoch1000_run.out 2>&1 

# echo "$(date) starting inferencing 1"
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_20-25-28/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE.toml \
#     --checkpoint ../checkpoints/20251005_20-25-28/epoch200 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_20-25-28_epoch200_temp_debug \
#     --prompt "A red sphere on top of a blue cube." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20251005_20-25-28_epoch200_debug.out 2>&1 

# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_20-25-28/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE.toml \
#     --checkpoint ../checkpoints/20251005_20-25-28/epoch100 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_20-25-28_epoch100_temp_debug \
#     --prompt "A red sphere on top of a blue cube." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20251005_20-25-28_epoch100_debug.out 2>&1 

# echo "$(date) starting inferencing 3"
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_20-25-28/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE.toml \
#     --checkpoint ../checkpoints/20251005_20-25-28/epoch500 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_20-25-28_epoch500_temp_debug \
#     --prompt "In realistic style, A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20251005_20-25-28_epoch500_run_real.out 2>&1 


# echo "$(date) starting inferencing 4"
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251005_20-25-28/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE.toml \
#     --checkpoint ../checkpoints/20251005_20-25-28/epoch500 \
#     --fps_values 3000 6000 10000  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251005_20-25-28_epoch500_temp_debug_fpsonly \
#     --prompt "In realistic style, A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/temp_20251005_20-25-28_epoch500_run_fpsonly_real.out 2>&1 