#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue
cd inference

# Shutter:  A group of monkeys swinging through jungle vines
# Aperture:  A steak sizzling on a hot grill, smoke rising
# Temp:  A time-lapse of the Northern Lights (aurora borealis) dancing in the sky


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch1000\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch1000 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch1000.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch800\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch800 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch800.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch600\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch600 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch600.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch400\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch400 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch400.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch200\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch200 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch200.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch150\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch150 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch150.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch100\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch100 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch100.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch50\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch50 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch50.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
#     --checkpoint ../checkpoints/20251115_08-01-31/epoch25\
#     --fps_values 0.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/corrupt_vis/20251115_08-01-31/epoch25 \
#     --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/shutter_20251115_08-01-31_epoch25.out 2>&1 



python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_08-01-31/epoch800\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_08-01-31/clean \
    --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251115_08-01-31_graft.out 2>&1 



python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_08-01-31/epoch800\
    --fps_values -1.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_08-01-31/clean \
    --prompt "A time-lapse of the Northern Lights (aurora borealis) dancing in the sky" \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251115_08-01-31_graft.out 2>&1 

