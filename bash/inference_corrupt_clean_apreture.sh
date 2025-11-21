#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue
cd inference

# Shutter:  A group of monkeys swinging through jungle vines
# Aperture:  A deer close to the camera beside a tree trunk, with a distant forest in the background.
# Temp:  A time-lapse of the Northern Lights (aurora borealis) dancing in the sky

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch800\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/epoch800 \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch800.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch400\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/epoch400 \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch400.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch200\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/epoch200 \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch200.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch100\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/epoch100 \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch100.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch50\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/epoch50 \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch50.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch800\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/epoch0 \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch800.out 2>&1 

python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_22-27-44/epoch800\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/corrupt_vis/20251115_22-27-44/clean \
    --prompt "A deer close to the camera beside a tree trunk, with a distant forest in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251115_22-27-44_epoch800.out 2>&1 