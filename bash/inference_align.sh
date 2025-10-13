#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue



echo "$(date) starting inferencing with simple combined checkpoint approach"
cd inference
python test_fps_align_textmagnitude.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-48-34/wan_SC_TARGET_14B_HUMAN_HOUSE.toml \
    --checkpoint ../checkpoints/20251008_06-48-34/epoch1000 \
    --fps_values 1 8 64 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251008_06-48-34_epoch1000_align_text \
    --prompt "Smiling man in blue polo on a lawn, suburban house behind." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/bokeh_20251008_06-48-34_epoch1000_original_align_text.out 2>&1 