#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference



python test_fps_graft_align.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_graft_align \
    --prompt "A bear swiping at a fish in a river." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29503 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_graft_align.out 2>&1 


python test_fps_graft_align.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_align \
    --prompt "A row of hanging lanterns sways gently above a quite cobblestone path." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_align_candel.out 2>&1 




python test_fps_graft_align.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft_align \
    --prompt "A man running on the beach." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_graft_align_debug.out 2>&1 