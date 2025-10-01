#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


echo "$(date) starting inferencing epoch 150"
cd inference
python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml \
    --checkpoint ../checkpoints/20250930_05-53-32/epoch150 \
    --fps_values 1 1.6 2 4 16 64 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250930_05-53-32_epoch150_bokeh \
    --prompt "A man running on the beach in front of the moutains." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20250930_05-53-32_epoch150_run.out 2>&1 


echo "$(date) starting inferencing epoch 100"
# cd inference
python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml \
    --checkpoint ../checkpoints/20250930_05-53-32/epoch100 \
    --fps_values 1 1.6 2 4 16 64 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250930_05-53-32_epoch100_bokeh \
    --prompt "A man running on the beach in front of the moutains." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20250930_05-53-32_epoch100_run.out 2>&1 

# echo "$(date) starting inferencing"
# # cd inference
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250928_22-18-11/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml \
#     --checkpoint ../checkpoints/20250928_22-18-11/epoch200 \
#     --fps_values 2000 4000 6000 8000 10000 12000 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20250928_22-18-11_epoch200_temp_beach \
#     --prompt "A man running on the beach." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/temp_20250928_22-18-11_epoch200_run_beach.out 2>&1 

# echo "$(date) starting training"
# cd ..
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN.toml \
#   > ./output/bokeh_nohup_log/bokeh_human_joint.out 2>&1