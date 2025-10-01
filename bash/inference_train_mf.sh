#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


echo "$(date) starting inferencing"
cd inference
python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20250928_22-18-11/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml \
    --checkpoint ../checkpoints/20250928_22-18-11/epoch200 \
    --fps_values 2000 4000 6000 8000 10000 12000 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250928_22-18-11_epoch200_temp \
    --prompt "A man running on the beach in front of the moutains." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20250928_22-18-11_epoch200_run.out 2>&1 

# echo "$(date) starting training"
# cd ..
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUNTAIN_FLOWER_ONLY.toml \
#   > ./output/temp_nohup_log/temp_mf_base_only.out 2>&1