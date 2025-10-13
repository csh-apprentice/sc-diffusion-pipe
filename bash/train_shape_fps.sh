#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

echo "$(date) starting traning!"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
  > ./output/fps_nohup_log/fps_shape_blur.out 2>&1