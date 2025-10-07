#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

echo "$(date) starting trainig!"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUE_RESUME.toml \
  > ./output/temp_nohup_log/temp_redbliue_resume.out 2>&1

echo "$(date) starting trainig!"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE_RESUME.toml \
  > ./output/temp_nohup_log/temp_redbluewhite_resume.out 2>&1

echo "$(date) all done"
