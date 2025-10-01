#!/usr/bin/env bash 
set -euo pipefail  # stop on first error; drop -e if you want to continue

echo "$(date) starting shutter speed"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN.toml \
  > ./output/bokeh_nohup_log/bokeh_human_joint.out 2>&1

echo "$(date) starting job2"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN_ONLY.toml \
  > ./output/bokeh_nohup_log/bokeh_human_base_only.out 2>&1

echo "$(date) all done"
