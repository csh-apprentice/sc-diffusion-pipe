#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

echo "$(date) starting resume fps only human!"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN_RESUME.toml \
  > ./output/bokeh_nohup_log/bokeh_human_resume.out 2>&1