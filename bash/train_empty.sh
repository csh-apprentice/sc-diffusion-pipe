#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

# echo "$(date) starting job1"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN_EMPTY.toml \
#   > ./output/bokeh_nohup_log/bokeh_human_empty_new.out 2>&1


echo "$(date) starting job2"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUNTAIN_FLOWER_EMPTY.toml \
  > ./output/temp_nohup_log/temp_mf_empty.out 2>&1

echo "$(date) all done"
