#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue

# echo "$(date) starting job1"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN.toml \
#   > ./output/bokeh_nohup_log/bokeh_human_joint.out 2>&1

# echo "$(date) starting job2"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN_ONLY.toml \
#   > ./output/bokeh_nohup_log/bokeh_human_base_only.out 2>&1


echo "$(date) starting job3"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml \
  > ./output/temp_nohup_log/temp_mf_joint_103.out 2>&1

# echo "$(date) starting job4"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUNTAIN_FLOWER_ONLY.toml \
#   > ./output/temp_nohup_log/temp_mf_base_only.out 2>&1

echo "$(date) all done"
