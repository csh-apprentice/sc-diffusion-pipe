Training Command:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_DUMMY_rank4.toml' > ./output/nohup_log/T3_output_rank4_FIX_v1.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_DUMMY.toml' > ./output/nohup_log/T3_output_rank8_v7.out 2>&1 &
Inference Command:
python inference/lora_inference.py