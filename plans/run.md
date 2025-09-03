Training Command:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SUBTASKS_COMPLETE.toml' > ./output/nohup_log/fix_magnitude.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/test_subtask4.toml' > ./output/nohup_log/test_subtask4.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_DUMMY.toml' > ./output/nohup_log/T3_output_rank8_v7.out 2>&1 &
Inference Command:
python inference/lora_inference.py

nohup python inference/lora_inference_sct2v_cross.py > output/nohup_inference/sct2v_inference_v3.out 2>&1 &

cd /root/workspace/sc-diffusion-pipe/inference

nohup python test_fps_multiple_experiments.py \
    --base_model /root/workspace/diffusion-pipe/cindy_ckpts/Wan2.1-T2V-14B \
    --checkpoint ../checkpoints/20250825_21-14-37/epoch100 \
    --fps_values 12 60 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/fps_comprehensive_test \
    --prompt "A man running on the beach" \
    --seed 42 \
    --port 29502 \
    > ../output/nohup_log/fps_comprehensive_experiment.out 2>&1 &
