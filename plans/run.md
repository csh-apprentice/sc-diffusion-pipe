Training Command:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SUBTASKS_COMPLETE.toml' > ./output/nohup_log/fix_magnitude.out 2>&1 &


nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE.toml' > ./output/nohup_log/sanity_shape.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE.toml' > ./output/nohup_log/sanity_shape.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/test_subtask4.toml' > ./output/nohup_log/test_subtask4.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_DUMMY.toml' > ./output/nohup_log/T3_output_rank8_v7.out 2>&1 &
Inference Command:
python inference/lora_inference.py

nohup python inference/lora_inference_sct2v_cross.py > output/nohup_inference/sct2v_inference_v3.out 2>&1 &

cd /root/workspace/sc-diffusion-pipe/inference

nohup python test_fps_multiple_experiments.py \
    --base_model /root/workspace/diffusion-pipe/cindy_ckpts/Wan2.1-T2V-14B \
    --checkpoint ../checkpoints/20250904_22-45-31/epoch100 \
    --fps_values 12 60 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250904_22-45-31_noneg \
    --prompt "A man running on the beach" \
    --seed 42 \
    --port 29502 \
    > ../output/nohup_log/fps_comprehensive_experiment_20250904_22-45-31_noneg.out 2>&1 &


python measure_motion_blur.py /root/workspace/sc-diffusion-pipe/datatset/mirflickr --recursive --max-side 1024 \
  --save-csv blur_stats.csv --save-plot blur_hist.png 


nohup deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_EXTREME_TEST.toml > output/nohup_log/test_rmsnorm_init.out 2>&1 &

python collect_exposure_times.py /root/workspace/sc-diffusion-pipe/datatset/mirflickr/meta/exif \
  --recursive \
  --num 25 \
  --save-csv exposures.csv \
  --save-plot exposures_hist.png \
  --show


python collect_exposure_times.py /root/workspace/sc-diffusion-pipe/datatset/mirflickr/meta/exif --recursive --num 10


python collect_exposure_times.py /root/workspace/sc-diffusion-pipe/datatset/mirflickr/meta/exif --recursive --filter 0.082 0.084 --out-list short_exposures.txt --log-bins

python generate_synthetic_blur_dataset.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shapes \
  --modes image \
  --samples_per_fps 60 \
  --min_speed 200 \
  --max_speed 200 \
  --caption_relations 



python generate_synthetic_blur_dataset.py \
  --out_dir /root/workspace/sc-diffusion-pipe/datatset/test_vid \
  --num_scenes 2 \
  --fps_bins 12 240 \
  --video --frames 16 \
  --width 512 --height 512 \
  --substeps 16 \
  --relations basic \
  --caption_blur_text none

nohup python test_fps_multiple_experiments.py \
    --base_model /root/workspace/diffusion-pipe/cindy_ckpts/Wan2.1-T2V-14B \
    --checkpoint ../checkpoints/20250906_21-53-11/epoch1000 \
    --fps_values 12 60 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250906_21-53-11 \
    --prompt "a red traingle is moving from left to right " \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --force_gate_one \
    > ../output/nohup_log/fps_comprehensive_experiment_20250906_21-53-11_gate_one.out 2>&1 &
