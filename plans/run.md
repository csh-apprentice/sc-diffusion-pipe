Training Command:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml' > ./output/nohup_log/sanity_shape_single_image_joint.out 2>&1 &


nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml' > ./output/nohup_log/sanity_shape_single_image.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BOTH_IMG.toml' > ./output/nohup_log/sanity_shape_both_image.out 2>&1 &

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
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shapes_both \
  --modes both \
  --samples_per_fps 60 \
  --align_img_video \
  --min_speed 100 \
  --max_speed 400 \
  --min_obj 1 \
  --max_objs 2 \
  --seed 91 \
  --ensure_static_video \
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

nohup python test_fps_multiple_experiments_align.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20250914_06-42-00/wan_SC_TARGET_14B_FPS_SHAPE_BOTH_VID.toml \
    --checkpoint ../checkpoints/20250914_06-42-00/epoch250 \
    --fps_values 12 60 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250914_06-42-00_epoch250_fix \
    --prompt "Video of a orange square at the top-right (moving), a purple star at the top-right (static).   " \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/fps_comprehensive_experiment_20250914_06-42-00_epoch250_neg.out 2>&1 &

nohup python test_fps_multiple_experiments_align.py \
    --config ../MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE.toml \
    --checkpoint ../checkpoints/20250909_20-49-48/epoch1000 \
    --fps_values 12 60 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250909_20-49-48_epoch1000_fix_neg \
    --prompt "A man running on a beach." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/fps_comprehensive_experiment_20250909_20-49-48_epoch1000_fix_neg_run.out 2>&1 &


nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_EXTREME_TEST.toml' > ./output/nohup_log/fix_gate_test.out 2>&1 &


tensorboard --logdir /root/workspace/sc-diffusion-pipe/checkpoints \
  --host 0.0.0.0 \
  --port 7007 \
  --load_fast=false



ssh -L 7007:127.0.0.1:7007 shihanc-shuttercontrol.workbench.prod.netflix.net
