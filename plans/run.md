Training Command:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_VID_BASEONLY.toml' > ./output/nohup_log/single_video_baseonly_6000.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml' > ./output/temp_nohup_log/temp_mf_all_104.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_BALL_KITCHEN.toml' > ./output/bokeh_nohup_log/ball_kitchen.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN_SOFTPLUS.toml' > ./output/bokeh_nohup_log/human_softplus.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUTAIN_FLOWER_SOFTPLUS.toml' > ./output/temp_nohup_log/mf_softplus.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUE2D.toml' > ./output/temp_nohup_log/redblue_deepthird2d.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE.toml' > ./output/temp_nohup_log/redbluewhite_deepthird.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml' > ./output/bokeh_nohup_log/moutain_flower_08.out 2>&1 &


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
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shapes_single_new \
  --modes both \
  --samples_per_fps 1 \
  --align_img_video \
  --min_speed 400 \
  --max_speed 400 \
  --min_obj 1 \
  --max_objs 1 \
  --seed 91 \
  --caption_relations 

  --ensure_static_video \




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
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20250917_23-08-48/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml \
    --checkpoint ../checkpoints/20250917_23-08-48/epoch1000 \
    --fps_values 12 60 240 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250917_23-08-48_epoch1000_fps_only_new \
    --prompt "A man running on the beach." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得 不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --fps_only \
    > ../output/nohup_log/fps_comprehensive_experiment_20250917_23-08-48_epoch1000_fps_only_new.out 2>&1 &

nohup python test_fps_multiple_experiments_align.py \
    --config ../MY_TOML/wan_SC_TARGET_14B_DUMMY_JOINT.toml \
    --checkpoint ../checkpoints/20250926_23-41-32/epoch1000 \
    --fps_values 1 4 32 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20250926_23-41-32_epoch1000_bokeh \
    --prompt "A man running on the beach." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20250926_23-41-32_epoch1000_run.out 2>&1 &



python resize_image.py \
  --input /root/workspace/BokehMe/K40/21/human.jpg \
  --output /root/workspace/sc-diffusion-pipe/dataset/human_bokeh/1.6/human.jpg\
  --width  512\
  --height  675

python resize_image.py \
  --input /root/workspace/BokehMe/inputs/mouflower.jpg \
  --output /root/workspace/BokehMe/inputs/mouflower_resize.jpg \
  --width  512\
  --height 364



nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_EXTREME_TEST.toml' > ./output/nohup_log/fix_gate_test.out 2>&1 &


tensorboard --logdir /root/workspace/sc-diffusion-pipe/checkpoints \
  --host 0.0.0.0 \
  --port 7007 \
  --load_fast=false



ssh -L 7007:127.0.0.1:7007 shihanc-shuttercontrol.workbench.prod.netflix.net


python make_palette_2x2.py \
  --out four_colors.png \
  --colors "#FF0000,#00FF00,#0000FF,#FFFF00"
