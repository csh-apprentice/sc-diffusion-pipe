Training Command:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe



bash /root/workspace/sc-diffusion-pipe/bash/run_principal_orthogonality.sh /root/workspace/sc-diffusion-pipe/output/principal_orthogonality/20251015_14-02-14/orthagnal_check.log

bash /root/workspace/sc-diffusion-pipe/bash/run_backbone_drift.sh /root/workspace/sc-diffusion-pipe/output/backbone_drift/K16/20251014_06-32-03/backbone_drift.log

bash /root/workspace/sc-diffusion-pipe/bash/run_backbone_drift_2.sh /root/workspace/sc-diffusion-pipe/output/backbone_drift/20251019_04-51-46/backbone_drift.log

bash /root/workspace/sc-diffusion-pipe/bash/compute_similarity_matrix.sh /root/workspace/sc-diffusion-pipe/output/similarity_matrix/20251028_21-19-57/epoch1000/similar_matrix.log

bash /root/workspace/sc-diffusion-pipe/bash/compute_similarity_matrix_2.sh /root/workspace/sc-diffusion-pipe/output/similarity_matrix/20251027_23-58-53/epoch1000/similar_matrix.log

bash /root/workspace/sc-diffusion-pipe/bash/analyze_similarity_matrix.sh /root/workspace/sc-diffusion-pipe/output/similarity_matrix/20251027_23-58-53/epoch1000 0.5

bash /root/workspace/sc-diffusion-pipe/bash/run_fps_condition_similarity.sh /root/workspace/sc-diffusion-pipe/output/fps_condition_similarity/20251024_22-29-25/epoch1000/fps_condition_similarity.log


bash /root/workspace/sc-diffusion-pipe/bash/run_ytext_analysis.sh /root/workspace/sc-diffusion-pipe/output/ytext_analysis/20250930_05-53-32/epoch1000/y_text_anlysis.log

bash /root/workspace/sc-diffusion-pipe/bash/compute_energy.sh /root/workspace/sc-diffusion-pipe/output/energy_compute/20251028_21-19-57/epoch1000/energy_compute.log


bash /root/workspace/sc-diffusion-pipe/bash/run_condition_eval.sh /root/workspace/sc-diffusion-pipe/output/condition_eval/20251028_21-19-57/epoch800/eval_condition.log


bash bash/run_subspace_analysis.sh \
    /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000 \
    -1.0 -0.5 0.0 0.5 1.0  \
    2>&1 | tee subspace_analysis_epoch1000.log

  b                                         
      /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
      /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000 \
      -1.0 -0.5 0.0 0.5 1.0 \
      2>&1 | tee subspace_analysis_epoch1000.log

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_SHUTTER_150_8f.toml' > ./output/fps_nohup_log/shutter_150_8f.out 2>&1 &


nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_ABLATION.toml' > ./output/fps_nohup_log/shutter_single_ablation.out 2>&1 &


nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_TRIGGER.toml' > ./output/fps_nohup_log/fps_shape_blur_trigger.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_FPS_HORSE_ABLATION.toml' > ./output/fps_nohup_log/shutter_horse.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_DEBUG_TRIGGER.toml' > ./output/fps_nohup_log/fps_shape_blur_debug_trigger.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH_SYNTHESIS.toml' > ./output/fps_bokeh_log/2dshape_bokeh.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_BASE.toml' > ./output/bokeh_nohup_log/3shapes_bokeh_base.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_30SCENE_1028.toml' > ./output/bokeh_nohup_log/3shapes_Bokeh_1028.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml' > ./output/bokeh_nohup_log/3shapes_Bokeh_1030_3.out 2>&1 &


nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_150SCENE_1103.toml' > ./output/bokeh_nohup_log/3shapes_Bokeh_150_1103.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_5.toml' > ./output/bokeh_nohup_log/3shapes_Bokeh_1030_5.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_BOKEH_150_DIVERSE.toml' > ./output/bokeh_nohup_log/bokeh_150_diverse.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml' > ./output/bokeh_nohup_log/SOLID_MINI_PART12.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_BOKEH_150_02.toml' > ./output/bokeh_nohup_log/bokeh_150_02.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_HUMAN.toml' > ./output/bokeh_nohup_log/bokeh_human_new.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_VID_BASEONLY.toml' > ./output/nohup_log/single_video_baseonly_6000.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_MOUTAIN_FLOWER.toml' > ./output/temp_nohup_log/temp_mf_all_104.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_TEMP_150_r8.toml' > ./output/temp_nohup_log/temp_150_r8.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_TEMP_BASE.toml' > ./output/temp_nohup_log/temp_base.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config temp_TOML/wan_SC_TARGET_14B_SHAPE_TEMP_REDBLUEWHITE_ALL.toml' > ./output/temp_nohup_log/temp_redbluewhite_all.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config focal_TOML/wan_SC_TARGET_14B_3SHAPES_FOCAL_ALL_RESUME.toml' > ./output/focal_nohup_log/focal_threeshapes_all_resume.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_SYN3SHAPES_ALL.toml' > ./output/bokeh_nohup_log/syn3shapes_bokeh_center_all.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_HUMAN_HOUSE_EMPTY.toml' > ./output/bokeh_nohup_log/humanhouse_bokeh_center_empty.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config focal_TOML/wan_SC_TARGET_14B_3SHAPES_FOCAL_RESUME.toml' > ./output/focal_nohup_log/3shapes_focal_resume.out 2>&1 &

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config bokeh_TOML/wan_SC_TARGET_14B_3SHAPES_LOWRANK.toml' > ./output/bokeh_nohup_log/3shapes_bokeh_lowrank.out 2>&1 &

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
  --input /root/workspace/BokehMe/K2/21/human.jpg \
  --output /root/workspace/sc-diffusion-pipe/dataset/human_bokeh/32/human.jpg\
  --width  512\
  --height  675

python resize_image.py \
  --input /root/workspace/BokehMe/inputs/mouflower.jpg \
  --output /root/workspace/BokehMe/inputs/mouflower_resize.jpg \
  --width  512\
  --height 364

python resize_image.py \
  --input /root/workspace/BokehMe/DPT/input/syn3shapes.png \
  --output /root/workspace/BokehMe/DPT/input/syn3shapes_resize.png \
  --width  640 \
  --height 462



nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config MY_TOML/wan_SC_TARGET_14B_FPS_EXTREME_TEST.toml' > ./output/nohup_log/fix_gate_test.out 2>&1 &


tensorboard --logdir /root/workspace/sc-diffusion-pipe/checkpoints \
  --host 0.0.0.0 \
  --port 7007 \
  --load_fast=false



ssh -L 7007:127.0.0.1:7007 shihanc-shuttercontrol.workbench.prod.netflix.net


python make_palette_2x2.py \
  --out four_colors.png \
  --colors "#FF0000,#00FF00,#0000FF,#FFFF00"



python generate_synthetic_blur_dataset_multi.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shutter_one_shot_150_8f/ \
  --modes video \
  --num_scales 150 \
  --samples_per_scale 1 \
  --num_frames 8 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_obj 1 \
  --max_objs 1 \
  --seed 42 \
  --bg_white_prob 1.0 \
  --caption_relations 


python generate_synthetic_blur_dataset_fps.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shutter_one_shot_ablation/ \
  --modes video \
  --fps_list 12,24,30,40,60,120,240 \
  --samples_per_fps 1 \
  --num_frames 8 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_objs 1 \
  --max_objs 1 \
  --seed 42 \
  --caption_relations 


python generate_synthetic_blur_dataset_full_random.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shapes_blur_random/9s12f \
  --modes video \
  --num_scales 9 \
  --samples_per_scale 2 \
  --num_frames 12 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_obj 1 \
  --max_objs 3 \
  --caption_relations 


python generate_synthetic_blur_dataset_pallete_random.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shutter_one_shot_150_8f/ \
  --modes video \
  --num_scales 9 \
  --samples_per_scale 1 \
  --num_frames 8 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_obj 1 \
  --max_objs 1 \
  --bg_white_prob 1.0 \
  --caption_relations 

python generate_synthetic_blur_dataset_final.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shapes_blur_new/1s4f \
  --modes video \
  --num_scales 1 \
  --samples_per_scale 2 \
  --num_frames 4 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_obj 1 \
  --max_objs 3 \
  --caption_relations 
  

python generate_temp_by_scale.py \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/2dshapes_temp_part2/9s \
  --num_scenes 3 \
  --num_scales 9 \
  --white-bg \
  --k-lo 2000 --k-hi 20000 --k-ref 6500 --preserve-luminance

python generate_temp_by_scale.py \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/temp_one_shot_150 \
  --num_scenes 1 \
  --num_scales 150 \
  --white-bg \
  --seed 38 \
  --min_shapes 3 \
  --max_shapes 3 \
  --k-lo 2000 --k-hi 20000 --k-ref 6500 --preserve-luminance 


python generate_temp_by_scale.py \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/benchmark_temp \
  --num_scenes 1 \
  --num_scales 150 \
  --white-bg \
  --seed 38 \
  --min_shapes 3 \
  --max_shapes 3 \
  --debug-scales 0.0 \
  --k-lo 2000 --k-hi 20000 --k-ref 6500 --preserve-luminance 

python apply_temp_shifts.py \
  --input_image /root/workspace/sc-diffusion-pipe/utils/synthesis/crisp_shapes.png \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/temp_one_shot_150 \
  --num_scales 150 \
  --k-lo 2000 \
  --k-hi 20000 \
  --k-ref 6500 \
  --preserve-luminance \
  --seed 38


python apply_temp_shifts.py \
  --input_image /root/workspace/sc-diffusion-pipe/dataset/mouflower_temp/6000/mouflower.jpg \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/mouflower_ablation \
  --num_scales 7 \
  --k-lo 2000 \
  --k-hi 20000 \
  --k-ref 6500 \
  --preserve-luminance \
  --seed 42


find /root/workspace/sc-diffusion-pipe/dataset/shapes_blur_trigger -type f -name "*.txt" -exec sed -i '1s;^;In [NFSC_ST_1018] style: ;' {} +

cd /root/workspace/sc-diffusion-pipe/dataset/temp_one_shot_150
find . -type d -exec cp /root/workspace/sc-diffusion-pipe/utils/synthesis/crisp_shapes.txt {} \;


./bash/run_yfps_analysis.sh ./analysis/yellow_light.log



python evaluate.py \
    --dimension 'subject_consistency' 'background_consistency'  'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' \
    --videos_path /root/workspace/sc-diffusion-pipe/output/50_49/20251008_20-21-45 \
    --mode=custom_input \
    --output_path "./50_49/evaluation_20251008_20-21-45/"

python evaluate.py \
    --dimension 'subject_consistency' 'background_consistency'  'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' \
    --videos_path /root/workspace/sc-diffusion-pipe/output/50_49/20251031_08-25-18 \
    --mode=custom_input \
    --output_path "./50_49/evaluation_20251031_08-25-18/"

python evaluate.py \
    --dimension 'subject_consistency' 'background_consistency'  'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' \
    --videos_path /root/workspace/sc-diffusion-pipe/output/50_49/clean_category_42 \
    --mode=custom_input \
    --output_path "./evaluation_clean/"


python evaluate.py \
    --dimension 'subject_consistency' 'background_consistency'  'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' \
    --videos_path /root/workspace/sc-diffusion-pipe/output/50_49/20251030_08-16-52 \
    --mode=custom_input \
    --output_path "./50_49/evaluation_20251030_08-16-52/"


python evaluate.py \
    --dimension 'subject_consistency' 'background_consistency'  'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' \
    --videos_path /root/workspace/sc-diffusion-pipe/output/50_49/20251023_00-19-38 \
    --mode=custom_input \
    --output_path "./50_49/evaluation_20251023_00-19-38/"



python metric/visualize_score_progression.py \
  --checkpoint_dirs scores/20251031_08-25-18 scores/20251008_20-21-45  \
  --labels "Human" "3 Shapes"  \
  --score_types ssf ss_fd dvs \
  --title_prefix "One Shot Training Progression" \
  --baseline scores/clean_category_99 \
  --output_dir plots/one_shot


python metric/visualize_score_progression.py \
  --checkpoint_dirs scores/20251031_08-25-18 scores/20251008_20-21-45  \
  --labels "Human" "3 Shapes"  \
  --score_types ssf ss_fd dvs \
  --title_prefix "One Shot Training Progression" \
  --baseline scores/clean_category_99 \
  --output_dir plots/one_shot



python metric/visualize_score_progression.py \
  --checkpoint_dirs scores/20251014_06-32-03 scores/20251030_08-16-52 scores/20251023_00-19-38  \
  --labels "Shutter Speed (30)" "Bokeh (30)" "Temp (30)"  \
  --score_types ssf ss_fd  \
  --title_prefix " Training Progression" \
  --baseline scores/clean_category_99 \
  --output_dir plots/30_shots


python metric/visualize_score_progression.py \
  --checkpoint_dirs scores/20251027_22-55-30 scores/20251030_08-16-52   \
  --labels "One shot" "30 shots" \
  --score_types ssf ss_fd  \
  --title_prefix " One shot vs. 30 shots Training Progression" \
  --baseline scores/clean_category_99 \
  --output_dir plots/ablation


python metric/visualize_score_progression.py \
  --checkpoint_dirs scores/20251014_06-32-03  \
  --labels "Shutter Speed (30)"   \
  --score_types ssf ss_fd  \
  --title_prefix " Training Progression" \
  --baseline scores/clean_category_99 \
  --output_dir plots/debug


cp -r /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1 /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1_focus


cp -r /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span2 /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span2_focus

cp -r /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span3 /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span3_focus


# 2. Loop through each .txt file in the copied folder and append text
for file in /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span3_focus/*.txt; do
    echo ", the camera focus on the foreground." >> "$file"
done