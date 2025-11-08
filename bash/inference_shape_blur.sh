#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue
cd inference

# echo "$(date) starting inferencing epoch 1000"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251016_19-16-00/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_RANDOM.toml \
#     --checkpoint ../checkpoints/20251016_19-16-00/epoch1000\
#     --fps_values -1.0 0.0 1.0\
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251016_19-16-00_epoch1000_neg \
#     --prompt "A soccer ball flying toward the goal as the goalie dives to block it." \
#     --negative_prompt "pure color background, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/shutter_20251016_19-16-00_epoch1000_neg.out 2>&1 


# echo "$(date) starting inferencing epoch 1000"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251018_23-02-25/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_RANDOM_TRIGGER.toml \
#     --checkpoint ../checkpoints/20251018_23-02-25/epoch500\
#     --fps_values 0.0  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251018_23-02-25_epoch500 \
#     --prompt "A ceiling fan spinning under warm yellow light." \
#     --negative_prompt "pure color background, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251018_23-02-25_epoch500_other.out 2>&1 

# echo "$(date) starting inferencing epoch 1000"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
#     --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251014_06-32-03_epoch1000_neg_allflat \
#     --prompt "A ceiling fan spinning under warm yellow light." \
#     --negative_prompt "flat red background, flat green background, flat blue background, flat yellow background, flat purple background, flat orange background, flat teal background, flat pink background, flat white background, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_negallflat.out 2>&1 

echo "$(date) starting inferencing epoch 1000"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
#     --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
#     --fps_values -0.5 0.5 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251014_06-32-03_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/ablation \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_other.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_fps_only \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span2 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --fps_only\
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_fpsonly.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_fps_only \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span3 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --fps_only\
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span3_fpsonly.out 2>&1 


# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251019_04-51-46/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_DEBUG.toml \
#     --checkpoint ../checkpoints/20251019_04-51-46/epoch1000\
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251019_04-51-46_epoch1000_neg_50 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/all_prompts \
#     --negative_prompt "pure color background, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251019_04-51-46_epoch1000_neg.out 2>&1 

# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251019_19-39-09/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_NEW.toml \
#     --checkpoint ../checkpoints/20251019_19-39-09/epoch1000\
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251019_19-39-09_epoch1000_neg_50 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/all_prompts \
#     --negative_prompt "pure color background, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251019_19-39-09_epoch1000_neg.out 2>&1 


# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251019_08-09-41/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_DEBUG_TRIGGER.toml \
#     --checkpoint ../checkpoints/20251019_08-09-41/epoch625\
#     --fps_values 0.0  \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251019_08-09-41_epoch625 \
#     --prompt "A ceiling fan spinning under warm yellow light." \
#     --negative_prompt "[NFSC_ST_1018]: a CNEWHCOA background, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251019_08-09-41_epoch600_other.out 2>&1 



# echo "$(date) starting inferencing epoch 1000"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251021_07-35-12/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_TRIGGER.toml \
#     --checkpoint ../checkpoints/20251021_07-35-12/epoch1000\
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251021_07-35-12_epoch1000_neg \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/all_prompts \
#     --negative_prompt "[NFSC_ST_1018] style, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251021_07-35-12_epoch1000_trigger_neg.out 2>&1 


# echo "$(date) starting inferencing epoch 1000"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251024_18-20-36/wan_SC_TARGET_14B_SHUTTER_150.toml\
#     --checkpoint ../checkpoints/20251024_18-20-36/epoch1000 \
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251024_18-20-36_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251024_18-20-36_epoch1000.out 2>&1 

# echo "$(date) starting inferencing epoch 1000"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251027_03-52-36/wan_SC_TARGET_14B_SHUTTER_150_4f.toml \
#     --checkpoint ../checkpoints/20251027_03-52-36/epoch1000 \
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251027_03-52-36_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251027_03-52-36_epoch1000.out 2>&1 

# echo "$(date) starting inferencing epoch 1000"
# # cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251027_23-58-53/wan_SC_TARGET_14B_SHUTTER_150_8f.toml \
#     --checkpoint ../checkpoints/20251027_23-58-53/epoch500 \
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251027_23-58-53_epoch500 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251027_23-58-53_epoch500.out 2>&1 



python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251106_09-42-01/wan_SC_TARGET_14B_FPS_HORSE_ABLATION.toml \
    --checkpoint ../checkpoints/20251106_09-42-01/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251106_09-42-01_epoch1000 \
    --prompt "A man running on the fild under sunshine." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251106_09-42-01_epoch1000.out 2>&1 
