#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference
# python test_fps_multiple_experiments_align.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251002_23-52-43/wan_SC_TARGET_14B_SHAPE_SINGLE_CENTER.toml \
#     --checkpoint ../checkpoints/20251002_23-52-43/epoch1000 \
#     --fps_values 1.4 2 2.8 4 5.6 8 11 \
#     --steps 30 \
#     --frames 49 \
#     --output_dir ../output/20251002_23-52-43_epoch1000_bokeh \
#     --prompt "A dog jumping over a wooden hurdle." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251002_23-52-43_epoch1000_dog.out 2>&1 


echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml \
#     --checkpoint ../checkpoints/20250930_05-53-32/epoch1000 \
#     --fps_values 1 8 64 \
#     --steps 50 \
#     --frames 16 \
#     --output_dir ../output/empty_prompts/20250930_05-53-32_epoch1000 \
#     --prompt "" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20250930_05-53-32_epoch1000_empty.out 2>&1 


# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-47-24/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_06-47-24/epoch1000 \
#     --fps_values 0.025  0.4 1.4 \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251008_06-47-24_epoch1000_debug \
#     --prompt "A red sphere is positioned in the foreground, a blue cube sits behind it at the center, and a green cylinder is placed further back in the background, all aligned diagonally on a gray surface under soft directional lighting." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_06-47-24_epoch1000_debug.out 2>&1 

# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_06-47-24/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_06-47-24/epoch1000 \
#     --fps_values 0.025  0.4 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_06-47-24_epoch1000_debug \
#     --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_06-47-24_epoch1000_run.out 2>&1 

# echo "$(date) starting inferencing 1"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251010_22-08-59/wan_SC_TARGET_14B_SYN3SHAPES_ALL.toml \
#     --checkpoint ../checkpoints/20251010_22-08-59/epoch2000 \
#     --fps_values 2  16 64 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251010_22-08-59_epoch2000_debug \
#     --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251010_22-08-59_epoch2000_debug.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.025 0.05 0.1 0.2 0.4 0.8 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000_fpsonly \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1 \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_span1_fps_only.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.03 0.05 0.08 0.12 0.15 0.3 0.5 0.6 0.8 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000_multi \
#     --prompt "Smiling man in blue polo on a lawn, suburban house behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_multi_prompts.out 2>&1


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values  0.025 0.05 0.1 0.2 0.4 0.8 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000_fpsonly \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span2 \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_span2_fps_only.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values  0.025 0.05 0.1 0.2 0.4 0.8 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000_fpsonly \
#     --prompt /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span3 \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_span3_fps_only.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.025 0.1 1.4 \
#     --steps 75 \
#     --frames 49 \
#     --output_dir ../output/20251008_20-21-45_epoch1000 \
#     --prompt "A dog jumping over a wooden hurdle, fences and trees behind." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_dog_75.out 2>&1 

# echo "$(date) starting inferencing"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_00-31-01/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_00-31-01/epoch400 \
#     --fps_values 0.1  0.4 2.8 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251008_00-31-01_epoch400_fpsonly \
#     --prompt "A dog jumping over a wooden hurdle in the foreground, behind is the grass field." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, A young woman in a fuzzy brown jacket and winter hat holds a cup at a lively outdoor Christmas market." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --fps_only \
#     > ../output/nohup_log/bokeh_20251008_00-31-01_epoch400_dog_fpsonly.out 2>&1 

# echo "$(date) starting inferencing"
# cd inference
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_00-31-25/wan_SC_TARGET_14B_3SHAPES_FOCAL.toml \
#     --checkpoint ../checkpoints/20251008_00-31-25/epoch500 \
#     --fps_values  40 50 65 \
#     --steps 30 \
#     --frames 16 \
#     --output_dir ../output/20251008_00-31-25_epoch500 \
#     --prompt "A red sphere is positioned in the foreground, a blue cube sits behind it at the center, and a green cylinder is placed further back in the background, all aligned diagonally on a gray surface under soft directional lighting." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_00-31-25_epoch500_debug.out 2>&1 

# echo "$(date) starting inferencing with simple combined checkpoint approach"
# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
#     --fps_values 0.025 0.05 0.1 1.4 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/empty_prompts/20251008_20-21-45_epoch1000 \
#     --prompt "" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_empty.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251020_07-56-38/wan_SC_TARGET_14B_3SHAPES_SIMPLE.toml \
#     --checkpoint ../checkpoints/20251020_07-56-38/epoch1000\
#     --fps_values 0.025 0.1 1.4  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251020_07-56-38_epoch1000 \
#     --prompt "A man in the foreground standing on the bridge, with a city full of colorful signs in the background." \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251020_07-56-38_epoch1000.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251021_08-22-01/wan_SC_TARGET_14B_SHAPE_SOLID_MINI.toml \
#     --checkpoint ../checkpoints/20251021_08-22-01/epoch1000 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251021_08-22-01_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251021_08-22-01_epoch1000_debug.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251021_21-55-42/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_RESUME.toml \
#     --checkpoint ../checkpoints/20251021_21-55-42/epoch750 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251021_21-55-42_epoch750 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251021_21-55-42_epoch750_debug.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-18-19/wan_SC_TARGET_14B_3SHAPES_30SCENE.toml \
#     --checkpoint ../checkpoints/20251023_07-18-19/epoch475 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251023_07-18-19_epoch100b0 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/quick_test \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251023_07-18-19_epoch475_all.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/wan_SC_TARGET_14B_3SHAPES_BASE.toml \
#     --checkpoint ../checkpoints/20251023_07-43-45/epoch1000 \
#     --fps_values 0.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251023_07-43-45_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251023_07-43-45_epoch1000_all.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251024_22-29-25/wan_SC_TARGET_14B_BOKEH_150.toml \
#     --checkpoint ../checkpoints/20251024_22-29-25/epoch450 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251024_22-29-25_epoch450 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/quick_test \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251024_22-29-25_epoch500_all.out 2>&1 


# echo "$(date) starting inferencing with simple combined checkpoint approach"
# python test_fps_combined_simple.py \
#     --config /root/workspace/sc-diffusion-pipe/bokeh_TOML/wan_SC_TARGET_14B_BOKEH_150_02.toml \
#     --checkpoint_base ../checkpoints/20251023_07-43-45/epoch1000 \
#     --checkpoint_fps ../checkpoints/20251026_07-09-05/epoch200 \
#     --fps_values -1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251026_07-09-05_epoch200 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251026_07-09-05_epoch200.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251027_03-23-30/wan_SC_TARGET_14B_BOKEH_150.toml \
#     --checkpoint ../checkpoints/20251027_03-23-30/epoch1000 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251027_03-23-30_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251027_03-23-30_epoch1000_all.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251027_22-55-30/wan_SC_TARGET_14B_BOKEH_150.toml \
#     --checkpoint ../checkpoints/20251027_22-55-30/epoch450 \
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251027_22-55-30_epoch450 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251027_22-55-30_epoch450_all.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251027_22-55-30/wan_SC_TARGET_14B_BOKEH_150.toml \
#     --checkpoint ../checkpoints/20251027_22-55-30/epoch450 \
#     --fps_values -1.0 0.0 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251027_22-55-30_epoch450 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251027_22-55-30_epoch450_all.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251028_08-08-10/wan_SC_TARGET_14B_3SHAPES_30SCENE_1028.toml \
#     --checkpoint ../checkpoints/20251028_08-08-10/epoch500 \
#     --fps_values 1.0 \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251028_08-08-10_epoch500 \
#     --prompt "Close shot of a golden retriever, suburban house behind" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251028_08-08-10_epoch500.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml\
#     --checkpoint ../checkpoints/20251028_21-19-57/epoch800 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251028_21-19-57_epoch800 \
#     --prompt "Close-up of a cat sitting by a rainy window, its face reflected on the glass as raindrops trickle down.”" \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251028_21-19-57_epoch800_newcat.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-25-27/wan_SC_TARGET_14B_BOKEH_150_DIVERSE.toml \
#     --checkpoint ../checkpoints/20251028_21-25-27/epoch800 \
#     --fps_values -1.0 0.0 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251028_21-25-27_epoch800 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/quick_test \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251028_21-25-27_epoch1000.out 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
#     --checkpoint ../checkpoints/20251030_08-16-52/epoch1000 \
#     --fps_values -1.0 -0.5 0.0 0.5 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251030_08-16-52_epoch1000_neg \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/fail_prompts \
#     --negative_prompt "Flat Color bounding, pure color background, synthetic style, unrealistic, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > befo 2>&1 

# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
#     --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
#     --fps_values -1.0 -0.5 0.0 0.5 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251014_06-32-03_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span3 \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_other.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
#     --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
#     --fps_values -1.0 -0.5 0.0 0.5 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251014_06-32-03_epoch1000_neg \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/fail_prompts \
#     --negative_prompt "Flat Color bounding, pure color background, synthetic style, unrealistic, ghost effect, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_other.out 2>&1 


# python test_fps_multiple_experiments_align_old.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-17-12/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_5.toml \
#     --checkpoint ../checkpoints/20251030_08-17-12/epoch1000 \
#     --fps_values -1.0 -0.5 0.0 0.5 1.0  \
#     --steps 50 \
#     --frames 49 \
#     --output_dir ../output/20251030_08-17-12_epoch1000 \
#     --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/all_prompts \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     > ../output/nohup_log/bokeh_20251030_08-17-12_epoch1000_all.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_fps_only \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span2 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --fps_only \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_span2_only.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_fps_only \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span3 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --fps_only \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_span3_only.out 2>&1 
