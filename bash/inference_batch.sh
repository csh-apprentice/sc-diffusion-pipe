#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference


echo "$(date) starting inferencing 2"
python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 30 \
    --frames 16 \
    --output_dir ../output/16steps/clean_category_42 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_clean_42_16frames.out 2>&1 

echo "$(date) starting inferencing 2"
python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/clean_category_42 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_clean_42_49frames.out 2>&1 


python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 30 \
    --frames 16 \
    --output_dir ../output/16steps/20251008_20-21-45 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251008_20-21-45_16frames.out 2>&1 

python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251008_20-21-45 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251008_20-21-45_49frames.out 2>&1 


python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251031_08-25-18/wan_SC_TARGET_14B_HUMAN.toml \
    --checkpoint ../checkpoints/20251031_08-25-18/epoch1000 \
    --fps_values 0.0 \
    --steps 30 \
    --frames 16 \
    --output_dir ../output/16steps/20251031_08-25-18 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251031_08-25-18_16frames.out 2>&1 

python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251031_08-25-18/wan_SC_TARGET_14B_HUMAN.toml \
    --checkpoint ../checkpoints/20251031_08-25-18/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251031_08-25-18 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251031_08-25-18_49frames.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_batch_prompts.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint_parent ../checkpoints/20251008_20-21-45 \
#     --checkpoint_prefix epoch \
#     --checkpoint_start 200 \
#     --checkpoint_end 1000 \
#     --checkpoint_interval 100 \
#     --fps_values 0.0 \
#     --steps 1 \
#     --frames 4 \
#     --output_dir ../output/onestep/20251008_20-21-45 \
#     --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/bokeh_20251008_20-21-45.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_batch_prompts.py \
#     --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
#     --checkpoint_parent ../checkpoints/20251008_20-21-45 \
#     --checkpoint_prefix epoch \
#     --checkpoint_start 200 \
#     --checkpoint_end 1000 \
#     --checkpoint_interval 100 \
#     --fps_values 0.0 \
#     --steps 1 \
#     --frames 4 \
#     --output_dir ../output/onestep/20251008_20-21-45 \
#     --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/bokeh_20251008_20-21-45.out 2>&1 


# echo "$(date) starting inferencing 2"
# python test_fps_batch_prompts.py \
#     --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
#     --checkpoint_parent ../checkpoints/20251023_00-19-38 \
#     --checkpoint_prefix epoch \
#     --checkpoint_start 200 \
#     --checkpoint_end 1000 \
#     --checkpoint_interval 100 \
#     --fps_values 0.0 \
#     --steps 1 \
#     --frames 4 \
#     --output_dir ../output/onestep/20251023_00-19-38 \
#     --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
#     --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
#     --seed 42 \
#     --port 29502 \
#     --width 512 \
#     --height 512 \
#     --base_only \
#     > ../output/nohup_log/bokeh_20251023_00-19-38.out 2>&1 


python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251024_18-52-42/wan_SC_TARGET_14B_TEMP_150.toml \
    --checkpoint_parent ../checkpoints/20251024_18-52-42 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251024_18-52-42 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/temp_20251024_18-52-42.out 2>&1 


python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251027_23-58-53/wan_SC_TARGET_14B_SHUTTER_150_8f.toml \
    --checkpoint_parent ../checkpoints/20251027_23-58-53 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251027_23-58-53 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251027_23-58-53.out 2>&1 


# echo "$(date) starting inferencing 2"
python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251027_22-55-30/wan_SC_TARGET_14B_BOKEH_150.toml \
    --checkpoint_parent ../checkpoints/20251027_22-55-30 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251027_22-55-30 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251027_22-55-30.out 2>&1 

python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml\
    --checkpoint_parent ../checkpoints/20251023_00-19-38 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251023_00-19-38 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251023_00-19-38.out 2>&1 



python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251030_08-16-52 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251030_08-16-52_49frames.out 2>&1 


python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml   \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251014_06-32-03 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251014_06-32-03_49frames.out 2>&1 

    


python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml   \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251023_00-19-38 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251023_00-19-38_49frames.out 2>&1 


python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251103_09-44-34/wan_SC_TARGET_14B_3SHAPES_150SCENE_1103.toml \
    --checkpoint_parent ../checkpoints/20251103_09-44-34 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251103_09-44-34 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251103_09-44-34.out 2>&1 



python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251106_09-42-00/wan_SC_TARGET_14B_FPS_HORSE_ABLATION.toml\
    --checkpoint_parent ../checkpoints/20251106_09-42-00 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251106_09-42-00 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251106_09-42-00.out 2>&1 



python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251106_09-42-00/wan_SC_TARGET_14B_FPS_HORSE_ABLATION.toml   \
    --checkpoint ../checkpoints/20251106_09-42-00/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251106_09-42-00 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251106_09-42-00_49frames.out 2>&1 



python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251107_00-49-00/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint_parent ../checkpoints/20251107_00-49-00 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251107_00-49-00 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/temp_20251107_00-49-00.out 2>&1 



python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251107_00-51-16/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml \
    --checkpoint_parent ../checkpoints/20251107_00-51-16 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251107_00-51-16 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/temp_20251107_00-51-16.out 2>&1 




python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251107_01-19-02/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_ABLATION.toml \
    --checkpoint_parent ../checkpoints/20251107_01-19-02 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251107_01-19-02 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251107_01-19-02.out 2>&1 




