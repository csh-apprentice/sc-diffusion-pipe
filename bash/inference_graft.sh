#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# echo "$(date) starting inferencing"
cd inference



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft \
    --prompt "A bear swiping at a fish in a river." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_grat.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/span3 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_debug_graft_span3.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span3 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_span3.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml   \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251008_20-21-45_epoch1000_graft \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251008_20-21-45_epoch1000_debug_graft_span1.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_medium_aperture \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1_medium_aperture \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_span1_med_ap.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 0.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_fail_foreground \
    --prompt "A person in the foreground under neon lights, with a city full of colorful signs in the background, the camera focus on the foreground." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_neon_foreground.out 2>&1 



    



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values  -1.0 0.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_fail_graft_medium_aperture \
    --prompt "A row of hanging lanterns sways gently above a quite cobblestone path, the camera focus on the foreground." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_candel_medium_00.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values  0.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_fail_graft_medium_aperture \
    --prompt "A row of hanging lanterns sways gently above a quite cobblestone path, medium aperture." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_candel_medium_00.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -2.0 -1.5  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_fail_graft \
    --prompt "A row of hanging lanterns sways gently above a quite cobblestone path." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_candel.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251031_08-25-18/wan_SC_TARGET_14B_HUMAN.toml   \
    --checkpoint ../checkpoints/20251031_08-25-18/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251031_08-25-18_epoch1000_graft \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/span1 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251031_08-25-18_epoch1000_debug_graft_span1.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml  \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000\
    --fps_values -1.0 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251023_00-19-38_epoch1000_graft \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/span2 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_debug_graft_span2.out 2>&1 



python test_fps_graft_align.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_graft_align \
    --prompt "A bear swiping at a fish in a river." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29503 \
    --width 512 \
    --height 512 \
    --align \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_graft_align.out 2>&1 


python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft_appendix_dog \
    --prompt "A dog chasing a ball across a field." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_appendix_dog.out 2>&1 

python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft_appendix_man_dog \
    --prompt "A man running with his dog on the field." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_appendix_man_dog.out 2>&1 


python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft_appendix_swan_full_sprint \
    --prompt "A low-angle, head-on shot of a majestic swan running directly toward the camera on a shimmering lake. It is flapping its large wings vigorously, creating a dramatic spray of water as it attempts to take off." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_appendix_swan.out 2>&1 

python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft_appendix/ped \
    --prompt "Pedestrians walking briskly across a busy city intersection with cars rushing fast in the background." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_appendix_ped.out 2>&1 



python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/ablation/warm \
    --prompt "A bear swiping at a fish in a river, shot with extreme warm temperature." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_grat.out 2>&1 


python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/abaltion/cool \
    --prompt "A bear swiping at a fish in a river, shot with extreme cool temperature." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_grat.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_abaltion/motion_blur/extreme \
    --prompt "A large, revolving glass door spinning as people walk through, with extreme motion blur." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --clean \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_appendix_extreme.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_abaltion/motion_blur/clean \
    --prompt "A large, revolving glass door spinning as people walk through." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --clean \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_span2_appendix_sharp.out 2>&1 




python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_11_16/ \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/11_16 \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_11_16.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/in2out \
    --prompt "A first-person view of opening a closed door from inside the home, revealing a lovely outdoor garden." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/in2out \
    --prompt "A first-person view of opening a closed door from inside the home, revealing a lovely outdoor garden." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 

python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/ani \
    --prompt "In Anime style, a girl reading book under the tree, butterflies flying around her." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 \
    --steps 50 \
    --frames 60 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/focuschange/dog \
    --prompt "A dog jumping over a wooden hurdle, fences and trees behind, the camera focus on the dog." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_foreground.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 60 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/focuschange/hurdle \
    --prompt "A dog jumping over a wooden hurdle, fences and trees behind, the camera focus on the hurdle." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_hurdle.out 2>&1 



python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_multi/ \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/multi_depth \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_multi.out 2>&1 



python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_multi/ \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/multi_depth \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_multi.out 2>&1 


python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_background/ \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/background \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_background.out 2>&1 

python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/ \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/move_change \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 



python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/dog \
    --prompt "A dog running toward the camera and jumping over a small hurdle in the foreground." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 

python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values  -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/fox \
    --prompt "A fox running toward the camera and clearing a log in the foreground, the camera focusing on the log." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 


python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/man \
    --prompt "A soccer player sprinting toward the camera and stepping past a cone in the foreground." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 


python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/dog_cfg5 \
    --prompt "A dog running toward the camera and jumping over a wooden hurdle in the foreground, the camera focusing on the dog." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --scale 5 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 



python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/dog_40 \
    --prompt "A dog running toward the camera and jumping over a wooden hurdle in the foreground, the camera focusing on the dog." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 40 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 

python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_move_change/dog_41 \
    --prompt "A dog running toward the camera and jumping over a wooden hurdle in the foreground, the camera focusing on the dog." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 41 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251030_08-16-52_epoch1000_move_change.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/mid \
    --prompt "A line of framed photos on a long hallway wall, the camera focusing on a close frame." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_frame.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/mid \
    --prompt "A line of coffee cups arranged from near to far, the camera focusing on one cup." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_cup.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/indoor \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/indoor \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/outdoor \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/indoor \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/style \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/style \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/change \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/change \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp_change.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/indoor \
    --prompt "A giant fluffy teddy bear sitting on the couch, surrounded by colorful ballons." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/clay_one \
    --prompt "A line of clay plots arranged from near to far on a stong ledge, the camera focusing on one clay pot." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_clay.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -0.6 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/clay_one_dirty \
    --prompt "A line of clay plots arranged from near to far on a stong ledge, the camera focusing on one clay pot." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_clay.out 2>&1 

python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/clay_one_dirty \
    --prompt "A line of clay plots arranged from near to far on a stong ledge, the camera focusing on one clay pot." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_clay.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/Anime \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/Anime \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp_anime.out 2>&1 


python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/Pixel \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/Pixel \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp_pixel.out 2>&1 



python test_fps_graft.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml \
    --checkpoint ../checkpoints/20251023_00-19-38/epoch1000 \
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/temp_20251023_00-19-38_epoch1000_temp_graft/supp/Multiple \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/temp_prompts/multiple \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/temp_20251023_00-19-38_epoch1000_supp_multiple.out 2>&1 


python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000/Multiple \
    --prompt_folder /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/multiple \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_multiple.out 2>&1 
# , the camera focusing on one clay pot.

python test_fps_graft.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_graft/market \
    --prompt "A time-lapse of crowds of people rushing through an open-air market, examining stalls." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_bird.out 2>&1 



python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values 0.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/clay_backbone \
    --prompt "A line of clay plots arranged from near to far on a stong ledge, the camera focusing on one clay pot." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_clay_clean.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0\
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_clean_fan \
    --prompt "A ceiling fan spinning under warm yellow light." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_clean_fan.out 2>&1 


python test_fps_batch_prompts.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/foreground_11_19 \
    --prompt_file /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/foreground.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_foreground.out 2>&1 


python test_fps_batch_prompts.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/move_11_19 \
    --prompt_file /root/workspace/sc-diffusion-pipe/prompt_folder/bokeh_prompts/move_11_19.txt\
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_move_11_19.out 2>&1 




python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251119_18-54-17/wan_SC_TARGET_14B_SHAPE_TEMP_ADAPTER_TRAIN.toml \
    --checkpoint ../checkpoints/20251119_18-54-17/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0\
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251119_18-54-17_epoch1000 \
    --prompt "A tiger jumps over a water stream." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/shutter_20251119_18-54-17_epoch1000_tiger.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_19-13-59/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0\
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251115_19-13-59_epoch1000 \
    --prompt "A tiger jumps over a water stream." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20251115_19-13-59_epoch1000_tiger.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251119_18-54-17/wan_SC_TARGET_14B_SHAPE_TEMP_ADAPTER_TRAIN.toml \
    --checkpoint ../checkpoints/20251119_18-54-17/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0\
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251119_18-54-17_epoch1000 \
    --prompt "A tiger jumps over a water stream." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/shutter_20251119_18-54-17_epoch1000_tiger.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_19-13-59/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0\
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251115_19-13-59_epoch1000 \
    --prompt "A tiger jumps over a water stream." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20251115_19-13-59_epoch1000_tiger.out 2>&1 



python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_19-13-59/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/slides/falcon_fast \
    --prompt "A falcon diving toward the ground, the landscape shifting fast beneath it as the camera stays centered on the bird, shot with extreme fast shutter speed." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20251115_19-13-59_epoch1000_falcon.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_19-13-59/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/slides/falcon_slow \
    --prompt "A falcon diving toward the ground, the landscape shifting fast beneath it as the camera stays centered on the bird, shot with extreme slow shutter speed." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20251115_19-13-59_epoch1000_falcon_slow.out 2>&1 



python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251119_18-23-14/wan_SC_TARGET_14B_BOKEH_SHAPE_ADAPTER_TRAIN.toml\
    --checkpoint ../checkpoints/20251119_18-23-14/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251119_18-23-14\
    --prompt "A falcon diving toward the ground, the landscape shifting fast beneath it as the camera stays centered on the bird, shot with extreme fast shutter speed." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251119_18-23-14_epoch1000_falcon.out 2>&1 



python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml\
    --checkpoint ../checkpoints/20251115_22-27-44/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251115_22-27-44\
    --prompt "Smiling man in blue polo on a lawn, suburban house behind." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251115_22-27-44_epoch1000_.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251119_18-23-14/wan_SC_TARGET_14B_BOKEH_SHAPE_ADAPTER_TRAIN.toml\
    --checkpoint ../checkpoints/20251119_18-23-14/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251119_18-23-14\
    --prompt "Smiling man in blue polo on a lawn, suburban house behind." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/bokeh_20251119_18-23-14_epoch1000_.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251119_17-45-29/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_ADAPTER_TRAIN.toml\
    --checkpoint ../checkpoints/20251119_17-45-29/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251119_17-45-29 \
    --prompt "A bicyle rider in full sprint on the suburban road." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/shutter_20251119_17-45-29_epoch1000_.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_07-34-10/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_ABLATION.toml\
    --checkpoint ../checkpoints/20251115_07-34-10/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251115_07-34-10_epoch1000 \
    --prompt "A bicyle rider in full sprint on the suburban road." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/shutter_20251115_07-34-10_epoch1000_.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_19-13-59/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/slides/falcon_motionblur \
    --prompt "A falcon diving toward the ground, the landscape shifting fast beneath it as the camera stays centered on the bird, shot with extreme motion blur." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20251115_19-13-59_epoch1000_falcon_motionblur.out 2>&1 

python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml \
    --checkpoint ../checkpoints/20251115_19-13-59/epoch1000\
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/slides/falcon_sharp \
    --prompt "A falcon diving toward the ground, the landscape shifting fast beneath it as the camera stays centered on the bird, video is sharp." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    > ../output/nohup_log/temp_20251115_19-13-59_epoch1000_falcon_sharp.out 2>&1 


python test_fps_multiple_experiments_align_old.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values 0.0\
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251014_06-32-03_epoch1000_clean_fan \
    --prompt "A ceiling fan spinning under warm yellow light." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/shutter_20251014_06-32-03_epoch1000_clean_fan.out 2>&1 



python test_fps_batch_prompts.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -1.0 -0.5 0.0 0.5 1.0  \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/20251030_08-16-52_epoch1000_graft_supp/FOV \
    --prompt_file /root/workspace/sc-diffusion-pipe/prompt_folder/shutter_prompts/FOV.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_FOV.out 2>&1 


python test_fps_multiple_experiments_align.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    --checkpoint ../checkpoints/20251014_06-32-03/epoch1000\
    --fps_values -0.5 0.5  \
    --steps 50 \
    --frames 49 \
    --output_dir /root/workspace/sc-diffusion-pipe/output/20251030_08-16-52_epoch1000_graft_focus/0002 \
    --prompt "A person in the foreground under neon lights, with a city full of colorful signs in the background, the camera focus on the foreground." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --graft \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_supp_neon.out 2>&1 


python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -0.5 0.0 0.5 \
    --steps 50 \
    --frames 49 \
    --output_dir /root/workspace/sc-diffusion-pipe/output/20251030_08-16-52_epoch1000_graft/0029 \
    --prompt "A boat sailing from the horizon, growing in size as it nears the coastline." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_0029.out 2>&1 

python test_fps_multiple_experiments_align_old.py  \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251030_08-16-52/wan_SC_TARGET_14B_3SHAPES_30SCENE_1030_3.toml  \
    --checkpoint ../checkpoints/20251030_08-16-52/epoch1000\
    --fps_values -0.5 0.5 \
    --steps 50 \
    --frames 49 \
    --output_dir /root/workspace/sc-diffusion-pipe/output/20251030_08-16-52_epoch1000_graft_focus/0016 \
    --prompt "A line of sunflowers in a field, stretching from close-up to the horizon, the camera focus on the foreground." \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_20251030_08-16-52_epoch1000_debug_graft_0016.out 2>&1 



