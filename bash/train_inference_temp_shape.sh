#!/bin/bash

# --- CONFIGURATION ---
# The PID of the training process you are currently waiting for.
PID_TO_WATCH=1459812 
# How many seconds to wait between checks.
CHECK_INTERVAL=600 

echo "Waiting for the previous training process (PID: $PID_TO_WATCH) to complete."
echo "Checking every $CHECK_INTERVAL seconds..."

# This loop will continue as long as the process with that PID exists.
# 'kill -0' is a special command that doesn't actually kill the process.
# It just checks if the process exists and if you have permission to signal it.
# It succeeds (returns 0) if the process is running, and fails (returns non-zero) if it's not.
while kill -0 $PID_TO_WATCH >/dev/null 2>&1; do
    echo "[$(date)] Process $PID_TO_WATCH is still running. Waiting..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "--------------------------------------------------------"
echo "Success! The previous training process (PID: $PID_TO_WATCH) has completed."
echo "Starting the new training now."
echo "--------------------------------------------------------"
echo ""


cd inference


python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251107_00-49-00/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml  \
    --checkpoint ../checkpoints/20251107_00-49-00/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251107_00-49-00 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/temp_20251107_00-49-00_49frames.out 2>&1 

