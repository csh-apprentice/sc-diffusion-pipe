#!/usr/bin/env bash
# Run y_fps analysis during inference
# This captures and analyzes y_text and y_fps outputs from FPS adapters
#
# Usage:
#   bash run_yfps_analysis.sh [LOG_FILE]
#
# Arguments:
#   LOG_FILE  Optional path to save log output (default: no log file, stdout only)
#
# Examples:
#   bash run_yfps_analysis.sh                           # Output to stdout only
#   bash run_yfps_analysis.sh y_fps_analysis.log       # Save to y_fps_analysis.log
#   bash run_yfps_analysis.sh logs/my_analysis.log     # Save to logs/my_analysis.log

set -e

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000"
OUTPUT_DIR="output/y_fps_analysis"
PROMPT="A ceiling fan spinning under warm yellow light."
LOG_FILE="${1:-}"  # Optional first argument for log file path

# Build the command
CMD="python inference/analyze_yfps_during_inference.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --prompt \"$PROMPT\" \
    --fps_values -1.0 0.0 1.0 \
    --steps 5 \
    --frames 17 \
    --width 640 \
    --height 384 \
    --seed 42 \
    --port 29505"

# Execute with or without logging to file
if [ -n "$LOG_FILE" ]; then
    echo "Running analysis and saving log to: $LOG_FILE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
else
    echo "Running analysis (output to stdout only)"
    eval "$CMD"
fi

echo ""
echo "===================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
if [ -n "$LOG_FILE" ]; then
    echo "Log saved to: $LOG_FILE"
fi
echo "===================="
