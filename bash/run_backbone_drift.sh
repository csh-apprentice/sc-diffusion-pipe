#!/usr/bin/env bash
# Run Backbone Drift Check (Intruder Dimension Analysis)
# This analyzes whether base LoRA introduced new high-ranking singular vectors
# that could indicate catastrophic forgetting of pretrained knowledge.
#
# Usage:
#   bash run_backbone_drift.sh [LOG_FILE]
#
# Arguments:
#   LOG_FILE  Optional path to save log output (default: no log file, stdout only)
#
# Examples:
#   bash run_backbone_drift.sh                           # Output to stdout only
#   bash run_backbone_drift.sh backbone_drift.log        # Save to backbone_drift.log
#   bash run_backbone_drift.sh logs/my_analysis.log      # Save to logs/my_analysis.log

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe


# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20250930_05-53-32"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250918_22-35-16/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_VID_NEW.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250918_22-35-16//epoch1000"
# OUTPUT_DIR="output/backbone_drift/20250918_22-35-16/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251007_00-35-06/wan_SC_TARGET_14B_BALL_KITCHEN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251007_00-35-06/epoch975"
# OUTPUT_DIR="output/backbone_drift/20251007_00-35-06/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20251008_20-21-45/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251011_07-53-15/wan_SC_TARGET_14B_2DSHAPE_TEMP_all.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251011_07-53-15/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20251011_07-53-15/"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000"
OUTPUT_DIR="output/backbone_drift/20251014_06-32-03"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/wan_SC_TARGET_14B_SHAPES_RANDOM.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20251015_14-02-14/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251019_04-51-46/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_DEBUG.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251019_04-51-46/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20251019_04-51-46/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251020_07-56-38/wan_SC_TARGET_14B_3SHAPES_SIMPLE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251020_07-56-38/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20251020_07-56-38/"

BLOCKS="2 5 15 20 27 33 39"
TOP_K="64"
THRESHOLD="0.5"
LOG_FILE="${1:-}"  # Optional first argument for log file path

# Build the command
CMD="python inference/backbone_drift_check.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --blocks $BLOCKS \
    --top_k $TOP_K \
    --threshold $THRESHOLD \
    --output_dir \"$OUTPUT_DIR\" \
    --device cuda"

# Execute with or without logging to file
if [ -n "$LOG_FILE" ]; then
    # Create log file directory if it doesn't exist
    LOG_DIR=$(dirname "$LOG_FILE")
    if [ ! -d "$LOG_DIR" ]; then
        echo "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
    echo "Running backbone drift check and saving log to: $LOG_FILE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
else
    echo "Running backbone drift check (output to stdout only)"
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
