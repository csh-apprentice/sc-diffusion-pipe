#!/usr/bin/env bash
# Run Principal Orthogonality Check
# This performs data-free analysis to verify functional orthogonality between
# the FPS adapter and backbone model using SVD + CKA metric.
#
# Usage:
#   bash run_principal_orthogonality.sh [LOG_FILE]
#
# Arguments:
#   LOG_FILE  Optional path to save log output (default: no log file, stdout only)
#
# Examples:
#   bash run_principal_orthogonality.sh                           # Output to stdout only
#   bash run_principal_orthogonality.sh orthogonality.log        # Save to orthogonality.log
#   bash run_principal_orthogonality.sh logs/my_analysis.log     # Save to logs/my_analysis.log

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250911_21-49-13/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250911_21-49-13/epoch1000"
# OUTPUT_DIR="output/principal_orthogonality/20250911_21-49-13"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/epoch1000"
# OUTPUT_DIR="output/principal_orthogonality/20250930_05-53-32"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/epoch1000"
# OUTPUT_DIR="output/principal_orthogonality/20251008_20-21-45/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000"
# OUTPUT_DIR="output/principal_orthogonality/20251014_06-32-03"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/wan_SC_TARGET_14B_SHAPES_RANDOM.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/epoch1000"
OUTPUT_DIR="output/principal_orthogonality/20251015_14-02-14/"

BLOCKS="27 33 39"
NUM_COMPONENTS="64"
FPS_CONDITION="1.0"
LOG_FILE="${1:-}"  # Optional first argument for log file path

# Build the command
CMD="python inference/principal_orthogonality_check.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --blocks $BLOCKS \
    --num_components $NUM_COMPONENTS \
    --fps_condition $FPS_CONDITION"

# Execute with or without logging to file
if [ -n "$LOG_FILE" ]; then
    # Create log file directory if it doesn't exist
    LOG_DIR=$(dirname "$LOG_FILE")
    if [ ! -d "$LOG_DIR" ]; then
        echo "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
    echo "Running principal orthogonality check and saving log to: $LOG_FILE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
else
    echo "Running principal orthogonality check (output to stdout only)"
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
