#!/usr/bin/env bash
# Compute Backbone Similarity Matrix (Step 1 of 2)
# This script computes and saves the similarity matrix for all blocks.
# Run this ONCE per checkpoint, then use analyze_similarity_matrix.sh
# with different thresholds for rapid experimentation.
#
# Usage:
#   bash compute_similarity_matrix.sh [LOG_FILE]

set -e

# Configuration

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251019_04-51-46/wan_SC_TARGET_14B_FPS_SHAPE_BLUR_DEBUG.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251019_04-51-46/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20251019_04-51-46/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/epoch1000"
# OUTPUT_DIR="output/backbone_drift/20250930_05-53-32"


# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251014_06-32-03"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251027_23-58-53/wan_SC_TARGET_14B_SHUTTER_150_8f.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251027_23-58-53/epoch1000"
OUTPUT_DIR="output/similarity_matrix/20251027_23-58-53/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-25-27/wan_SC_TARGET_14B_BOKEH_150_DIVERSE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-25-27/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251028_21-25-27/epoch1000"
TOP_K=64
DEVICE="cuda"

# Optional: specify blocks explicitly (leave empty to auto-detect all blocks with LoRA)
# BLOCKS="1 15 27 33 39"
BLOCKS=""

echo "========================================================================"
echo "COMPUTE BACKBONE SIMILARITY MATRIX"
echo "========================================================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top-k vectors: $TOP_K"
echo "========================================================================"

# Build the command
CMD="python inference/compute_backbone_similarity_matrix.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --top_k $TOP_K \
    --output_dir \"$OUTPUT_DIR\" \
    --device $DEVICE"

# Add blocks if specified
if [ -n "$BLOCKS" ]; then
    CMD="$CMD --blocks $BLOCKS"
    echo "Blocks: $BLOCKS (specified)"
else
    echo "Blocks: auto-detect all blocks with LoRA"
fi

echo ""
echo "This may take several minutes depending on the number of blocks..."
echo ""

# Optional log file argument
LOG_FILE="${1:-}"

# Execute with or without logging to file
if [ -n "$LOG_FILE" ]; then
    # Create log file directory if it doesn't exist
    LOG_DIR=$(dirname "$LOG_FILE")
    if [ ! -d "$LOG_DIR" ]; then
        echo "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
    echo "Running computation and saving log to: $LOG_FILE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
else
    echo "Running computation (output to stdout only)"
    eval "$CMD"
fi

echo ""
echo "========================================================================"
echo "COMPUTATION COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run analyze_similarity_matrix.sh with different thresholds"
echo "  2. Example: bash bash/analyze_similarity_matrix.sh <matrix_dir> 0.5"
echo ""
