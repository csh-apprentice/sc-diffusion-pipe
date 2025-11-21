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

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20250930_05-53-32"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251011_07-53-15/wan_SC_TARGET_14B_2DSHAPE_TEMP_all.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251011_07-53-15/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251011_07-53-15/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251007_00-35-06/wan_SC_TARGET_14B_BALL_KITCHEN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251007_00-35-06/epoch975"
# OUTPUT_DIR="output/similarity_matrix/20251007_00-35-06/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251008_20-21-45/"


# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251014_06-32-03/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251014_06-32-03"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/wan_SC_TARGET_14B_SHAPES_RANDOM.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251015_14-02-14/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251021_08-22-01/wan_SC_TARGET_14B_SHAPE_SOLID_MINI.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251021_08-22-01/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251021_08-22-01/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251021_19-21-34/wan_SC_TARGET_14B_2DSHAPE_TEMP.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251021_19-21-34/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251021_19-21-34/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251021_21-55-42/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_RESUME.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251021_21-55-42/epoch500"
# OUTPUT_DIR="output/similarity_matrix/20251021_21-55-42/epoch500"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251023_00-19-38/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-18-19/wan_SC_TARGET_14B_3SHAPES_30SCENE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-18-19/epoch475"
# OUTPUT_DIR="output/similarity_matrix/20251023_07-18-19/epoch475"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/wan_SC_TARGET_14B_3SHAPES_BASE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251023_07-43-45/"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_18-52-42/wan_SC_TARGET_14B_TEMP_150.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_18-52-42/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251024_18-52-42/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_18-52-42/wan_SC_TARGET_14B_TEMP_150.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_18-52-42/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251024_18-52-42/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_22-29-25/wan_SC_TARGET_14B_BOKEH_150.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_22-29-25/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251024_22-29-25/epoch1000"


# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251026_07-26-24/wan_SC_TARGET_14B_TEMP_BASE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251026_07-26-24/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251026_07-26-24/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251027_03-23-30/wan_SC_TARGET_14B_BOKEH_150.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251027_03-23-30/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251027_03-23-30/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/epoch800"
# OUTPUT_DIR="output/similarity_matrix/20251028_21-19-57/epoch800"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_07-33-48/wan_SC_TARGET_14B_FPS_HORSE_ABLATION.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_07-33-48/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251115_07-33-48/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_07-34-10/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_ABLATION.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_07-34-10/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251115_07-34-10/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/wan_SC_TARGET_14B_MF_TEMP_ABLATION.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_08-01-31/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251115_08-01-31/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/wan_SC_TARGET_14B_SHAPE_TEMP_ABLATION.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_19-13-59/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251115_19-13-59/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/wan_SC_TARGET_14B_BOKEH_SHAPE_ABLATION.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-27-44/epoch1000"
# OUTPUT_DIR="output/similarity_matrix/20251115_22-27-44/epoch1000"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-28-03/wan_SC_TARGET_14B_BOKEH_HUMAN_ABLATION.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251115_22-28-03/epoch1000"
OUTPUT_DIR="output/similarity_matrix/20251115_22-28-03/epoch1000"




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
