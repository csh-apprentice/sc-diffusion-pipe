#!/usr/bin/env bash
# Run FPS Condition Similarity Analysis
# This analyzes how the FPS adapter's dominant singular vector changes
# across different FPS conditioning values.
#
# Usage:
#   bash run_fps_condition_similarity.sh [LOG_FILE]
#
# Arguments:
#   LOG_FILE  Optional path to save log output (default: no log file, stdout only)
#
# Examples:
#   bash run_fps_condition_similarity.sh                              # Output to stdout only
#   bash run_fps_condition_similarity.sh condition_similarity.log     # Save to condition_similarity.log

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-pipe

# Configuration
# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/wan_SC_TARGET_14B_SHAPES_RANDOM.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251015_14-02-14/epoch1000"
# OUTPUT_DIR="output/fps_condition_similarity/20251015_14-02-14"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_22-29-25/wan_SC_TARGET_14B_BOKEH_150.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251024_22-29-25/epoch1000"
OUTPUT_DIR="output/fps_condition_similarity/20251024_22-29-25/epoch1000"

# Analysis parameters
BLOCKS="27 33 39"
CONDITIONS="-1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0"
NUM_TEST_VECTORS=128
DEVICE="cuda"

LOG_FILE="${1:-}"  # Optional first argument for log file path

echo "========================================================================"
echo "FPS CONDITION SIMILARITY ANALYSIS"
echo "========================================================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Blocks: $BLOCKS"
echo "Conditions: $CONDITIONS"
echo "Num test vectors: $NUM_TEST_VECTORS"
echo "========================================================================"
echo ""

# Build the command
CMD="python inference/compute_fps_condition_similarity.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --blocks $BLOCKS \
    --conditions $CONDITIONS \
    --num_test_vectors $NUM_TEST_VECTORS \
    --device $DEVICE"

# Execute with or without logging to file
if [ -n "$LOG_FILE" ]; then
    # Create log file directory if it doesn't exist
    LOG_DIR=$(dirname "$LOG_FILE")
    if [ ! -d "$LOG_DIR" ]; then
        echo "Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
    echo "Running FPS condition similarity analysis and saving log to: $LOG_FILE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
else
    echo "Running FPS condition similarity analysis (output to stdout only)"
    eval "$CMD"
fi

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
if [ -n "$LOG_FILE" ]; then
    echo "Log saved to: $LOG_FILE"
fi
echo ""
echo "Check the output directory for:"
echo "  - fps_condition_similarity_blockXX.png: Heatmaps for each block"
echo "  - fps_singular_values_across_conditions.png: Singular value trends"
echo "  - similarity_matrix_blockXX.npy: Raw similarity matrices"
echo "  - analysis_summary.txt: Summary statistics"
echo ""
