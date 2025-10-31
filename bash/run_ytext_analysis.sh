#!/usr/bin/env bash
# Analyze Y_text Change After LoRA Fine-tuning
#
# This script analyzes how y_text output changes after LoRA fine-tuning:
# 1. Similarity Matrix: Compares top-k y_text_lora vectors to y_text_clean
# 2. Singular Spectrum: Bird's eye view of singular values (clean vs LoRA)
#
# Usage:
#   bash run_ytext_analysis.sh [LOG_FILE]

set -e

# Configuration

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/epoch1000"
OUTPUT_DIR="output/ytext_analysis/20250930_05-53-32/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/wan_SC_TARGET_14B_3SHAPES_BASE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/epoch1000"
# OUTPUT_DIR="output/ytext_analysis/20251023_07-43-45/epoch1000"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/epoch800"
# OUTPUT_DIR="output/ytext_analysis/20251028_21-19-57/epoch800"

TOP_K=64
NUM_TEST_VECTORS=128
DEVICE="cuda"

# Optional: specify blocks explicitly (leave empty to auto-detect all blocks with LoRA)
# BLOCKS="1 15 27 33 39"
BLOCKS=""

echo "========================================================================"
echo "ANALYZE Y_TEXT CHANGE AFTER LORA FINE-TUNING"
echo "========================================================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Top-k vectors: $TOP_K"
echo "Test vectors: $NUM_TEST_VECTORS"
echo "========================================================================"

# Build the command
CMD="python inference/analyze_ytext_change.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --top_k $TOP_K \
    --num_test_vectors $NUM_TEST_VECTORS \
    --device $DEVICE"

# Add blocks if specified
if [ -n "$BLOCKS" ]; then
    CMD="$CMD --blocks $BLOCKS"
    echo "Blocks: $BLOCKS (specified)"
else
    echo "Blocks: auto-detect all blocks with LoRA"
fi

echo ""
echo "This will generate:"
echo "  1. Similarity heatmap (y_text_lora vs y_text_clean)"
echo "  2. Singular spectrum plots (clean vs LoRA for each block)"
echo ""
echo "This may take several minutes depending on the number of blocks..."
echo ""

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

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
    echo "Running analysis and saving log to: $LOG_FILE"
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
else
    echo "Running analysis (output to stdout only)"
    eval "$CMD"
fi

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - ytext_similarity_heatmap.png: Heatmap showing max similarities"
echo "  - ytext_singular_spectrum_all.png: Combined spectrum plot"
echo "  - ytext_singular_spectrum_blockX.png: Individual spectrum plots"
echo "  - ytext_similarity_matrix.csv: Numerical similarity data"
echo "  - ytext_analysis_data.npz: Numpy arrays for further analysis"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
