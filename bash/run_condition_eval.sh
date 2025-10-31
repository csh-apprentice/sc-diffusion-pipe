#!/usr/bin/env bash
# Data-Free Evaluation Metrics for Joint LoRA & Adapter Finetuning
#
# This script computes quantitative metrics to find the "sweet point" checkpoint:
# 1. CPD (Content Primitive Drift) Score: Measures backbone health (should stay ~1.0)
# 2. Conditional Disparity Score: Measures adapter learning (should rise to ~1.0)
# 3. Supporting Metrics: Intruder Count, Effective Rank, Magnitude Ratio
#
# The "Sweet Point" is where Disparity Score plateaus AND CPD Score is still high.
#
# Usage:
#   bash run_condition_eval.sh [LOG_FILE]

set -e

# Configuration

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/wan_SC_TARGET_14B_3SHAPES_BASE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/epoch1000"
# OUTPUT_DIR="output/condition_eval/20251023_07-43-45/epoch1000"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/epoch800"
OUTPUT_DIR="output/condition_eval/20251028_21-19-57/epoch800"

NUM_COMPONENTS=64
LOW_COND=0.025    # Sharp/low condition
HIGH_COND=1.4     # Bokeh/high condition
DEVICE="cuda"

# Optional: specify blocks explicitly (leave empty to auto-detect with FPS adapter)
# BLOCKS="27 33 39"
BLOCKS=""

echo "========================================================================"
echo "DATA-FREE EVALUATION METRICS"
echo "========================================================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Condition range: [$LOW_COND, $HIGH_COND]"
echo "Principal components: $NUM_COMPONENTS"
echo "========================================================================"

# Build the command
CMD="python inference/compute_condition_eval_metrics.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_components $NUM_COMPONENTS \
    --low_cond $LOW_COND \
    --high_cond $HIGH_COND \
    --device $DEVICE"

# Add blocks if specified
if [ -n "$BLOCKS" ]; then
    CMD="$CMD --blocks $BLOCKS"
    echo "Blocks: $BLOCKS (specified)"
else
    echo "Blocks: auto-detect all blocks with FPS adapter"
fi

echo ""
echo "This will compute:"
echo "  1. CPD Score (Content Primitive Drift - Backbone Health)"
echo "  2. Disparity Score (Conditional Separation - Adapter Learning)"
echo "  3. Supporting metrics (Intruder Count, Effective Rank, Magnitude Ratio)"
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
echo "  - evaluation_metrics.png: Combined plot with all metrics"
echo "  - evaluation_metrics.csv: Numerical results"
echo "  - evaluation_metrics.npz: Numpy arrays for further analysis"
echo ""
echo "Interpretation:"
echo "  - CPD Score ~1.0 = Backbone healthy (no catastrophic forgetting)"
echo "  - Disparity Score ~1.0 = Adapter trained (distinct outputs per condition)"
echo "  - Sweet Point = High CPD + High Disparity"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
