#!/usr/bin/env bash
# Compute Backbone Energy Change After Fine-tuning
# This script computes the normalized energy (sum of squared singular values)
# for each weight matrix (q, k, v, o) in the backbone after LoRA fine-tuning.
#
# Outputs 4 plots showing normalized energy vs block index, where:
#   - Energy = 1.0 means no change (pretrained baseline)
#   - Energy > 1.0 means increased energy after training
#   - Energy < 1.0 means decreased energy after training
#
# Usage:
#   bash compute_energy.sh [LOG_FILE]

set -e

# Configuration

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/wan_SC_TARGET_14B_HUMAN.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20250930_05-53-32/epoch1000"
# OUTPUT_DIR="output/energy/20250930_05-53-32"

# CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/wan_SC_TARGET_14B_3SHAPES_BASE.toml"
# CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251023_07-43-45/epoch1000"
# OUTPUT_DIR="output/energy/20251023_07-43-45/epoch1000"

CONFIG_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/wan_SC_TARGET_14B_SHAPE_SOLID_MINI_PART12.toml"
CHECKPOINT_PATH="/root/workspace/sc-diffusion-pipe/checkpoints/20251028_21-19-57/epoch1000"
OUTPUT_DIR="output/energy/20251028_21-19-57/epoch1000"

DEVICE="cuda"

# Optional: specify blocks explicitly (leave empty to auto-detect all blocks with LoRA)
# BLOCKS="1 15 27 33 39"
BLOCKS=""

echo "========================================================================"
echo "COMPUTE BACKBONE ENERGY CHANGE"
echo "========================================================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"

# Build the command
CMD="python inference/compute_backbone_energy.py \
    --config \"$CONFIG_PATH\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
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
echo "Computing energies for all weight matrices (q, k, v, o)..."
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
echo "Generated files:"
echo "  - energy_comparison.png: Combined plot with all 4 weight matrices"
echo "  - energy_q.png, energy_k.png, energy_v.png, energy_o.png: Individual plots"
echo "  - energy_summary.csv: Numerical results in CSV format"
echo "  - energy_data.npz: Numpy arrays for further analysis"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
