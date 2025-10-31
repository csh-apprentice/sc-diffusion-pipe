#!/bin/bash
#
# Run complete conditional subspace analysis pipeline
#
# Usage:
#   bash bash/run_subspace_analysis.sh <config_toml> <checkpoint_path> <fps_values...>
#
# Example:
#   bash bash/run_subspace_analysis.sh \
#       fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
#       outputs/checkpoint/step_10000 \
#       12 24 60 120 240

set -e  # Exit on error

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config_toml> <checkpoint_path> <fps_values...>"
    echo ""
    echo "Example:"
    echo "  $0 fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml outputs/checkpoint/step_10000 12 24 60 120 240"
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
shift 2
FPS_VALUES="$@"

# Generate output directory name from checkpoint
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
OUTPUT_DIR="output/subspace_analysis_${CHECKPOINT_NAME}"

echo "=============================================================================="
echo "CONDITIONAL SUBSPACE ANALYSIS PIPELINE"
echo "=============================================================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "FPS values: $FPS_VALUES"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================================================="
echo ""

# Step 1: Extract conditional subspaces
echo "Step 1/2: Extracting conditional subspaces..."
echo "------------------------------------------------------------------------------"
python inference/analyze_conditional_subspace.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --fps_values $FPS_VALUES \
    --output_dir "$OUTPUT_DIR" \
    --port 29502

if [ $? -ne 0 ]; then
    echo "❌ Subspace extraction failed!"
    exit 1
fi

echo ""
echo "✅ Step 1 complete: Subspaces extracted"
echo ""

# Step 2: Run geometric analyses
echo "Step 2/2: Running geometric analyses..."
echo "------------------------------------------------------------------------------"
python inference/analyze_conditional_geometry.py \
    --subspace_file "${OUTPUT_DIR}/subspace_data.pkl" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --port 29503

if [ $? -ne 0 ]; then
    echo "❌ Geometric analysis failed!"
    exit 1
fi

echo ""
echo "✅ Step 2 complete: Geometric analyses done"
echo ""

# Summary
echo "=============================================================================="
echo "✅ ANALYSIS COMPLETE!"
echo "=============================================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - subspace_data.pkl            : Raw subspace data"
echo "  - metadata.json                : Analysis metadata"
echo "  - dimensionality_results.csv   : Dimensionality statistics"
echo "  - dimensionality_analysis.png  : Dimensionality plots"
echo "  - orthogonality_results.npz    : Orthogonality matrices"
echo "  - orthogonality_heatmaps.png   : Orthogonality heatmaps"
echo "  - orthogonality_evolution.png  : Orthogonality across blocks"
echo "  - backbone_alignment_results.csv : Backbone alignment statistics"
echo "  - backbone_alignment.png       : Backbone alignment plots"
echo "  - summary_report.txt           : Text summary report"
echo ""
echo "View the summary report:"
echo "  cat ${OUTPUT_DIR}/summary_report.txt"
echo "=============================================================================="
