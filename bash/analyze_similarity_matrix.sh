#!/usr/bin/env bash
# Analyze Precomputed Backbone Similarity Matrix (Step 2 of 2)
# This script loads a precomputed similarity matrix and applies a threshold
# to detect intruder dimensions. This is FAST and can be run multiple times
# with different thresholds.
#
# Usage:
#   bash analyze_similarity_matrix.sh <matrix_dir> <threshold> [OPTIONS]
#
# Example:
#   bash analyze_similarity_matrix.sh output/similarity_matrix/20251015_14-02-14 0.5
#   bash analyze_similarity_matrix.sh output/similarity_matrix/20251015_14-02-14 0.3 --verbose
#   bash analyze_similarity_matrix.sh output/similarity_matrix/20251015_14-02-14 0.7 --plot_individual

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash analyze_similarity_matrix.sh <matrix_dir> <threshold> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  matrix_dir:  Directory containing precomputed similarity matrix"
    echo "  threshold:   Cosine similarity threshold (e.g., 0.5)"
    echo ""
    echo "Options:"
    echo "  --verbose:         Print detailed per-block results"
    echo "  --plot_individual: Generate individual plots for each block/weight"
    echo "  --no_heatmap:      Skip heatmap generation"
    echo "  --output_dir DIR:  Custom output directory"
    echo ""
    echo "Examples:"
    echo "  bash analyze_similarity_matrix.sh output/similarity_matrix/20251015_14-02-14 0.5"
    echo "  bash analyze_similarity_matrix.sh output/similarity_matrix/20251015_14-02-14 0.3 --verbose --plot_individual"
    exit 1
fi

MATRIX_DIR="$1"
THRESHOLD="$2"
shift 2  # Remove first two arguments

# Remaining arguments are passed to Python script
EXTRA_ARGS="$@"

echo "========================================================================"
echo "ANALYZE BACKBONE SIMILARITY MATRIX"
echo "========================================================================"
echo "Matrix directory: $MATRIX_DIR"
echo "Threshold: $THRESHOLD"
echo "Extra arguments: $EXTRA_ARGS"
echo "========================================================================"
echo ""

# Check if matrix directory exists
if [ ! -d "$MATRIX_DIR" ]; then
    echo "❌ ERROR: Matrix directory not found: $MATRIX_DIR"
    echo ""
    echo "Did you run compute_similarity_matrix.sh first?"
    exit 1
fi

# Check if similarity matrix file exists
if [ ! -f "$MATRIX_DIR/similarity_matrix.npy" ]; then
    echo "❌ ERROR: similarity_matrix.npy not found in: $MATRIX_DIR"
    echo ""
    echo "The matrix directory should contain:"
    echo "  - similarity_matrix.npy"
    echo "  - block_indices.npy"
    echo "  - metadata.npz"
    echo "  - config.txt"
    exit 1
fi

# Build and run command
CMD="python inference/analyze_similarity_matrix.py \
    --matrix_dir \"$MATRIX_DIR\" \
    --threshold $THRESHOLD \
    $EXTRA_ARGS"

echo "Running analysis..."
echo ""
eval "$CMD"

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $MATRIX_DIR/analysis_threshold_${THRESHOLD//./_}/"
echo ""
echo "To try a different threshold, simply run:"
echo "  bash bash/analyze_similarity_matrix.sh $MATRIX_DIR <new_threshold>"
echo ""
