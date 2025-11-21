#!/usr/bin/env bash
# Compare Backbone Similarity Matrices (Realistic vs Synthetic)
# This script compares two precomputed similarity matrices and visualizes
# intruder dimensions for V matrix only.
#
# Usage:
#   bash compare_similarity_matrices.sh <realistic_dir> <synthetic_dir> <threshold> [OPTIONS]
#
# Example:
#   bash compare_similarity_matrices.sh \
#     output/similarity_matrix/realistic_20251015_14-02-14 \
#     output/similarity_matrix/synthetic_20251015_15-30-22 \
#     0.5
#
#   bash compare_similarity_matrices.sh \
#     output/similarity_matrix/realistic_20251015_14-02-14 \
#     output/similarity_matrix/synthetic_20251015_15-30-22 \
#     0.5 \
#     --output_dir output/comparison_results

set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: bash compare_similarity_matrices.sh <realistic_dir> <synthetic_dir> <threshold> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  realistic_dir:  Directory containing similarity matrix from realistic dataset"
    echo "  synthetic_dir:  Directory containing similarity matrix from synthetic dataset"
    echo "  threshold:      Cosine similarity threshold (e.g., 0.5)"
    echo ""
    echo "Options:"
    echo "  --output_dir DIR:  Custom output directory (default: auto-generated)"
    echo "  --verbose:         Print detailed analysis"
    echo ""
    echo "Examples:"
    echo "  bash compare_similarity_matrices.sh \\"
    echo "    output/similarity_matrix/realistic_20251015_14-02-14 \\"
    echo "    output/similarity_matrix/synthetic_20251015_15-30-22 \\"
    echo "    0.5"
    exit 1
fi

REALISTIC_DIR="$1"
SYNTHETIC_DIR="$2"
THRESHOLD="$3"
shift 3  # Remove first three arguments

# Remaining arguments are passed to Python script
EXTRA_ARGS="$@"

echo "========================================================================"
echo "COMPARE BACKBONE SIMILARITY MATRICES"
echo "========================================================================"
echo "Realistic dataset matrix: $REALISTIC_DIR"
echo "Synthetic dataset matrix: $SYNTHETIC_DIR"
echo "Threshold: $THRESHOLD"
echo "Extra arguments: $EXTRA_ARGS"
echo "========================================================================"
echo ""

# Check if realistic directory exists
if [ ! -d "$REALISTIC_DIR" ]; then
    echo "❌ ERROR: Realistic matrix directory not found: $REALISTIC_DIR"
    echo ""
    echo "Please provide a valid directory containing similarity_matrix.npy"
    exit 1
fi

# Check if synthetic directory exists
if [ ! -d "$SYNTHETIC_DIR" ]; then
    echo "❌ ERROR: Synthetic matrix directory not found: $SYNTHETIC_DIR"
    echo ""
    echo "Please provide a valid directory containing similarity_matrix.npy"
    exit 1
fi

# Check if similarity matrix files exist
if [ ! -f "$REALISTIC_DIR/similarity_matrix.npy" ]; then
    echo "❌ ERROR: similarity_matrix.npy not found in: $REALISTIC_DIR"
    exit 1
fi

if [ ! -f "$SYNTHETIC_DIR/similarity_matrix.npy" ]; then
    echo "❌ ERROR: similarity_matrix.npy not found in: $SYNTHETIC_DIR"
    exit 1
fi

# Build and run command
CMD="python inference/compare_similarity_matrices.py \
    --realistic_dir \"$REALISTIC_DIR\" \
    --synthetic_dir \"$SYNTHETIC_DIR\" \
    --threshold $THRESHOLD \
    $EXTRA_ARGS"

echo "Running comparison analysis..."
echo ""
eval "$CMD"

echo ""
echo "========================================================================"
echo "COMPARISON COMPLETE"
echo "========================================================================"
echo ""
