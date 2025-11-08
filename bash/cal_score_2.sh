#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch100 \
    --output_file scores/20251031_08-25-18/epoch100/scores.json

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch200 \
    --output_file scores/20251031_08-25-18/epoch200/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch300 \
    --output_file scores/20251031_08-25-18/epoch300/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch400 \
    --output_file scores/20251031_08-25-18/epoch400/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch500 \
    --output_file scores/20251031_08-25-18/epoch500/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch600 \
    --output_file scores/20251031_08-25-18/epoch600/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch700 \
    --output_file scores/20251031_08-25-18/epoch700/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch800 \
    --output_file scores/20251031_08-25-18/epoch800/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch900 \
    --output_file scores/20251031_08-25-18/epoch900/scores.json    

python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251031_08-25-18/epoch1000 \
    --output_file scores/20251031_08-25-18/epoch1000/scores.json    




