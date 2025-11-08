#!/usr/bin/env bash
set -euo pipefail  # stop on first error; drop -e if you want to continue


# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch100 \
#     --output_file scores/20251008_20-21-45/epoch100/scores.json

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch200 \
#     --output_file scores/20251008_20-21-45/epoch200/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch300 \
#     --output_file scores/20251008_20-21-45/epoch300/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch400 \
#     --output_file scores/20251008_20-21-45/epoch400/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch500 \
#     --output_file scores/20251008_20-21-45/epoch500/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch600 \
#     --output_file scores/20251008_20-21-45/epoch600/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch700 \
#     --output_file scores/20251008_20-21-45/epoch700/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch800 \
#     --output_file scores/20251008_20-21-45/epoch800/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch900 \
#     --output_file scores/20251008_20-21-45/epoch900/scores.json    

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251008_20-21-45/epoch1000 \
#     --output_file scores/20251008_20-21-45/epoch1000/scores.json    


# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251031_08-25-18/epoch800 \
#     --output_file scores/20251031_08-25-18/epoch800/scores.json

# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251031_08-25-18/epoch1000 \
#     --output_file scores/20251031_08-25-18/epoch1000/scores.json


# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/clean_category_99 \
#     --output_file scores/clean_category_99/scores.json

# for epoch in 100 200 300 400 500 600 700 800 900 1000; do
# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251027_22-55-30/epoch${epoch} \
#     --output_file scores/20251027_22-55-30/epoch${epoch}/scores.json
# done

# for epoch in 100 200 300 400 500 600 700 800 900 1000; do
# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251023_00-19-38/epoch${epoch} \
#     --output_file scores/20251023_00-19-38/epoch${epoch}/scores.json
# done


# for epoch in 100 200 300 400 500 600 700 800 900 1000; do
# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251014_06-32-03/epoch${epoch} \
#     --output_file scores/20251014_06-32-03/epoch${epoch}/scores.json
# done

# for epoch in 100 200 300 400 500 600 700 800 900 1000; do
# python metric/video_score_calculator_extended.py \
#     --orig_parent_dir output/onestep/clean_category_42 \
#     --adapt_parent_dir output/onestep/20251030_08-16-52/epoch${epoch} \
#     --output_file scores/20251030_08-16-52/epoch${epoch}/scores.json
# done


for epoch in 100 200 300 400 500 600 700 800 900 1000; do
python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251106_09-42-01/epoch${epoch} \
    --output_file scores/20251106_09-42-01/epoch${epoch}/scores.json
done


for epoch in 100 200 300 400 500 600 700 800 900 1000; do
python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251107_00-49-00/epoch${epoch} \
    --output_file scores/20251107_00-49-00/epoch${epoch}/scores.json
done

for epoch in 100 200 300 400 500 600 700 800 900 1000; do
python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251107_00-51-16/epoch${epoch} \
    --output_file scores/20251107_00-51-16/epoch${epoch}/scores.json
done

for epoch in 100 200 300 400 500 600 700 800 900 1000; do
python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251107_01-19-02/epoch${epoch} \
    --output_file scores/20251107_01-19-02/epoch${epoch}/scores.json
done