We already have all the videos output for each of the checkpoint/clean backbone, now it's time to calculate the score by comparing them against the clean backbone. An example script of how to do it for a single video is listed in /root/workspace/sc-diffusion-pipe/metric/video_score_calculator.py. What we need to do is to extend it a little bit:

it curretnly accept ORIG_DIR and ADAPT_DIR, where it assume we Assumes videos files are named identically in both folder, that is good strategy, keep it, but let's add another mode, saying ORIG_PARENT_DIR and ADAPT_PARENT_DIR.

In this case, we would expect this two parent folders would have same layout


for example: 
ORIG_PARENT_DIR=/root/workspace/sc-diffusion-pipe/output/onestep/clean_category_42
ADAPT_PARENT_DIR=/root/workspace/sc-diffusion-pipe/output/onestep/20251008_20-21-45/epoch1000


where we have identical subfolders, in each subfolder we have exact videos with the identical naming.

So what i want you to extend is let's calculate score for each subdirectory,
then calculate a total score assume you see all the videos in a big videos in all the subfolders. In this case

we will caulctae scores for 
"animal"
"architecture"
.....

"vehicles"

and a total score "total"

Write teh scores in a dict file given a output file name path