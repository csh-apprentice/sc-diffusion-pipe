Great, now we are trying to create a new inference script but borrowing most of the inference code from /root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align_old.py.

The only changes is how we read the prompts and how we save the output,
Instead of reading the prompts from a prompt_folder where each prompt is listed as a txt file, i want you to support two reading mode:

1. txt file base: This is a big koint prompt file where each line represent a independeent prompt, an example is in /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/all_category.txt

So the output should be written as the number_of_line_cond.txt
If we use clean flag or base flag, then just neglect cond,
sabe it directly as number_of_line.txt, eg: 00001.mp4

2. Dir root base: Given a folder path name, eg: /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/prompts_per_category, where we have multiple txt file in this folder, each txt file is still in joint format. 

Give a save root A, for 2. you should create subfolders naming as the prefix of these txt files under root A, and save the output video naming as the number of line (optional condition) in the corresponing subfolder.