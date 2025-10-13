I made a fatal mistake when writing the ALIGN_INFERENCE.md, which I attach in below 

"I still suspect it's the unequal signal strength that weaken our inference result when we try to use fps_only mode in the /root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align.py. Let's manually align the ||y_fps||/||y_text|| when an additional arg 

align = "true" is given (default is false)

when the align = "true" and we are in fps_only mode.

We will first go through the forward pass using the both base lora and fps adapter, so we can calculate the magnitude of 
mboth_i=||y_text|| in each activate block (i is the index).


Then in the inferencing, we go through another forward pass using the clean wan backbone (without base lora), so we can calculate the new magnitude of the text by mfps_i=||y_text||,
then instead adding the y_fps and y_text by 
y=y_text+g(\alpha)*y_fps, we do 

y=y_text+g(\alpha)*y_fps*mfps_i/mboth_i and complete the forward pass to get the inferecing result."

I need to fix that:

## Subtask 1: Logging down all the ratios when applying both base lora and fps adapter

There are more paramters that we need to log down since we also have multiple denoising steps, define

rboth_ijk=||y_fps_ijk||/||y_text_ij|| when applying both base lora and fps adapter, where i refers to the denoising steps index, and j refers to the the block index, k refers to the fps index we are using now (we may have multiple index when we are inferencing with multiplt condition fps values).

To make comparisons, we can run that inferencing and save the output as what we did when loading and using both base lora and adapter.

## Subatsk2: Aligning the ratio in new inference

After inferencing that video, we do a strict magnitude match when the forwar pass to DIT blocks in each denoising step and each block and each fps condition:

we first calculate the
rfps_ijk=||y_fps_ijk||/||y_text_ij||, note that although it shares the same formulation as subtask 1 it is, the y_fps and y_text is different since we don't apply the base lora and only use fps adapter in this forward pass, hence the query (q) is different.

then update the y_fps to 
y_fps_ijk=y_fps_ijk*rboth_ijk / rfps_ijk, in this case, in each denoising step and each block, the ratio of the y_fps and y_text match.

## Subtask 3: Debugging Info

I still like to know how the ratio compares over time, so please print out the rboth_ijk / rfps_ijk, if we have too may ratio numbers to print out, let's only focus on the mean value, for each k
print out the mean of ratio among all the blocks in each denoising timestep.



