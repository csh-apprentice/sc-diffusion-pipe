We will restart the align task.

We used to work on the /root/workspace/sc-diffusion-pipe/plans/ALIGN_INFERENCE_NEW.md task and the code is right, but the inferencing result is not good, so I am thinking about new align magnitude, I will only change how the ratio is defined and keep any other settings the same, you can read thorugh the whole task description to notice the difference.

In fact, the different is how we define the rboth_ijk and rfps_ijk.

To do the task, we copy the code /root/workspace/sc-diffusion-pipe/inference/test_fps_align_new.py naming the mew code as test_fps_align_textmagnitude.py and change the ratio calculation.


## Subtask 1: Logging down all the ratios when applying both base lora and fps adapter

There are more paramters that we need to log down since we also have multiple denoising steps, define

rboth_ijk=||y_text_ijk|| when applying both base lora and fps adapter, where i refers to the denoising steps index, and j refers to the the block index, k refers to the fps index we are using now (we may have multiple index when we are inferencing with multiplt condition fps values).

To make comparisons, we can run that inferencing and save the output as what we did when loading and using both base lora and adapter.

## Subatsk2: Aligning the ratio in new inference

After inferencing that video, we do a strict magnitude match when the forwar pass to DIT blocks in each denoising step and each block and each fps condition:

we first calculate the
rfps_ijk=||y_text_ijk||, note that although it shares the same formulation as subtask 1 it is, the  y_text is different since we don't apply the base lora and only use fps adapter in this forward pass, hence the query (q) is different.

then update the y_fps to 
y_fps_ijk=y_fps_ijk*rboth_ijk / rfps_ijk, in this case, in each denoising step and each block, we either amplify or weaken the ompact of the y_fps





