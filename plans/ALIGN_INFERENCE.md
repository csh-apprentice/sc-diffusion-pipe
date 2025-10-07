I still suspect it's the unequal signal strength that weaken our inference result when we try to use fps_only mode in the /root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align.py. Let's manually align the ||y_fps||/||y_text|| when an additional arg 

align = "true" is given (default is false)

when the align = "true" and we are in fps_only mode.

We will first go through the forward pass using the both base lora and fps adapter, so we can calculate the magnitude of 
mboth_i=||y_text|| in each activate block (i is the index).


Then in the inferencing, we go through another forward pass using the clean wan backbone (without base lora), so we can calculate the new magnitude of the text by mfps_i=||y_text||,
then instead adding the y_fps and y_text by 
y=y_text+g(\alpha)*y_fps, we do 

y=y_text+g(\alpha)*y_fps*mfps_i/mboth_i and complete the forward pass to get the inferecing result.