I want to restart my alignment task. you can reference to the 
/root/workspace/sc-diffusion-pipe/inference/test_fps_align_textmagnitude.py where we did a similar task 
for plan /root/workspace/sc-diffusion-pipe/plans/ALIGN_INFERENCE_TEXT.md

## Subtask 1 Plus: Logging down ratios when applying base lora & clean backbone

Instead of running multiple denoising steps and use that fps adapter, i will go easy to avoid any confiusion, so given an inference condition c, we will go through one simple denoising step.

In this step, you should figure out the ratio for our fps adapter injected block i:

r_text_i=||y_text_base_lora_fps_adapter||/||y_text_clean_fps_adapter||
r_cond_i=||y_fps_base_lora_fps_adapter||/||y_fps_clean_fps_adapter||

for each fps adapter injected block


## Subatsk2: Aligning the ratio in new inference

Then we just simply run the inference using only fps adapter on a clean backbone by changing the gate value

new_gate_i=old_gate_i*r_text_i*r_cond_i

for each injected block i









