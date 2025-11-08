I would also like to try the grafting by only inferenceing with the fps adapter and the base lora in the same position as fps_adapter:

for example: if the fps adapter is injected in only deepest thrid layers, then we only need to load the base lor located in deepest thrid layers and fps adapter.




[updated]

This method works well now, let's also add the /root/workspace/sc-diffusion-pipe/plans/ALIGN_INFERNCE_TEXT_ONESTEP.md idea, please copy the /root/workspace/sc-diffusion-pipe/inference/test_fps_graft.py and work from that clean code

you should go trhough the subtask 1

||y_text_base_lora_only||/||y_text_base_lora_partial||

so you can calculate the ratios Then we just simply run the inference using fps adapter and the partial base lora by changing the gate value

new_gate_i=old_gate_i*
||y_text_base_lora_only||/||y_text_base_lora_partial||


If you need reference of the code, please refers to the 

/root/workspace/sc-diffusion-pipe/inference/test_fps_align_onestep.py