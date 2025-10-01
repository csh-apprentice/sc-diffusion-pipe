We need to define new resume workflow for our trianing pipeline, currently our training pipeline only support resuming from a checkpoint that is defined in [adapter] block, for example, take alook at the code in /root/workspace/sc-diffusion-pipe/MY_TOML/wan_SC_TARGET_14B_MONICA.toml, we can uncomment the init_from_existing line, so the resume looks like this:

[adapter]
type = 'lora'
rank = 32
# Dtype for the LoRA weights you are training.
dtype = 'bfloat16'
# You can initialize the lora weights from a previously trained lora.
init_from_existing = '/root/dev/diffusion-pipe/hunyuan_video_test/20250107_19-28-56/epoch42'

However, this workflow only works when we define the adapter, also it's unclear if it supports resume from both the base lora and our new fps-adapter.

Under this background, we need to define our new resume workflow, I divide the task into several subtasks:

## Subtask 1: Understand how our current resume workflow works
If we keep the original definition of the config file, defining the init_from_existing in the [adapter] block, does the moedl supports also loading the fps related params? If so, do we need to keep the config align with our previous training config? i.e., if we use the base lora [rank 32] and fps adapter [rank 32]
, we still need to define these setting in order to resume from training. 

## Subtask 2: Auto resume
based on the understanding from teh subtask 1, figure out a workflow that can load in the saved checkpoint and apply it onto the model, but only train the params that our current config file defined, you need to pass the following cases:

Case 1:
Previous config: base lora + fps dapter
Current config: bade lora + fps adapter

This is the old case when we have the same definition in previous config and current config both the config enable the training on the base lora and the fps adapter, so inheriently, our resume training should train starting fron the checkpoint.

Case 2:
Previous config: fps adapter
Current config: fps adapter

This is the old case when we have the same definition in previous config and current config both the config enable the trainingonly fps adapter, so inheriently, our resume training should train starting fron the checkpoint.


Case 3:
Previous config: base lora 
Current config: bade lora 

This is the old case when we have the same definition in previous config and current config both the config enable the training only the base lora, so inheriently, our resume training should train starting from the checkpoint.

Case 2: 
Previous config: base lora + fps dapter
Current config: fps adapter
In this case, we should be able to load both the base lora and fps adpater, but freeze the base lora and only train on the fps_adapter starting fron teh fps_adapter params our checkpoint saved.s

Case 2:
Previous config: base lora
Current config: fps adpater
In this case, we should be able to load base lora but freeze it and only train on teh new fps adapter.


