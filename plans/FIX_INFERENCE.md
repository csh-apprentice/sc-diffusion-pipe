I found serious misalignment in how we define the toml and how we inference, hence, we can never see the difference out there.

To make it clear, I want you to first copy and inference/test_fps_multiple_experiments.py to inference/test_fps_multiple_experiments_align.py, then work on test_fps_multiple_experiments_align.py.

Few things you need to correct:
Subtask1: Align with how we train the model, don't parse the arg manually to initalize the WanPipeline and inferecing. But accept a new input of the path to the TOML file. (You can use MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_IMG.toml as testing).

Subtask2: Carefully go through all the toml definition, what are the parameters/settings we should use to override the default in inferecing? Please use those setting/params for inferencing.

Subtask3: Model is never put in eval mode
A few layers (norms/projections) can behave subtly differently in train vs eval. Always set eval for sampling.
wan_t2v.transformer.eval()
wan_t2v.vae.model.eval()
wan_t2v.text_encoder.model.eval()


Subtask4: Autocast dtype mismatch
You trained & load the transformer in bfloat16, but your sampling loop uses with torch.autocast('cuda') (default is float16). That silent cast can destabilize the denoiser and cause saturation.


Subtask5: Find all necessary settings:
can you take a look at the models/wan/wan.py on how it use the settings in config file, I am pretty sure you miss some settings, includes but not limited to fps_tau_transform, fps_tau_scale, fps_gate_mode etc.  I want perfect align so the settings never get override by the default setting if they wre given in the config file.