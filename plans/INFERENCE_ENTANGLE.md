I did both experiemnt of training only teh fps related params and joint training of both lora backbone and the the fps injected blocks, found that the joint traininng is superior. This is because our training dataset has some distribution shift compared to that of the original WAN. 
(/root/workspace/sc-diffusion-pipe/MY_TOML/dataset_SC_TARGET_SHAPE_SINGLE_IMG.toml) is an example of joint training that gives good result when overfitting a single image.

Ideally, Joint training can disentangle the context and ,otion blur, i.e. and backbone lora updates is only optimzaing the gap between context while the injected lora params focus on the motion blur only. To test how far out joint traininig goes so far, I want to extend our currect inferencing script of /root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align.py

## Subtask 1: Support selected adapting of checkpoint
I want to add two args in our inferencing script, 
-- fps_only : If given this arg, we will first look into the given config to see whether our saved checkpoint has fps related params for training, if so, We would override the setting in config, even if we do joint training, we only load in the fps related params and don't inference on the backbone lra params. If we don't find fps params definition in config, throw out an error.

-- base_only : Similar to fps_only, when given this arg, we 
will first look into the given config to see whether our saved checkpoint enable base lora in training, if so, We would override the setting in config, even if we do joint training, we only load in the base lora related params and don't inference on the fps params. If we don't find base lora params definition in config, throw out an error.