# diffusion-pipe
A pipeline parallel training script for diffusion models.

Models supported: SDXL, Flux, LTX-Video, HunyuanVideo (t2v), Cosmos, Lumina Image 2.0, Wan2.1 (t2v and i2v), Chroma, HiDream, Stable Diffusion 3, Cosmos-Predict2, OmniGen2, Flux Kontext, Wan2.2, Qwen-Image.

## Features
- Pipeline parallelism, for training models larger than can fit on a single GPU
- Useful metrics logged to Tensorboard
- Compute metrics on a held-out eval set, for measuring generalization
- Training state checkpointing and resuming from checkpoint
- Efficient multi-process, multi-GPU pre-caching of latents and text embeddings
- Seemlessly supports both image and video models in a unified way
- Easily add new models by implementing a single subclass

## Recent changes
- 2025-08-07
  - Fix Flux training error caused by a breaking change in Diffusers. Make sure to update requirements.
- 2025-08-06
  - Support Qwen-Image.
  - Slight speed improvement to Automagic optimizer.
- 2025-07-29
  - Support Wan2.2.
    - The 5B is tested and fully validated on t2i training.
    - All other models and modes (A14B, i2v, timestep ranges) are tested to confirm they run and that the loss looks reasonable, but proper learning hasn't been validated yet.
- 2025-07-14
  - Merge dev branch into main. Lots of changes that aren't relevant for most users. Recommended to use ```--regenerate_cache``` (or delete the cache folders) after update.
      - If something breaks, please raise an issue and use the last known good commit in the meanwhile: ```git checkout 6940992455bb3bb2b88cd6e6c9463e7469929a70```
  - Loading speed and throughput improvements for dataset caching. Will only make a big difference for very large datasets.
  - Various dataset features and improvements to support large-scale training. Still testing, not documented yet.
  - Add ```--trust_cache``` flag that will blindly load cached metadata files if they exist, without checking if any files changed. Can make dataset loading faster for large datasets, but you must be sure nothing in the dataset has changed since last caching. You probably don't have a large enough dataset for this to be useful.
  - Add torch compile option that can speed up models. Not tested with all models.
  - Add support for edit datasets and Flux Kontext. See supported models doc for details.
- 2025-06-27
  - OmniGen2 LoRA training is supported, but only via standard t2i training.
  - Refactored Cosmos-Predict2 implementation to align with other rectified flow models. The only effective change is that the loss weighting is slightly different.
- 2025-06-14
  - Cosmos-Predict2 t2i LoRA training is supported. As usual, see the supported models doc for details.
  - Added option for using float8_e5m2 as the transformer_dtype.
- 2025-06-10
  - Stable Diffusion 3 LoRA training is supported.
  - Pinned Deepspeed version to fix error caused by Deepspeed 0.17.1.
- 2025-05-22
  - Add Automagic optimizer
  - Support i2v training for LTX-Video. Thanks @GallenShao for the PR!
  - Support multiple shuffling of tags when caching text embeddings. Credit to @gitmylo for the PR.

## Windows support
It will be difficult or impossible to make training work on native Windows. This is because Deepspeed only has [partial Windows support](https://github.com/microsoft/DeepSpeed/blob/master/blogs/windows/08-2024/README.md). Deepspeed is a hard requirement because the entire training script is built around Deepspeed pipeline parallelism. However, it will work on Windows Subsystem for Linux, specifically WSL 2. If you must use Windows I recommend trying WSL 2.

## Installing
Clone the repository:
```
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
```

If you alread cloned it and forgot to do --recurse-submodules:
```
git submodule init
git submodule update
```

Install Miniconda: https://docs.anaconda.com/miniconda/

Create the environment:
```
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
```

Install PyTorch first. It is not listed in the requirements file, because certain GPUs sometimes need different versions of PyTorch or CUDA, and you might have to find a combination that works for your hardware. As of this writing (August 7, 2025), PyTorch 2.7.1 with CUDA 12.8 works on my 4090, and is compatible with flash-attn 2.8.1:
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
Note: as of right now (August 7, 2025), I can't get PyTorch 2.8.0 to work with any version of flash attention. And the latest flash attention 2.8.2 doesn't work with any version of PyTorch. Torch 2.7.1 + flash attention 2.8.1 works (that flash attention version is pinned in requirements.txt). You can always try to install flash attention from source after installing PyTorch, to attempt to avoid these kinds of errors from the pre-packaged wheels.

Install nvcc: https://anaconda.org/nvidia/cuda-nvcc. Probably try to make it match the CUDA version of PyTorch.

Install the rest of the dependencies:
```
pip install -r requirements.txt
```

### Cosmos requirements
NVIDIA Cosmos (the original Cosmos video model, not Cosmos-Predict2) additionally requires TransformerEngine. Cosmos-Predict2 doesn't require TransformerEngine, but it will use it when available for a slight speed increase.

This dependency isn't in the requirements file. You probably need to set some environment variables for it to install correctly. The following command worked for me:
```
C_INCLUDE_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn/include:$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH pip install --no-build-isolation transformer_engine[pytorch]
```
Edit the paths above for your conda environment.

## Dataset preparation
A dataset consists of one or more directories containing image or video files, and corresponding captions. You can mix images and videos in the same directory, but it's probably a good idea to separate them in case you need to specify certain settings on a per-directory basis. Caption files should be .txt files with the same base name as the corresponding media file, e.g. image1.png should have caption file image1.txt in the same directory. If a media file doesn't have a matching caption file, a warning is printed, but training will proceed with an empty caption.

For images, any image format that can be loaded by Pillow should work. For videos, any format that can be loaded by ImageIO should work. Note that this means **WebP videos are not supported**, because ImageIO can't load multi-frame WebPs.

## Supported models
See the [supported models doc](./docs/supported_models.md) for more information on how to configure each model, the options it supports, and the format of the saved LoRAs.

## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does. [This config file](./examples/main_example.toml) is the main example with all of the comments. [This dataset config file](./examples/dataset.toml) has the documentation for the dataset options.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the `--resume_from_checkpoint` flag. You can also specify a specific checkpoint folder name after the flag to resume from that particular checkpoint (e.g. `--resume_from_checkpoint "20250212_07-06-40"`). This option is particularly useful if you have run multiple training sessions with different datasets and want to resume from a specific training folder.

Please note that resuming from checkpoint uses the **config file on the command line**, not the config file saved into the output directory. You are responsible for making sure that the config file you pass in matches what was previously used.

## Output files
A new directory will be created in ```output_dir``` for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

## Reducing VRAM requirements
The [wan_14b_min_vram.toml](./examples/wan_14b_min_vram.toml) example file has all of these settings enabled.
- Use AdamW8BitKahan optimizer:
  ```
  [optimizer]
  type = 'AdamW8bitKahan'
  lr = 5e-5
  betas = [0.9, 0.99]
  weight_decay = 0.01
  stabilize = false
  ```
- Use block swapping if the model supports it: ```blocks_to_swap = 32```
- Try the expandable_segments feature in the CUDA memory allocator:
  - ```PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /home/you/path/to/config.toml```
  - I've seen this help a lot when training on video with multiple aspect ratio buckets.
  - On my system, sometimes this causes random CUDA failures. If training gets through a few steps though, it will train indefinitely without failures. Very weird.
- Use unsloth activation checkpointing: ```activation_checkpointing = 'unsloth'```

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

Three flags are relevant for caching. ```--cache_only``` does the caching flow, then exits without training anything. ```--regenerate_cache``` forces cache regeneration. ```--trust_cache``` will blindly load the cached metadata files, without checking if any data files have changed via the fingerprint. This can speed up loading for very large datasets (100,000+ images), but you must make sure nothing in the dataset has changed.

## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs.

## SCPIPE ENV:
You can either follow the env guide from the main repo or try our snapshot:
/root/workspace/sc-diffusion-pipe/scpipe_env

## How to use scpipe
To train the model on 2 GPU (A100E 80 GB):

nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=2 train.py --deepspeed --config fps_TOML/wan_SC_TARGET_14B_SHUTTER_30S.toml' > ./output/fps_nohup_log/shutter_30S.out 2>&1 &

## How to create the dataset:

### Shutter

shape_blur dataset comes from /root/workspace/sc-diffusion-pipe/utils/synthesis/generate_synthetic_blur_dataset_pallete_random.py, the videos are not fully align but still gives good result

python generate_synthetic_blur_dataset_pallete_random.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shutter_one_shot_150_8f/ \
  --modes video \
  --num_scales 9 \
  --samples_per_scale 1 \
  --num_frames 8 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_obj 1 \
  --max_objs 3 \
  --bg_white_prob 1.0 \
  --caption_relations 


(if you want a fixed fps list)
python generate_synthetic_blur_dataset_fps.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shutter_one_shot_slides/ \
  --modes video \
  --fps_list 12,24,30,40,60,120,240 \
  --samples_per_fps 1 \
  --num_frames 12 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_objs 1 \
  --max_objs 1 \
  --seed 40 \
  --caption_relations 



To create the align video:
python generate_synthetic_blur_dataset_pallete_align.py \
  --out_dir /root/workspace/sc-diffusion-pipe/dataset/shutter_100s4f \
  --modes video \
  --num_scales 100 \
  --samples_per_scale 1 \
  --num_frames 4 \
  --align_img_video \
  --min_speed 1000 \
  --max_speed 1000 \
  --min_obj 1 \
  --max_objs 3 \
  --seed 42 \
  --caption_relations 

num_scales: how many condition for each scene
samples_per_scale: how many scenes
num_frames: How many frame in each video

### APERTURE:
Please refer to the google drive (blender script bokeh_dataset_box_1030.py) (dataset/creation_code)

### TEMPERATURE:
(USE ALL WHITE BACKGROUND)
python generate_temp_by_scale.py \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/2dshapes_temp_part2/9s \
  --num_scenes 3 \
  --num_scales 9 \
  --white-bg \
  --k-lo 2000 --k-hi 20000 --k-ref 6500 --preserve-luminance


adjust temperature fom existed:

python apply_temp_shifts.py \
  --input_image /root/workspace/sc-diffusion-pipe/dataset/mouflower_temp/6000/mouflower.jpg \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/mouflower_ablation \
  --num_scales 7 \
  --k-lo 2000 \
  --k-hi 20000 \
  --k-ref 6500 \
  --preserve-luminance \
  --seed 42

Or even uniform space:
python apply_temp_shifts.py \
  --input_image /root/workspace/sc-diffusion-pipe/dataset/mouflower_temp/6000/mouflower.jpg \
  --output_dir /root/workspace/sc-diffusion-pipe/dataset/real_syn/temp/mouflower_ablation \
  --num_scales 7 \
  --k-lo 2000 \
  --k-hi 20000 \
  --k-ref 6500 \
  --preserve-luminance \
  --uniform

## Exsited Dataset
### Shutter: 
For some reason, shapes_blur is sota, but we also have a fully align dataset shutter_30 scenes

### APERTURE:
3shapes_bokeh_30scenes_1030_3

### Temperature:
2dshapes_temp_part1 and 2dshapes_temp_part2

## How to inference
We have four inference script now, may beed future merging:

CLEAN (Original Wan2.1 Backbone) 
FPS_ONLY (Only apply the trained adapter)
BASE_ONLY (Ony apply the trained backbone Lora)
GRAFT (The clean inference method we mention in paper)
No ARG: default, dirty inference method we mention in paper


inference/test_fps_multiple_experiments_align_old.py (no graft mode)
inference/test_fps_multiple_experiments_align.py (no clean mode)
inference/test_fps_graft.py (no clean mode)
inference/test_fps_batch_prompts.py: support a prompt file line by line (all mode support)

## How to evaluate

We mention FEP and SVP in the paper:

### For FEP score:
you should run the on th eoriginal backbone first:

python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 30 \
    --frames 16 \
    --output_dir ../output/16steps/clean_category_42 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_clean_42_16frames.out 2>&1 

Then change the seed to 99 to set the baseline for FEP score.



To evaluete the changing for each model, run the following command this will generate FEP score for epcoh 100, 200, 300 ,... 1000 for Vbench 800 prompts in 8 subfolders.




python test_fps_batch_prompts.py \
    --config  /root/workspace/sc-diffusion-pipe/checkpoints/20251023_00-19-38/wan_SC_TARGET_14B_2DSHAPE_TEMP_30S.toml\
    --checkpoint_parent ../checkpoints/20251023_00-19-38 \
    --checkpoint_prefix epoch \
    --checkpoint_start 100 \
    --checkpoint_end 1000 \
    --checkpoint_interval 100 \
    --fps_values 0.0 \
    --steps 1 \
    --frames 4 \
    --output_dir ../output/onestep/20251023_00-19-38 \
    --prompt_dir ../utils/VBench/prompts/prompts_per_category  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/shutter_20251023_00-19-38.out 2>&1 

Then run: (see also bash/cal_score.sh)

for epoch in 100 200 300 400 500 600 700 800 900 1000; do
python metric/video_score_calculator_extended.py \
    --orig_parent_dir output/onestep/clean_category_42 \
    --adapt_parent_dir output/onestep/20251106_09-42-01/epoch${epoch} \
    --output_file scores/20251106_09-42-01/epoch${epoch}/scores.json
done

### For SVP score:
You should generate results with 50 denoising steps and 49 frames 
on hour 96 prompts (metric/high_quality_prompts_96.txt):

Baseline:
python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/clean_category_42 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --clean \
    > ../output/nohup_log/bokeh_clean_42_49frames.out 2>&1 


python test_fps_batch_prompts.py \
    --config /root/workspace/sc-diffusion-pipe/checkpoints/20251008_20-21-45/wan_SC_TARGET_14B_3SHAPES.toml \
    --checkpoint ../checkpoints/20251008_20-21-45/epoch1000 \
    --fps_values 0.0 \
    --steps 50 \
    --frames 49 \
    --output_dir ../output/50_49/20251008_20-21-45 \
    --prompt_file /root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt  \
    --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走." \
    --seed 42 \
    --port 29502 \
    --width 512 \
    --height 512 \
    --base_only \
    > ../output/nohup_log/bokeh_20251008_20-21-45_49frames.out 2>&1 

we use the XCLIP/VQA/Vbench score:

XCLIP score: metric/xclip_score_calculator.py
VQA score: https://github.com/linzhiqiu/t2v_metrics

Put the metric/calculate_vqa_score.py under the /root/workspace/sc-diffusion-pipe/utils/t2v_metrics

Vbench score: follow the vbench git repo : https://github.com/Vchitect/VBench

python evaluate.py \
    --dimension 'subject_consistency' 'background_consistency'  'motion_smoothness' 'dynamic_degree' 'aesthetic_quality' 'imaging_quality' \
    --videos_path /root/workspace/sc-diffusion-pipe/output/50_49/20251030_08-16-52_graft \
    --mode=custom_input \
    --output_path "./50_49/evaluation_20251030_08-16-52_graft/"

### For results visualization:

Refer to /root/workspace/sc-diffusion-pipe/visualize/line_plot
and /root/workspace/sc-diffusion-pipe/visualize/bar_plot


## How to analysis

### HYP1: 
bash /root/workspace/sc-diffusion-pipe/bash/compute_similarity_matrix.sh /root/workspace/sc-diffusion-pipe/output/similarity_matrix/20251115_22-28-03/epoch1000/similar_matrix.log

code: inference/compute_backbone_similarity_matrix.py

## HYP2:
/root/workspace/sc-diffusion-pipe/bash/run_principal_orthogonality.sh

bash /root/workspace/sc-diffusion-pipe/bash/run_principal_orthogonality.sh /root/workspace/sc-diffusion-pipe/output/principal_orthogonality/20251008_20-21-45/orthagnal_check.log



