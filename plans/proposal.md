Project Title
Incorporating Frame Rate–Based Motion Blur Conditioning into Wan2.1 T2V Model via LoRA Fine-Tuning

Background
Wan2.1 is a state-of-the-art text-to-video model employing a large DiT backbone, cross-attention to a frozen text encoder, and a latent-space VAE for efficient video synthesis. Currently, it lacks explicit conditioning on motion blur, a significant cinematic effect strongly correlated with frame rate (fps) and shutter speed. The objective is to enable controllable motion blur at inference time, commencing with a 360° shutter assumption where blur is solely determined by fps.

Goal
To fine-tune Wan2.1 using LoRA to accept an fps scalar as a conditioning input, thereby enabling the modulation of motion blur intensity without necessitating retraining of the entire model.

Method
Data:
Source: 240 fps “sharp” videos.
Blur synthesis: Consecutive frames are averaged to simulate lower fps with motion blur (e.g., 12, 24, 40, 60, 120, 240 fps).
Folder structure:
dataset_root/
12/clip1.mp4 + clip1.txt
24/clip2.mp4 + clip2.txt
...

Prompt reuse: All synthetic fps variants derived from the same clip share an identical caption.
Conditioning signal:
Compute tau_rel = 240 / target_fps (unitless) for each sample.
This value will serve as the scalar control for blur intensity.
Model changes:
An FpsConditioning MLP will be added to embed tau_rel.
This will be injected into the deepest approximately one-third of cross-attention blocks via a parallel K′/V′ attention stream (consistent with Bokeh Diffusion–style).
LoRA will be applied to K′/V′ projections to minimize memory consumption.
Training:
Only the fps-conditioning module and LoRA weights will be fine-tuned.
The Wan base model will remain frozen.
Loss function: The standard DDPM/DiT noise prediction loss will be employed.
Evaluation:
With a fixed prompt and seed, fps values will be swept to assess monotonic blur changes, temporal stability, and content consistency.
Expected Outcome
A Wan2.1 LoRA adapter that, when provided with an fps value, can generate videos with controllable motion blur intensity.



