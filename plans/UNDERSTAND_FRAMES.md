To clearly nuderstand how our backbone works, I need to figure out how our backbone strategy when parsing differentn frames of videos or images.

# Subtask 1: Understand how the frame bukets works
Take a looks at my latest training output when using the toml from 
MY_TOML/wan_SC_TARGET_14B_FPS_SHAPE_SINGLE_VID.toml
The output is written in 
output/nohup_log/sanity_shape_single_video_joint.out
You can read the first 100 lines. My question is 
why I have this notification in output: video with frames=8 is being skipped because it is too short
video with frames=0 is being skipped because it is too shortvideo with frames=16 is being skipped because it is too short

video with frames=4 is being skipped because it is too short
If like the comment said " For video training, you need to configure frame buckets (similar to aspect ratio buckets). There will always
be a frame bucket of 1 for images. Videos will be assigned to the first frame bucket that the video is greater than or equal to in length. But videos are never assigned to the image frame bucket (1); if the video is very short it would just be dropped." Our training videos all have 12 frames and thus should all be dropped, why the waring said they are gave 0,8,16 though, also the training is working means the model is using some data for traiing, what does it use?

# Subtask 2: Understand how the images are used for training
So for teh speical case, teh image has one frame and should be assign to the first frame bucket, I wonder in the training, how doe sthe model use it for training? Do we use it as a video with 1 frame or we duplicate that one frame so it can be easily parse in to the encoder. Help me figure that out.

# Subtask 3: Understand how the num_repeats take effect
In our MY_TOML/dataset_SC_TARGET_SHAPE_SINGLE_VID.toml, we set num_repeats = 10, I wonder if that's duplciat the same training sample with the same config, or duplicating teh training sample with different seed.

---

# FINDINGS (Analysis completed)

## Root Cause Analysis: Framerate Mismatch in Frame Bucket Calculation

### The Core Issue
The frame bucket warnings occur due to a **framerate mismatch** between the dataset calculation logic and actual video properties:

**Dataset Frame Calculation Logic** (utils/dataset.py:694):
```python
frames = int(self.framerate * meta['duration'])
```

Where:
- `self.framerate = 16` (hardcoded for WAN model)
- `meta['duration']` varies based on video FPS to maintain 12 actual frames

### Actual Video Analysis
All videos contain **exactly 12 frames**, but have different durations to maintain their target FPS:

| Video FPS | Duration | Actual Frames | Calculated Frames (16×duration) | Result |
|-----------|----------|---------------|----------------------------------|--------|
| 12fps     | 1.0s     | 12           | 16                              | Rejected (frames=16 too short) |
| 24fps     | 0.5s     | 12           | 8                               | Rejected (frames=8 too short) |
| 40fps     | 0.3s     | 12           | 4                               | Rejected (frames=4 too short) |
| 60fps     | 0.2s     | 12           | 3                               | Rejected (frames=3 too short) |
| 120fps    | 0.1s     | 12           | 1                               | **Accepted** (treated as image) |
| 240fps    | 0.05s    | 12           | 0                               | Rejected (frames=0 too short) |

### Why Training Still Works
- Only the **120fps video** gets accepted (calculated as 1 frame, treated as image)
- Training cache shows `(1.0, 512, 512, 1)` indicating single-frame data processing
- The model receives correct FPS conditioning value (120) from folder structure

## Subtask Answers

### Subtask 1: Frame Bucket Warnings
**Answer**: Videos are rejected due to incorrect frame calculation using fixed 16fps assumption against varying video durations, despite all containing 12 actual frames.

### Subtask 2: Single-Frame Image Processing  
**Answer**: Images (1 frame) are processed as true single-frame sequences without duplication. They get assigned to frame bucket 1 and processed through the same pipeline as videos but with temporal dimension = 1.

### Subtask 3: num_repeats Effect
**Answer**: `num_repeats = 10` creates 10 identical copies of the same samples. The dataset size is multiplied by 10, but each copy uses the same image+caption+FPS configuration. Randomization comes from training noise and data augmentation, not from the repeats themselves.

## Critical Discovery: Temporal Resampling Issue

### The Problem
During video loading, the code performs temporal resampling:
```python
# models/base.py:95-98
for frame in imageio.v3.imiter(filepath_or_file, fps=self.framerate):
video = imageio.v3.imiter(filepath_or_file, fps=self.framerate)
```

### Impact Analysis
When `fps=16` is applied to videos:
- **12fps video (1.0s)** → Resampled to 16 frames via interpolation
- **24fps video (0.5s)** → Resampled to 8 frames via downsampling  
- **Original temporal structure is lost**

### Data Corruption Effects
1. **Frame Interpolation**: Creates synthetic frames that don't exist in original data
2. **Inconsistent Training**: All videos get resampled toward 16fps, losing FPS-specific motion characteristics
3. **FPS Conditioning Mismatch**: Model sees 16fps motion but receives conditioning signal for original FPS (12, 24, etc.)

### Current Status Assessment
- If all training samples are effectively processed at 16fps due to resampling, the current setup may be internally consistent
- However, the FPS conditioning signals (12, 24, 40, 60, 120, 240) from folder names may not match the actual temporal content the model observes
- This could explain why FPS conditioning effects are subtle - the model sees similar temporal patterns regardless of FPS label

### Recommended Future Actions
1. **Option 1**: Remove fps parameter from imageio.v3.imiter() to preserve original temporal structure
2. **Option 2**: Adjust frame buckets to accommodate actual frame counts: `frame_buckets = [1, 12, 33, 65, 87]`  
3. **Option 3**: Verify if current 16fps resampling approach is intentional for temporal consistency

### Two-FPS System Discovery
The codebase uses **two separate FPS concepts**:
1. **Model framerate** (`self.framerate = 16`): Used for frame bucketing and video resampling
2. **FPS conditioning values**: Extracted from folder names (12, 24, 40, 60, 120, 240) for training conditioning

This dual-FPS system may be causing the disconnect between expected FPS behavior and actual training results.