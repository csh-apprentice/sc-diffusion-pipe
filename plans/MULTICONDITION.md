Now we need o consider to support multi condition rather than single scalar.

How it works? Now our condition t2v model only supports a single scalar condition, model reads the input from our designed dataset like this:

<data_root>/
  12/   video_x.mp4, video_x.txt, ...
  24/   ...
  40/   ...
  60/   ...
  120/  ...
  240/  ...

Our extension from the orginal code base adding supports to detect if there are subfolders under the given the given directory path, for example:

path = "/root/workspace/sc-diffusion-pipe/dataset/3shapes_bokeh"

To support multi-condition, I redesign the dataset layout, so it would be some thing like this:

<data_root>/
  12_0.1/   video_x.mp4, video_x.txt, ...
  24_0.2/   ...
  40_0.3/   ...
  60_0.4/   ...
  120_0.5/  ...
  240_0.6/  ...

You can see the name of each subfoder is not neccessary a floating number now, it can be several numbers connected with "_", for example for subfolder 12_0.1, that means we have two condition 12 and 0.1. If there is only one condition, the case fall back to our old dataset layout.

We use to save the scalar as a floating number & single size tensor, [fps], we can extend it to [fps1,fps2,...fpsn] if we have n condition by checking the directory naming.

We also have several extended settings for condition,

1. fps_tau_transform, configurable transform for our scalar fps, we extend it to accept "tensor" like config:
for example, we can use fps_tau_transform = "centerlog1p" for one condition tranining, we can also have 
fps_tau_transform=["centerlog1p","raw"] if we have two conditions, for design robustness, we shouldn't break our old design intergrity, hence if we only have a single condition, a string input should still be valid.

For multi-condition case,  tranform method will be seperately apply to corresponding condition.

2.fps_tau_scale & fps_reference_fps, used to be floating inputs, should still support it in one condition case, in multi- condition, also extend them to tensor, for example

fps_tau_scale = [1.0,2.0]
fps_reference_fps=[240.0,2.0]

for a 2 condition case, so 1.0 and 240 should be apply to condition1, 2.0 and 2.0 should be assigned to condition2

3. FpsConditioning class, in this class, we use to         
self.lin1 = nn.Linear(1, hidden)
1 referes to that single consition, now since we should support multi-condition that input dimension should be decided by how many conditions we have when parsing the dataset.


To test, an example trainig dataset with 2 conditions is listed in /root/workspace/sc-diffusion-pipe/dataset/2shapes_shutter_bokeh

An toml file is written in /root/workspace/sc-diffusion-pipe/shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH.toml

