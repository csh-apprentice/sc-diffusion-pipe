So the problem is if we left gate_alpha as a training parameter, the gate is decreasing over training, and it seems to mute the impact of y_fps. To solve this problem, can you:
## SubTask 1: Adding fixed gate option
We should provide multiple choices for gate in training.
Let's make it configurable, in TOML file:
have two choice now(we may extend it support more methods)
gate_mode_learned
gate_mode_fixed

If specify gate_mode_learned, then just follow our old setting, our param traininig starts frpm fps_adapter_gate_init.

If specify gate_mode_fixed, should provide another parameter
fps_gate_fixed in TOML file, for example:

fps_gate_fixed=0.5

In this case, in our training, our gate value is set to 0.5 as a fixed value.