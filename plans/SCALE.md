Let's adding a scale option in the cross attention adding.
Add a option under the toml [model] right next to where we define the fps_adapter_gate_init:
fps_scale = "true" or "false"
default is false

Subtask 1: False Logic
If the fps_scale is "false", we keep the original logc, but I want to change the "learned" mode, rename it as "sigmoid", since we are using sigmoid function in current learning mode.

Subtask 2: True Logic
If the fps_scale is "true", we scale the y_fps_norm so it will have the same magnitude as the y_text before we add them together 

but be attention that y_fps_norm may be zero at the start, so maybe adding a small magnitude all ones vector on the y_fps_norm.

Subtask 3: Extending Activiatin choice
Except for the "sigmoid", let's also add the following two activation methods:

'identity' gate=alpha
'relu', gate=RELU(alpha)
'silu' gate=SiLU (alpha)

all these options should be able to parse through the config file and read properly

Subtask 3: warm up option
since we need a cold start at first, the learning is unstable at first, to solve it, let's also define a "warmup_steps" together with the "fps_scale", default is 100, so we would 
have 
w(t)=min(1,t/K), K is the warmup_steps total, t is the current step,
y=y_text+(w(t)*g(alpha))y_fps_scale

Subtask 4: Extending Activiatin choice
let's also add the following  activation methods:

'softplus' gate=softplus(alpha)


