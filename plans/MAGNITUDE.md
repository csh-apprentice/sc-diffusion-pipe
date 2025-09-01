To enhance the strength of the fps injection, I propose to change the following:

## Subtask 1: Extend how we define compute_tau_rel
We should make use of log considering the shutter speed.
An suggested changes for the code around compute_tau_rel is to changes it to
```
import torch
from typing import Optional, Union

Number = Union[float, int]
TensorOrNumber = Union[torch.Tensor, Number]

def compute_tau_rel(
    fps: TensorOrNumber,
    *,
    # If you don't want a reference, pass reference_fps=None (uses tau = 1/fps)
    reference_fps: Optional[Number] = 240.0,
    # Transform to apply to tau (exposure proxy)
    transform: str = "log1p",          # options: "raw" | "log" | "log1p" | "neglogfps"
    # Multiplicative scaling (kept as a multiplier, as you prefer)
    scale: float = 0.33333334,         # ≈ 1/3 → keeps log1p(240/fps) ~ [0, ~3] → [0, ~1]
    # Numerical stability
    eps: float = 1e-6,
) -> TensorOrNumber:
    """
    Compute a scalar for fps-conditioning.

    tau_raw = (reference_fps / fps) if reference_fps is not None else (1 / fps)

    transform:
      - "raw":      tau_raw
      - "log":      log(tau_raw + eps)
      - "log1p":    log1p(tau_raw)              # recommended
      - "neglogfps": -log(fps + eps)            # equivalent to log(1/fps) up to a constant

    Returns same type as input (torch.Tensor or float/int).
    """
    is_tensor = isinstance(fps, torch.Tensor)
    x = fps.to(torch.float32) if is_tensor else torch.tensor(float(fps), dtype=torch.float32)

    # exposure proxy tau
    if reference_fps is None:
        tau = 1.0 / (x + eps)
    else:
        tau = float(reference_fps) / (x + eps)

    # transform
    if transform == "raw":
        y = tau
    elif transform == "log":
        y = torch.log(tau + eps)
    elif transform == "log1p":
        y = torch.log1p(tau)          # smooth & stable for our τ ∈ {1,2,4,6,10,20}
    elif transform == "neglogfps":
        y = -torch.log(x + eps)       # similar behavior; no reference needed
    else:
        raise ValueError(f"Unknown transform: {transform}")

    # multiplicative scaling (your preference)
    if scale is not None:
        y = y * float(scale)

    if is_tensor:
        return y
    return float(y.item())
```

## SubTask2: fps_adapter_num_tokens 
Adding support for fps_adapter_num_tokens which you should add in the TOML file and read from that like "fps_adapter_rank". So this means How many conditioning tokens per head you generate from the fps embedding c_fps. Right now we project c_fps [B, D_c] → K′ and V′ of shape [B, H, 1, d]. That “1” is num_tokens=1.
If you set fps_adapter_num_tokens = N, then instead of one virtual token, the adapter produces N basis tokens:

K′, V′ → [B, H, N, d].

The attention then distributes across those N learned tokens.

This increases expressivity: the model can learn different “aspects” of blur conditioning, not just one global offset.

There would be some tradeoffs:
1 (default): lightweight, acts like a single global control knob per head.

4: richer, allows the fps signal to influence attention in more nuanced ways.

Higher values = more parameters + memory, but often more capacity.

## SubTask3: Use fps_embed_dim instead of dim

Add a config fps_embed_dim (default 256). Set the last layer of self.fps_conditioning to output fps_embed_dim. Then pass that to the adapter via fps_conditioning_dim=fps_embed_dim.

## SubTask4: Rewrite our fps mlp
Create a class FpsConditioning(nn.Module) with:

__init__(self, out_dim: int, hidden: int = 64)
Layers: lin1 = Linear(1, hidden), act = SiLU(), ln = LayerNorm(hidden), lin2 = Linear(hidden, out_dim).

Special init inside __init__ (wrap in torch.no_grad()):

kaiming_uniform_(lin1.weight, a=0.0, mode='fan_in', nonlinearity='relu'); lin1.bias = 0

ln.weight = 1, ln.bias = 0

lin2.weight = 0, lin2.bias = 0 # zero-disturbance head

forward(self, x): return lin2( ln( SiLU( lin1(x) ) ) )

Replace the old self.fps_conditioning = nn.Sequential(...) with:

self.fps_conditioning = FpsConditioning(out_dim=fps_conditioning_dim, hidden=64)

Ensure your blanket Xavier init does not touch this module (either run the blanket loop before constructing it, or explicitly skip its submodules when looping).

In model.py init_weights(), ensure self.fps_conditioning’s layers use special init: lin1.weight = kaiming_uniform_(fan_in, relu/SiLU), lin1.bias=0; LayerNorm weight=1, bias=0; lin2.weight=0, lin2.bias=0. Make sure the blanket Xavier loop does not overwrite these (either run the blanket loop before constructing self.fps_conditioning, or skip these specific modules when looping).


Also make fps_condition_hidden a configureble parameter in TOML.


## SubTask5: change the fps_adapter_gate_init
fps_adapter_gate_init also impacts on how fast our model learns, given our slow learing case, our Gate init is too open at start.Make it a config and default to a small, but non-zero opening: −2.0

## SubTask 6: Add LoRA scale

In FPSCrossAttentionAdapter.__init__, add lora_alpha (configurable, default 16/32) and precompute self.lora_scale = lora_alpha / rank. In forward, multiply k_fps_proj and v_fps_proj by self.lora_scale.

## SubTask 7: Remove the Layernorm in FPS condition
 did some simple tests with our FPS MLP and noticed that using a LayerNorm right after the first linear layer weakens the fps signal at initialization. Specifically, because we initialize the bias of that linear layer to zero, normalization wipes out the proportional differences (e.g. fps=12 vs fps=60), leaving only directional variations. The SiLU nonlinearity recovers some distinction, but the signal still comes through weaker.
For consistency, I’d propose removing LayerNorm here. The time embedding MLP in Wan2.1 — which performs the very similar task of turning a scalar timestep into a high-dimensional embedding — follows the simpler Linear → SiLU → Linear design, and this would align our fps conditioning path with that proven structure.


## Optional

Increase the learing rate



