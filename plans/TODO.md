# Task 1: Dataset Hook: 
Help me to extend the codebase so now it supports reading datasets with fps. Also you should store that fps somewhere in the class as it is goona to use for future model injection.

# Task 2: Conditioning Encoder 
Turn the scalar fps (or tau_rel = 240 / fps) from the batch into a learned conditioning embedding c_fps and thread it through your Wan2.1 forward pass so later (Step 3) we can inject it into cross-attention without refactors. You should write tau_rel into a seperate function since we may have differnent mapping rather than 240/ fps later,  Map scalar tau_rel → conditioning embedding vector can use a simple mlp like this:
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim)
        )

# Task 3 — Inject FPS Conditioning into Cross-Attention

## Core Idea (Disentangled Addition)
- Keep **Queries (Q)** unchanged.
- Add a **parallel cross-attention stream** with **new Keys/Values (K′, V′)** derived from the fps embedding `c_fps`. You can mimic how thw wan2.1 model cembed on the text prompts on the corss attention.
- Final cross-attention output:
  \[
  y = \mathrm{Attn}(Q, K_{\text{text}}, V_{\text{text}})
    \;+\; g \cdot \mathrm{Attn}(Q, K', V')
  \]
  - `y_txt = Attn(Q, K_text, V_text)`
  - `y_fps = Attn(Q, K′, V′)`
  - Gate `g = σ(α)` (learnable scalar per block, init 0)

This ensures the fps signal is **disentangled** from text semantics and **inactive at init**.



## LoRA Structure (to avoid OOM)
Each projection uses low-rank decomposition:
- `Wk′ = B_k A_k`, `Wv′ = B_v A_v`
- Shapes:
  - `A_k: D_c × r`, `B_k: r × (H*d)`
  - `A_v: D_c × r`, `B_v: r × (H*d)`
- Rank `r` is small (e.g., 4–8).

### ✅ LoRA Initialization (classic, safe)
- **Down (A):** random init (e.g., Normal(0, 0.01))  
- **Up (B):** zeros  
- Ensures:
  - Initial delta = 0 → no behavior change
  - Gradients flow: `B` gets gradients immediately, then `A` updates after first steps
- Keep **gate α = 0** at start as a second safety.

---

## Block Selection
- Apply fps adapter **only in deepest ~⅓** of cross-attn layers (e.g., layers 27–39 of 40).
- Make this configurable. 
---

## TOML Config Knobs
```toml

fps_adapter_rank = 4                              # LoRA rank r

fps_adapter_gate_init = 0.0                       # gate α init

fps_condition_blocks = "deepest_third"            # or explicit number (eg: 10 means the deepest 10 blocks)

```

# Task 4: Create new inference script
Now we need to create a new inferenec script that not only accecpt the text prompts but also a user specified "fps" number (scalar) as input. 
You should:
1. First check our code logic to see if our training save the fps adapter & mlp params.
2. Copy inference/lora_inference.py to inference/lora_inference_sct2v_cross.py, understand how our code is working to inference on the basic wan t2v task then extend it accepting the "fps".


Note, if we do save the checkpoints includes the injected params, we can test it using the checkpoints saved in we under checkpoints/20250821_17-46-33 folder for epoch 1.


