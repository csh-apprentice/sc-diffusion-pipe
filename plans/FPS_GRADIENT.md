# FPS LoRA Gradient Issue Diagnosis

## What’s Happening

In our FPS adapter implementation, we followed the standard LoRA trick:  
- Down projection `A` initialized randomly.  
- Up projection `B` initialized to all zeros.  

This is correct for LoRA, but **our context is different**:

- In standard LoRA (e.g., on attention Q/K/V), the main weight `W` is non-zero, so the output is:  
  \n**y = xW + (xA)B**  
  Even if `B=0`, the gradient flows through `W`, and training updates `B` immediately.

- In our FPS adapter, there is **no base weight** for the conditioning path. The only path is:  
  \n**y_fps = (xA)B**  
  If `B=0` initially, then `y_fps=0` → the output does not affect the loss → gradients to both `A` and `B` vanish.

This is the “dead branch” problem: the FPS adapter branch contributes nothing, so no learning happens.

---

## Why This Doesn’t Happen in Basic LoRA

In normal LoRA applied to an existing attention layer:  
- Base output `y_base = xW` is already non-zero.  
- LoRA adds a residual term `(xA)B`.  
- Even though `B=0` initially, the gradient of the loss w.r.t. `B` is non-zero because it sits on top of a functioning path (`xW`).  
- This allows `B` to start learning immediately.

In FPS adapter, there is **no base weight** for the conditioning path, so initializing `B=0` kills the gradient.

---

## What To Do

We must avoid the **all-zero initialization** of `B` in the FPS adapter.

Options:

1. **Small Random Init for B**  
   - Initialize `B` (the up-projection) with a very small random Gaussian or Xavier init.  
   - This ensures `y_fps` is small but non-zero, letting gradients flow.


---

## Next Step

- Fix initialization of `B` in the FPS adapter to small random instead of zeros.  
