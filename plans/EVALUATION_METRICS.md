You can resort to /root/workspace/sc-diffusion-pipe/bash/compute_similarity_matrix.sh and /root/workspace/sc-diffusion-pipe/bash/run_principal_orthogonality.sh or our source code in /root/workspace/sc-diffusion-pipe/models/wan/model.py if you needed code reference:

note that we didn't implement the compute_tau_rel in our previous file, so maybe use that before you feed our cond into MLP, transform method can be reded from config file.


# Data-Free Metrics for Joint LoRA & Adapter Finetuning

This file outlines quantitative, data-free metrics to evaluate the success of a joint finetuning strategy. The primary goal is to find the "sweet point" checkpoint.

**The "Sweet Point" Definition:** The ideal checkpoint is one where the **conditional adapter is fully trained** (it has learned the physical effect) *and* the **backbone is not corrupted** (it has not suffered "catastrophic forgetting").

We need two primary scores to find this.

---

## 1. Metric 1: "Content Primitive Drift" (CPD) Score

* **Purpose:** Measures backbone health and "catastrophic forgetting." [cite_start]It quantifies how much the finetuned model's core "content space" has drifted from the *original* pre-trained model. [cite: 439]
* **Interpretation:**
    * `CPD_Score` $\approx$ 1.0: **Good.** The backbone is clean and has not been corrupted.
    * `CPD_Score` $\ll$ 1.0: **Bad.** The backbone is corrupted and has "forgotten" its original knowledge.

### Algorithm (Data-Free)

1.  **Get "Ground Truth" Primitives:**
    * Load the *original pre-trained* backbone weights ($W_q, W_k, W_v$).
    * Perform SVD to get their top $N$ principal components (e.g., $N=64$):
        * `q_pre_trained`, `k_pre_trained`, `v_pre_trained`

2.  **Get "Adapted" Primitives:**
    * Load your *finetuned checkpoint's* adapted backbone weights ($W'_q, W'_k, W'_v$, where $W' = W + \Delta W_{lora}$).
    * Perform SVD to get their top $N$ principal components:
        * `q_adapted`, `k_adapted`, `v_adapted`

3.  **Calculate Score:**
    * Simulate the principal *text-based* attention output for both models:
        * `y_text_pre_trained = Attention(q_pre_trained, k_pre_trained, v_pre_trained)`
        * `y_text_adapted = Attention(q_adapted, k_adapted, v_adapted)`
    * **`CPD_Score = CosineSimilarity(y_text_pre_trained, y_text_adapted)`**

---

## 2. Metric 2: "Conditional Disparity Score"

* **Purpose:** Measures if the conditional adapter has *learned* its task. It quantifies the separation between the adapter's outputs for the two extreme conditions (e.g., max bokeh vs. max sharp).
* **Interpretation:**
    * `Disparity_Score` $\approx$ 1.0: **Good.** The adapter is trained and can produce two highly distinct effects.
    * `Disparity_Score` $\approx$ 0.0: **Bad.** The adapter is untrained or has collapsed (e.g., outputs the same thing for all conditions).

### Algorithm (Data-Free)

1.  **Get Primitives:**
    * Load your finetuned checkpoint's adapted query weights ($W'_q$) and get the top $N$ principal queries, `q_test`.
    * Load your finetuned `MLP_cond` (`FpsConditioning`) and `Adapter_cond` (`FPSCrossAttentionAdapter`).

2.  **Generate Extreme Signals:**
    * **Ultra-Sharp Signal (`c = LOW_COND`):**
        * `e_cond_sharp = MLP_cond(c = LOW_COND)`
        * `K_sharp, V_sharp = Adapter_cond(e_cond_sharp)`
        * `y_cond_sharp = Attention(q_test, K_sharp, V_sharp)`
    * **Ultra-Bokeh Signal (`c = HIGH_COND`):**
        * `e_cond_bokeh = MLP_cond(c = HIGH_COND)`
        * `K_bokeh, V_bokeh = Adapter_cond(e_cond_bokeh)`
        * `y_cond_bokeh = Attention(q_test, K_bokeh, V_bokeh)`

3.  **Calculate Score:**
    * **`Disparity_Score = 1.0 - CosineSimilarity(y_cond_sharp, y_cond_bokeh)`**

---

## 3. Finding the "Sweet Point" (How to Use These Scores)

You must use **both** scores together for early stopping.

1.  Save checkpoints at regular intervals (e.g., every 50-100 epochs).
2.  Run the analysis for both `CPD_Score` and `Disparity_Score` on each checkpoint.
3.  Plot the results on a single graph:

    * **`CPD_Score` (Backbone Health):** Starts at 1.0. You want it to **stay high**. A drop means overfitting.
    * **`Disparity_Score` (Condition Learning):** Starts at 0.0. You want it to **rise and plateau**.



**The "Sweet Point" is the epoch where the `Disparity_Score` has reached its maximum plateau, AND the `CPD_Score` is still at its maximum (before it starts to drop).**

---

## 4. Supporting Heuristics

These metrics (from your paper) are also useful for confirming your analysis:

* **Intruder Count (from Alg. 2):**
    * **Purpose:** Measures "destructive forgetting."
    * **Metric:** `$N_{intruders}$`. Should be low, ideally 0.
    * **Use:** Plot `Epoch` vs. `Intruder Count`. This plot should mirror the `CPD_Score` plot (when intruders go up, CPD score goes down).

* **Effective Rank of $y_{cond}$ (from Alg. 3):**
    * [cite_start]**Purpose:** Checks for the "Bulldozer Effect" (memorizing content). [cite: 576-577]
    * **Metric:** `$\mathcal{R}_{cond}$`. Should be very low ($\approx$ 1.0).
    * **Use:** Plot `Epoch` vs. `Effective Rank`. If this starts to climb, the adapter is memorizing content.

* **Magnitude Ratio (from Table 1):**
    * [cite_start]**Purpose:** Checks for the "Bulldozer Effect" (domination). [cite: 491]
    * **Metric:** `Magnitude(y_cond) / Magnitude(y_text)`. Should be very small.

---

## 5. Dataset Heuristic

* **Purpose:** To quantify the complexity of a training dataset *before* training to balance diversity and simplicity.
* **Metric:** "Pre-Training Content Rank"
* **Algorithm:**
    1.  Encode all training frames using a pre-trained encoder (e.g., VAE or CLIP).
    2.  Stack all embeddings into a single matrix.
    3.  Compute the **Effective Rank** of this matrix.
* **Interpretation:**
    * [cite_start]**Too Low (e.g., 1-5):** "Practically limited," won't generalize. [cite: 342]
    * [cite_start]**Too High (e.g., 50+):** High risk of "catastrophic forgetting." [cite: 343]
    * [cite_start]**"Sweet Spot" (e.g., 10-20):** Good balance of diversity and simplicity. [cite: 336]