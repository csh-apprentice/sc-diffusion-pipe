The Plan: The "Backbone Intruder Dimension" Check
This analysis directly measures how much the final adapted backbone has spectrally deviated from the original pre-trained backbone.

Objective: To prove that the base LoRA update ($\Delta W_{lora}$) primarily makes small perturbations to the existing singular vectors of the pre-trained weights, rather than introducing new, high-ranking, and dissimilar singular vectors ("intruder dimensions"). A low number of intruders is the mathematical signature of a "clean," non-destructive fine-tuning process that avoids catastrophic forgetting.

Methodology:

This plan is a direct application of Algorithm 1 from the paper, but applied specifically to your backbone weights.

1. Gather the Weights: For this test, you need two sets of weight matrices. For each adapted attention block (e.g., 27, 33, 39), you will extract the same weight matrix (e.g., cross_attn.q.weight).

W_pre: The weight matrix from the original, unmodified Wan 2.1 checkpoint.

W_lora: The final weight matrix from your fully trained model. This weight is the sum of the original pre-trained weight and your learned base LoRA update ($W_{lora} = W_{pre} + \text{lora_B} @ \text{lora_A}$).

2. Compute SVD for Both:

Calculate the SVD of the original weight to get its basis of singular vectors: $W_{pre} = U_{pre} S_{pre} V_{pre}^T$.

Calculate the SVD of the final adapted weight to get its new basis of singular vectors: $W_{lora} = U_{lora} S_{lora} V_{lora}^T$.

3. Search for Intruders:

Iterate through the top k (e.g., k=64) most important singular vectors from the final adapted backbone (the columns of $U_{lora}$).

For each vector $u_{lora_j}$, calculate its maximum cosine similarity to any of the original singular vectors in $U_{pre}$.

max_similarity = max_i( |cos(u_lora_j, u_pre_i)| )

4. Count the Intruders:

If max_similarity is below a certain threshold $\epsilon$ (e.g., 0.5), then $u_{lora_j}$ is an intruder dimension. It represents a new, powerful direction that did not exist in the original model.

Count the total number of intruders found in the top k vectors.

Interpretation of Results
Success (Low Intruder Count, e.g., 0-2): This is the ideal outcome. It proves that your base LoRA learned its new condition=0 baseline by making only small adjustments to the pre-existing spectral structure. It "re-tuned" the existing knowledge rather than overwriting it. This is a clean, non-destructive update that preserves the model's general capabilities.

Failure (High Intruder Count): This would be a red flag. It would indicate that in order to learn the condition=0 baseline for your synthetic data, the backbone LoRA was forced to overfit and learn entirely new, powerful features specific to that simple dataset. These intruders are the mathematical signature of catastrophic forgetting, and they would likely harm the model's performance on general, out-of-distribution prompts.