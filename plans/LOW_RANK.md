Analysis 2: Low-Rank Representation Check

This analysis verifies that the adapter has learned an efficient and compact representation for the conditional effect. It measures the "effective dimensionality" of the signal the adapter can produce.

Methodology: The "Effective Rank" Test

Generate the Conditional Value Tensor:

For a strong condition (e.g., c=1), use the trained adapter to deterministically generate the value tensor v_fps(c). This tensor has a shape like [num_tokens, num_heads, head_dim].

Reshape for SVD:

Flatten the v_fps(c) tensor into a 2D matrix, V_mat, of shape [num_tokens * num_heads, head_dim]. The rows of this matrix represent all the possible "instruction" vectors the adapter can contribute to the attention output.

Compute Singular Value Spectrum:

Perform an SVD on V_mat to obtain its singular values, S. These values, sorted in descending order, represent the magnitude of each principal direction within the adapter's output space.

Analyze the Spectrum:

Plot: Create a scree plot of the singular values to visualize their decay.

Calculate Effective Rank: Compute the effective rank by counting the number of singular values above a relative threshold (e.g., greater than 1% of the largest singular value).

Interpretation

Success (Low Effective Rank): The scree plot shows a sharp "elbow," and the calculated effective rank is a small number (e.g., 1-4). This is the ideal outcome. It proves the model has learned the "essence" of the conditional effect in a low-dimensional, efficient representation, which is a strong indicator of good generalization.

Failure (High Effective Rank): The singular values decay slowly, and the effective rank is high. This suggests the adapter has learned a noisy, complex, and high-dimensional representation, likely due to overfitting on the training data. This indicates a higher risk of the adapter producing brittle or artifact-prone results.