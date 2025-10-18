Experimental Plan C: Data-Free Analysis via Conditional Subspace Characterization

You can work from this codebase: /root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align_old.py

Objective:
To empirically measure the properties of the FPS-conditional pathway by directly analyzing the geometric properties of its potential output space. This data-free approach aims to characterize the adapter's learned function and its separation from the backbone's functional space.

Methodology:
This plan is based on the key insight that for any given query q, the output y_fps = Attention(q, k_fps, v_fps) is a linear combination of the value vectors in v_fps. Therefore, all possible outputs of the conditional branch are confined to the linear subspace spanned by these value vectors. We will pre-calculate and analyze this "conditional subspace" for each FPS condition.

1. Conditional Subspace Generation:
For a set of discrete FPS conditions, $c \in \{c_1, c_2, ...\}$, we will perform the following pre-calculation for each adapted attention block $j$:

Generate the deterministic value tensor $v_{fps}^{(j)}(c)$ using the adapter's projection layers.

Reshape $v_{fps}^{(j)}(c)$ (which has shape [Num_Tokens, Num_Heads, Head_Dim]) into a matrix $V_{c}^{(j)}$ whose columns form an orthonormal basis for the conditional subspace for that block and condition. This can be done via SVD or QR decomposition.

2. Direct Analysis of the Conditional Subspace:
Once we have the basis matrices $V_{c}^{(j)}$ for each condition and block, we can perform the following data-free analyses:

Effective Dimensionality: We will compute the SVD of $v_{fps}^{(j)}(c)$ and analyze its singular value spectrum. The effective rank will tell us the dimensionality of the conditional information. A low effective rank implies the adapter has learned a compact representation for the motion blur effect.

Orthogonality Between Conditions: For two different conditions $c_1$ and $c_2$, we will measure the geometric alignment between their respective subspaces, defined by $V_{c1}^{(j)}$ and $V_{c2}^{(j)}$. We will compute the principal angles between these subspaces.

Interpretation: Large angles (close to 90 degrees) would prove that the adapter has learned to represent different conditions in distinct, non-interfering subspaces, indicating a clean separation of effects.

Alignment with Backbone Functional Space: To measure potential entanglement with the backbone, we will compare the conditional subspace with the dominant output directions of the backbone's attention blocks.

First, we will perform an SVD on the backbone's output projection matrix (o.weight) to find its top left singular vectors, which form a basis for its primary output subspace, $U_{backbone}^{(j)}$.

Then, we will compute the principal angles between the conditional subspace $V_{c}^{(j)}$ and the backbone's output subspace $U_{backbone}^{(j)}$.

Interpretation: Consistently large angles would suggest that the adapter's outputs are structurally orthogonal to the backbone's primary outputs, providing strong, data-free evidence of functional disentanglement. Small angles would indicate a higher risk of feature-space collision and context drift.

Advantages of this Plan:
This approach provides a rigorous, quantitative, and geometrically intuitive way to analyze the adapter's learned behavior. By pre-calculating and saving the basis matrices for each condition, we can perform these analyses offline, entirely removing the dependency on sampling prompts and running full inference loops.