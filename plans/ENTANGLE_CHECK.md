Experimental Plan: Quantifying Functional Entanglement in the Cross-Attention Adapter


You can work from this codebase: /root/workspace/sc-diffusion-pipe/inference/test_fps_multiple_experiments_align_old.py

Objective:
To empirically measure the degree of functional entanglement between the primary text-conditional pathway (y_text) and the FPS-conditional pathway (y_fps) within our modified attention blocks. This analysis aims to verify the hypothesis that context drift in generated videos correlates with a higher degree of alignment (i.e., lower orthogonality) between these two signal streams.

Methodology:
We will instrument the inference pipeline to capture key metrics from the internal state of each FPSCrossAttentionAdapter at every step of the denoising process. For a given input prompt and a fixed FPS condition, we will perform the following analysis:

1. Instrumentation and Data Capture:
The FPSCrossAttentionAdapter module will be modified to intercept the two intermediate activation tensors immediately before they are combined:

$y_{text}$: The output of the standard text cross-attention, representing the content signal.

$y_{fps}$: The output of the FPS-conditional cross-attention, representing the motion blur signal.

During a full denoising sequence of $T$ steps, we will log these tensors for each of the $N_{blocks}$ adapted attention blocks. Let $y_{text}^{(i,j)}$ and $y_{fps}^{(i,j)}$ denote the tensors captured at denoising step $i$ (where $i \in \{1, ..., T\}$) from the adapter in block $j$ (where $j \in \{1, ..., N_{blocks}\}$).

2. Metric Calculation:
For each captured pair ($y_{text}^{(i,j)}$, $y_{fps}^{(i,j)}$), we will compute two key metrics:

Signal Magnitude ($M$): The total energy or influence of each pathway, measured by the Frobenius norm of the activation tensor.

$M_{text}^{(i,j)} = ||y_{text}^{(i,j)}||_F$

$M_{fps}^{(i,j)} = ||y_{fps}^{(i,j)}||_F$

Alignment Score ($S$): The functional overlap between the two pathways, measured by the average cosine similarity between their corresponding token-level feature vectors. After reshaping the tensors to [tokens, features], the score is defined as:

$S_{ij} = \mathbb{E}[\cos(\text{normalize}(y_{text}^{(i,j)}), \text{normalize}(y_{fps}^{(i,j)}))]$

3. Aggregation and Final Output:
After completing the full inference loop for a single prompt, we will aggregate the collected data to analyze the model's behavior over time. For each denoising step $i$, we will compute the expected value (average) of our metrics across all $N_{blocks}$ adapted blocks:

Average Alignment per Step ($S_i$):
$S_i = \frac{1}{N_{blocks}} \sum_{j=1}^{N_{blocks}} S_{ij}$

Average Magnitude per Step ($M^{(i)}$):
$M_{text}^{(i)} = \frac{1}{N_{blocks}} \sum_{j=1}^{N_{blocks}} M_{text}^{(i,j)}$
$M_{fps}^{(i)} = \frac{1}{N_{blocks}} \sum_{j=1}^{N_{blocks}} M_{fps}^{(i,j)}$

The final output of this analysis for a given prompt will be a time-series table and corresponding plots illustrating how the average alignment ($S_i$) and the average magnitudes ($M_{text}^{(i)}$, $M_{fps}^{(i)}$) evolve across the denoising steps.

Interpretation:
This procedure will allow us to characterize the adapter's behavior quantitatively. A successful, disentangled model is expected to exhibit a consistently low alignment score ($S_i \approx 0$) across all denoising steps, indicating functional orthogonality. Conversely, prompts that result in context drift are hypothesized to show a significantly higher alignment score, providing a quantitative link between feature entanglement and generation artifacts.