It may also worth to look at how our singular value specturm changes in y_text, just like what we did in  /root/workspace/sc-diffusion-pipe/bash/run_principal_orthogonality.sh, but instead of comparing the y_text and y_fps orthagnality & singular values.

I want you to compare between the clean y_fps (clean backbone) and our finetuned y_fps (+lora backbone):

1. Similarity Matrix, 
Iterate through the top k (e.g., k=64) most important y_{text_lora}

For each vector $y_{text_lora}$, calculate its maximum cosine similarity to any of the original y_text in.

max_similarity = max_i( |cos(y_text_lora_j, y_text_i)| )

Also return me the heatmap like what we did in /root/workspace/sc-diffusion-pipe/bash/compute_similarity_matrix.sh, we only have one heatmap for text, not seperate it to k,q,v,o this time.

2. Singular Specture bird view,

I want to have a bird view of the singular values specture comparison of finetuned y_text and original y_text, show all the singular values in two lines on one plot for each block. Don't normalize since that's they should be roughly in same scale.