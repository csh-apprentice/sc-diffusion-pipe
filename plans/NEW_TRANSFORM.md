We need to design some new transform in the compute_tau_rel function in models/wan/model.py:

can you add this new transform?


transform = "centerlog1p"
y=sign(reference_fps-x)*torch.log1p(abs(tau-1))


noted that after you add this code, you should also make  sure
transform option configurable, check the code if you need to add new logic.