Task 1: Dataset Hook: Help me to extend the codebase so now it supports reading datasets with fps. Also you should store that fps somewhere in the class as it is goona to use for future model injection.

Task 2: Conditioning Encoder: Turn the scalar fps (or tau_rel = 240 / fps) from the batch into a learned conditioning embedding c_fps and thread it through your Wan2.1 forward pass so later (Step 3) we can inject it into cross-attention without refactors. You should write tau_rel into a seperate function since we may have differnent mapping rather than 240/ fps later,  Map scalar tau_rel → conditioning embedding vector can use a simple mlp like this:
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim)
        )