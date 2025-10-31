# Subspace Analysis - Quick Start Guide

## 🚀 One Command to Rule Them All

```bash
bash bash/run_subspace_analysis.sh \
    fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
    outputs/checkpoint/step_10000 \
    12 24 60 120 240
```

**That's it!** Results in `output/subspace_analysis_step_10000/`

---

## 📊 What You Get

```
output/subspace_analysis_step_10000/
├── summary_report.txt                  ← Read this first!
├── dimensionality_analysis.png         ← Conditioning efficiency
├── orthogonality_heatmaps.png          ← Condition separation
├── backbone_alignment.png              ← Entanglement analysis
├── subspace_data.pkl                   ← Raw data
└── *.csv files                         ← Detailed statistics
```

---

## 🎯 Quick Interpretation

### Read the summary report:
```bash
cat output/subspace_analysis_step_10000/summary_report.txt
```

### Look for these:

**✅ Good Conditioning:**
```
Mean rank ratio: 0.25         → Compact (efficient)
Mean pairwise angle: 72°      → Strong separation
Mean minimum angle: 68°       → Disentangled
```

**❌ Bad Conditioning:**
```
Mean rank ratio: 0.85         → Redundant
Mean pairwise angle: 25°      → Weak separation
Mean minimum angle: 22°       → Entangled
```

---

## 🔧 Common Scenarios

### Scenario 1: Multi-Condition Model
```bash
bash bash/run_subspace_analysis.sh \
    shutter_bokeh_TOML/wan_SC_TARGET_14B_2SHAPES_SHUTTER_BOKEH.toml \
    outputs/multi_cond/step_5000 \
    0.5 0.02  1.0 0.05  0.125 0.1
    # Auto-groups as: [0.5, 0.02], [1.0, 0.05], [0.125, 0.1]
```

### Scenario 2: Fast Analysis (Skip Backbone)
```bash
# Step 1: Extract subspaces
python inference/analyze_conditional_subspace.py \
    --config <config.toml> \
    --checkpoint <checkpoint_path> \
    --fps_values 12 24 60 \
    --output_dir output/fast_analysis

# Step 2: Analyze (no pipeline loading)
python inference/analyze_conditional_geometry.py \
    --subspace_file output/fast_analysis/subspace_data.pkl \
    --skip_backbone \
    --output_dir output/fast_analysis
```

### Scenario 3: Compare Checkpoints
```bash
for step in 1000 5000 10000 15000; do
    bash bash/run_subspace_analysis.sh \
        fps_TOML/wan_SC_TARGET_14B_FPS_SHAPE_BLUR.toml \
        outputs/checkpoint/step_${step} \
        12 24 60
done

# Compare summary reports:
grep "Mean rank ratio" output/subspace_analysis_*/summary_report.txt
```

---

## 📚 Need More Help?

- **Full documentation:** `inference/README_SUBSPACE_ANALYSIS.md`
- **Implementation plan:** `summary/CONDITION_EVAL_IMPLEMENTATION_PLAN.md`
- **Implementation summary:** `summary/SUBSPACE_ANALYSIS_IMPLEMENTATION_SUMMARY.md`

---

## 🐛 Troubleshooting

**"No FPS adapters found"**
```bash
# Check if checkpoint has FPS parameters
python -c "import safetensors; \
ckpt = safetensors.torch.load_file('checkpoint.safetensors'); \
print([k for k in ckpt.keys() if 'fps' in k.lower()][:5])"
```

**Out of memory**
```bash
# Use CPU or skip backbone
python inference/analyze_conditional_geometry.py \
    --subspace_file <file.pkl> \
    --skip_backbone \
    --output_dir <output>
```

**Analysis is slow**
```bash
# Analyze fewer conditions
bash bash/run_subspace_analysis.sh \
    <config> <checkpoint> \
    12 60 240  # Only 3 conditions instead of 5
```

---

## 💡 Pro Tips

1. **Start small**: Test with 3-4 conditions first
2. **Skip backbone**: Saves time, backbone analysis often optional
3. **Check summary first**: Don't dive into CSVs until you read `summary_report.txt`
4. **Compare checkpoints**: Use this to pick best checkpoint geometrically
5. **Integrate training**: Run every 1000 steps to monitor conditioning health

---

**Questions?** Check `inference/README_SUBSPACE_ANALYSIS.md` for detailed guide!
