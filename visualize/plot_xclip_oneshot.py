#!/usr/bin/env python3
# CVPR-style grouped bar chart (title top, legend bottom, clean serif aesthetic)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- Data ----------------
tasks = ["Shutter", "Aperture", "Temperature"]
baseline = np.array([0.522, 0.522, 0.522])
synthetic = np.array([0.352, 0.343, 0.431])
real      = np.array([0.096, 0.021, 0.281])

drop_syn  = (baseline - synthetic) / baseline * 100.0
drop_real = (baseline - real)      / baseline * 100.0

# -------------- Style -----------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "PT Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,  # embed TrueType for CVPR PDF
    "ps.fonttype": 42,
})

# Classic muted palette (neutral gray, academic blue, brick red)
C_BASE = "#8A8A8A"
C_SYN  = "#2B6CA3"
C_REAL = "#A33A2B"

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)
x = np.arange(len(tasks))
w = 0.24

b1 = ax.bar(x - w, baseline,  width=w, color=C_BASE, edgecolor="black", linewidth=0.8, label="Baseline (Wan)")
b2 = ax.bar(x + 0.00, synthetic, width=w, color=C_SYN,  edgecolor="black", linewidth=0.8, label="Synthetic")
b3 = ax.bar(x + w, real,      width=w, color=C_REAL, edgecolor="black", linewidth=0.8, label="Real")

# Axes & labels
ax.set_title("VQA Score by Condition and Domain", pad=20, loc="center")
ax.set_ylabel("VQA Score")
ax.set_xticks(x, tasks)
ax.set_ylim(0, 0.65)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)
ax.yaxis.grid(True, linestyle="-", linewidth=0.45, alpha=0.18)
ax.set_axisbelow(True)

# ---------- Smart annotations ----------
def annotate(bars, pct=None, color="#222", y_gap_pt=6, two_lines=False):
    """Use annotate with point offsets to prevent overlaps"""
    for i, r in enumerate(bars):
        h  = r.get_height()
        xc = r.get_x() + r.get_width()/2
        if pct is None or not two_lines:
            txt = f"{h:.3f}"
        else:
            txt = f"{h:.3f}\n(-{pct[i]:.1f}%)"
        ax.annotate(
            txt, xy=(xc, h), xytext=(0, y_gap_pt),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, color=color, linespacing=1.25,
            clip_on=False
        )

annotate(b1, color="#222", y_gap_pt=4, two_lines=False)
annotate(b2, pct=drop_syn,  color="#1f3f66", y_gap_pt=6, two_lines=True)
annotate(b3, pct=drop_real, color="#6b1e14", y_gap_pt=6, two_lines=True)

# --- Legend in a single row at the very bottom ---
leg = ax.legend(
    ncols=3,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    handlelength=1.5,
    columnspacing=1.0,
    fontsize=11
)

# --- Adjust layout so nothing overlaps ---
plt.subplots_adjust(bottom=0.28, top=0.88)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

# ---------------- Save ----------------
fig.savefig("vqa_grouped_final.pdf", bbox_inches="tight")
fig.savefig("vqa_grouped_final.png", dpi=300, bbox_inches="tight")
print("Saved: vqa_grouped_final.pdf, vqa_grouped_final.png")
