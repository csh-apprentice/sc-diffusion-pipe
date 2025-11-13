import matplotlib.pyplot as plt
import numpy as np

# --- 1. Style Setup (for CVPR / Bar Plot Match) ---

# Set font to serif to match LaTeX paper.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,  # Base font size
    "axes.labelsize": 10,
    "xtick.labelsize": 10, # Category labels should be readable
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# --- 2. Data Definition ---

# Categories and Colors
categories = ['Shutter', 'Aperture', 'Temperature']
colors = {
    'BASELINE': "#8A8A8A",
    'SYN': "#2B6CA3",
    'REAL': "#A33A2B"
}

# Data for each group
baseline_scores = [25.390, 25.390, 25.390]
syn_scores = [24.777, 25.105, 25.015]
real_scores = [23.278, 19.824, 24.456]

# --- 3. Create the Plot ---

# Create a figure with a 4:3 aspect ratio
fig, ax = plt.subplots(figsize=(6, 4.5))

# Set up bar positions
x = np.arange(len(categories))  # The label locations
bar_width = 0.25  # The width of the bars
bar_padding = 0.05 # Padding between bars

# Bar positions for each group
pos_base = x - (bar_width + bar_padding / 2)
pos_syn = x
pos_real = x + (bar_width + bar_padding / 2)

# Plot the bars
rects_base = ax.bar(pos_base, baseline_scores, bar_width, label='Baseline', color=colors['BASELINE'])
rects_syn = ax.bar(pos_syn, syn_scores, bar_width, label='Syn', color=colors['SYN'])
rects_real = ax.bar(pos_real, real_scores, bar_width, label='Real', color=colors['REAL'])

# Add bar labels on top of each bar, in the bar's color
# 'fmt' formats the number to 3 decimal places
# 'padding' adds a small space above the bar
# 'fontsize' is slightly smaller to prevent crowding
ax.bar_label(rects_base, fmt='%.3f', color=colors['BASELINE'], padding=3, fontsize=7)
ax.bar_label(rects_syn, fmt='%.3f', color=colors['SYN'], padding=3, fontsize=7)
ax.bar_label(rects_real, fmt='%.3f', color=colors['REAL'], padding=3, fontsize=7)

# --- 4. Final Styling & Saving ---

# Set labels (no title as requested)
ax.set_ylabel('XCLIP Score') # Updated Y-axis label

# Set x-axis ticks to be the category names
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Set y-axis limits
all_scores = baseline_scores + syn_scores + real_scores
# Increase top limit (e.g., * 1.2) to make space for the bar labels
# Also set a reasonable bottom limit (e.g., 0 or just below min)
min_score = min(all_scores)
max_score = max(all_scores)
ax.set_ylim(bottom=min_score * 0.9, top=max_score * 1.05) # Focus on the data range

# Add a legend
ax.legend(loc='upper right')

# Add a light horizontal grid (behind the bars)
ax.grid(True, axis='y', linestyle=':', alpha=0.7)
ax.set_axisbelow(True) # Send grid behind bars

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Use tight_layout to ensure labels don't get cut off when saving
plt.tight_layout()

# Save the figure as a PDF (vector format, best for LaTeX)
plt.savefig("xclip_score_bar_plot.pdf", bbox_inches='tight')

# Also save as a high-DPI PNG for quick viewing
plt.savefig("xclip_score_bar_plot.png", dpi=300, bbox_inches='tight')

# Show the plot
print("Plot saved as 'xclip_score_bar_plot.pdf' and 'xclip_score_bar_plot.png'")
plt.show()