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

# Categories and Colors (reusing from XCLIP as requested)
categories = ['Shutter', 'Aperture', 'Temperature']
colors = {
    'BASELINE': "#8A8A8A",
    'CLEAN': "#297373",  # Using SYN color for CLEAN
    'DIRTY': "#ff8552"   # Using REAL color for DIRTY
}

# Data for each group
baseline_scores = [0.522, 0.522, 0.522]
clean_scores = [0.521, 0.513, 0.532]
dirty_scores = [0.453, 0.427, 0.550]

# --- 3. Create the Plot ---

# Create a figure with a 4:3 aspect ratio
fig, ax = plt.subplots(figsize=(6, 4.5))

# Set up bar positions
x = np.arange(len(categories))  # The label locations
bar_width = 0.25  # The width of the bars
bar_padding = 0.05 # Padding between bars

# Bar positions for each group
pos_base = x - (bar_width + bar_padding / 2)
pos_clean = x
pos_dirty = x + (bar_width + bar_padding / 2)

# Plot the bars
rects_base = ax.bar(pos_base, baseline_scores, bar_width, label='Baseline', color=colors['BASELINE'])
rects_clean = ax.bar(pos_clean, clean_scores, bar_width, label='Clean', color=colors['CLEAN'])
rects_dirty = ax.bar(pos_dirty, dirty_scores, bar_width, label='Dirty', color=colors['DIRTY'])

# Add bar labels on top of each bar, in the bar's color
# 'fmt' formats the number to 3 decimal places
# 'padding' adds a small space above the bar
# 'fontsize' is slightly smaller to prevent crowding
ax.bar_label(rects_base, fmt='%.3f', color=colors['BASELINE'], padding=3, fontsize=7)
ax.bar_label(rects_clean, fmt='%.3f', color=colors['CLEAN'], padding=3, fontsize=7)
ax.bar_label(rects_dirty, fmt='%.3f', color=colors['DIRTY'], padding=3, fontsize=7)

# --- 4. Final Styling & Saving ---

# Set labels (no title as requested)
ax.set_ylabel('VQA Score') # Y-axis label

# Set x-axis ticks to be the category names
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Set y-axis limits
all_scores = baseline_scores + clean_scores + dirty_scores
# Increase top limit (e.g., * 1.3) to make space for the bar labels
ax.set_ylim(bottom=0, top=max(all_scores) * 1.3) # 0 to ~20% above max

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
plt.savefig("vqa_score_clean_dirty_bar_plot.pdf", bbox_inches='tight')

# Also save as a high-DPI PNG for quick viewing
plt.savefig("vqa_score_clean_dirty_bar_plot.png", dpi=300, bbox_inches='tight')

# Show the plot
print("Plot saved as 'vqa_score_clean_dirty_bar_plot.pdf' and 'vqa_score_clean_dirty_bar_plot.png'")
plt.show()