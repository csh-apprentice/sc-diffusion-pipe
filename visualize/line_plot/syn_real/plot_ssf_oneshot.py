import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines # Need this for the handle check

# --- 1. Style Setup (for CVPR / Bar Plot Match) ---

# Set font to serif to match LaTeX paper.
# We just request 'serif' and let Matplotlib find the best available
# one on the system, like DejaVu Serif. This avoids font warnings.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,  # Base font size
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# --- 2. Data Definition ---

# Constants from your data
BASELINE = 0.9530
STEPS_PER_EPOCH = 80
EPOCHS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Raw data
shutter_syn_raw = [0.9834, 0.9799, 0.9752, 0.9755, 0.9708, 0.9670, 0.9635, 0.9564, 0.9517, 0.9489]
shutter_real_raw = [0.9598, 0.9469, 0.9434, 0.9323, 0.9273, 0.9191, 0.9184, 0.9155, 0.9130, 0.9116]
bokeh_syn_raw = [0.9720, 0.9625, 0.9578, 0.9472, 0.9419, 0.9395, 0.9366, 0.9310, 0.9255, 0.9232]
bokeh_real_raw = [0.9010, 0.8750, 0.8571, 0.8514, 0.8436, 0.8380, 0.8367, 0.8329, 0.8291, 0.8233]
temp_syn_raw = [0.9720, 0.9706, 0.9700, 0.9616, 0.9634, 0.9581, 0.9529, 0.9516, 0.9493, 0.9491]
temp_real_raw = [0.9365, 0.9200, 0.9093, 0.9026, 0.8963, 0.8924, 0.8912, 0.8914, 0.8894, 0.8843]

# --- 3. Prepare Data for Plotting ---

# Calculate x-axis in "k steps" (e.g., 80k)
# We divide by 1000 to make the x-axis labels cleaner (e.g., "80" instead of "80000")
steps_k = [e * STEPS_PER_EPOCH / 1000 for e in EPOCHS]

# Prepend the Step 0 (with SSF = 1.0) to all data
plot_steps = [0] + steps_k
shutter_syn = [1.0] + shutter_syn_raw
shutter_real = [1.0] + shutter_real_raw
bokeh_syn = [1.0] + bokeh_syn_raw
bokeh_real = [1.0] + bokeh_real_raw
temp_syn = [1.0] + temp_syn_raw
temp_real = [1.0] + temp_real_raw

# --- 4. Create the Plot ---

# Create a figure with a 4:3 aspect ratio
# (e.g., 6 inches wide, 4.5 inches tall)
fig, ax = plt.subplots(figsize=(6, 4.5))

# Plot the 6 data lines
# Using 'tab:blue', 'tab:orange', 'tab:green' for clear, distinct colors.

# Shutter (Blue, Circle marker)
ax.plot(plot_steps, shutter_syn, label='Shutter (Syn)', color='tab:blue', linestyle='solid', marker='o', markersize=5)
ax.plot(plot_steps, shutter_real, label='Shutter (Real)', color='tab:blue', linestyle='dashed', marker='o', markersize=5)

# Bokeh (Orange, Square marker)
ax.plot(plot_steps, bokeh_syn, label='Aperture (Syn)', color='tab:orange', linestyle='solid', marker='s', markersize=5)
ax.plot(plot_steps, bokeh_real, label='Aperture (Real)', color='tab:orange', linestyle='dashed', marker='s', markersize=5)

# Temperature (Green, Triangle marker)
ax.plot(plot_steps, temp_syn, label='Temperature (Syn)', color='tab:green', linestyle='solid', marker='^', markersize=5)
ax.plot(plot_steps, temp_real, label='Temperature (Real)', color='tab:green', linestyle='dashed', marker='^', markersize=5)

# Plot the baseline
ax.axhline(y=BASELINE, color='black', linestyle='dotted', label='Baseline (SSF)')

# --- 5. Final Styling & Saving ---

# Set labels (no title as requested)
ax.set_xlabel('Steps (k)')
ax.set_ylabel('SSF Score')

# Set x-axis ticks to be explicit and clean
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels(['0', '20k', '40k', '60k', '80k'])

# Set y-axis limits to give a good view of the data
# Find min and add some padding
data_min = min(bokeh_real_raw + temp_real_raw) # Find the absolute min
ax.set_ylim(bottom=data_min - 0.02, top=1.02) # Start just below min, end just above 1.0

# Add a legend
# ax.legend(loc='lower left') # Old call

# --- New Legend Logic (Markers on plot, not in legend) ---
# Get handles and labels
handles, labels = ax.get_legend_handles_labels()

# Create new handles for the legend, copying properties but removing markers
new_handles = []
for h in handles:
    # Check if it's a Line2D object (which our plots are)
    if isinstance(h, mlines.Line2D):
        # Create a new Line2D object for the legend
        # Copy color, linestyle, linewidth
        # Set marker to 'None' (empty string) to hide it
        new_handle = mlines.Line2D([], [],
                                color=h.get_color(),
                                linestyle=h.get_linestyle(),
                                linewidth=h.get_linewidth(),
                                marker='None') # This is the key change
        new_handles.append(new_handle)
    else:
        # For other handle types (like the baseline's axhline), just pass them through
        new_handles.append(h)

# Add the new legend
ax.legend(handles=new_handles, labels=labels, loc='lower left')
# --- End New Legend Logic ---


# Add a light grid for readability
ax.grid(True, linestyle=':', alpha=0.7)

# Use tight_layout to ensure labels don't get cut off when saving
plt.tight_layout()

# Save the figure as a PDF (vector format, best for LaTeX)
plt.savefig("ssf_score_vs_steps.pdf", bbox_inches='tight')

# Also save as a high-DPI PNG for quick viewing
plt.savefig("ssf_score_vs_steps.png", dpi=300, bbox_inches='tight')

# Show the plot
print("Plot saved as 'ssf_score_vs_steps.pdf' and 'ssf_score_vs_steps.png'")
plt.show()