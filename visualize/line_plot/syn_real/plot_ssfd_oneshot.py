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
BASELINE = 6.20
# *** CORRECTION APPLIED HERE: Changed from 80 to 8 ***
STEPS_PER_EPOCH = 8
EPOCHS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Raw data
shutter_syn_raw = [2.45, 3.05, 3.59, 3.67, 4.39, 5.01, 5.55, 6.78, 7.55, 8.09]
shutter_real_raw = [6.02, 8.54, 9.34, 11.56, 12.50, 14.19, 14.40, 15.05, 15.84, 16.31]
bokeh_syn_raw = [4.44, 6.12, 6.91, 8.90, 9.99, 10.62, 11.17, 12.37, 13.53, 14.15]
bokeh_real_raw = [23.91, 33.45, 40.52, 42.19, 45.57, 45.58, 48.78, 50.75, 52.02, 55.47]
temp_syn_raw = [4.43, 4.45, 4.47, 5.91, 5.70, 6.23, 7.20, 7.79, 8.19, 8.22]
temp_real_raw = [10.57, 14.06, 17.56, 20.15, 22.54, 24.71, 25.76, 26.12, 27.35, 29.08]

# --- 3. Prepare Data for Plotting ---

# Calculate x-axis in "k steps" (e.g., 0.8k, 1.6k, ..., 8.0k)
# We divide by 1000 to keep the x-axis values in thousands of steps
steps_k = [e * STEPS_PER_EPOCH / 1000 for e in EPOCHS]

# Prepend the Step 0 (with SSFD = 0.0) to all data
plot_steps = [0] + steps_k
shutter_syn = [0.0] + shutter_syn_raw
shutter_real = [0.0] + shutter_real_raw
bokeh_syn = [0.0] + bokeh_syn_raw
bokeh_real = [0.0] + bokeh_real_raw
temp_syn = [0.0] + temp_syn_raw
temp_real = [0.0] + temp_real_raw

# --- 4. Create the Plot ---

# Create a figure with a 4:3 aspect ratio
# (e.g., 6 inches wide, 4.5 inches tall)
fig, ax = plt.subplots(figsize=(6, 4.5))

# Plot the 6 data lines
# Using 'tab:blue', 'tab:orange', 'tab:green' for clear, distinct colors.

# Shutter (Blue, Circle marker)
ax.plot(plot_steps, shutter_syn, label='Shutter (Syn)', color='tab:blue', linestyle='solid', marker='o', markersize=5)
ax.plot(plot_steps, shutter_real, label='Shutter (Real)', color='tab:blue', linestyle='dashed', marker='o', markersize=5)

# Aperture (Orange, Square marker)
ax.plot(plot_steps, bokeh_syn, label='Aperture (Syn)', color='tab:orange', linestyle='solid', marker='s', markersize=5)
ax.plot(plot_steps, bokeh_real, label='Aperture (Real)', color='tab:orange', linestyle='dashed', marker='s', markersize=5)

# Temperature (Green, Triangle marker)
ax.plot(plot_steps, temp_syn, label='Temperature (Syn)', color='tab:green', linestyle='solid', marker='^', markersize=5)
ax.plot(plot_steps, temp_real, label='Temperature (Real)', color='tab:green', linestyle='dashed', marker='^', markersize=5)

# Plot the baseline
ax.axhline(y=BASELINE, color='black', linestyle='dotted', label='Baseline (SSFD)')

# --- 5. Final Styling & Saving ---

# Set labels (no title as requested)
ax.set_xlabel('Steps (k)')
ax.set_ylabel('SSFD Score')

# *** NEW X-AXIS TICKS AND LABELS FOR 0k to 8.0k range (8 steps per epoch) ***
# Set x-axis ticks explicitly
tick_values = np.linspace(0, 8, 5) # Generates [0.0, 2.0, 4.0, 6.0, 8.0]
ax.set_xticks(tick_values)
# Format labels to show one decimal place for clarity, e.g., '2.0', '4.0', etc.
ax.set_xticklabels([f'{x:.1f}' for x in tick_values])
# Set X-axis limit to the maximum calculated step value
ax.set_xlim(left=0, right=8.0)
# *******************************************************

# Set y-axis limits to give a good view of the data
# Start at 0, end just above the max value
data_max = max(bokeh_real_raw)
ax.set_ylim(bottom=0, top=data_max * 1.05) # e.g., 0 to ~60

# Add a legend
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
ax.legend(handles=new_handles, labels=labels, loc='upper left')
# --- End New Legend Logic ---


# Add a light grid for readability
ax.grid(True, linestyle=':', alpha=0.7)

# Use tight_layout to ensure labels don't get cut off when saving
plt.tight_layout()

# Save the figure as a PDF (vector format, best for LaTeX)
plt.savefig("ssfd_score_vs_steps.pdf", bbox_inches='tight')

# Also save as a high-DPI PNG for quick viewing
plt.savefig("ssfd_score_vs_steps.png", dpi=300, bbox_inches='tight')

# Show the plot
print("Plot saved as 'ssfd_score_vs_steps.pdf' and 'ssfd_score_vs_steps.png'")
plt.show()