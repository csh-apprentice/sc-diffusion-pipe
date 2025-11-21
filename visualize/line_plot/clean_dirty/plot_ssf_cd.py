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
STEPS_PER_EPOCH = 18 # Updated
EPOCHS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Raw data
shutter_dirty_raw = [0.9718, 0.9652, 0.9619, 0.9597, 0.9583, 0.9569, 0.9538, 0.9538, 0.9472, 0.9490]
shutter_clean_raw = [0.9945, 0.9937, 0.9930, 0.9915, 0.9905, 0.9891, 0.9883, 0.9867, 0.9857, 0.9844]
aperture_dirty_raw = [0.9280, 0.9230, 0.9165, 0.9102, 0.9087, 0.9034, 0.8940, 0.8995, 0.8966, 0.8896]
aperture_clean_raw = [0.9914, 0.9878, 0.9846, 0.9814, 0.9786, 0.9750, 0.9719, 0.9687, 0.9659, 0.9632]
temp_dirty_raw = [0.9465, 0.9440, 0.9408, 0.9352, 0.9295, 0.9277, 0.9308, 0.9296, 0.9269, 0.9213]
temp_clean_raw = [0.9920, 0.9896, 0.9878, 0.9858, 0.9834, 0.9804, 0.9800, 0.9781, 0.9779, 0.9762]

# --- 3. Prepare Data for Plotting ---

# Calculate x-axis in "k steps" (e.g., 180k)
# We divide by 1000 to make the x-axis labels cleaner
steps_k = [e * STEPS_PER_EPOCH / 1000 for e in EPOCHS] # Now goes 18, 36, ..., 180

# Prepend the Step 0 (with SSF = 1.0) to all data
plot_steps = [0] + steps_k
shutter_clean = [1.0] + shutter_clean_raw
shutter_dirty = [1.0] + shutter_dirty_raw
aperture_clean = [1.0] + aperture_clean_raw
aperture_dirty = [1.0] + aperture_dirty_raw
temp_clean = [1.0] + temp_clean_raw
temp_dirty = [1.0] + temp_dirty_raw

# --- 4. Create the Plot ---

# Create a figure with a 4:3 aspect ratio
# (e.g., 6 inches wide, 4.5 inches tall)
fig, ax = plt.subplots(figsize=(6, 4.5))

# Plot the 6 data lines
# Using 'tab:blue', 'tab:orange', 'tab:green' for clear, distinct colors.
# 'Clean' = solid, 'Dirty' = dashed

# Shutter (Blue, Circle marker)
ax.plot(plot_steps, shutter_clean, label='Shutter (Clean)', color='tab:blue', linestyle='solid', marker='o', markersize=5)
ax.plot(plot_steps, shutter_dirty, label='Shutter (Dirty)', color='tab:blue', linestyle='dashed', marker='o', markersize=5)

# Aperture (Orange, Square marker)
ax.plot(plot_steps, aperture_clean, label='Aperture (Clean)', color='tab:orange', linestyle='solid', marker='s', markersize=5)
ax.plot(plot_steps, aperture_dirty, label='Aperture (Dirty)', color='tab:orange', linestyle='dashed', marker='s', markersize=5)

# Temperature (Green, Triangle marker)
ax.plot(plot_steps, temp_clean, label='Temperature (Clean)', color='tab:green', linestyle='solid', marker='^', markersize=5)
ax.plot(plot_steps, temp_dirty, label='Temperature (Dirty)', color='tab:green', linestyle='dashed', marker='^', markersize=5)

# Plot the baseline
ax.axhline(y=BASELINE, color='black', linestyle='dotted', label='Baseline (SSF)')

# --- 5. Final Styling & Saving ---

# Set labels (no title as requested)
ax.set_xlabel('Steps (k)')
ax.set_ylabel('SSF Score')

# Set x-axis ticks to be explicit and clean for the new range
# *** NEW X-AXIS TICKS AND LABELS FOR 0k to 18.0k range ***
# Set x-axis ticks explicitly
tick_values = np.linspace(0, 18, 6) # [0.0, 3.6, 7.2, 10.8, 14.4, 18.0]
ax.set_xticks(tick_values)
# Format labels to show one decimal place for clarity, e.g., '3.6', '7.2', etc.
ax.set_xticklabels([f'{x:.1f}' for x in tick_values])
# Set X-axis limit to the maximum calculated step value
ax.set_xlim(left=0, right=18.0)

# Set y-axis limits to give a good view of the data
# Find min and add some padding
data_min = min(aperture_dirty_raw + temp_dirty_raw) # Find the absolute min
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
plt.savefig("ssf_score_clean_dirty_vs_steps.pdf", bbox_inches='tight')

# Also save as a high-DPI PNG for quick viewing
plt.savefig("ssf_score_clean_dirty_vs_steps.png", dpi=300, bbox_inches='tight')

# Show the plot
print("Plot saved as 'ssf_score_clean_dirty_vs_steps.pdf' and 'ssf_score_clean_dirty_vs_steps.png'")
plt.show()