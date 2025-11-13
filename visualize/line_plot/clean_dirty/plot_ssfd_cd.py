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
STEPS_PER_EPOCH = 180 # Updated
EPOCHS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Raw data
shutter_dirty_raw = [4.02, 4.86, 5.32, 5.66, 5.85, 6.17, 6.86, 6.86, 8.15, 8.01]
shutter_clean_raw = [0.86, 1.0, 1.12, 1.37, 1.56, 1.87, 1.99, 2.30, 2.50, 2.78]
aperture_dirty_raw = [12.01, 13.12, 15.15, 17.19, 17.59, 19.22, 22.34, 20.76, 21.66, 24.37]
aperture_clean_raw = [1.38, 2.02, 2.65, 3.33, 3.90, 4.68, 5.44, 6.22, 6.93, 7.65]
temp_dirty_raw = [8.49, 8.99, 9.50, 10.77, 11.17, 11.86, 11.49, 11.87, 12.52, 13.49]
temp_clean_raw = [1.32, 1.74, 2.08, 2.56, 3.10, 3.73, 3.85, 4.24, 4.21, 4.72]


# --- 3. Prepare Data for Plotting ---

# Calculate x-axis in "k steps" (e.g., 180k)
# We divide by 1000 to make the x-axis labels cleaner
steps_k = [e * STEPS_PER_EPOCH / 1000 for e in EPOCHS] # Now goes 18, 36, ..., 180

# Prepend the Step 0 (with SSFD = 0.0) to all data
plot_steps = [0] + steps_k
shutter_clean = [0.0] + shutter_clean_raw
shutter_dirty = [0.0] + shutter_dirty_raw
aperture_clean = [0.0] + aperture_clean_raw
aperture_dirty = [0.0] + aperture_dirty_raw
temp_clean = [0.0] + temp_clean_raw
temp_dirty = [0.0] + temp_dirty_raw

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
ax.axhline(y=BASELINE, color='black', linestyle='dotted', label='Baseline (SSFD)')

# --- 5. Final Styling & Saving ---

# Set labels (no title as requested)
ax.set_xlabel('Steps (k)')
ax.set_ylabel('SSFD Score')

# Set x-axis ticks to be explicit and clean for the new range
ax.set_xticks([0, 36, 72, 108, 144, 180])
ax.set_xticklabels(['0', '36k', '72k', '108k', '144k', '180k'])

# Set y-axis limits to give a good view of the data
# Start at 0, end just above the max value
data_max = max(aperture_dirty_raw)
ax.set_ylim(bottom=0, top=data_max * 1.05) # e.g., 0 to ~25

# Add a legend
# ax.legend(loc='upper left') # Old call

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
plt.savefig("ssfd_score_clean_dirty_vs_steps.pdf", bbox_inches='tight')

# Also save as a high-DPI PNG for quick viewing
plt.savefig("ssfd_score_clean_dirty_vs_steps.png", dpi=300, bbox_inches='tight')

# Show the plot
print("Plot saved as 'ssfd_score_clean_dirty_vs_steps.pdf' and 'ssfd_score_clean_dirty_vs_steps.png'")
plt.show()