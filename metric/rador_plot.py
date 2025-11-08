import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors

def plot_comparison_radars(all_data, benchmark_data):
    """
    Creates a 1x2 radar plot for SSF and SS-FD scores, normalized
    against a benchmark.
    """
    
    # Get the category labels from the benchmark
    labels = list(benchmark_data.keys())
    num_vars = len(labels)
    
    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # --- Aesthetic Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use a high-quality colormap and cycle line styles
    colors = plt.cm.get_cmap('tab20', len(all_data))
    linestyles = ['-', '--', ':', '-.']
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(figsize=(22, 10), ncols=2, subplot_kw=dict(polar=True))
    
    # --- Plot 1: SSF (Normalized vs. Benchmark) ---
    ax1.set_title('SSF (Single-Step Fidelity)\nNormalized to Baseline (Higher is Better)', size=16, color='black', y=1.1)
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1], [label.capitalize() for label in labels], size=11)
    ax1.set_rlabel_position(0)
    # Set Y-limit to 1.25 to see models that *outperform* the baseline
    ax1.set_ylim(0, 1.25) 
    ax1.set_yticks([0.5, 1.0])
    ax1.set_yticklabels(["50%", "100% (Baseline)"], color="gray", size=10)
    ax1.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
    ax1.spines['polar'].set_visible(False) # Remove outer circle

    
    # --- Plot 2: SS-FD (Normalized vs. Benchmark) ---
    ax2.set_title('SS-FD (Single-Step Fréchet Distance)\nNormalized to Baseline (Higher is Better)', size=16, color='black', y=1.1)
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1], [label.capitalize() for label in labels], size=11)
    ax2.set_rlabel_position(0)
    ax2.set_ylim(0, 1.25) # Also set to 1.25
    ax2.set_yticks([0.5, 1.0])
    ax2.set_yticklabels(["50%", "100% (Baseline)"], color="gray", size=10)
    ax2.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
    ax2.spines['polar'].set_visible(False) # Remove outer circle
    
    # --- Plot data for each checkpoint ---
    for i, (checkpoint_name, categories) in enumerate(all_data.items()):
        color = colors(i % 20)
        linestyle = linestyles[i // 20 % len(linestyles)]
        
        # --- Normalize data against the benchmark ---
        # SSF: (checkpoint_score / baseline_score). > 1.0 is better.
        ssf_values = [categories[label]['ssf'] / benchmark_data[label]['ssf'] for label in labels]
        ssf_values += ssf_values[:1] # Complete the loop
        
        # SS-FD: (baseline_score / checkpoint_score). > 1.0 is better.
        ssfd_values = [benchmark_data[label]['ss_fd'] / categories[label]['ss_fd'] for label in labels]
        ssfd_values += ssfd_values[:1] # Complete the loop
        
        # Plot on ax1 (SSF)
        ax1.plot(angles, ssf_values, color=color, linewidth=2, linestyle=linestyle, label=checkpoint_name)
        ax1.fill(angles, ssf_values, color=color, alpha=0.1)
        
        # Plot on ax2 (SS-FD)
        ax2.plot(angles, ssfd_values, color=color, linewidth=2, linestyle=linestyle, label=checkpoint_name)
        ax2.fill(angles, ssfd_values, color=color, alpha=0.1)

    # Add a single legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(all_data) // 2 or 1, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout(pad=3.0)
    
    # Save and show
    output_filename = "checkpoint_comparison_radar_1x2.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"\nRadar plot saved to {output_filename}")
    plt.show()

def main():
    # --- Configuration ---
    CHECKPOINT_DIR = Path("/root/workspace/sc-diffusion-pipe/metric/sanity")
    BENCHMARK_FILE = CHECKPOINT_DIR / "clean.json"
    # ---------------------
    
    all_data = {}
    benchmark_data = None
    
    # --- Load Benchmark Data First ---
    if not BENCHMARK_FILE.exists():
        print(f"Error: Benchmark file not found at {BENCHMARK_FILE}")
        print("Please create this file from your 'clean-vs-clean' run.")
        
        # Create dummy data for example, reflecting your scores
        print("Creating dummy data to demonstrate...")
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        # Baseline (Clean)
        dummy_baseline = {
            "categories": { "animal": {"ssf": 0.948, "ss_fd": 6.2}, "architecture": {"ssf": 0.948, "ss_fd": 6.2},
                            "food": {"ssf": 0.948, "ss_fd": 6.2}, "human": {"ssf": 0.948, "ss_fd": 6.2},
                            "lifestyle": {"ssf": 0.948, "ss_fd": 6.2}, "plant": {"ssf": 0.948, "ss_fd": 6.2},
                            "scenery": {"ssf": 0.948, "ss_fd": 6.2}, "vehicles": {"ssf": 0.948, "ss_fd": 6.2} }
        }
        with open(BENCHMARK_FILE, 'w') as f:
            json.dump(dummy_baseline, f, indent=2)
        
        # Good Checkpoint
        dummy_good = {
            "categories": { "animal": {"ssf": 0.924, "ss_fd": 14.1}, "architecture": {"ssf": 0.924, "ss_fd": 14.1},
                            "food": {"ssf": 0.924, "ss_fd": 14.1}, "human": {"ssf": 0.914, "ss_fd": 14.1},
                            "lifestyle": {"ssf": 0.915, "ss_fd": 14.1}, "plant": {"ssf": 0.921, "ss_fd": 14.1},
                            "scenery": {"ssf": 0.931, "ss_fd": 14.1}, "vehicles": {"ssf": 0.930, "ss_fd": 14.1} }
        }
        with open(CHECKPOINT_DIR / "good_checkpoint.json", 'w') as f:
            json.dump(dummy_good, f, indent=2)
            
        # Bad Checkpoint
        dummy_bad = {
            "categories": { "animal": {"ssf": 0.753, "ss_fd": 55.4}, "architecture": {"ssf": 0.753, "ss_fd": 55.4},
                            "food": {"ssf": 0.753, "ss_fd": 55.4}, "human": {"ssf": 0.753, "ss_fd": 55.4},
                            "lifestyle": {"ssf": 0.753, "ss_fd": 55.4}, "plant": {"ssf": 0.753, "ss_fd": 55.4},
                            "scenery": {"ssf": 0.753, "ss_fd": 55.4}, "vehicles": {"ssf": 0.753, "ss_fd": 55.4} }
        }
        with open(CHECKPOINT_DIR / "bad_checkpoint.json", 'w') as f:
            json.dump(dummy_bad, f, indent=2)
            
        print(f"Created dummy files: 'baseline.json', 'good_checkpoint.json', 'bad_checkpoint.json'")
    
    # Load the benchmark data
    with open(BENCHMARK_FILE, 'r') as f:
        benchmark_data = json.load(f).get("categories")
        if not benchmark_data:
            print(f"Error: 'categories' key not found in {BENCHMARK_FILE}")
            return
            
    print(f"Loaded benchmark from {BENCHMARK_FILE.name}")
    
    # --- Load Checkpoint Data ---
    json_files = list(CHECKPOINT_DIR.glob("*.json"))
    
    for f_path in json_files:
        if f_path.resolve() == BENCHMARK_FILE.resolve():
            continue
            
        checkpoint_name = f_path.stem
        with open(f_path, 'r') as f:
            data = json.load(f)
            
        categories = data.get("categories")
        if not categories:
            print(f"Warning: 'categories' key not found in {f_path}, skipping.")
            continue
            
        all_data[checkpoint_name] = categories
            
    if not all_data:
        print(f"No checkpoint files found in {CHECKPOINT_DIR} (other than benchmark). Exiting.")
        return

    print(f"Found {len(all_data)} checkpoints to compare against baseline.")

    plot_comparison_radars(all_data, benchmark_data)

if __name__ == "__main__":
    main()

