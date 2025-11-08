import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

# --- This function is copied directly from calculate_scores.py ---
@torch.no_grad()
def get_video_embedding(video_path, clip_model, processor, device):
    """
    Decodes a video file frame-by-frame, gets CLIP embeddings,
    and returns the mean embedding for the whole video.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        batch_of_images = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame from BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            batch_of_images.append(pil_image)
        
        cap.release()

        if not batch_of_images:
            print(f"Error: No frames read from {video_path}")
            return None

        # Process all images at once
        inputs = processor(
            images=batch_of_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Get embeddings for all frames
        all_frame_embeddings = clip_model.get_image_features(**inputs)

        # Average to get single video embedding
        video_embedding = torch.mean(all_frame_embeddings, dim=0)
        
        return video_embedding.cpu().numpy()

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def visualize_embeddings_per_checkpoint(full_df, baseline_name="baseline"):
    """
    Fits t-SNE *only* on the baseline data, then transforms all other
    checkpoints to that same 2D space for a true comparison.
    """
    checkpoints = sorted(full_df['checkpoint'].unique())
    num_checkpoints = len(checkpoints)
    
    if num_checkpoints == 0:
        print("No data to plot.")
        return
    if baseline_name not in checkpoints:
        print(f"Error: Baseline checkpoint '{baseline_name}' not found in data.")
        print(f"Found checkpoints: {checkpoints}")
        return

    print(f"\nFound {num_checkpoints} checkpoints: {checkpoints}")
    print(f"Using '{baseline_name}' as the ground-truth for t-SNE components.")
    
    # --- 1. Isolate Baseline and Other Checkpoints ---
    df_baseline = full_df[full_df['checkpoint'] == baseline_name].copy()
    df_others = full_df[full_df['checkpoint'] != baseline_name].copy()
    
    if len(df_baseline) < 2:
        print(f"Error: Baseline checkpoint '{baseline_name}' has < 2 data points. Cannot run t-SNE.")
        return

    # --- 2. Run t-SNE: FIT on Baseline, TRANSFORM on Others ---
    n_samples_baseline = len(df_baseline)
    perplexity_value = min(30.0, float(n_samples_baseline - 1))
    print(f"\nFitting t-SNE on {n_samples_baseline} 'baseline' embeddings (perplexity={perplexity_value})...")
    
    tsne = TSNE(
        n_components=2, 
        perplexity=perplexity_value, 
        max_iter=1000,
        init='pca', 
        learning_rate='auto', 
        random_state=42
    )
    
    # FIT and TRANSFORM the baseline data
    baseline_embeddings_array = np.array(df_baseline['embedding'].tolist())
    baseline_embeddings_2d = tsne.fit_transform(baseline_embeddings_array)
    
    # Add coordinates to the baseline DataFrame
    df_baseline['x'] = baseline_embeddings_2d[:, 0]
    df_baseline['y'] = baseline_embeddings_2d[:, 1]
    
    # Now, just TRANSFORM the other data
    print(f"Transforming {len(df_others)} embeddings from other checkpoints...")
    other_embeddings_array = np.array(df_others['embedding'].tolist())
    
    # Add a check for empty array
    if other_embeddings_array.size == 0:
        print("No other checkpoints found to transform. Plotting baseline only.")
    else:
        other_embeddings_2d = tsne.transform(other_embeddings_array)
        # Add coordinates to the other DataFrame
        df_others['x'] = other_embeddings_2d[:, 0]
        df_others['y'] = other_embeddings_2d[:, 1]
    
    # Combine them back into a single, unified DataFrame
    full_df_mapped = pd.concat([df_baseline, df_others])
    
    # Get global min/max for setting plot limits
    x_min, x_max = full_df_mapped['x'].min(), full_df_mapped['x'].max()
    y_min, y_max = full_df_mapped['y'].min(), full_df_mapped['y'].max()
    padding = (x_max - x_min) * 0.05 # 5% padding
    
    print("t-SNE complete. Generating plots...")

    # --- 3. Create Subplots Grid ---
    ncols = int(np.ceil(np.sqrt(num_checkpoints)))
    nrows = int(np.ceil(num_checkpoints / ncols))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 8), squeeze=False)
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    # Get a consistent color palette for categories
    categories = sorted(full_df['category'].unique())
    palette = sns.color_palette('tab10', n_colors=len(categories))

    # --- 4. Plot for each checkpoint (using shared 2D components) ---
    for i, checkpoint_name in enumerate(checkpoints):
        ax = axes[i]
        
        df_checkpoint = full_df_mapped[full_df_mapped['checkpoint'] == checkpoint_name]
        
        # --- 5. Plot with Seaborn ---
        sns.scatterplot(
            data=df_checkpoint,
            x='x',
            y='y',
            hue='Category',
            hue_order=categories, # Consistent category order
            palette=palette,      # Consistent category colors
            s=50,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
            ax=ax
        )
        
        ax.set_title(checkpoint_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1 (Fit to Baseline)', fontsize=10)
        ax.set_ylabel('t-SNE Component 2 (Fit to Baseline)', fontsize=10)
        ax.grid(color="gray", linestyle=":", linewidth=0.5, alpha=0.7)
        
        # Set shared axis limits
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Hide legend from individual plots
        if ax.get_legend():
            ax.get_legend().remove()

    # --- 6. Clean up unused subplots ---
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    # --- 7. Create one central legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(categories), bbox_to_anchor=(0.5, -0.02), fontsize=12, title="Categories", title_fontsize=14)

    fig.suptitle('t-SNE Visualization (Projected onto Baseline Components)', fontsize=20, fontweight='bold', y=1.03)
    plt.tight_layout(pad=3.0, rect=[0, 0.05, 1, 1]) # Adjust rect to make space for legend
    
    output_filename = "embedding_distribution_by_category_PROJECTED.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"\nScatter plot grid saved to {output_filename}")
    plt.show()


def main():
    # --- Configuration ---
    # IMPORTANT: Organize your videos in subfolders like this:
    # ./all_rendered_videos/
    #    ├── baseline/  <-- This name MUST match BASELINE_NAME
    #    │   ├── animal/
    #    │   └── ...
    #    ├── good_checkpoint/
    #    │   ├── animal/
    #    │   └── ...
    #
    ROOT_VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/metric/sanity_single_vid")
    BASELINE_NAME = "/root/workspace/sc-diffusion-pipe/metric/sanity_single_vid/clean" # The exact name of your clean run folder
    VIDEO_EXTENSION = "*.mp4" # Change if you use .gif, .avi, etc.
    # ---------------------

    # --- 1. Load CLIP Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model and processor...")
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    print("CLIP models loaded.")
    
    all_data = [] # Will be a list of dictionaries

    # --- 2. Find directories and process videos ---
    if not ROOT_VIDEOS_DIR.exists():
        print(f"Error: Root video directory not found: {ROOT_VIDEOS_DIR}")
        print("Please create dummy data to run this example.")
        # Create dummy data
        ROOT_VIDEOS_DIR.mkdir(exist_ok=True)
        (ROOT_VIDEOS_DIR / "clean" / "animal").mkdir(parents=True, exist_ok=True)
        (ROOT_VIDEOS_DIR / "clean" / "human").mkdir(parents=True, exist_ok=True)
        (ROOT_VIDEOS_DIR / "good_checkpoint" / "animal").mkdir(parents=True, exist_ok=True)
        (ROOT_VIDEOS_DIR / "good_checkpoint" / "human").mkdir(parents=True, exist_ok=True)
        (ROOT_VIDEOS_DIR / "bad_checkpoint" / "animal").mkdir(parents=True, exist_ok=True)
        (ROOT_VIDEOS_DIR / "bad_checkpoint" / "human").mkdir(parents=True, exist_ok=True)
        print(f"Created dummy directories in {ROOT_VIDEOS_DIR}")
        print("Please add your rendered videos (e.g., .mp4) to these folders.")
        return

    # Use .glob() to find all video files recursively
    video_files = list(ROOT_VIDEOS_DIR.glob(f"**/{VIDEO_EXTENSION}"))
    if not video_files:
        print(f"Error: No video files found in {ROOT_VIDEOS_DIR} or its subfolders.")
        return

    print(f"Found {len(video_files)} total video files to process...")

    for video_path in tqdm(video_files, desc="Processing all videos"):
        try:
            # Path structure: .../checkpoint_name/category_name/video.mp4
            category_name = video_path.parent.name
            checkpoint_name = video_path.parent.parent.name
            
            # Simple validation
            if checkpoint_name == ROOT_VIDEOS_DIR.name:
                print(f"Skipping video in root: {video_path}. Please place in subfolder.")
                continue

            embedding = get_video_embedding(video_path, clip_model, processor, device)
            
            if embedding is not None:
                all_data.append({
                    "embedding": embedding,
                    "checkpoint": checkpoint_name,
                    "category": category_name
                })
        except Exception as e:
            print(f"Error parsing path {video_path}: {e}")

    if not all_data:
        print("No embeddings were successfully generated. Exiting.")
        return
    # --- 3. Convert to DataFrame and Visualize ---
    full_df = pd.DataFrame(all_data)
    visualize_embeddings_per_checkpoint(full_df, baseline_name=BASELINE_NAME)

if __name__ == "__main__":
    main()

