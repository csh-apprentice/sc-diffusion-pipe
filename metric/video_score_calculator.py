import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import numpy as np
from scipy import linalg
from tqdm import tqdm
from PIL import Image
import cv2  # Import OpenCV

# --- 1. Fréchet Distance Helper ---
# A standard implementation to calculate Fréchet Distance
# --- 1. Fréchet Distance Helper ---
# A standard implementation to calculate Fréchet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Fréchet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might have complex components
    # --- FIX: Updated linalg.sqrtm call ---
    # Newer SciPy versions (>=1.18.0) removed the 'disp' argument
    # and only return one value, not a tuple.
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    except linalg.LinAlgError:
        # Handle cases where the matrix might be singular
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# --- 2. Core Logic: Video Embedding ---
@torch.no_grad()
def get_video_embedding(video_path, clip_model, clip_processor, device):
    """
    Reads a video file frame by frame, gets CLIP embeddings
    for each frame, and returns the mean-pooled embedding
    for the entire video.
    """
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # 1. Read all frames from video
    batch_of_images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from OpenCV's BGR format to PIL's RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_of_images.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    if not batch_of_images:
        print(f"Warning: No frames read from {video_path}")
        return None

    # 2. Process all decoded images for CLIP
    # We can process all frames at once, letting the processor handle
    # batching and conversion to PIL Images internally if needed.
    inputs = clip_processor(
        images=batch_of_images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 3. Get embeddings for all frames in batches
    all_frame_embeddings = []
    
    # Process in CLIP batches to avoid OOM
    CLIP_BATCH_SIZE = 64
    pixel_values = inputs['pixel_values']
    
    for i in range(0, len(pixel_values), CLIP_BATCH_SIZE):
        i_end = min(i + CLIP_BATCH_SIZE, len(pixel_values))
        batch = pixel_values[i:i_end]
        frame_embeddings = clip_model.get_image_features(pixel_values=batch)
        all_frame_embeddings.append(frame_embeddings)

    # Concat all frame embeddings: [num_frames, embedding_dim]
    all_frame_embeddings = torch.cat(all_frame_embeddings, dim=0)

    # 4. Average to get the single semantic embedding for the video
    video_embedding = torch.mean(all_frame_embeddings, dim=0)
    
    return video_embedding

# --- 3. Main Script ---
def main():
    # --- Configuration ---
    # Point these to your folders of rendered video files
    ORIG_DIR = Path("/root/workspace/sc-diffusion-pipe/output/onestep/debug_clean_42")

    # ADAPT_DIR = Path("/root/workspace/sc-diffusion-pipe/output/onestep/debug_bad_check")
    #ADAPT_DIR = Path("/root/workspace/sc-diffusion-pipe/output/onestep/debug_good_check")
    ADAPT_DIR = Path("/root/workspace/sc-diffusion-pipe/output/onestep/debug_clean_99")
    
    # Assumes files are named identically in both folders
    # e.g., "prompt_001.mp4", "prompt_002.gif", ...
    # You can use "*.mp4", "*.gif", etc.
    FILE_EXTENSION = "*.mp4" 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading CLIP model...")
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    # VAE DECODER IS NO LONGER NEEDED
    
    # --- Calculate SSF (Single-Step Fidelity) ---
    print("\n" + "="*30)
    print("Calculating SSF (Single-Step Fidelity) Score...")
    print(" (High is good, 1.0 is perfect)")
    print("="*30)
    
    total_ssf_score = 0.0
    pair_count = 0
    all_orig_embeddings = []
    all_adapt_embeddings = []

    orig_files = sorted(list(ORIG_DIR.glob(FILE_EXTENSION)))
    if not orig_files:
        print(f"Error: No files found in {ORIG_DIR} with extension {FILE_EXTENSION}")
        return

    for orig_path in tqdm(orig_files, desc="Calculating SSF"):
        adapt_path = ADAPT_DIR / orig_path.name
        
        if not adapt_path.exists():
            print(f"Warning: Skipping {orig_g_path.name}, no matching file in {ADAPT_DIR}")
            continue
            
        # Get the single semantic vector for each video
        emb_orig = get_video_embedding(orig_path, clip_model, clip_processor, device)
        emb_adapt = get_video_embedding(adapt_path, clip_model, clip_processor, device)
        
        if emb_orig is None or emb_adapt is None:
            print(f"Skipping pair {orig_path.name} due to read error.")
            continue
            
        # Store for SS-FD calculation later
        all_orig_embeddings.append(emb_orig)
        all_adapt_embeddings.append(emb_adapt)

        # Calculate per-prompt score
        score = F.cosine_similarity(emb_orig.unsqueeze(0), emb_adapt.unsqueeze(0))
        # print(f"  {orig_path.name}: {score.item():.4f}") # Uncomment for per-file scores
        
        total_ssf_score += score.item()
        pair_count += 1

    if pair_count == 0:
        print("\nError: No matching pairs found to calculate SSF.")
        return

    avg_ssf_score = total_ssf_score / pair_count
    print("\n" + "*"*30)
    print(f"Average SSF Score: {avg_ssf_score:.6f}")
    print("*"*30)

    # --- Calculate SS-FD (Single-Step Fréchet Distance) ---
    print("\n" + "="*30)
    print("Calculating SS-FD (Single-Step Fréchet Distance)...")
    print(" (Low is good, 0.0 is perfect)")
    print("="*30)

    # Stack all embeddings into two big numpy arrays
    feat_orig = torch.stack(all_orig_embeddings).detach().cpu().numpy()
    feat_adapt = torch.stack(all_adapt_embeddings).detach().cpu().numpy()
    
    # Calculate mean and covariance
    mu_orig = np.mean(feat_orig, axis=0)
    sigma_orig = np.cov(feat_orig, rowvar=False)
    
    mu_adapt = np.mean(feat_adapt, axis=0)
    sigma_adapt = np.cov(feat_adapt, rowvar=False)

    # Calculate Fréchet Distance
    ss_fd_score = calculate_frechet_distance(mu_orig, sigma_orig, mu_adapt, sigma_adapt)
    
    print("\n" + "*"*30)
    print(f"SS-FD Score: {ss_fd_score:.6f}")
    print("*"*30)

if __name__ == "__main__":
    main()

