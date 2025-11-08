import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

# --- 1. Re-usable Embedding Function ---
@torch.no_grad()
def get_video_embedding(video_path, clip_image_model, clip_processor, device):
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
        inputs = clip_processor(
            images=batch_of_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Get embeddings for all frames
        all_frame_embeddings = clip_image_model.get_image_features(**inputs)

        # Average to get single video embedding
        video_embedding = torch.mean(all_frame_embeddings, dim=0)
        
        return video_embedding

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

@torch.no_grad()
def get_text_embedding(text_prompt, clip_text_model, clip_tokenizer, device):
    """
    Gets the CLIP text embedding for a single prompt string.
    """
    inputs = clip_tokenizer(
        [text_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    text_embedding = clip_text_model.get_text_features(**inputs)
    return text_embedding.squeeze(0) # Remove batch dim

# --- 3. Main Script ---
def main():
    # --- Configuration ---
    # Put your high-quality rendered videos here.
    # e.g., ./high_quality_videos/good_checkpoint/00001.mp4
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/16steps/clean_category_42/animal")
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/16steps/20251031_08-25-18/animal")
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/16steps/20251008_20-21-45/animal")

    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/50_49/clean_category_42")
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/50_49/20251031_08-25-18")
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/50_49/20251008_20-21-45")
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/50_49/20251014_06-32-03")
    VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/50_49/20251023_00-19-38")
    # VIDEOS_DIR = Path("/root/workspace/sc-diffusion-pipe/output/50_49/20251030_08-16-52")

    
    # Path to the text file containing your prompts, one per line.
    PROMPT_FILE = Path("/root/workspace/sc-diffusion-pipe/utils/VBench/prompts/high_quality_prompts_96.txt") 
    
    # Optional: Change to "*.gif" or "*.avi" if needed
    VIDEO_EXTENSION = "*.mp4" 
    # ---------------------

    # --- 1. Load All CLIP Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model and processor...")
    clip_model_name = "openai/clip-vit-large-patch14"
    
    # We need all three parts:
    clip_image_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_text_model = clip_image_model # They are the same model
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    print("CLIP models loaded.")

    # --- 2. Load Prompts from File ---
    if not PROMPT_FILE.exists():
        print(f"Error: Prompt file not found at {PROMPT_FILE}")
        return
        
    print(f"Loading prompts from {PROMPT_FILE}...")
    try:
        prompts = PROMPT_FILE.read_text().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()] # Clean up lines
        if not prompts:
            print("Error: Prompt file is empty.")
            return
        print(f"Loaded {len(prompts)} prompts.")
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    # --- 3. Find video paths ---
    video_files = list(VIDEOS_DIR.glob(f"**/{VIDEO_EXTENSION}"))
    if not video_files:
        print(f"Error: No video files with extension {VIDEO_EXTENSION} found in {VIDEOS_DIR}")
        return
        
    print(f"Found {len(video_files)} videos to evaluate...")

    all_clip_scores = []
    
    # Pre-cache text embeddings to avoid re-calculating
    text_embedding_cache = {}

    for video_path in tqdm(video_files, desc="Calculating CLIP Scores"):
        
        # --- 4. Get Text Prompt from Filename Index ---
        # "00001.mp4" -> "00001"
        video_id_str = video_path.stem
        
        try:
            # Convert "00001" -> 1
            video_index = int(video_id_str)
            # Get the prompt from the list (user said 1-indexed)
            prompt_index_in_list = video_index - 1
            
            if not (0 <= prompt_index_in_list < len(prompts)):
                print(f"Warning: Skipping {video_path.name}. ID '{video_index}' is out of range for prompts file (1 to {len(prompts)}).")
                continue
                
            text_prompt = prompts[prompt_index_in_list]
            
        except ValueError:
            print(f"Warning: Skipping {video_path.name}. Filename stem '{video_id_str}' is not a valid integer.")
            continue
        except IndexError:
            # This case is technically covered by the bounds check, but good to have
            print(f"Warning: Skipping {video_path.name}. Index {prompt_index_in_list} out of bounds.")
            continue
        
        # --- 5. Get Text Embedding ---
        if text_prompt not in text_embedding_cache:
            text_embedding_cache[text_prompt] = get_text_embedding(
                text_prompt, clip_text_model, clip_tokenizer, device
            )
        text_emb = text_embedding_cache[text_prompt]

        # --- 6. Get Video Embedding ---
        video_emb = get_video_embedding(
            video_path, clip_image_model, clip_processor, device
        )
        
        if video_emb is None:
            print(f"Warning: Skipping {video_path.name}, could not get video embedding.")
            continue
            
        # --- 7. Calculate CLIP Score (Cosine Similarity) ---
        # Normalize both embeddings for a true cosine similarity
        video_emb_norm = F.normalize(video_emb, p=2, dim=0)
        text_emb_norm = F.normalize(text_emb, p=2, dim=0)
        
        score = F.cosine_similarity(video_emb_norm, text_emb_norm, dim=0)
        all_clip_scores.append(score.item())

    # --- 8. Calculate Final Average Score ---
    if not all_clip_scores:
        print("Error: No scores were calculated.")
        return
        
    final_avg_clip_score = np.mean(all_clip_scores)
    
    print("\n--- Evaluation Complete ---")
    print(f"Checkpoint Directory: {VIDEOS_DIR.name}")
    print(f"Total Videos Evaluated: {len(all_clip_scores)}")
    print(f"Average CLIP Score: {final_avg_clip_score:.6f}")

if __name__ == "__main__":
    main()

