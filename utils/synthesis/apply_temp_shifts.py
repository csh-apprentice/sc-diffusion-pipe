#!/usr/bin/env python3
"""
Applies color temperature shifts to an existing image based on sampled scales.

Assumes the input image is at the reference temperature (K_ref).
Maps scales s in [-1,1] to Kelvin using a perceptually-uniform Mired-based mapping.
Outputs shifted images into scale-named subfolders.
"""

import os, math, argparse, random
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ------------------ Color Temperature Functions (Copied) ------------------

def kelvin_to_rgb_fairchild(K: float) -> np.ndarray:
    """Approximate mapping from Kelvin to RGB multipliers. Returns float32 [R,G,B] in [0,1]."""
    K = float(K)
    temp = K / 100.0
    if temp <= 66.0:
        R = 255.0
        G = max(0.0, 99.47 * math.log(temp) - 161.12)
        B = max(0.0, 138.52 * math.log(max(temp - 10.0, 1e-6)) - 305.04)
    # Corrected Fairchild logic for temp > 66 (added missing range and fixed logic)
    elif temp > 66.0: 
        R = 329.70 * ((temp - 60.0) ** -0.1332) # Adjusted exponent slightly based on common sources
        G = 288.12 * ((temp - 60.0) ** -0.0755) # Adjusted exponent slightly based on common sources
        B = 255.0
    # Note: The original script had a gap between 66 and 88, and complex logic > 88. 
    # This simplified version is more standard for the >66K range.
    rgb = np.clip(np.array([R, G, B], dtype=np.float32), 0.0, 255.0) / 255.0
    return rgb

def apply_temperature_bgr(
    img_bgr: np.ndarray,
    K: float,
    K_ref: float = 6500.0,
    preserve_luminance: bool = True,
) -> np.ndarray:
    """
    Apply color-temperature shift to a BGR uint8 image (full-frame camera WB).
    """
    if K == K_ref: # If target is same as reference, return original
        return img_bgr.copy()
        
    g_ref = kelvin_to_rgb_fairchild(K_ref)
    g_tgt = kelvin_to_rgb_fairchild(K)
    gains = (g_tgt / (g_ref + 1e-6)).astype(np.float32)  # RGB gains relative to reference

    img = img_bgr.astype(np.float32) / 255.0
    r = img[..., 2] * gains[0]
    g = img[..., 1] * gains[1]
    b = img[..., 0] * gains[2]
    out = np.stack([b, g, r], axis=-1)

    if preserve_luminance:
        # Keep average luminance roughly constant (Rec.709)
        # Ensure input dimensions are compatible for broadcasting if needed
        Y_in  = 0.2126 * (img[..., 2]) + 0.7152 * (img[..., 1]) + 0.0722 * (img[..., 0])
        Y_out = 0.2126 * r           + 0.7152 * g           + 0.0722 * b
        
        # Calculate mean luminance safely, avoiding division by zero
        mean_Y_in = np.mean(Y_in)
        mean_Y_out = np.mean(Y_out)
        gain_y = (mean_Y_in + 1e-6) / (mean_Y_out + 1e-6)
        out = out * gain_y

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)

# ------------------ Condition Space + Mapping (Copied) ------------------

def stratified_scales(n: int, rng: random.Random) -> List[float]:
    """Bin-uniform *random* sampling over [-1,1]: one random s per bin."""
    if n <= 0:
        raise ValueError("--num_scales must be positive.")
    lo, hi = -1.0, 1.0
    w = (hi - lo) / float(n)
    xs = []
    for i in range(n):
        a = lo + i * w
        b = a + w
        xs.append(rng.uniform(a, b))
    return xs

def map_scale_to_kelvin(s: float, k_lo: float, k_hi: float) -> float:
    """
    Perceptually-uniform mapping via Mired (Micro Reciprocal Degrees).
    Maps s in [-1, 1] linearly to Mired space, then converts back to Kelvin.
    """
    if not (k_hi > k_lo > 0):
        raise ValueError("Require 0 < K_lo < K_hi.")

    # Convert Kelvin bounds to Mired bounds
    m_warm = 1_000_000.0 / k_lo
    m_cool = 1_000_000.0 / k_hi

    # Linearly interpolate the scalar 's' in Mired space [m_warm, m_cool]
    mired = m_warm + (s + 1.0) * (m_cool - m_warm) / 2.0

    # Convert the resulting Mired value back to Kelvin
    kelvin = 1_000_000.0 / max(mired, 1e-10)
    
    return float(kelvin)

# ------------------ Main Logic ------------------

def main():
    ap = argparse.ArgumentParser(description="Apply temperature shifts to an input image.")
    ap.add_argument("--input_image", type=str, required=True, help="Path to the input image file.")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save the shifted images.")
    
    # Scale and Kelvin parameters (same as before)
    ap.add_argument("--num_scales", type=int, default=9, help="Number of stratified scale samples.")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible scales.")
    ap.add_argument("--k-lo", type=float, required=True, help="Minimum Kelvin value (maps to s=-1).")
    ap.add_argument("--k-hi", type=float, required=True, help="Maximum Kelvin value (maps to s=+1).")
    ap.add_argument("--k-ref", type=float, default=6500.0, help="Reference Kelvin of the input image.")
    ap.add_argument("--preserve-luminance", action="store_true", help="Attempt to preserve average image luminance.")
    
    args = ap.parse_args()

    # --- Seeding ---
    rng = random.Random(args.seed)

    # --- Load Input Image ---
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        return
    
    base_bgr = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if base_bgr is None:
        print(f"Error: Could not read input image from {args.input_image}")
        return
        
    print(f"Loaded input image: {args.input_image} (shape: {base_bgr.shape})")

    # --- Generate Scales ---
    scales = stratified_scales(args.num_scales, rng=rng)
    folder_names = [f"{s:.3f}" for s in scales]
    
    print(f"[info] scales: {', '.join(folder_names)}")
    print(f"[info] mapping: s=-1 → {map_scale_to_kelvin(-1,args.k_lo,args.k_hi):.0f}K, "
          f"s=0 → {map_scale_to_kelvin(0,args.k_lo,args.k_hi):.0f}K, "
          f"s=+1 → {map_scale_to_kelvin(+1,args.k_lo,args.k_hi):.0f}K  (K_ref={args.k_ref:.0f})")

    # --- Apply Shifts and Save ---
    input_basename = os.path.basename(args.input_image)
    input_stem, input_ext = os.path.splitext(input_basename)

    for s, folder in zip(scales, folder_names):
        K = map_scale_to_kelvin(s, args.k_lo, args.k_hi)
        
        # Apply the temperature shift
        img_out = apply_temperature_bgr(base_bgr, K=float(K), K_ref=args.k_ref,
                                        preserve_luminance=args.preserve_luminance)

        # Create output directory and save
        out_dir = os.path.join(args.output_dir, folder)
        os.makedirs(out_dir, exist_ok=True)
        
        # Use the original filename in the output folder
        img_path = os.path.join(out_dir, f"{input_stem}{input_ext}") 

        ok = cv2.imwrite(img_path, img_out)
        if not ok:
            print(f"Warning: Failed to write output image to {img_path}")
        else:
            print(f"Saved: {img_path} (K={K:.0f})")

    print("Done.")

if __name__ == "__main__":
    main()