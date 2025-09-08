#!/usr/bin/env python3
# measure_motion_blur.py
# Parallel motion-blur estimation with tqdm progress bar.
# Author: ChatGPT

import argparse
import math
import os
from pathlib import Path
import csv
import concurrent.futures as cf

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import find_peaks

# --------------------------- Utility ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path, recursive: bool):
    if recursive:
        yield from (p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        yield from (p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)

def read_gray(path: Path, max_side: int = None) -> np.ndarray:
    """Read image, convert to grayscale float32 in [0,1], optional max-side resize."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        if max_side is not None and max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def hanning2d(h, w):
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)

def variance_of_laplacian(img: np.ndarray) -> float:
    kern = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]], dtype=np.float32)
    img_p = np.pad(img, 1, mode="reflect")
    lap = (kern[0,1]*img_p[:-2,1:-1] +
           kern[1,0]*img_p[1:-1,:-2] +
           kern[1,2]*img_p[1:-1,2:] +
           kern[2,1]*img_p[2:,1:-1] +
           kern[1,1]*img_p[1:-1,1:-1])
    return float(lap.var())

def tenengrad(img: np.ndarray) -> float:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return float(np.mean(gx**2 + gy**2))

def structure_tensor_orientation(field: np.ndarray, sigma_smooth=2.0):
    gx = sobel(field, axis=1, mode="reflect")
    gy = sobel(field, axis=0, mode="reflect")
    Jxx = gaussian_filter(gx*gx, sigma=sigma_smooth)
    Jxy = gaussian_filter(gx*gy, sigma=sigma_smooth)
    Jyy = gaussian_filter(gy*gy, sigma=sigma_smooth)
    Jxx_m = float(np.mean(Jxx))
    Jxy_m = float(np.mean(Jxy))
    Jyy_m = float(np.mean(Jyy))
    theta = 0.5 * math.atan2(2*Jxy_m, (Jxx_m - Jyy_m))  # [-pi/2, pi/2]
    trace = Jxx_m + Jyy_m
    diff = (Jxx_m - Jyy_m)
    coherence = math.sqrt((diff*diff + 4*Jxy_m*Jxy_m)) / (trace + 1e-12)
    return theta, coherence

def fft_log_spectrum(img: np.ndarray):
    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    return np.log(np.abs(F) + 1e-8)

def real_cepstrum_2d(img: np.ndarray):
    F = np.fft.fft2(img)
    logS = np.log(np.abs(F) + 1e-8)
    C = np.fft.ifft2(logS)
    C = np.real(C)
    return np.fft.fftshift(C)

def sample_line_from_center(arr: np.ndarray, angle_rad: float, max_r: int = None):
    h, w = arr.shape
    cy, cx = (h-1)/2.0, (w-1)/2.0
    if max_r is None:
        max_r = int(min(cx, cy))
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    rs = np.arange(0, max_r, dtype=np.float32)
    xs = cx + rs * dx
    ys = cy + rs * dy
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    x0 = np.clip(x0, 0, w-1); x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1); y1 = np.clip(y1, 0, h-1)
    wa = (x1 - xs) * (y1 - ys)
    wb = (xs - x0) * (y1 - ys)
    wc = (x1 - xs) * (ys - y0)
    wd = (xs - x0) * (ys - y0)
    values = (arr[y0, x0] * wa +
              arr[y0, x1] * wb +
              arr[y1, x0] * wc +
              arr[y1, x1] * wd)
    return rs, values

def estimate_motion_angle_and_length(img_gray: np.ndarray):
    h, w = img_gray.shape
    if min(h, w) < 64:
        return (np.nan, np.nan, {"reason": "too_small"})
    win = hanning2d(h, w)
    img_w = (img_gray - np.mean(img_gray)) * win

    S = fft_log_spectrum(img_w)
    S_hp = S - gaussian_filter(S, sigma=3.0)

    stripe_theta, coherence = structure_tensor_orientation(S_hp, sigma_smooth=2.5)
    motion_theta = (stripe_theta + math.pi/2.0) % math.pi
    angle_deg = float(np.degrees(motion_theta))

    C = real_cepstrum_2d(img_w)
    # robust normalization
    lo, hi = np.percentile(C, [5, 95])
    Cn = (C - lo) / (hi - lo + 1e-8)

    max_r = int(min(h, w) * 0.45)
    rs, prof = sample_line_from_center(Cn, motion_theta, max_r=max_r)
    ignore = max(3, int(0.01 * max(h, w)))
    r_search = rs[ignore:]
    p_search = prof[ignore:]

    prom = max(0.05, 0.5 * float(np.std(p_search)))
    peaks, _ = find_peaks(p_search, prominence=prom, distance=2)
    if peaks.size == 0:
        perp_theta = (motion_theta + math.pi/2.0) % math.pi
        rs2, prof2 = sample_line_from_center(Cn, perp_theta, max_r=max_r)
        r_search2 = rs2[ignore:]
        p_search2 = prof2[ignore:]
        peaks2, _ = find_peaks(p_search2, prominence=prom, distance=2)
        if peaks2.size == 0:
            return (angle_deg, np.nan, {"coherence": coherence, "reason": "no_cepstrum_peak"})
        return (angle_deg, float(r_search2[peaks2[0]]), {"coherence": coherence, "alt_axis": True})

    return (angle_deg, float(r_search[peaks[0]]), {"coherence": coherence, "alt_axis": False})

# --------------------------- Worker ---------------------------

def _init_worker(singlethread_blas: bool = False):
    # Keep each worker single-threaded to avoid oversubscription
    if singlethread_blas:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def process_one(path_str: str, max_side: int):
    p = Path(path_str)
    try:
        img = read_gray(p, max_side=max_side)
        if np.std(img) < 1e-3:
            angle_deg, length_px, aux = (np.nan, np.nan, {"reason":"flat"})
        else:
            angle_deg, length_px, aux = estimate_motion_angle_and_length(img)

        vol = variance_of_laplacian(img)
        ten = tenengrad(img)

        return {
            "path": str(p),
            "angle_deg": angle_deg,
            "length_px": length_px,
            "laplacian_var": vol,
            "tenengrad": ten,
            "coherence": aux.get("coherence", np.nan),
            "note": aux.get("reason", "ok") + ("_alt" if aux.get("alt_axis", False) else "")
        }
    except Exception as e:
        return {
            "path": str(p),
            "angle_deg": np.nan,
            "length_px": np.nan,
            "laplacian_var": np.nan,
            "tenengrad": np.nan,
            "coherence": np.nan,
            "note": f"error:{type(e).__name__}"
        }

# --------------------------- Main CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Estimate motion blur magnitude across a folder (parallel) and plot distribution.")
    ap.add_argument("folder", type=str, help="Folder with images")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--max-side", type=int, default=1024, help="Resize so max side <= this (speeds up FFT)")
    ap.add_argument("--save-csv", type=str, default="blur_stats.csv", help="Output CSV path")
    ap.add_argument("--save-plot", type=str, default="blur_hist.png", help="Output histogram image path")
    ap.add_argument("--show", action="store_true", help="Show histogram window")
    ap.add_argument("--bins", type=int, default=40, help="Histogram bins for blur length")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers (processes). Set 1 to disable.")
    ap.add_argument("--singlethread-blas", action="store_true", help="Limit BLAS/FFT to 1 thread per worker.")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    paths = list(list_images(root, args.recursive))
    if not paths:
        raise SystemExit("No images found.")

    rows = []

    # Parallel path
    if args.workers and args.workers > 1:
        with cf.ProcessPoolExecutor(max_workers=args.workers,
                                    initializer=_init_worker,
                                    initargs=(args.singlethread-blas if False else args.singlethread_blas,)) as ex:
            futures = [ex.submit(process_one, str(p), args.max_side) for p in paths]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Analyzing", smoothing=0.05):
                rows.append(fut.result())
    else:
        # Sequential fallback (still with tqdm)
        for p in tqdm(paths, desc="Analyzing"):
            rows.append(process_one(str(p), args.max_side))

    # Save CSV
    csv_path = Path(args.save_csv)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Wrote {csv_path.resolve()}")

    # Plot distribution of motion-blur length (pixels)
    lengths = np.array([r["length_px"] for r in rows], dtype=np.float32)
    lengths = lengths[~np.isnan(lengths)]
    if lengths.size == 0:
        print("[-] No valid motion-blur length estimates to plot.")
        return

    plt.figure(figsize=(7.5, 4.8))
    plt.hist(lengths, bins=args.bins)
    plt.xlabel("Estimated motion-blur length (pixels)")
    plt.ylabel("Count of images")
    plt.title(f"Motion blur distribution — {root.name}  (N={lengths.size} valid)")
    plt.tight_layout()
    plot_path = Path(args.save_plot)
    plt.savefig(plot_path, dpi=150)
    print(f"[+] Saved histogram to {plot_path.resolve()}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
