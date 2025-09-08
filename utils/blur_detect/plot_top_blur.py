#!/usr/bin/env python3
# plot_top_blur.py
# Visualize the N most motion-blurred images (from CSV) and overlay motion path.
# Author: ChatGPT

import argparse
import csv
import math
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# --------------------------- CSV utils ---------------------------

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def load_rows(csv_path: Path) -> List[Dict]:
    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = []
        for r in rdr:
            rows.append({
                "path": r.get("path", ""),
                "angle_deg": _to_float(r.get("angle_deg", "nan")),
                "length_px": _to_float(r.get("length_px", "nan")),
                "laplacian_var": _to_float(r.get("laplacian_var", "nan")),
                "tenengrad": _to_float(r.get("tenengrad", "nan")),
                "coherence": _to_float(r.get("coherence", "nan")),
                "note": r.get("note", ""),
            })
    return rows

# --------------------------- Ranking ---------------------------

def rank_rows(rows: List[Dict], metric: str, num: int, fallback: str = "tenengrad") -> List[Dict]:
    """
    Select top 'num' most blurred images.
    metric: 'length_px' (descending) recommended.
            If NaN for all, fall back to 'tenengrad' (ascending) or 'laplacian_var' (ascending).
    """
    arr = np.array([r.get(metric, float("nan")) for r in rows], dtype=np.float64)
    valid = ~np.isnan(arr)
    selected: List[Dict] = []

    if valid.any():
        # Sort by chosen metric: length_px (bigger = more blur), others (smaller = more blur)
        if metric == "length_px":
            order = np.argsort(-arr, kind="mergesort")  # stable
        else:
            order = np.argsort(arr, kind="mergesort")
        for idx in order:
            if np.isnan(arr[idx]):
                continue
            selected.append(rows[idx])
            if len(selected) >= num:
                return selected

    # Fallback if insufficient
    fb = fallback
    fb_arr = np.array([rows[i].get(fb, float("nan")) for i in range(len(rows))], dtype=np.float64)
    fb_valid = ~np.isnan(fb_arr)
    if fb_valid.any():
        # For these sharpness metrics, lower = blurrier
        order = np.argsort(fb_arr, kind="mergesort")
        for idx in order:
            if np.isnan(fb_arr[idx]):
                continue
            if rows[idx] not in selected:
                selected.append(rows[idx])
                if len(selected) >= num:
                    break
    return selected

# --------------------------- Imaging ---------------------------

def load_thumbnail(path: Path, max_side: int):
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = 1.0
        if max(w, h) > max_side:
            s = max_side / float(max(w, h))
            im = im.resize((max(1, int(round(w*s))), max(1, int(round(h*s)))), Image.BICUBIC)
        return im, s

def draw_motion_overlay(ax, width, height, angle_deg, length_px_scaled):
    """
    Draw a motion path centered in the image:
      - a line of length 'length_px_scaled'
      - oriented by 'angle_deg' (0° = +x; CCW)
    """
    if np.isnan(angle_deg) or np.isnan(length_px_scaled) or length_px_scaled <= 0:
        return
    cx, cy = width / 2.0, height / 2.0
    theta = math.radians(angle_deg)
    dx = math.cos(theta) * (length_px_scaled / 2.0)
    dy = math.sin(theta) * (length_px_scaled / 2.0)
    x0, y0 = cx - dx, cy - dy
    x1, y1 = cx + dx, cy + dy

    # line + arrow
    ax.plot([x0, x1], [y0, y1], linewidth=2)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', linewidth=2))

# --------------------------- Copy helpers ---------------------------

def copy_or_link(paths: List[Path], out_dir: Path, symlink: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dst = out_dir / p.name
        try:
            if symlink:
                if dst.exists():
                    dst.unlink()
                os.symlink(os.path.abspath(p), dst)
            else:
                shutil.copy2(p, dst)
        except Exception as e:
            print(f"[warn] Failed to {'link' if symlink else 'copy'} {p} -> {dst}: {type(e).__name__}")

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot the N most motion-blurred images with overlayed blur path.")
    ap.add_argument("csv", type=str, help="CSV from measure_motion_blur.py")
    ap.add_argument("--num", type=int, default=20, help="Number of images to show")
    ap.add_argument("--metric", choices=["length_px", "tenengrad", "laplacian_var"], default="length_px",
                    help="Metric to rank by (length_px desc; others asc)")
    ap.add_argument("--fallback", choices=["tenengrad", "laplacian_var"], default="tenengrad",
                    help="Fallback metric if primary has NaNs")
    ap.add_argument("--thumb-max-side", type=int, default=512, help="Max side of thumbnails for plotting")
    ap.add_argument("--cols", type=int, default=5, help="Grid columns")
    ap.add_argument("--save-grid", type=str, default="top_blur_grid.png", help="Output mosaic path")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--print-paths", action="store_true", help="Print file paths to stdout")
    ap.add_argument("--copy-to", type=str, help="Copy selected images to this folder")
    ap.add_argument("--symlink", action="store_true", help="Use symlinks instead of copies (with --copy-to)")
    ap.add_argument("--cap", type=int, default=None, help="Cap overlay length to this many pixels AFTER scaling")
    ap.add_argument("--title", type=str, default=None, help="Custom figure title")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No rows in CSV.")

    selected = rank_rows(rows, metric=args.metric, num=args.num, fallback=args.fallback)
    if not selected:
        raise SystemExit("Could not select any images (all metrics NaN?)")

    # Prepare grid
    n = len(selected)
    cols = max(1, args.cols)
    rows_n = (n + cols - 1) // cols

    # Heuristic figure size (inches)
    cell = args.thumb_max_side / 100.0  # ~5.12 inches if max_side=512
    fig_w = max(4.0, cols * cell * 1.05)
    fig_h = max(3.0, rows_n * cell * 1.15)

    fig, axes = plt.subplots(rows_n, cols, figsize=(fig_w, fig_h), squeeze=False, dpi=args.dpi)
    axes = axes.flatten()

    shown_paths = []

    for i, (ax, item) in enumerate(zip(axes, tqdm(selected, desc="Rendering"))):
        p = Path(item["path"])
        try:
            im, scale = load_thumbnail(p, args.thumb_max_side)
            w, h = im.size
            ax.imshow(im)
            ax.axis("off")

            # Overlay motion path
            angle = item.get("angle_deg", float("nan"))
            length = item.get("length_px", float("nan"))
            L = length * scale if not np.isnan(length) else float("nan")
            if args.cap is not None and not np.isnan(L):
                L = min(L, float(args.cap))

            draw_motion_overlay(ax, w, h, angle, L)

            # Caption
            base = p.name
            angle_txt = "N/A" if np.isnan(angle) else f"{angle:.1f}°"
            len_txt = "N/A" if np.isnan(length) else f"{length:.1f}px"
            coh = item.get("coherence", float("nan"))
            coh_txt = "" if np.isnan(coh) else f", coh={coh:.2f}"
            ax.set_title(f"{base}\nL={len_txt}, θ={angle_txt}{coh_txt}", fontsize=8)

            shown_paths.append(p)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{p.name}\n{type(e).__name__}", ha="center", va="center", fontsize=9)
            ax.axis("off")

    # Hide any leftover axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=12)

    fig.tight_layout()
    out = Path(args.save_grid)
    fig.savefig(out, bbox_inches="tight")
    print(f"[+] Saved mosaic to {out.resolve()}")

    # Print paths
    if args.print_paths:
        print("\n# Top blurred image paths:")
        for p in shown_paths:
            print(str(p))

    # Copy / link selection
    if args.copy_to:
        dst = Path(args.copy_to)
        copy_or_link(shown_paths, dst, symlink=bool(args.symlink))
        print(f"[+] {'Linked' if args.symlink else 'Copied'} {len(shown_paths)} files to {dst.resolve()}")

if __name__ == "__main__":
    main()
