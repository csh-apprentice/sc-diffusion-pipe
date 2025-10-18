#!/usr/bin/env python3
# Sharp images; camera-WB over full frame; EXACT temperature method from kelvin_sweep_foldered.py.
# Bin-uniform *random* sampling: s ~ stratified_scales(n) over [-1,1].
# Map s -> Kelvin in [K_lo, K_hi] via centered-log. (s=-1 ~ K_lo, s=+1 ~ K_hi, s=0 ~ sqrt)
# Output per s: <out>/{+/-0.XXX}/scene_XXXX.{png,txt}
# Captions: "Image of a red circle at the top-left, a green star at the bottom-right, ..."

import os, math, argparse, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ------------------ YOUR temperature functions (unaltered behavior) ------------------

def kelvin_to_rgb_fairchild(K: float) -> np.ndarray:
    """Approximate mapping from Kelvin to RGB multipliers. Returns float32 [R,G,B] in [0,1]."""
    K = float(K)
    temp = K / 100.0
    if temp <= 66.0:
        R = 255.0
        G = max(0.0, 99.47 * math.log(temp) - 161.12)
        B = max(0.0, 138.52 * math.log(max(temp - 10.0, 1e-6)) - 305.04)
    elif temp <= 88.0:
        R = 0.5 * (255.0 + 329.70 * ((temp - 60.0) ** -0.1933))
        G = 0.5 * (288.12 * ((temp - 60.0) ** -0.1155) + 99.47 * math.log(temp) - 161.12)
        B = 0.5 * (138.52 * math.log(max(temp - 10.0, 1e-6)) - 305.04 + 255.0)
    else:
        R = 329.70 * ((temp - 60.0) ** -0.1933)
        G = 288.12 * ((temp - 60.0) ** -0.1155)
        B = 255.0
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
    If K == K_ref => near-identity.
    """
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
        Y_in  = 0.2126 * (img[..., 2]) + 0.7152 * (img[..., 1]) + 0.0722 * (img[..., 0])
        Y_out = 0.2126 * r           + 0.7152 * g           + 0.0722 * b
        gain_y = (Y_in.mean() + 1e-6) / (Y_out.mean() + 1e-6)
        out = out * gain_y

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)

# ------------------ condition space + mapping ------------------

def stratified_scales(n: int, seed: Optional[int] = None) -> List[float]:
    """Your bin-uniform *random* sampling over [-1,1]: one random s per bin."""
    if n <= 0:
        raise ValueError("--num_scales must be positive.")
    rng = random.Random(seed)
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
    Centered-log mapping:
      log K = log g + s * 0.5 * log(k_hi / k_lo),  with g = sqrt(k_lo * k_hi)
    """
    if not (k_hi > k_lo > 0):
        raise ValueError("Require 0 < K_lo < K_hi.")
    g = math.sqrt(k_lo * k_hi)
    half_span = 0.5 * math.log(k_hi / k_lo)
    return float(math.exp(math.log(g) + s * half_span))

# ------------------ scene synthesis (sharp; palette with names) ------------------

@dataclass
class Shape:
    kind: str
    color_name: str
    color_bgr: Tuple[int,int,int]
    cx: int
    cy: int
    size: int
    rot_deg: float

WHITE = ("white", (255,255,255))
# Named palette (RGB → BGR); shapes will EXCLUDE white; BG may be white unless --white-bg is False and random picks non-white.
PALETTE_RGB_NAMED = [
    ("red",    (230,  57,  70)),
    ("green",  ( 87, 187, 138)),
    ("blue",   ( 69, 123, 157)),
    ("yellow", (251, 191,  36)),
    ("purple", (139,  92, 246)),
    ("orange", (245, 158,  11)),
    ("teal",   ( 13, 148, 136)),
    ("pink",   (236,  72, 153)),
    ("white",  (255, 255, 255)),
]
PALETTE_BGR_NAMED = [(name, (rgb[2], rgb[1], rgb[0])) for (name, rgb) in PALETTE_RGB_NAMED]

KINDS = ("circle","square","triangle","star")

def star_pts(cx: float, cy: float, r: float, inner_ratio=0.5, n=5):
    rin = r * inner_ratio
    pts=[]
    a0 = -math.pi/2
    step = math.pi / n
    for i in range(2*n):
        rr = r if (i%2==0) else rin
        a  = a0 + i*step
        pts.append([cx + rr*math.cos(a), cy + rr*math.sin(a)])
    return np.array(pts, dtype=np.float32)

def draw_shape(img: np.ndarray, s: Shape):
    col = tuple(int(c) for c in s.color_bgr)
    if s.kind == "circle":
        cv2.circle(img, (s.cx, s.cy), s.size, col, -1, lineType=cv2.LINE_AA)
    elif s.kind in ("square","triangle"):
        cx, cy, a = s.cx, s.cy, s.size
        if s.kind == "square":
            pts = np.array([[cx-a,cy-a],[cx+a,cy-a],[cx+a,cy+a],[cx-a,cy+a]], dtype=np.float32)
        else:
            pts = np.array([[cx,cy-a],[cx-0.866*a,cy+0.5*a],[cx+0.866*a,cy+0.5*a]], dtype=np.float32)
        if s.rot_deg:
            t  = np.deg2rad(s.rot_deg)
            R  = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]], np.float32)
            pts = (pts - np.array([[cx,cy]], np.float32)) @ R.T + np.array([[cx,cy]], np.float32)
        cv2.fillPoly(img, [pts.astype(np.int32)], col, lineType=cv2.LINE_AA)
    elif s.kind == "star":
        pts = star_pts(s.cx, s.cy, r=s.size, inner_ratio=0.5, n=5)
        if s.rot_deg:
            t  = np.deg2rad(s.rot_deg)
            R  = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]], np.float32)
            pts = (pts - np.array([[s.cx,s.cy]], np.float32)) @ R.T + np.array([[s.cx,s.cy]], np.float32)
        cv2.fillPoly(img, [pts.astype(np.int32)], col, lineType=cv2.LINE_AA)
    else:
        raise ValueError(f"Unknown shape kind: {s.kind}")

def pick_background(white_only: bool, rng: random.Random) -> Tuple[str,Tuple[int,int,int]]:
    if white_only:
        return WHITE
    name, bgr = rng.choice(PALETTE_BGR_NAMED)  # may be white or colored
    return (name, bgr)

def make_scene(W:int,H:int,min_n:int,max_n:int, white_bg: bool, rng: random.Random) -> Tuple[np.ndarray,List[Shape],str]:
    # background
    bg_name, bg_bgr = pick_background(white_bg, rng)
    img = np.full((H,W,3), bg_bgr, np.uint8)

    # shape colors: EXCLUDE white
    nonwhite = [(n,b) for (n,b) in PALETTE_BGR_NAMED if n != "white"]
    n = rng.randint(min_n, max_n)
    cols = rng.sample(nonwhite, k=min(n, len(nonwhite)))

    margin = 60
    shapes: List[Shape] = []
    for i in range(n):
        kind = rng.choice(KINDS)
        cname, cbgr = cols[i % len(cols)]
        cx   = rng.randint(margin, W - margin)
        cy   = rng.randint(margin, H - margin)
        size = int(rng.uniform(0.06*min(W,H), 0.12*min(W,H)))  # scale randomized
        rot  = rng.uniform(0, 360) if kind != "circle" else 0.0
        s = Shape(kind, cname, cbgr, cx, cy, size, rot)
        shapes.append(s)
        draw_shape(img, s)
    return img, shapes, bg_name

# ------------------ captions (short, natural) ------------------

def loc_name(x: int, y: int, W: int, H: int) -> str:
    # 3x3 grid → "top-left", "center-right", etc.
    col = "left" if x < W/3 else ("right" if x > 2*W/3 else "center")
    row = "top"  if y < H/3 else ("bottom" if y > 2*H/3 else "middle")
    return f"{row}-{col}"

def caption_for(shapes: List[Shape], W: int, H: int, bg_name: str) -> str:
    parts = []
    for sh in shapes:
        parts.append(f"a {sh.color_name} {sh.kind} at the {loc_name(sh.cx, sh.cy, W, H)}")
    who = ", ".join(parts)
    # If background is white, say nothing; else mention background color briefly.
    if bg_name != "white":
        return f"Image of {who} on a {bg_name} background."
    else:
        return f"Image of {who}."

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser(description="Sharp images; camera-WB full-frame; exact Kelvin math; bin-uniform random scales.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_scenes", type=int, required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--min_shapes", type=int, default=2)
    ap.add_argument("--max_shapes", type=int, default=4)

    # Scales in [-1,1]
    ap.add_argument("--num_scales", type=int, default=9, help="Number of stratified bins in [-1,1].")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")

    # Kelvin mapping + reference for WB
    ap.add_argument("--k-lo", type=float, required=True)
    ap.add_argument("--k-hi", type=float, required=True)
    ap.add_argument("--k-ref", type=float, default=6500.0)
    ap.add_argument("--preserve-luminance", action="store_true")

    # Background control
    ap.add_argument("--white-bg", action="store_true", help="Force pure-white background; otherwise random from palette.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Build *random* scales via stratified bins
    scales = stratified_scales(args.num_scales, seed=args.seed)
    # Folder names are just the value (no prefix)
    folder_names = [f"{s:+.3f}" for s in scales]

    print(f"[info] scales: {', '.join(folder_names)}")
    print(f"[info] mapping: s=-1 → {map_scale_to_kelvin(-1,args.k_lo,args.k_hi):.0f}K, "
          f"s=0 → {map_scale_to_kelvin(0,args.k_lo,args.k_hi):.0f}K, "
          f"s=+1 → {map_scale_to_kelvin(+1,args.k_lo,args.k_hi):.0f}K  (K_ref={args.k_ref:.0f})")

    # Pre-create per-scale dirs (image+caption in SAME folder)
    for folder in folder_names:
        os.makedirs(os.path.join(args.output_dir, folder), exist_ok=True)

    W, H = args.width, args.height
    for idx in range(args.num_scenes):
        base_bgr, shapes, bg_name = make_scene(W, H, args.min_shapes, args.max_shapes, args.white_bg, rng)

        for s, folder in zip(scales, folder_names):
            K = map_scale_to_kelvin(s, args.k_lo, args.k_hi)
            img_out = apply_temperature_bgr(base_bgr, K=float(K), K_ref=args.k_ref,
                                            preserve_luminance=args.preserve_luminance)

            out_dir = os.path.join(args.output_dir, folder)
            stem = f"scene_{idx:04d}"
            img_path = os.path.join(out_dir, f"{stem}.png")
            txt_path = os.path.join(out_dir, f"{stem}.txt")

            ok = cv2.imwrite(img_path, img_out)
            if not ok:
                raise RuntimeError(f"Failed to write {img_path}")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_for(shapes, W, H, bg_name))

            print(f"saved: {img_path} | {txt_path}")

    print("Done.")

if __name__ == "__main__":
    main()
