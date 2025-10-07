#!/usr/bin/env python3
"""
make_crisp_shapes.py
Create a single RGB image with 3 sharp shapes (red, green, blue), no blur/bokeh.

Usage:
  python make_crisp_shapes.py --out my_image.png --width 512 --height 512 --seed 42
  (Optional) --caption writes my_image.txt with a short description.
"""

import argparse, math, random
from typing import Tuple, List
from dataclasses import dataclass
from PIL import Image, ImageDraw

@dataclass
class ShapeSpec:
    kind: str     # "circle" | "square" | "triangle" | "star"
    color: Tuple[int,int,int]
    cx: float
    cy: float
    size: float

KINDS = ["circle", "square", "triangle", "star"]

COLORS = [
    ("red",   (230, 57, 70)),
    ("green", (87, 187, 138)),
    ("blue",  (69, 123, 157)),
]

def star_points(cx: float, cy: float, r_outer: float, inner_ratio: float = 0.5, num_points: int = 5):
    r_inner = r_outer * inner_ratio
    pts = []
    angle0 = -math.pi / 2.0
    step = math.pi / num_points
    for i in range(num_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        a = angle0 + i * step
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def draw_shape(draw: ImageDraw.ImageDraw, s: ShapeSpec):
    cx, cy, r = s.cx, s.cy, s.size
    col = s.color
    if s.kind == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif s.kind == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif s.kind == "triangle":
        p1 = (cx, cy - r)
        p2 = (cx - 0.866*r, cy + 0.5*r)
        p3 = (cx + 0.866*r, cy + 0.5*r)
        draw.polygon([p1, p2, p3], fill=col)
    elif s.kind == "star":
        pts = star_points(cx, cy, r_outer=r, inner_ratio=0.5, num_points=5)
        draw.polygon(pts, fill=col)

def roughly_non_overlapping(existing: List[ShapeSpec], cx: float, cy: float, r: float, min_gap: float) -> bool:
    for e in existing:
        # conservative: treat all as circles of radius size
        dist2 = (cx - e.cx)**2 + (cy - e.cy)**2
        need = (r + e.size + min_gap)
        if dist2 < need * need:
            return False
    return True

def location_bucket(cx: float, cy: float, W: int, H: int) -> str:
    horiz = "left" if cx < W/2 else "right"
    vert  = "top" if cy < H/2 else "bottom"
    return f"{vert}-{horiz}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="crisp_shapes.png")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caption", action="store_true", help="Also write a .txt caption next to the image.")
    args = ap.parse_args()

    W, H = args.width, args.height
    rng = random.Random(args.seed)

    # Pick 3 distinct kinds, 3 fixed colors (red/green/blue)
    kinds = rng.sample(KINDS, k=3)
    colors = [c for _, c in COLORS]  # [red, green, blue] in fixed order
    color_names = [n for n, _ in COLORS]

    # Sample non-overlapping positions/sizes
    shapes: List[ShapeSpec] = []
    margin = 60
    min_r, max_r = 40, 70
    min_gap = 20  # pixels between shapes

    for i in range(3):
        kind = kinds[i]
        color = colors[i]
        for _attempt in range(200):
            r = rng.uniform(min_r, max_r)
            cx = rng.uniform(margin + r, W - margin - r)
            cy = rng.uniform(margin + r, H - margin - r)
            if roughly_non_overlapping(shapes, cx, cy, r, min_gap):
                shapes.append(ShapeSpec(kind=kind, color=color, cx=cx, cy=cy, size=r))
                break
        else:
            # fallback (should be rare)
            r = (min_r + max_r) / 2
            shapes.append(ShapeSpec(kind=kind, color=color, cx=W/2 + i*5, cy=H/2 + i*5, size=r))

    # Render (sharp, no blur)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGB")
    for s in shapes:
        draw_shape(draw, s)

    img.save(args.out)

    if args.caption:
        parts = []
        for s, cname in zip(shapes, color_names):
            parts.append(f"a {cname} {s.kind} at the {location_bucket(s.cx, s.cy, W, H)}")
        cap = "Image of " + ", ".join(parts) + ". (All shapes sharp)"
        with open(args.out.rsplit(".", 1)[0] + ".txt", "w") as f:
            f.write(cap)

    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
