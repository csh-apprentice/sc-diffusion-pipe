#!/usr/bin/env python3
# make_palette_2x2.py
# Create a 2x2 palette image: each quadrant filled with a color.

from PIL import Image, ImageDraw
import argparse

def parse_color(s):
    # Accept "r,g,b" or hex like "#FFAA00"
    s = s.strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
    if "," in s:
        r, g, b = map(int, s.split(","))
        return (r, g, b)
    raise ValueError(f"Unrecognized color format: {s}")

def main():
    ap = argparse.ArgumentParser(description="Create a 2x2 color palette image.")
    ap.add_argument("--width", type=int, default=512, help="Image width")
    ap.add_argument("--height", type=int, default=512, help="Image height")
    ap.add_argument("--out", type=str, default="palette_2x2.png", help="Output filename")
    ap.add_argument("--c00", type=str, default="#E63946", help="Top-left color")
    ap.add_argument("--c01", type=str, default="#57BB8A", help="Top-right color")
    ap.add_argument("--c10", type=str, default="#457B9D", help="Bottom-left color")
    ap.add_argument("--c11", type=str, default="#F59E0B", help="Bottom-right color")
    args = ap.parse_args()

    W, H = args.width, args.height
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    c00 = parse_color(args.c00)
    c01 = parse_color(args.c01)
    c10 = parse_color(args.c10)
    c11 = parse_color(args.c11)

    midx, midy = W // 2, H // 2

    # Top-left
    draw.rectangle([0, 0, midx, midy], fill=c00)
    # Top-right
    draw.rectangle([midx, 0, W, midy], fill=c01)
    # Bottom-left
    draw.rectangle([0, midy, midx, H], fill=c10)
    # Bottom-right
    draw.rectangle([midx, midy, W, H], fill=c11)

    img.save(args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
