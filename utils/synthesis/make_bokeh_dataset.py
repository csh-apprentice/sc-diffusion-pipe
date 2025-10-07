#!/usr/bin/env python3
"""
Synthetic Bokeh (aperture / depth-of-field) dataset generator.

Key features
- Blur radius scales with diopter offset *and* 1 / f_number:
    blur_px ≈ |1/z - 1/focus| * (1 / f_number) * aperture_scale_px
  → smaller f-number (e.g., f/1.4) = blurrier; larger f-number (e.g., f/11) = sharper.

- Focus distance options:
    * median   : plane at median object depth
    * random   : uniform in [depth_near_m, depth_far_m]
    * fixed    : user-provided --focus_fixed_m
    * nearest  : plane at nearest object’s depth (nearest is always sharp)

- 2D motion (x,y) for videos; depths are constant per scene.

Folder layout
  <out>/images/f1.4/scene_000123.png, scene_000123.txt, ...
  <out>/videos/f1.4/scene_000123.mp4, scene_000123.txt, ...
A JSON config is also saved to <out>/configs/<timestamp>.json for reproducibility/debugging.
"""

import argparse, math, os, random, json, time, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# mp4 writer (optional)
try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception as e:
    _HAS_IMAGEIO = False
    warnings.warn(f"imageio.v3 not available ({e}); videos will be saved as PNG frames.")

# -----------------------
# Shapes & palette
# -----------------------

KINDS = ["circle", "square", "triangle", "star"]

COLOR_PALETTE = [
    ("green",  ( 87, 187, 138)),
    ("red",    (230,  57,  70)),
    ("blue",   ( 69, 123, 157)),
    ("yellow", (251, 191,  36)),
    ("purple", (139,  92, 246)),
    ("orange", (245, 158,  11)),
    ("teal",   ( 13, 148, 136)),
    ("pink",   (236,  72, 153)),
]

@dataclass
class Shape:
    kind: str
    color_name: str
    color_rgb: Tuple[int, int, int]
    x: float
    y: float
    z: float      # meters
    vx: float     # px/s (2D motion only)
    vy: float
    size: float   # radius-ish in px

# -----------------------
# Geometry / rendering
# -----------------------

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

def draw_one_shape_rgba(W: int, H: int, s: Shape) -> Image.Image:
    """Render a single shape into an RGBA layer (no blur yet)."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer, "RGBA")
    cx, cy, r = s.x, s.y, s.size
    col = (*s.color_rgb, 255)

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
    return layer

def render_scene_bokeh(W: int,
                       H: int,
                       shapes: List[Shape],
                       focus_m: float,
                       f_number: float,
                       aperture_scale_px: float) -> np.ndarray:
    """
    Composite shapes from far->near. For each shape, blur layer by:
        blur_px = |1/z - 1/focus| * (1/f_number) * aperture_scale_px
    """
    # Clamp & factors
    inv_f = 1.0 / max(1e-6, float(focus_m))
    f_factor = 1.0 / max(1e-6, float(f_number))  # smaller f-number -> larger factor

    # Sort by depth far->near so nearer shapes overpaint
    order = sorted(range(len(shapes)), key=lambda i: shapes[i].z, reverse=True)

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))  # white background
    for idx in order:
        s = shapes[idx]
        layer = draw_one_shape_rgba(W, H, s)

        inv_z = 1.0 / max(1e-6, float(s.z))
        diopters = abs(inv_z - inv_f)

        blur_px = diopters * f_factor * float(aperture_scale_px)
        # Cap extreme radii for stability; you can expose this if needed.
        blur_px = min(blur_px, 128.0)

        if blur_px > 0.25:
            layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

        canvas.alpha_composite(layer)

    return np.array(canvas.convert("RGB"), dtype=np.uint8)

# -----------------------
# Motion / scene sampling
# -----------------------

def move_shapes_xy(shapes: List[Shape], dt: float, W: int, H: int, margin: float = 40.0):
    """Advance x,y by dt with bounce on image borders. Depth z stays constant."""
    for s in shapes:
        s.x += s.vx * dt
        s.y += s.vy * dt
        if s.x < margin:
            s.x = margin + (margin - s.x)
            s.vx = abs(s.vx)
        if s.x > W - margin:
            s.x = (W - margin) - (s.x - (W - margin))
            s.vx = -abs(s.vx)
        if s.y < margin:
            s.y = margin + (margin - s.y)
            s.vy = abs(s.vy)
        if s.y > H - margin:
            s.y = (H - margin) - (s.y - (H - margin))
            s.vy = -abs(s.vy)

def sample_scene(W: int,
                 H: int,
                 num_objs_range: Tuple[int, int],
                 speed_px_s: Tuple[float, float],
                 depth_near_m: float,
                 depth_far_m: float,
                 rng: random.Random) -> List[Shape]:
    n = rng.randint(*num_objs_range)
    shapes: List[Shape] = []

    used_colors = rng.sample(COLOR_PALETTE, k=min(n, len(COLOR_PALETTE)))
    margin = 60

    for i in range(n):
        kind = rng.choice(KINDS)
        cname, crgb = used_colors[i % len(COLOR_PALETTE)]
        size = rng.uniform(18, 40)
        x = rng.uniform(margin, W - margin)
        y = rng.uniform(margin, H - margin)
        z = rng.uniform(depth_near_m, depth_far_m)  # meters
        speed = rng.uniform(*speed_px_s)
        theta = rng.uniform(0, 2 * math.pi)
        vx = speed * math.cos(theta)
        vy = speed * math.sin(theta)
        shapes.append(Shape(kind, cname, crgb, x, y, z, vx, vy, size))
    return shapes

# -----------------------
# Captions
# -----------------------

def location_bucket(x: float, y: float, W: int, H: int) -> str:
    horiz = "left" if x < W/2 else "right"
    vert  = "top" if y < H/2 else "bottom"
    return f"{vert}-{horiz}"

def caption_for_scene(shapes: List[Shape], for_video: bool) -> str:
    # Example:
    # "Image of a green square at the bottom-right (depth≈3.2m), a red star at the top-left (depth≈1.1m)."
    parts = []
    for s in shapes:
        pos = location_bucket(s.x, s.y, W=512, H=512)
        parts.append(f"a {s.color_name} {s.kind} at the {pos} (depth≈{s.z:.2f}m)")
    who = ", ".join(parts)
    return f"{'Video' if for_video else 'Image'} of {who}."

# -----------------------
# IO helpers
# -----------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_image(path: str, rgb: np.ndarray):
    Image.fromarray(rgb, mode="RGB").save(path)

def save_video_mp4(path: str, frames_rgb: List[np.ndarray], fps: int):
    if _HAS_IMAGEIO:
        iio.imwrite(path, np.stack(frames_rgb, axis=0), fps=fps, codec="libx264", quality=8)
    else:
        seq_dir = os.path.splitext(path)[0] + "_frames"
        ensure_dir(seq_dir)
        for i, f in enumerate(frames_rgb):
            Image.fromarray(f).save(os.path.join(seq_dir, f"{i:04d}.png"))
        warnings.warn(f"No mp4 writer; saved PNG frames to {seq_dir}")

# -----------------------
# Focus selection
# -----------------------

def pick_focus_distance(shapes: List[Shape],
                        mode: str,
                        depth_near_m: float,
                        depth_far_m: float,
                        fixed_m: Optional[float]) -> float:
    zs = [s.z for s in shapes]
    if mode == "median":
        return float(np.median(zs))
    elif mode == "random":
        return random.uniform(depth_near_m, depth_far_m)
    elif mode == "fixed":
        if fixed_m is None:
            raise ValueError("focus_mode 'fixed' requires --focus_fixed_m")
        return float(fixed_m)
    elif mode == "nearest":
        return float(min(zs))
    else:
        raise ValueError(f"Unknown focus mode: {mode}")

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser("Synthetic Bokeh dataset generator (2D, depth-of-field)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--modes", type=str, default="image", help="image, video, both")
    ap.add_argument("--apertures", type=str, default="1.4,2.8,5.6,11",
                    help="Comma-separated f-numbers (e.g., 1.4,2,2.8,4,5.6,8,11)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    # Scene / motion / depth
    ap.add_argument("--min_objs", type=int, default=2)
    ap.add_argument("--max_objs", type=int, default=4)
    ap.add_argument("--min_speed", type=float, default=40.0, help="px/s for videos")
    ap.add_argument("--max_speed", type=float, default=140.0, help="px/s for videos")
    ap.add_argument("--depth_near_m", type=float, default=0.7)
    ap.add_argument("--depth_far_m", type=float, default=6.0)

    # Focus
    ap.add_argument("--focus_mode", choices=["median", "random", "fixed", "nearest"], default="median")
    ap.add_argument("--focus_fixed_m", type=float, default=None)

    # Blur scaling
    ap.add_argument("--aperture_scale_px", type=float, default=100.0,
                    help="Global scale mapping diopters*(1/f) to pixels of Gaussian blur.")

    # Dataset sizes
    ap.add_argument("--samples_per_aperture", type=int, default=100, help="scenes per aperture per mode")

    # Video params
    ap.add_argument("--video_fps", type=int, default=16)
    ap.add_argument("--video_frames", type=int, default=33)

    # Repro
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    modes = args.modes.lower()
    do_image = modes in ("image", "both")
    do_video = modes in ("video", "both")

    apertures = [float(x.strip()) for x in args.apertures.split(",") if x.strip()]
    W, H = args.width, args.height

    # Roots
    img_root = os.path.join(args.out_dir, "images")
    vid_root = os.path.join(args.out_dir, "videos")
    cfg_root = os.path.join(args.out_dir, "configs")
    if do_image: ensure_dir(img_root)
    if do_video: ensure_dir(vid_root)
    ensure_dir(cfg_root)

    # Save run config for debugging
    run_cfg = vars(args).copy()
    run_cfg["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
    cfg_path = os.path.join(cfg_root, f"run_{run_cfg['timestamp']}.json")
    with open(cfg_path, "w") as f:
        json.dump(run_cfg, f, indent=2)

    # Generate
    total = args.samples_per_aperture
    for idx in tqdm(range(total), desc="sampling scenes"):
        # per-scene RNG so per-aperture variants share same base scene
        rng = random.Random(args.seed * 1_000_003 + idx * 97 + 20251002)

        # sample one base scene
        shapes_base = sample_scene(
            W, H,
            num_objs_range=(args.min_objs, args.max_objs),
            speed_px_s=(args.min_speed, args.max_speed),
            depth_near_m=args.depth_near_m,
            depth_far_m=args.depth_far_m,
            rng=rng
        )

        # choose focus distance (shared across apertures for this scene)
        focus_m = pick_focus_distance(shapes_base, args.focus_mode,
                                      args.depth_near_m, args.depth_far_m,
                                      args.focus_fixed_m)

        for fnum in apertures:
            f_dir = f"{fnum:g}"    # f1.4 -> f1_4 folder naming
            if do_image:
                out_dir = os.path.join(img_root, f_dir); ensure_dir(out_dir)
                # copy scene (so video motion doesn't affect image)
                shapes_img = [Shape(**vars(s)) for s in shapes_base]
                rgb = render_scene_bokeh(W, H, shapes_img, focus_m, fnum, args.aperture_scale_px)
                name = f"scene_{idx:06d}"
                save_image(os.path.join(out_dir, f"{name}.png"), rgb)
                cap = caption_for_scene(shapes_img, for_video=False)
                with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
                    f.write(cap)

            if do_video:
                out_dir = os.path.join(vid_root, f_dir); ensure_dir(out_dir)
                shapes_vid = [Shape(**vars(s)) for s in shapes_base]
                frames = []
                dt = 1.0 / float(args.video_fps)
                for t in range(args.video_frames):
                    rgb = render_scene_bokeh(W, H, shapes_vid, focus_m, fnum, args.aperture_scale_px)
                    frames.append(rgb)
                    move_shapes_xy(shapes_vid, dt, W, H)
                name = f"scene_{idx:06d}"
                save_video_mp4(os.path.join(out_dir, f"{name}.mp4"), frames, args.video_fps)
                cap = caption_for_scene(shapes_base, for_video=True)
                with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
                    f.write(cap)

    print("Done.")

if __name__ == "__main__":
    main()
