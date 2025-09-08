#!/usr/bin/env python3
import argparse, math, os, random, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Try to enable mp4 writing; otherwise we’ll fall back to PNG sequence
try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception as e:
    _HAS_IMAGEIO = False
    warnings.warn(f"imageio.v3 not available ({e}); video will be saved as PNG frames.")

# -----------------------
# Shapes & palette
# -----------------------

KINDS = ["circle", "square", "triangle", "star"]

COLOR_PALETTE = [
    ("red",      (230,  57,  70)),
    ("green",    ( 87, 187, 138)),
    ("blue",     ( 69, 123, 157)),
    ("yellow",   (251, 191,  36)),
    ("purple",   (139,  92, 246)),
    ("orange",   (245, 158,  11)),
    ("teal",     ( 13, 148, 136)),
    ("pink",     (236,  72, 153)),
]

@dataclass
class Shape:
    kind: str
    color_name: str
    color_rgb: Tuple[int, int, int]
    x: float
    y: float
    vx: float
    vy: float
    size: float  # radius-ish


# -----------------------
# Geometry utils
# -----------------------

def star_points(cx: float, cy: float, r_outer: float, inner_ratio: float = 0.5, num_points: int = 5):
    """Return vertices (x,y) for a regular star centered at (cx,cy)."""
    r_inner = r_outer * inner_ratio
    pts = []
    angle0 = -math.pi / 2.0
    step = math.pi / num_points
    for i in range(num_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        a = angle0 + i * step
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


# -----------------------
# Rendering
# -----------------------

def render_shapes_pil(W: int, H: int, shapes: List[Shape]) -> np.ndarray:
    """Render all shapes on an RGBA canvas and return uint8 RGB ndarray (H,W,3)."""
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for s in shapes:
        cx, cy, r = s.x, s.y, s.size
        col = (*s.color_rgb, 255)

        if s.kind == "circle":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=col)

        elif s.kind == "square":
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.rectangle(bbox, fill=col)

        elif s.kind == "triangle":
            p1 = (cx, cy - r)
            p2 = (cx - 0.866*r, cy + 0.5*r)
            p3 = (cx + 0.866*r, cy + 0.5*r)
            draw.polygon([p1, p2, p3], fill=col)

        elif s.kind == "star":
            pts = star_points(cx, cy, r_outer=r, inner_ratio=0.5, num_points=5)
            draw.polygon(pts, fill=col)

    rgb = np.array(canvas.convert("RGB"), dtype=np.uint8)
    return rgb


def move_shapes(shapes: List[Shape], dt: float, W: int, H: int, margin: float = 40.0):
    """Advance positions by dt with bounce at borders."""
    for s in shapes:
        s.x += s.vx * dt
        s.y += s.vy * dt
        # Bounce if outside margins
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


def render_exposure(shapes0: List[Shape],
                    t0: float,
                    exposure_s: float,
                    substeps: int,
                    W: int,
                    H: int) -> np.ndarray:
    """
    Integrate (average) substep renders over [t0, t0+exposure_s] to simulate motion blur.
    Shapes are deep-copied so the caller's list is not mutated.
    """
    # Clone shapes to avoid side effects
    shapes = [Shape(**vars(s)) for s in shapes0]

    # Move to exposure start (if your timeline has a separate shutter offset)
    # Here t0 is already the start for simplicity.

    acc = np.zeros((H, W, 3), dtype=np.float32)
    dt = exposure_s / max(1, substeps)

    for i in range(substeps):
        # Render current substep
        frame = render_shapes_pil(W, H, shapes)
        acc += frame.astype(np.float32)
        # Advance to next substep
        move_shapes(shapes, dt, W, H)

    acc /= float(substeps)
    return np.clip(acc, 0, 255).astype(np.uint8)


# -----------------------
# Scene sampling
# -----------------------

def sample_scene(W: int,
                 H: int,
                 num_objs_range: Tuple[int, int],
                 speed_px_s: Tuple[float, float],
                 allow_static: bool,
                 ensure_one_static: bool = False,
                 static_prob: float = 0.2) -> List[Shape]:
    """
    Sample a scene with N shapes. If allow_static=True, some shapes can be static.
    If ensure_one_static=True, guarantees at least one static shape.
    """
    n = random.randint(*num_objs_range)
    shapes: List[Shape] = []

    used_colors = random.sample(COLOR_PALETTE, k=min(n, len(COLOR_PALETTE)))
    static_idx = random.randrange(n) if (allow_static and ensure_one_static) else None

    # Generous margin to keep tips/corners on canvas
    margin = 60

    for i in range(n):
        kind = random.choice(KINDS)
        cname, crgb = used_colors[i % len(COLOR_PALETTE)]
        size = random.uniform(18, 40)

        x = random.uniform(margin, W - margin)
        y = random.uniform(margin, H - margin)

        speed = random.uniform(*speed_px_s)
        theta = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(theta)
        vy = speed * math.sin(theta)

        make_static = False
        if allow_static:
            if ensure_one_static and i == static_idx:
                make_static = True
            elif (not ensure_one_static) and (random.random() < static_prob):
                make_static = True

        if make_static:
            vx, vy = 0.0, 0.0

        shapes.append(Shape(kind, cname, crgb, x, y, vx, vy, size))

    return shapes


# -----------------------
# Captions
# -----------------------

def location_bucket(x: float, y: float, W: int, H: int) -> str:
    horiz = "left" if x < W/2 else "right"
    vert  = "top" if y < H/2 else "bottom"
    return f"{vert}-{horiz}"

def make_caption(shapes: List[Shape],
                 fps: int,
                 exposure_factor: float,
                 include_relations: bool,
                 include_blur:bool,
                 for_video: bool) -> str:
    """
    exposure_factor = 1.0 means 360° shutter ⇒ exposure = 1/fps
    """
    exposure_ms = 1000.0 * (exposure_factor / float(fps))
    summary = []
    for s in shapes:
        pos = location_bucket(s.x, s.y, W=512, H=512)  # we only need coarse language
        moving = (abs(s.vx) + abs(s.vy)) > 1e-6
        mv = "moving" if moving else "static"
        if include_relations:
            summary.append(f"a {s.color_name} {s.kind} at the {pos} ({mv})")
        else:
            summary.append(f"a {s.color_name} {s.kind} ({mv})")

    who = ", ".join(summary)
    base = f"{'Video' if for_video else 'Image'} of {who}. "
    blur = (
        f"Motion blur corresponds to ~{exposure_ms:.1f} ms exposure at {fps} fps "
        f"(≈ {exposure_factor*360:.0f}° shutter)."
    )
    if include_blur:
        return base + blur
    else:
        return base

# -----------------------
# Main generation
# -----------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_image(path: str, rgb: np.ndarray):
    Image.fromarray(rgb, mode="RGB").save(path)

def save_video_mp4(path: str, frames_rgb: List[np.ndarray], fps: int):
    if _HAS_IMAGEIO:
        iio.imwrite(path, np.stack(frames_rgb, axis=0), fps=fps, codec="libx264", quality=8)
    else:
        # Fallback: write as PNG sequence into a folder next to mp4 path
        seq_dir = os.path.splitext(path)[0] + "_frames"
        ensure_dir(seq_dir)
        for i, f in enumerate(frames_rgb):
            Image.fromarray(f).save(os.path.join(seq_dir, f"{i:04d}.png"))
        warnings.warn(f"No mp4 writer; saved PNG frames to {seq_dir}")

def main():
    ap = argparse.ArgumentParser("Synthetic FPS/Shutter dataset generator (2D shapes)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--modes", type=str, default="image",
                    help="Choose from: image, video, both")
    ap.add_argument("--fps_list", type=str, default="12,24,40,60,120,240")
    ap.add_argument("--samples_per_fps", type=int, default=100)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    # Video options
    ap.add_argument("--video_frames", type=int, default=16)
    ap.add_argument("--video_stride_frames", type=int, default=1,
                    help="Advance timeline by this many frames between outputs (usually 1)")
    ap.add_argument("--ensure_static_video", action="store_true",
                    help="Guarantee exactly one static object in video scenes (images remain all-moving).")
    ap.add_argument("--static_prob_video", type=float, default=0.2,
                    help="If not ensuring exactly one static, probability any given object is static in video mode.")

    # Exposure / blur
    ap.add_argument("--exposure_factor", type=float, default=1.0,
                    help="Exposure time multiplier: exposure = exposure_factor / fps. 1.0 ≈ 360° shutter; 0.5 ≈ 180°.")
    ap.add_argument("--substeps", type=int, default=16,
                    help="Number of shutter integration samples per frame (higher = smoother blur).")

    # Motion & scene
    ap.add_argument("--min_speed", type=float, default=60.0, help="px/s")
    ap.add_argument("--max_speed", type=float, default=220.0, help="px/s")
    ap.add_argument("--min_objs", type=int, default=2)
    ap.add_argument("--max_objs", type=int, default=4)

    # Captions
    ap.add_argument("--caption_relations", action="store_true",
                    help="Include relative positions in captions.")
    ap.add_argument("--caption_blur_note", action="store_true",
                    help="Explicitly mention blur/fps in caption (on by default in this script).")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    modes = args.modes.lower()
    do_image = modes in ("image", "both")
    do_video = modes in ("video", "both")

    fps_values = [int(x.strip()) for x in args.fps_list.split(",") if x.strip()]

    W, H = args.width, args.height
    samples = args.samples_per_fps

    if do_image:
        print("==> Generating IMAGE dataset")
    if do_video:
        print("==> Generating VIDEO dataset")

    # Root dirs
    img_root = os.path.join(args.out_dir, "images")
    vid_root = os.path.join(args.out_dir, "videos")
    if do_image: ensure_dir(img_root)
    if do_video: ensure_dir(vid_root)

    # For each FPS bucket
    for fps in fps_values:
        if do_image:
            out_img_dir = os.path.join(img_root, f"{fps}")
            ensure_dir(out_img_dir)
        if do_video:
            out_vid_dir = os.path.join(vid_root, f"{fps}")
            ensure_dir(out_vid_dir)

        pbar = tqdm(range(samples), desc=f"fps={fps} ({'img' if do_image else ''}{'+' if do_image and do_video else ''}{'vid' if do_video else ''})")

        for idx in pbar:
            # 1) Sample a “base” scene; images: all moving; videos: optionally include a static object
            shapes_img = sample_scene(
                W, H,
                num_objs_range=(args.min_objs, args.max_objs),
                speed_px_s=(args.min_speed, args.max_speed),
                allow_static=False,                 # images: all-moving for pure motion→blur mapping
                ensure_one_static=False
            )
            shapes_vid = sample_scene(
                W, H,
                num_objs_range=(args.min_objs, args.max_objs),
                speed_px_s=(args.min_speed, args.max_speed),
                allow_static=True,                  # videos can have static objects
                ensure_one_static=args.ensure_static_video,
                static_prob=args.static_prob_video
            )

            # 2) Compute exposure
            exposure_s = args.exposure_factor / float(fps)  # 1/fps scaled by shutter angle factor

            # 3) IMAGE: one blurred exposure from t0
            if do_image:
                t0 = 0.0
                img = render_exposure(shapes_img, t0, exposure_s, args.substeps, W, H)
                img_name = f"scene_{idx:06d}.png"
                save_image(os.path.join(out_img_dir, img_name), img)

                if args.caption_blur_note or args.caption_relations:
                    cap = make_caption(
                        shapes_img, fps=fps, exposure_factor=args.exposure_factor,
                        include_relations=args.caption_relations, include_blur=False, for_video=False
                    )
                    with open(os.path.join(out_img_dir, f"scene_{idx:06d}.txt"), "w") as f:
                        f.write(cap)

            # 4) VIDEO: T frames, each with per-frame exposure integration
            if do_video:
                T = args.video_frames
                frames = []
                # Deep copy so images branch doesn’t affect videos
                shapes = [Shape(**vars(s)) for s in shapes_vid]

                # Timeline: frame k starts at tk = k / fps, exposure: [tk, tk + exposure]
                # Between frames, advance by stride * (1/fps)
                # (Objects already moved within each exposure integration)
                for k in range(T):
                    tk = (k * args.video_stride_frames) / float(fps)
                    # For video we don’t separately move between frames;
                    # render_exposure mutates a local copy internally, so we manually
                    # advance by dt_frame after each frame to keep continuity.
                    # Here we choose dt_frame = args.video_stride_frames / fps
                    frame = render_exposure(shapes, tk, exposure_s, args.substeps, W, H)
                    frames.append(frame)
                    # Advance the *master* shapes to the next frame start time:
                    dt_frame = args.video_stride_frames / float(fps)
                    move_shapes(shapes, dt_frame, W, H)

                # Save as mp4 (or PNGs)
                vid_name = f"scene_{idx:06d}.mp4"
                save_video_mp4(os.path.join(out_vid_dir, vid_name), frames, fps=fps)

                if args.caption_blur_note or args.caption_relations:
                    cap = make_caption(
                        shapes_vid, fps=fps, exposure_factor=args.exposure_factor,
                        include_relations=args.caption_relations, include_blur=False, for_video=True
                    )
                    with open(os.path.join(out_vid_dir, f"scene_{idx:06d}.txt"), "w") as f:
                        f.write(cap)

    print("Done.")

if __name__ == "__main__":
    main()
