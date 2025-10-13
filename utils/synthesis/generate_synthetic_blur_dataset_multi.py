#!/usr/bin/env python3
"""
Synthetic FPS/Shutter dataset generator (range-driven, fixed #frames, centered-log scaling)

Key points:
- Specify clip length by number of frames: --num_frames (e.g., 8 or 4).
- Frame mid-times are spaced using a virtual timeline rate --sim_fps (default 16 Hz),
  independent of the shutter control. The shutter exposure window still uses 1/fps_mapped.
- Accepts --fps_lo/--fps_hi and --num_scales; samples scale s in [-1,1] (stratified),
  maps to concrete fps via centered log mapping (geo-centered at sqrt(lo*hi)).
- Randomizes background color with a contrast guard and mentions it in the caption.
- Writes images/videos and captions into scale-named folders:
  out_root/images/scale_{+/-x.xxx}/..., out_root/videos/scale_{+/-x.xxx}/...
"""

import argparse, math, os, random, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

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
BG_EXTRA = [("white", (255, 255, 255))]

@dataclass
class Shape:
    kind: str
    color_name: str
    color_rgb: Tuple[int, int, int]
    x: float
    y: float
    vx: float
    vy: float
    size: float

# -----------------------
# Centered-log map [-1,1] <-> [fps_lo,fps_hi]
# -----------------------

def center_from_range(fps_lo: float, fps_hi: float) -> float:
    return math.sqrt(fps_lo * fps_hi)

def scale_from_fps(fps: float, fps_lo: float, fps_hi: float) -> float:
    c = center_from_range(fps_lo, fps_hi)
    denom = math.log(fps_hi / c)
    return max(-1.0, min(1.0, math.log(max(fps/c, 1e-12)) / max(denom, 1e-12)))

def fps_from_scale(s: float, fps_lo: float, fps_hi: float) -> float:
    c = center_from_range(fps_lo, fps_hi)
    base = fps_hi / c  # == sqrt(hi/lo)
    return c * (base ** s)

# -----------------------
# Stratified sampling over [-1,1]
# -----------------------

def stratified_scales(n: int, seed: Optional[int] = None) -> List[float]:
    rng = random.Random(seed)
    lo, hi = -1.0, 1.0
    w = (hi - lo) / float(n)
    xs = []
    for i in range(n):
        a = lo + i * w
        b = a + w
        xs.append(rng.uniform(a, b))
    return xs

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

def render_shapes_pil(W: int, H: int, shapes: List[Shape], bg_rgb: Tuple[int,int,int]) -> np.ndarray:
    from PIL import Image, ImageDraw
    canvas = Image.new("RGBA", (W, H), (*bg_rgb, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for s in shapes:
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
    return np.array(canvas.convert("RGB"), dtype=np.uint8)

# -----------------------
# Time evolution with bouncing
# -----------------------

def move_shapes(shapes: List[Shape], dt: float, W: int, H: int, margin: float = 40.0):
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

def advance_time(shapes: List[Shape], dt: float, W: int, H: int, max_step_s: float):
    if dt <= 0: return
    n = int(dt // max_step_s)
    rem = dt - n * max_step_s
    for _ in range(n):
        move_shapes(shapes, max_step_s, W, H)
    if rem > 0:
        move_shapes(shapes, rem, W, H)

# -----------------------
# Scene sampling
# -----------------------

def sample_scene(W: int,
                 H: int,
                 num_objs_range: Tuple[int, int],
                 speed_px_s: Tuple[float, float],
                 allow_static: bool,
                 ensure_one_static: bool = False,
                 static_prob: float = 0.2,
                 rng: Optional[random.Random] = None) -> List[Shape]:
    rng = rng or random
    n = rng.randint(*num_objs_range)
    shapes: List[Shape] = []
    used_colors = rng.sample(COLOR_PALETTE, k=min(n, len(COLOR_PALETTE)))
    static_idx = rng.randrange(n) if (allow_static and ensure_one_static) else None
    margin = 60
    for i in range(n):
        kind = rng.choice(KINDS)
        cname, crgb = used_colors[i % len(COLOR_PALETTE)]
        size = rng.uniform(18, 40)
        x = rng.uniform(margin, W - margin)
        y = rng.uniform(margin, H - margin)
        speed = rng.uniform(*speed_px_s)
        theta = rng.uniform(0, 2 * math.pi)
        vx = speed * math.cos(theta)
        vy = speed * math.sin(theta)
        make_static = False
        if allow_static:
            if ensure_one_static and i == static_idx:
                make_static = True
            elif (not ensure_one_static) and (rng.random() < static_prob):
                make_static = True
        if make_static:
            vx, vy = 0.0, 0.0
        shapes.append(Shape(kind, cname, crgb, x, y, vx, vy, size))
    return shapes

def energize_static_shapes(shapes: List[Shape],
                           min_speed: float,
                           max_speed: float,
                           rng: random.Random):
    for s in shapes:
        if abs(s.vx) + abs(s.vy) <= 1e-6:
            speed = rng.uniform(min_speed, max_speed)
            theta = rng.uniform(0, 2 * math.pi)
            s.vx = speed * math.cos(theta)
            s.vy = speed * math.sin(theta)

# -----------------------
# Background sampling with contrast guard
# -----------------------

def rgb_delta_rms(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    return math.sqrt(((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) / 3.0) / 255.0

def choose_background(shapes: List[Shape],
                      bg_palette: List[Tuple[str,Tuple[int,int,int]]],
                      white_prob: float,
                      min_rms_diff: float = 0.12,
                      rng: Optional[random.Random] = None) -> Tuple[str,Tuple[int,int,int]]:
    rng = rng or random
    candidates = (BG_EXTRA + COLOR_PALETTE) if rng.random() < white_prob else (COLOR_PALETTE + BG_EXTRA)
    for _ in range(16):
        name, rgb = rng.choice(candidates)
        if all(rgb_delta_rms(rgb, s.color_rgb) >= min_rms_diff for s in shapes):
            return name, rgb
    return ("white", (255,255,255))

# -----------------------
# Captions
# -----------------------

def location_bucket(x: float, y: float, W: int, H: int) -> str:
    horiz = "left" if x < W/2 else "right"
    vert  = "top" if y < H/2 else "bottom"
    return f"{vert}-{horiz}"

def make_caption(shapes: List[Shape],
                 fps: float,
                 exposure_factor: float,
                 include_relations: bool,
                 include_blur: bool,
                 for_video: bool,
                 bg_name: str) -> str:
    exposure_ms = 1000.0 * (exposure_factor / float(fps))
    summary = []
    for s in shapes:
        pos = location_bucket(s.x, s.y, W=512, H=512)
        moving = (abs(s.vx) + abs(s.vy)) > 1e-6
        mv = "moving" if moving else "static"
        summary.append(f"a {s.color_name} {s.kind} at the {pos} ({mv})" if include_relations
                       else f"a {s.color_name} {s.kind} ({mv})")
    who = ", ".join(summary)
    base = f"{'Video' if for_video else 'Image'} of {who} on a {bg_name} background. "
    blur = (f"Motion blur corresponds to ~{exposure_ms:.1f} ms exposure at {fps:.1f} fps "
            f"(≈ {exposure_factor*360:.0f}° shutter).")
    return base + (blur if include_blur else "")

# -----------------------
# Frame mid-times from (#frames, sim_fps)
# -----------------------

def frame_mid_times_fixed(num_frames: int, sim_fps: float) -> List[float]:
    """
    Returns num_frames mid-times spaced at Δt = 1/sim_fps:
      t_k = (k + 0.5) / sim_fps, k=0..T-1
    Total simulated duration = num_frames / sim_fps.
    """
    dt = 1.0 / float(sim_fps)
    return [(k + 0.5) * dt for k in range(num_frames)]

# -----------------------
# IO helpers
# -----------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_image(path: str, rgb: np.ndarray):
    Image.fromarray(rgb, mode="RGB").save(path)

def save_video_mp4(path: str, frames_rgb: List[np.ndarray], fps_out: int):
    """
    Save RGB frames as an MP4 using imageio if available; otherwise fall back to PNG frames.
    Ensures the parent directory exists to avoid FileNotFoundError from the backend.
    """
    # Ensure parent dir exists (prevents FileNotFoundError from imageio)
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

    if _HAS_IMAGEIO:
        # Make sure frames are a proper contiguous HxWx3 uint8 stack
        if len(frames_rgb) == 0:
            raise ValueError("save_video_mp4: no frames provided")
        arr = np.stack(frames_rgb, axis=0)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"Expected frames as [T,H,W,3] uint8, got {arr.shape} {arr.dtype}")

        # Some encoders require even dimensions; pad if necessary
        T, H, W, C = arr.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            padH = H + pad_h
            padW = W + pad_w
            pad = np.zeros((T, padH, padW, 3), dtype=np.uint8)
            pad[:, :H, :W, :] = arr
            arr = pad

        # Write
        iio.imwrite(path, arr, fps=fps_out, codec="libx264", quality=8)
    else:
        # Fallback: PNG sequence next to the intended mp4
        seq_dir = os.path.splitext(path)[0] + "_frames"
        os.makedirs(seq_dir, exist_ok=True)
        for i, f in enumerate(frames_rgb):
            Image.fromarray(f).save(os.path.join(seq_dir, f"{i:04d}.png"))
        warnings.warn(f"No mp4 writer; saved PNG frames to {seq_dir}")


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser("Synthetic FPS/Shutter dataset generator (fps range + fixed #frames)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--modes", type=str, default="image", help="image, video, both")

    # NEW: fps range & scale sampling
    ap.add_argument("--fps_lo", type=float, default=10.0)
    ap.add_argument("--fps_hi", type=float, default=250.0)
    ap.add_argument("--num_scales", type=int, default=9, help="# of stratified scale samples in [-1,1]")

    # NEW: video temporal layout
    ap.add_argument("--num_frames", type=int, default=8, help="Frames per video clip (e.g., 8 or 4)")
    ap.add_argument("--sim_fps", type=float, default=16.0,
                    help="Virtual timeline rate to place frame mid-times (independent of shutter)")

    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    # Exposure integration
    ap.add_argument("--samples_per_exposure", type=int, default=32,
                    help="Uniform samples within each exposure window (integration)")
    ap.add_argument("--max_step_s", type=float, default=0.005,
                    help="Integrator max step when advancing motion")

    # Images / videos
    ap.add_argument("--samples_per_scale", type=int, default=10,
                    help="Number of scenes per scale bucket (per mode)")
    ap.add_argument("--ensure_static_video", action="store_true")
    ap.add_argument("--static_prob_video", type=float, default=0.2)
    ap.add_argument("--fps_out_video", type=int, default=16, help="Encoded mp4 fps")

    # Image/video alignment
    ap.add_argument("--align_img_video", action="store_true")
    ap.add_argument("--images_force_all_moving", dest="images_force_all_moving",
                    action="store_true", default=True)
    ap.add_argument("--no_images_force_all_moving", dest="images_force_all_moving",
                    action="store_false")

    # Motion & scene
    ap.add_argument("--min_speed", type=float, default=60.0, help="px/s")
    ap.add_argument("--max_speed", type=float, default=220.0, help="px/s")
    ap.add_argument("--min_objs", type=int, default=2)
    ap.add_argument("--max_objs", type=int, default=4)

    # Background control
    ap.add_argument("--bg_white_prob", type=float, default=0.8,
                    help="Probability to use white background; otherwise pick from palette")
    ap.add_argument("--bg_min_rms_diff", type=float, default=0.12,
                    help="Min RMS color distance (0..1) between bg and any shape color")

    # Captions
    ap.add_argument("--caption_relations", action="store_true")
    ap.add_argument("--caption_blur_note", action="store_true")

    # Repro
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    modes = args.modes.lower()
    do_image = modes in ("image","both")
    do_video = modes in ("video","both")
    align_mv = args.align_img_video and do_image and do_video

    # Precompute scales and mapped fps values
    scales = stratified_scales(args.num_scales, seed=args.seed)
    fps_per_scale = [fps_from_scale(s, args.fps_lo, args.fps_hi) for s in scales]
    W, H = args.width, args.height

    # Roots
    img_root = os.path.join(args.out_dir, "images")
    vid_root = os.path.join(args.out_dir, "videos")
    if do_image: ensure_dir(img_root)
    if do_video: ensure_dir(vid_root)

    if do_image: print("==> Generating IMAGE dataset")
    if do_video:
        print("==> Generating VIDEO dataset")
        print(f"    Using fixed {args.num_frames} frames at sim_fps={args.sim_fps} Hz "
              f"(simulated duration = {args.num_frames/args.sim_fps:.3f}s)")
        mid_times_fixed = frame_mid_times_fixed(args.num_frames, args.sim_fps)

    pbar = tqdm(range(args.samples_per_scale), desc="sampling base scenes")
    for idx in pbar:
        # Deterministic seeds per index
        seed_shared = args.seed * 1_000_003 + idx * 97 + 100003
        seed_img    = args.seed * 1_000_003 + idx * 97 + 1
        seed_vid    = args.seed * 1_000_003 + idx * 97 + 2

        # Sample bases
        if align_mv:
            rng_shared = random.Random(seed_shared)
            shared_base = sample_scene(
                W, H,
                num_objs_range=(args.min_objs, args.max_objs),
                speed_px_s=(args.min_speed, args.max_speed),
                allow_static=True,
                ensure_one_static=args.ensure_static_video,
                static_prob=args.static_prob_video,
                rng=rng_shared
            )
            shapes_img_base = [Shape(**vars(s)) for s in shared_base]
            shapes_vid_base = [Shape(**vars(s)) for s in shared_base]
            if args.images_force_all_moving:
                rng_fix = random.Random(seed_img ^ 0xBEEF)
                energize_static_shapes(shapes_img_base, args.min_speed, args.max_speed, rng_fix)
        else:
            rng_img = random.Random(seed_img)
            rng_vid = random.Random(seed_vid)
            shapes_img_base = sample_scene(
                W, H,
                num_objs_range=(args.min_objs, args.max_objs),
                speed_px_s=(args.min_speed, args.max_speed),
                allow_static=False,
                ensure_one_static=False,
                rng=rng_img
            )
            shapes_vid_base = sample_scene(
                W, H,
                num_objs_range=(args.min_objs, args.max_objs),
                speed_px_s=(args.min_speed, args.max_speed),
                allow_static=True,
                ensure_one_static=args.ensure_static_video,
                static_prob=args.static_prob_video,
                rng=rng_vid
            )

        # Choose background (same bg per sample across all scales for fairness)
        rng_bg = random.Random(seed_shared ^ 0xABCDEF)
        bg_name, bg_rgb = choose_background(
            shapes_img_base if do_image else shapes_vid_base,
            bg_palette=COLOR_PALETTE + BG_EXTRA,
            white_prob=args.bg_white_prob,
            min_rms_diff=args.bg_min_rms_diff,
            rng=rng_bg
        )

        # --- IMAGE ---
        if do_image:
            # Use the midpoint of the simulated duration for the image exposure center
            t_mid = 0.5 * (args.num_frames / args.sim_fps)
            for s_val, fps in zip(scales, fps_per_scale):
                scale_dir = os.path.join(img_root, f"{s_val:0.3f}")
                ensure_dir(scale_dir)

                exposure_span = 1.0 / float(fps)
                t_start = max(0.0, t_mid - 0.5 * exposure_span)
                s_local = [Shape(**vars(s)) for s in shapes_img_base]
                advance_time(s_local, t_start, W, H, args.max_step_s)

                acc = np.zeros((H, W, 3), dtype=np.float32)
                dt = exposure_span / max(1, args.samples_per_exposure)
                for _ in range(args.samples_per_exposure):
                    acc += render_shapes_pil(W, H, s_local, bg_rgb).astype(np.float32)
                    advance_time(s_local, dt, W, H, args.max_step_s)
                img = np.clip(acc / float(args.samples_per_exposure), 0, 255).astype(np.uint8)

                stem = f"scene_{idx:06d}.png"
                save_image(os.path.join(scale_dir, stem), img)

                if args.caption_blur_note or args.caption_relations:
                    cap = make_caption(
                        shapes_img_base, fps=fps, exposure_factor=1.0,
                        include_relations=args.caption_relations,
                        include_blur=args.caption_blur_note, for_video=False,
                        bg_name=bg_name
                    )
                    with open(os.path.join(scale_dir, f"scene_{idx:06d}.txt"), "w") as f:
                        f.write(cap)

        # --- VIDEO ---
        if do_video:
            mid_times = frame_mid_times_fixed(args.num_frames, args.sim_fps)
            for s_val, fps in zip(scales, fps_per_scale):
                scale_dir = os.path.join(vid_root, f"{s_val:0.3f}")
                ensure_dir(scale_dir)

                exposure_span = 1.0 / float(fps)
                s_local = [Shape(**vars(s)) for s in shapes_vid_base]
                t_cur = 0.0
                frames: List[np.ndarray] = []
                for t_mid in mid_times:
                    t_start = max(0.0, t_mid - 0.5 * exposure_span)
                    advance_time(s_local, t_start - t_cur, W, H, args.max_step_s); t_cur = t_start

                    acc = np.zeros((H, W, 3), dtype=np.float32)
                    dt = exposure_span / max(1, args.samples_per_exposure)
                    for _ in range(args.samples_per_exposure):
                        acc += render_shapes_pil(W, H, s_local, bg_rgb).astype(np.float32)
                        advance_time(s_local, dt, W, H, args.max_step_s); t_cur += dt
                    frame = np.clip(acc / float(args.samples_per_exposure), 0, 255).astype(np.uint8)
                    frames.append(frame)

                vid_name = f"scene_{idx:06d}.mp4"
                save_video_mp4(os.path.join(scale_dir, vid_name), frames, fps_out=args.fps_out_video)

                if args.caption_blur_note or args.caption_relations:
                    cap = make_caption(
                        shapes_vid_base, fps=fps, exposure_factor=1.0,
                        include_relations=args.caption_relations,
                        include_blur=args.caption_blur_note, for_video=True,
                        bg_name=bg_name
                    )
                    with open(os.path.join(scale_dir, f"scene_{idx:06d}.txt"), "w") as f:
                        f.write(cap)

    print("Done.")

if __name__ == "__main__":
    main()
