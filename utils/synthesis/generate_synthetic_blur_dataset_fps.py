#!/usr/bin/env python3
"""
Synthetic FPS/Shutter dataset generator (2D shapes, absolute-time synthesis)

This version uses --num_frames (integer) instead of --duration_s.
- For FPS f and 360° shutter, frame k integrates over [t_k - 1/(2f), t_k + 1/(2f)].
- Two timeline modes:
  * per_fps        : each FPS uses its own (k+0.5)/f mid-times → exactly num_frames frames.
  * align_min_fps  : use mid-times of the smallest FPS for ALL videos → same #frames & timings across FPS;
                     only exposure span (1/f) differs.
- Optional image/video context alignment: --align_img_video.
"""

import argparse, math, os, random, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Try to enable mp4 writing; otherwise fallback to PNG sequence
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

def render_shapes_pil(W: int, H: int, shapes: List[Shape]) -> np.ndarray:
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
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
    """Advance positions by dt with one-step bounce handling (small dt recommended)."""
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
    """Advance shapes by dt using small substeps to avoid skipping multiple bounces."""
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
                 include_blur: bool,
                 for_video: bool) -> str:
    exposure_ms = 1000.0 * (exposure_factor / float(fps))
    summary = []
    for s in shapes:
        pos = location_bucket(s.x, s.y, W=512, H=512)
        moving = (abs(s.vx) + abs(s.vy)) > 1e-6
        mv = "moving" if moving else "static"
        summary.append(f"a {s.color_name} {s.kind} at the {pos} ({mv})" if include_relations
                       else f"a {s.color_name} {s.kind} ({mv})")
    who = ", ".join(summary)
    base = f"{'Video' if for_video else 'Image'} of {who}. "
    blur = (
        f"Motion blur corresponds to ~{exposure_ms:.1f} ms exposure at {fps} fps "
        f"(≈ {exposure_factor*360:.0f}° shutter)."
    )
    return base + (blur if include_blur else "")

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
# Frame time scheduling (num_frames-based)
# -----------------------

def frame_mid_times_per_fps(num_frames: int, fps: int) -> List[float]:
    """
    Generate exactly num_frames mid-exposure times for this fps:
    t_k = (k + 0.5) / fps
    """
    return [(k + 0.5) / float(fps) for k in range(num_frames)]

def frame_mid_times_align_min(num_frames: int, min_fps: int) -> List[float]:
    """Same mid-times for all fps, based on smallest fps."""
    return frame_mid_times_per_fps(num_frames, min_fps)

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser("Synthetic FPS/Shutter dataset generator (num-frames version)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--modes", type=str, default="image", help="image, video, both")
    ap.add_argument("--fps_list", type=str, default="12,24,40,60,120,240")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    # Synthesis timeline (now frame-count based)
    ap.add_argument("--num_frames", type=int, default=72,
                    help="Number of frames to synthesize per sequence for each FPS.")
    ap.add_argument("--samples_per_exposure", type=int, default=32,
                    help="Uniform samples within each exposure window (integration).")
    ap.add_argument("--max_step_s", type=float, default=0.005,
                    help="Integrator max step (s) when advancing to absolute times.")
    ap.add_argument("--timeline", choices=["per_fps","align_min_fps"], default="align_min_fps",
                    help="per_fps: each fps uses its own times; align_min_fps: use smallest fps times for all.")

    # Images / videos
    ap.add_argument("--samples_per_fps", type=int, default=100,
                    help="Number of scenes per FPS bucket (per mode).")
    ap.add_argument("--ensure_static_video", action="store_true",
                    help="Guarantee exactly one static object in video scenes (when not aligning with images).")
    ap.add_argument("--static_prob_video", type=float, default=0.2)

    # Image/video alignment
    ap.add_argument("--align_img_video", action="store_true",
                    help="When modes=both, reuse ONE shared base scene per index across image & video.")
    ap.add_argument("--images_force_all_moving", dest="images_force_all_moving",
                    action="store_true", default=True,
                    help="(Default) If aligned, force static shapes to move in image branch.")
    ap.add_argument("--no_images_force_all_moving", dest="images_force_all_moving",
                    action="store_false",
                    help="If aligned, keep static shapes static in images too.")

    # Motion & scene
    ap.add_argument("--min_speed", type=float, default=60.0, help="px/s")
    ap.add_argument("--max_speed", type=float, default=220.0, help="px/s")
    ap.add_argument("--min_objs", type=int, default=2)
    ap.add_argument("--max_objs", type=int, default=4)

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

    fps_values = [float(x.strip()) for x in args.fps_list.split(",") if x.strip()]
    min_fps = min(fps_values)
    W, H = args.width, args.height

    # Derive an effective duration from num_frames and the smallest FPS
    # (this keeps the image mid-time logic consistent)
    effective_duration_s = args.num_frames / float(min_fps)

    # Roots
    img_root = os.path.join(args.out_dir, "images")
    vid_root = os.path.join(args.out_dir, "videos")
    if do_image: ensure_dir(img_root)
    if do_video: ensure_dir(vid_root)

    if do_image: print("==> Generating IMAGE dataset")
    if do_video:
        print("==> Generating VIDEO dataset")
        print(f"    Timeline mode: {args.timeline} (num_frames={args.num_frames}, samples/exposure={args.samples_per_exposure})")
        if args.timeline == "align_min_fps":
            print(f"    Using smallest FPS = {min_fps} to define shared frame mid-times")

    pbar = tqdm(range(args.samples_per_fps), desc="sampling base scenes")
    for idx in pbar:
        # Deterministic seeds
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

        # --- IMAGE ---
        if do_image:
            out_img_dirs = {}
            for fps in fps_values:
                out_img_dir = os.path.join(img_root, f"{fps}")
                ensure_dir(out_img_dir)
                out_img_dirs[fps] = out_img_dir

            # Render a single image at the mid-time of the whole (effective) duration
            t_mid = 0.5 * effective_duration_s
            for fps in fps_values:
                exposure_span = 1.0 / float(fps)
                t_start = max(0.0, t_mid - 0.5 * exposure_span)
                s_local = [Shape(**vars(s)) for s in shapes_img_base]
                # advance to window start
                advance_time(s_local, t_start, W, H, args.max_step_s)
                # integrate
                acc = np.zeros((H, W, 3), dtype=np.float32)
                dt = exposure_span / max(1, args.samples_per_exposure)
                for _ in range(args.samples_per_exposure):
                    acc += render_shapes_pil(W, H, s_local).astype(np.float32)
                    advance_time(s_local, dt, W, H, args.max_step_s)
                img = np.clip(acc / float(args.samples_per_exposure), 0, 255).astype(np.uint8)

                save_image(os.path.join(out_img_dirs[fps], f"scene_{idx:06d}.png"), img)
                if args.caption_blur_note or args.caption_relations:
                    cap = make_caption(
                        shapes_img_base, fps=fps, exposure_factor=1.0,
                        include_relations=args.caption_relations,
                        include_blur=args.caption_blur_note, for_video=False
                    )
                    with open(os.path.join(out_img_dirs[fps], f"scene_{idx:06d}.txt"), "w") as f:
                        f.write(cap)

        # --- VIDEO ---
        if do_video:
            # Precompute shared frame mid-times
            base_times = (frame_mid_times_align_min(args.num_frames, int(min_fps))
                          if args.timeline == "align_min_fps"
                          else None)

            for fps in fps_values:
                out_vid_dir = os.path.join(vid_root, f"{fps}"); ensure_dir(out_vid_dir)
                exposure_span = 1.0 / float(fps)

                # Determine mid-times for this fps
                if base_times is not None:
                    mid_times = base_times
                else:
                    mid_times = frame_mid_times_per_fps(args.num_frames, int(fps))

                # Build local state starting from t=0 and march forward in absolute time
                s_local = [Shape(**vars(s)) for s in shapes_vid_base]
                t_cur = 0.0
                frames: List[np.ndarray] = []
                for t_mid in mid_times:
                    t_start = max(0.0, t_mid - 0.5 * exposure_span)
                    advance_time(s_local, t_start - t_cur, W, H, args.max_step_s); t_cur = t_start
                    # integrate uniformly within the shutter window
                    acc = np.zeros((H, W, 3), dtype=np.float32)
                    dt = exposure_span / max(1, args.samples_per_exposure)
                    for _ in range(args.samples_per_exposure):
                        acc += render_shapes_pil(W, H, s_local).astype(np.float32)
                        advance_time(s_local, dt, W, H, args.max_step_s); t_cur += dt
                    frame = np.clip(acc / float(args.samples_per_exposure), 0, 255).astype(np.uint8)
                    frames.append(frame)

                vid_name = f"scene_{idx:06d}.mp4"
                # You can still fix output fps to 16 for viewing
                save_video_mp4(os.path.join(out_vid_dir, vid_name), frames, fps=16)

                if args.caption_blur_note or args.caption_relations:
                    cap = make_caption(
                        shapes_vid_base, fps=int(fps), exposure_factor=1.0,
                        include_relations=args.caption_relations,
                        include_blur=args.caption_blur_note, for_video=True
                    )
                    with open(os.path.join(out_vid_dir, f"scene_{idx:06d}.txt"), "w") as f:
                        f.write(cap)

    print("Done.")

if __name__ == "__main__":
    main()
