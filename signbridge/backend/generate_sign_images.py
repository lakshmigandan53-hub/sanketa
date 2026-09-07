# -*- coding: utf-8 -*-
"""
SANKETA — Sign Skeleton Image Generator
backend/generate_sign_images.py

Generates realistic ISL hand skeleton reference images from our OWN collected
landmark data (.npy files). Uses OpenCV + MediaPipe drawing utilities.

For STATIC signs: renders the median/representative pose from 200 samples.
For DYNAMIC signs: renders a 3-frame GIF showing motion trajectory.

Output: web/public/signs/{WORD}.png and web/public/signs/{WORD}.gif

These images are derived from our own ISL training data — no copyright issues.
"""

import os, sys, json
import numpy as np
import cv2
from pathlib import Path

# Try to import PIL for GIF generation
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] PIL not available — GIFs will not be generated. Install with: pip install Pillow")

BACKEND_DIR = Path(__file__).parent
STATIC_DIR  = BACKEND_DIR / "datasets" / "raw" / "member_01"
DYNAMIC_DIR = BACKEND_DIR / "datasets" / "raw" / "member_01_dynamic"
OUTPUT_DIR  = BACKEND_DIR.parent / "web" / "public" / "signs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe hand connections (21 landmarks)
# Based on MediaPipe Hands topology
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17),          # Palm
]

IMG_SIZE = 256
BACKGROUND_COLOR = (15, 20, 35)   # Dark background matching app theme
LANDMARK_COLOR   = (0, 229, 255)  # Cyan — matches --primary CSS variable
CONNECTION_COLOR = (0, 180, 210)
FINGERTIP_COLOR  = (255, 120, 50) # Orange fingertips
FINGERTIPS       = {4, 8, 12, 16, 20}
WRIST            = 0

WORD_LABELS = {
    "HELLO":     "👋 Hello",
    "YES":       "✅ Yes",
    "NO":        "❌ No",
    "WATER":     "💧 Water",
    "THANK_YOU": "🙏 Thank You",
    "PLEASE":    "🤲 Please",
    "SORRY":     "😔 Sorry",
    "HELP":      "🆘 Help",
}


def landmarks_to_pixel(arr_63, size=IMG_SIZE, margin=30):
    """Convert flat (63,) landmark array to pixel coordinates."""
    lm = arr_63.reshape(21, 3)
    xs = lm[:, 0]
    ys = lm[:, 1]

    # Normalize to fill the canvas
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = max(x_max - x_min, 1e-5)
    y_range = max(y_max - y_min, 1e-5)

    # Keep aspect ratio
    scale = (size - 2 * margin) / max(x_range, y_range)
    px_x = ((xs - x_min) * scale + margin).astype(int)
    px_y = ((ys - y_min) * scale + margin).astype(int)

    # Center
    cx = (px_x.min() + px_x.max()) // 2
    cy = (px_y.min() + px_y.max()) // 2
    off_x = size // 2 - cx
    off_y = size // 2 - cy
    px_x = np.clip(px_x + off_x, 0, size - 1)
    px_y = np.clip(px_y + off_y, 0, size - 1)

    return list(zip(px_x.tolist(), px_y.tolist()))


def draw_hand_skeleton(img, pixels, alpha=1.0):
    """Draw hand skeleton on OpenCV image."""
    # Draw connections
    for a, b in HAND_CONNECTIONS:
        if a < len(pixels) and b < len(pixels):
            pt1 = pixels[a]
            pt2 = pixels[b]
            cv2.line(img, pt1, pt2, CONNECTION_COLOR, 2, cv2.LINE_AA)

    # Draw landmarks
    for i, pt in enumerate(pixels):
        color = FINGERTIP_COLOR if i in FINGERTIPS else (80, 200, 255)
        r = 5 if i in FINGERTIPS else (7 if i == WRIST else 4)
        cv2.circle(img, pt, r, color, -1, cv2.LINE_AA)
        # Highlight wrist
        if i == WRIST:
            cv2.circle(img, pt, r + 3, (100, 150, 255), 1, cv2.LINE_AA)

    return img


def get_representative_sample(class_dir, n_pick=5):
    """Get the median-like representative samples from a class directory."""
    files = sorted(class_dir.glob("*.npy"))
    if not files:
        return []

    # Sample evenly through the collected data
    n = len(files)
    indices = [int(n * i / n_pick) for i in range(n_pick)]
    samples = []
    for idx in indices:
        try:
            arr = np.load(files[idx]).astype(np.float32)
            if arr.shape == (21, 3):
                arr = arr.flatten()
            if arr.shape == (63,) and not np.isnan(arr).any():
                samples.append(arr)
        except Exception:
            pass
    return samples


def render_static_sign(word, samples):
    """Render representative static sign — composite of a few samples."""
    # Use the middle sample as primary
    mid = samples[len(samples) // 2]
    img = np.full((IMG_SIZE, IMG_SIZE, 3), BACKGROUND_COLOR, dtype=np.uint8)

    pixels = landmarks_to_pixel(mid)
    img = draw_hand_skeleton(img, pixels)

    # Add word label at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    display = word.replace("_", " ")
    (tw, th), _ = cv2.getTextSize(display, font, 0.65, 2)
    tx = (IMG_SIZE - tw) // 2
    ty = IMG_SIZE - 18
    cv2.putText(img, display, (tx, ty), font, 0.65, (200, 220, 255), 2, cv2.LINE_AA)

    # Add type indicator
    cv2.putText(img, "ISL", (8, 22), font, 0.5, (0, 180, 180), 1, cv2.LINE_AA)

    return img


def render_dynamic_frames(word, class_dir, n_frames=4):
    """Render N frames showing the motion trajectory of a dynamic sign."""
    files = sorted(class_dir.glob("*.npy"))
    if not files:
        return []

    # Pick a representative sequence
    mid_file = files[len(files) // 2]
    try:
        seq = np.load(mid_file).astype(np.float32)  # (30, 63)
        if seq.shape != (30, 63):
            return []
    except Exception:
        return []

    # Pick evenly-spaced frames from the sequence
    frame_indices = [int(30 * i / n_frames) for i in range(n_frames)]
    frames = []

    for fi in frame_indices:
        img = np.full((IMG_SIZE, IMG_SIZE, 3), BACKGROUND_COLOR, dtype=np.uint8)
        pixels = landmarks_to_pixel(seq[fi])
        img = draw_hand_skeleton(img, pixels)

        # Frame counter
        font = cv2.FONT_HERSHEY_SIMPLEX
        display = word.replace("_", " ")
        (tw, th), _ = cv2.getTextSize(display, font, 0.65, 2)
        tx = (IMG_SIZE - tw) // 2
        ty = IMG_SIZE - 18
        cv2.putText(img, display, (tx, ty), font, 0.65, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"ISL  frame {fi+1}/{30}", (8, 22), font, 0.42, (0, 180, 180), 1, cv2.LINE_AA)

        frames.append(img)

    return frames


def save_png(img, path):
    cv2.imwrite(str(path), img)
    print(f"  [PNG] Saved: {path.name} ({path.stat().st_size // 1024} KB)")


def save_gif(frames_bgr, path, duration_ms=300):
    if not PIL_AVAILABLE:
        print(f"  [GIF] Skipped (PIL not available): {path.name}")
        return
    pil_frames = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))

    if pil_frames:
        pil_frames[0].save(
            str(path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"  [GIF] Saved: {path.name} ({path.stat().st_size // 1024} KB)")


def main():
    print("=" * 60)
    print("SANKETA — Sign Skeleton Image Generator")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    generated = []
    failed = []

    # ── Static signs (single-frame PNG) ──────────────────────────────────
    static_words = ["HELLO", "YES", "NO", "WATER"]
    print(f"\n--- Generating STATIC sign PNGs ---")
    for word in static_words:
        class_dir = STATIC_DIR / word
        if not class_dir.exists():
            print(f"  SKIP {word}: directory not found")
            failed.append(word)
            continue

        samples = get_representative_sample(class_dir, n_pick=5)
        if not samples:
            print(f"  SKIP {word}: no valid samples")
            failed.append(word)
            continue

        img = render_static_sign(word, samples)
        out_png = OUTPUT_DIR / f"{word}.png"
        save_png(img, out_png)
        generated.append(f"{word}.png")

    # ── Dynamic signs (multi-frame GIF + single PNG) ──────────────────────
    dynamic_words = ["THANK_YOU", "PLEASE", "SORRY", "HELP"]
    print(f"\n--- Generating DYNAMIC sign GIFs ---")
    for word in dynamic_words:
        class_dir = DYNAMIC_DIR / word
        if not class_dir.exists():
            # Fallback: use static samples (same word exists in member_01)
            class_dir = STATIC_DIR / word
            if not class_dir.exists():
                print(f"  SKIP {word}: no data found")
                failed.append(word)
                continue
            print(f"  Using static fallback for {word}")
            samples = get_representative_sample(class_dir, n_pick=5)
            if not samples:
                print(f"  SKIP {word}: no valid samples")
                failed.append(word)
                continue
            img = render_static_sign(word, samples)
            out_png = OUTPUT_DIR / f"{word}.png"
            save_png(img, out_png)
            generated.append(f"{word}.png")
            continue

        # Generate multi-frame GIF from sequence
        frames = render_dynamic_frames(word, class_dir, n_frames=4)
        if not frames:
            print(f"  SKIP {word}: could not render frames")
            failed.append(word)
            continue

        # Save GIF
        out_gif = OUTPUT_DIR / f"{word}.gif"
        save_gif(frames, out_gif, duration_ms=280)
        if out_gif.exists():
            generated.append(f"{word}.gif")

        # Also save first frame as PNG fallback
        out_png = OUTPUT_DIR / f"{word}.png"
        save_png(frames[0], out_png)
        generated.append(f"{word}.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GENERATED: {len(generated)} files")
    for f in generated:
        print(f"  [OK] {f}")
    if failed:
        print(f"\nFAILED: {len(failed)}")
        for f in failed:
            print(f"  [FAIL] {f}")

    # Save manifest
    manifest = {
        "generated": generated,
        "failed": failed,
        "output_dir": str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"\nManifest saved to {OUTPUT_DIR}/manifest.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
