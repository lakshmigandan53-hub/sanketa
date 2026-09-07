# -*- coding: utf-8 -*-
"""
SANKETA — ISL Data Collection Tool
backend/collect_data.py

Collects real hand landmark samples per member per word from the webcam.

Usage:
  python collect_data.py --member member_01
  python collect_data.py --member member_01 --samples 200
  python collect_data.py --member member_01 --samples 200 --camera 0

Controls during collection:
  SPACE    — capture current frame as a sample
  N        — next word
  P        — previous word
  R        — show current word again (same word)
  Q / ESC  — quit (all progress is saved)

Output:
  datasets/raw/<member_id>/<WORD>/sample_XXXX.npy
  Each file is float32 array shape (63,) — raw MediaPipe x,y,z landmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Suppress TF / MediaPipe noise before any import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent
_CONFIG_PATH = _BACKEND_DIR / "config" / "words.json"
_DATASET_DIR = _BACKEND_DIR / "datasets" / "raw"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils
_mp_draw_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        logger.error("Config file not found: %s", _CONFIG_PATH)
        sys.exit(1)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_member_config(config: dict, member_id: str) -> dict:
    members = config.get("members", {})
    if member_id not in members:
        valid = ", ".join(sorted(members.keys()))
        logger.error("Member '%s' not found in config. Valid members: %s", member_id, valid)
        sys.exit(1)
    return members[member_id]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def count_samples(word_dir: Path) -> int:
    if not word_dir.exists():
        return 0
    return sum(1 for f in word_dir.iterdir() if f.suffix == ".npy")


def next_sample_path(word_dir: Path) -> Path:
    word_dir.mkdir(parents=True, exist_ok=True)
    idx = count_samples(word_dir)
    return word_dir / f"sample_{idx:04d}.npy"


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_landmarks(results) -> np.ndarray | None:
    """
    Returns (63,) float32 or None if no hand found.
    Preprocessing: raw MediaPipe normalized coords, flattened.
    This must match sign_recognition.py / mediapipe_service.py exactly.
    """
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand.landmark],
        dtype=np.float32,
    )  # (21, 3)
    return coords.flatten()  # (63,)


def is_valid_landmark(vec: np.ndarray) -> bool:
    """Basic sanity checks on the 63-feature vector."""
    if vec is None or vec.shape != (63,):
        return False
    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
        return False
    if np.allclose(vec, 0.0):
        return False
    return True


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

_CYAN   = (0, 200, 255)
_GREEN  = (0, 220, 80)
_RED    = (0, 60, 220)
_GRAY   = (150, 150, 150)
_WHITE  = (255, 255, 255)
_DARK   = (18, 18, 18)
_ORANGE = (0, 165, 255)


def draw_hud(
    frame: np.ndarray,
    word: str,
    word_type: str,
    word_idx: int,
    word_total: int,
    member_name: str,
    collected: int,
    target: int,
    hand_ok: bool,
    status_msg: str,
    status_color: tuple,
    fps: float,
    delay_active: bool,
    all_words: list[str],
    word_dir: Path,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Top banner
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 88), _DARK, -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, "SANKETA - ISL Data Collection",
                (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.68, _CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{member_name}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}",
                (w - 100, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Word {word_idx + 1}/{word_total}",
                (w - 100, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GRAY, 1, cv2.LINE_AA)

    # Current word box
    bx, by = 10, 100
    cv2.rectangle(frame, (bx, by), (bx + 180, by + 100), _DARK, -1)
    cv2.rectangle(frame, (bx, by), (bx + 180, by + 100), _CYAN, 2)
    display_word = word.replace("_", " ")
    font_scale = 0.65 if len(display_word) > 8 else 0.9
    cv2.putText(frame, display_word,
                (bx + 8, by + 48), cv2.FONT_HERSHEY_DUPLEX, font_scale, _GREEN, 2, cv2.LINE_AA)
    type_color = _ORANGE if word_type == "dynamic" else _CYAN
    cv2.putText(frame, f"[{word_type.upper()}]",
                (bx + 8, by + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, type_color, 1, cv2.LINE_AA)

    # Progress bar
    pct = min(collected / max(target, 1), 1.0)
    bar_x, bar_y = 200, 108
    bar_w = w - bar_x - 20
    bar_h = 26
    fill_col = _GREEN if pct >= 1.0 else _CYAN
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h), fill_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    cv2.putText(frame, f"{collected}/{target} samples  ({pct * 100:.0f}%)",
                (bar_x + 6, bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1, cv2.LINE_AA)

    # Hand status
    hs_col = _GREEN if hand_ok else _RED
    hs_txt = "Hand: DETECTED [OK]" if hand_ok else "Hand: NOT VISIBLE - show hand!"
    cv2.putText(frame, hs_txt,
                (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hs_col, 1, cv2.LINE_AA)

    # Capture cooldown indicator
    if delay_active:
        cv2.putText(frame, "[WAIT] COOLDOWN ...",
                    (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _ORANGE, 1, cv2.LINE_AA)

    # Status line
    cv2.putText(frame, status_msg,
                (200, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

    # Word list sidebar (right) — scrolling window for up to 20 words
    sx = w - 175
    sy = 105
    max_visible = min(12, len(all_words))  # show up to 12 words at a time
    # Center the window on the current word
    half = max_visible // 2
    start_i = max(0, min(word_idx - half, len(all_words) - max_visible))
    end_i   = start_i + max_visible
    cv2.putText(frame, f"Words ({word_idx+1}/{len(all_words)}):",
                (sx, sy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY, 1, cv2.LINE_AA)
    if start_i > 0:
        cv2.putText(frame, "  ^ more", (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _GRAY, 1, cv2.LINE_AA)
        sy += 14
    for i in range(start_i, end_i):
        if i >= len(all_words):
            break
        wrd = all_words[i]
        cnt = count_samples(_DATASET_DIR / word_dir.parent.name / wrd)
        done = cnt >= target
        cur  = i == word_idx
        col  = _GREEN if done else (_CYAN if cur else _GRAY)
        prefix = "> " if cur else "  "
        wlabel = wrd.replace("_", " ")
        if len(wlabel) > 10:
            wlabel = wlabel[:9] + "."
        cv2.putText(frame, f"{prefix}{wlabel} {cnt}",
                    (sx, sy + (i - start_i) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    if end_i < len(all_words):
        cv2.putText(frame, "  v more",
                    (sx, sy + max_visible * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _GRAY, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(frame, "SPACE=capture | N=next | P=prev | Q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, _GRAY, 1, cv2.LINE_AA)

    # Variation tip for static signs
    if word_type == "static":
        cv2.putText(frame, "TIP: Vary hand angle, distance, lighting between captures",
                    (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _ORANGE, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "TIP: Capture at different points during the sign motion",
                    (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _ORANGE, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="collect_data.py",
        description="SANKETA - ISL hand landmark data collector",
    )
    p.add_argument("--member", required=True,
                   help="Member ID, e.g. member_01")
    p.add_argument("--samples", type=int, default=None,
                   help="Target samples per word (default: from words.json)")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device ID (default: 0)")
    p.add_argument("--delay", type=float, default=None,
                   help="Min seconds between captures (default: from words.json)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config()
    member_cfg = get_member_config(config, args.member)

    target_samples = args.samples or config.get("target_samples_per_word", 200)
    capture_delay  = args.delay  or config.get("capture_delay_seconds", 0.4)
    member_name    = member_cfg.get("display_name", args.member)
    words          = member_cfg["words"]
    word_types     = member_cfg.get("word_types", {})

    member_dir = _DATASET_DIR / args.member

    print()
    print("=" * 60)
    print(f"  SANKETA - Data Collection")
    print(f"  Member  : {member_name}  ({args.member})")
    print(f"  Words   : {len(words)}")
    print(f"  Target  : {target_samples} samples/word")
    print(f"  Delay   : {capture_delay}s between captures")
    print(f"  Camera  : {args.camera}")
    print(f"  Output  : {member_dir}")
    print("=" * 60)
    print()
    print("Controls: SPACE=capture | N=next word | P=prev | Q=quit")
    print()

    # Find starting word (first incomplete)
    start_idx = 0
    for i, w in enumerate(words):
        if count_samples(member_dir / w) < target_samples:
            start_idx = i
            break
    else:
        print("  [OK] All words already have enough samples!")
        print("  Run: python validate_dataset.py")
        print("  Then: python train_static_20class.py")
        return

    print(f"  Resuming from word: {words[start_idx]}")
    print()

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d. Try --camera 1 or 2.", args.camera)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    word_idx      = start_idx
    status_msg    = "Show your hand, then press SPACE to capture"
    status_color  = _GRAY
    last_capture  = 0.0
    prev_time     = time.time()
    fps           = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera read failed.")
            break

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = cv2.flip(frame, 1)  # mirror

        # MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                _mp_draw.draw_landmarks(
                    frame, hl, _mp_hands.HAND_CONNECTIONS,
                    _mp_draw_styles.get_default_hand_landmarks_style(),
                    _mp_draw_styles.get_default_hand_connections_style(),
                )

        landmarks = extract_landmarks(results)
        hand_ok = is_valid_landmark(landmarks)

        word     = words[word_idx]
        wtype    = word_types.get(word, "static")
        word_dir = member_dir / word
        collected = count_samples(word_dir)

        # Auto-advance when word is complete
        if collected >= target_samples:
            word_idx += 1
            if word_idx >= len(words):
                print("\n  [OK] All words for this member collected!")
                print("  Run: python validate_dataset.py")
                print("  Then: python train_static_20class.py")
                break
            word = words[word_idx]
            word_dir = member_dir / word
            collected = count_samples(word_dir)
            status_msg   = f"Word complete! Now: {word.replace('_', ' ')}"
            status_color = _CYAN

        delay_active = (time.time() - last_capture) < capture_delay

        frame = draw_hud(
            frame, word, wtype, word_idx, len(words),
            member_name, collected, target_samples,
            hand_ok, status_msg, status_color, fps,
            delay_active, words, word_dir,
        )

        cv2.imshow(f"SANKETA - {member_name} Data Collection (Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key == ord(" "):  # SPACE - capture
            if not hand_ok:
                status_msg   = "No hand detected - show hand first!"
                status_color = _RED
            elif delay_active:
                remaining = capture_delay - (time.time() - last_capture)
                status_msg   = f"Wait {remaining:.1f}s before next capture"
                status_color = _ORANGE
            else:
                path = next_sample_path(word_dir)
                np.save(str(path), landmarks)
                last_capture = time.time()
                collected += 1
                status_msg   = f"[OK] Saved sample {collected}/{target_samples}"
                status_color = _GREEN
                logger.info("Saved %s", path.name)

        elif key in (ord("n"), ord("N")):
            word_idx = min(word_idx + 1, len(words) - 1)
            status_msg   = f"-> {words[word_idx].replace('_', ' ')}"
            status_color = _CYAN

        elif key in (ord("p"), ord("P")):
            word_idx = max(word_idx - 1, 0)
            status_msg   = f"<- {words[word_idx].replace('_', ' ')}"
            status_color = _CYAN

        elif key in (ord("r"), ord("R")):
            status_msg   = f"Same word: {words[word_idx].replace('_', ' ')}"
            status_color = _CYAN

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Summary
    print()
    print("=" * 60)
    print(f"  Session Summary — {member_name}")
    print("=" * 60)
    total = 0
    complete = 0
    for w in words:
        cnt = count_samples(member_dir / w)
        total += cnt
        done = cnt >= target_samples
        if done:
            complete += 1
        flag = "[OK]" if done else f"({cnt}/{target_samples})"
        print(f"  {w:<20} {cnt:>4} samples  {flag}")
    print(f"\n  {complete}/{len(words)} words complete   {total} total samples")
    if complete == len(words):
        print("\n  [OK] Ready! Run: python validate_dataset.py")
        print("  Then: python train_static_20class.py")
    else:
        print(f"\n  Re-run: python collect_data.py --member {args.member} to continue.")
    print("=" * 60)


if __name__ == "__main__":
    main()
