# -*- coding: utf-8 -*-
"""
SANKETA — Dynamic ISL Sequence Data Collection Tool
backend/collect_data_dynamic.py

Collects multi-frame sequences of hand landmarks for dynamic ISL words
(e.g., THANK_YOU, PLEASE, SORRY, HELP).

Contract:
  - 30 frames per sequence
  - 21 landmarks x (x, y, z) = 63 features per frame
  - Saved sample shape: (30, 63), float32
  - Output directory: datasets/raw/<member_id>_dynamic/<WORD>/sequence_XXXX.npy
  - MediaPipe landmark preprocessing is 100% identical to static pipeline.

Controls:
  SPACE    — Record one complete 30-frame sequence (with 1s prep countdown)
  N        — Next dynamic word
  P        — Previous dynamic word
  Q / ESC  — Quit (progress is saved)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Suppress TF / MediaPipe logs
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
_DATASETS_RAW_DIR = _BACKEND_DIR / "datasets" / "raw"

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
# Constants
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30       # 30 frames per sequence
MAX_MISSING_FRAMES = 5     # Reject sequence if hand lost in >5 of 30 frames (16.7%)

_CYAN   = (0, 200, 255)
_GREEN  = (0, 220, 80)
_RED    = (0, 60, 220)
_GRAY   = (150, 150, 150)
_WHITE  = (255, 255, 255)
_DARK   = (18, 18, 18)
_ORANGE = (0, 165, 255)
_YELLOW = (0, 230, 255)


# ---------------------------------------------------------------------------
# Config Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        logger.error("Config file not found: %s", _CONFIG_PATH)
        sys.exit(1)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_dynamic_words_for_member(config: dict, member_id: str) -> tuple[str, list[str]]:
    members = config.get("members", {})
    if member_id not in members:
        valid = ", ".join(sorted(members.keys()))
        logger.error("Member '%s' not found in config. Valid: %s", member_id, valid)
        sys.exit(1)

    member_cfg = members[member_id]
    display_name = member_cfg.get("display_name", member_id)
    all_words = member_cfg.get("words", [])
    word_types = member_cfg.get("word_types", {})

    # Filter only dynamic words
    dynamic_words = [w for w in all_words if word_types.get(w) == "dynamic"]
    if not dynamic_words:
        # Fallback to all words if none explicitly marked dynamic
        logger.warning("No words marked as 'dynamic' for %s. Using all words.", member_id)
        dynamic_words = all_words

    return display_name, dynamic_words


# ---------------------------------------------------------------------------
# Storage Helpers
# ---------------------------------------------------------------------------

def get_dynamic_member_dir(member_id: str) -> Path:
    """e.g. backend/datasets/raw/member_01_dynamic"""
    folder_name = f"{member_id}_dynamic"
    return _DATASETS_RAW_DIR / folder_name


def count_sequences(word_dir: Path) -> int:
    if not word_dir.exists():
        return 0
    return sum(1 for f in word_dir.iterdir() if f.suffix == ".npy")


def next_sequence_path(word_dir: Path) -> Path:
    word_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in word_dir.glob("*.npy"))
    idx = count_sequences(word_dir)
    while f"sequence_{idx:04d}.npy" in existing:
        idx += 1
    return word_dir / f"sequence_{idx:04d}.npy"


# ---------------------------------------------------------------------------
# Landmark Preprocessing (Identical to static & inference pipeline)
# ---------------------------------------------------------------------------

def extract_landmarks(results) -> np.ndarray | None:
    """
    Returns (63,) float32 or None if no hand found.
    Raw MediaPipe normalized coords: 21 landmarks x [x, y, z].
    """
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand.landmark],
        dtype=np.float32,
    )  # (21, 3)
    return coords.flatten()  # (63,)


def is_valid_landmark(vec: np.ndarray | None) -> bool:
    if vec is None or vec.shape != (63,):
        return False
    if np.any(np.isnan(vec)) or np.any(np.inf == vec):
        return False
    if np.allclose(vec, 0.0):
        return False
    return True


def repair_sequence(frames: list[np.ndarray | None]) -> np.ndarray | None:
    """
    Validates a 30-frame sequence and repairs minor missing frames via forward/backward fill.
    Returns (30, 63) float32 array, or None if too many missing frames (> MAX_MISSING_FRAMES).
    """
    if len(frames) != SEQUENCE_LENGTH:
        return None

    missing_indices = [i for i, f in enumerate(frames) if f is None or not is_valid_landmark(f)]
    if len(missing_indices) > MAX_MISSING_FRAMES:
        return None

    repaired = [f.copy() if (f is not None and is_valid_landmark(f)) else None for f in frames]

    # Forward fill
    last_valid = None
    for i in range(SEQUENCE_LENGTH):
        if repaired[i] is not None:
            last_valid = repaired[i]
        elif last_valid is not None:
            repaired[i] = last_valid.copy()

    # Backward fill for any initial missing frames
    next_valid = None
    for i in range(SEQUENCE_LENGTH - 1, -1, -1):
        if repaired[i] is not None:
            next_valid = repaired[i]
        elif next_valid is not None:
            repaired[i] = next_valid.copy()

    # Final check: all 30 frames must be valid (63,)
    for f in repaired:
        if f is None or not is_valid_landmark(f):
            return None

    arr = np.array(repaired, dtype=np.float32)  # Shape (30, 63)
    return arr


# ---------------------------------------------------------------------------
# HUD Rendering
# ---------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    word: str,
    word_idx: int,
    word_total: int,
    member_name: str,
    collected: int,
    target: int,
    hand_ok: bool,
    recording: bool,
    rec_frame_idx: int,
    countdown_val: int | None,
    status_msg: str,
    status_color: tuple,
    fps: float,
    all_words: list[str],
    member_dir: Path,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Top overlay
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 88), _DARK, -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)

    cv2.putText(frame, "SANKETA — Dynamic ISL Sequence Collector",
                (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, _CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{member_name} [DYNAMIC]",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}",
                (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Word {word_idx + 1}/{word_total}",
                (w - 110, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GRAY, 1, cv2.LINE_AA)

    # Word box
    bx, by = 10, 100
    cv2.rectangle(frame, (bx, by), (bx + 190, by + 105), _DARK, -1)
    cv2.rectangle(frame, (bx, by), (bx + 190, by + 105), _ORANGE, 2)
    display_word = word.replace("_", " ")
    font_scale = 0.65 if len(display_word) > 8 else 0.85
    cv2.putText(frame, display_word,
                (bx + 8, by + 45), cv2.FONT_HERSHEY_DUPLEX, font_scale, _WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "[DYNAMIC SEQUENCE]",
                (bx + 8, by + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _ORANGE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Seq: {collected} / {target}",
                (bx + 8, by + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA)

    # Progress bar (Sequences)
    pct = min(collected / max(target, 1), 1.0)
    bar_x, bar_y = 215, 106
    bar_w = w - bar_x - 180
    bar_h = 24
    fill_col = _GREEN if pct >= 1.0 else _ORANGE
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + bar_h), fill_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    cv2.putText(frame, f"Completed: {collected}/{target} ({pct * 100:.0f}%)",
                (bar_x + 8, bar_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _WHITE, 1, cv2.LINE_AA)

    # Hand detection status
    hs_col = _GREEN if hand_ok else _RED
    hs_txt = "Hand: DETECTED ✓" if hand_ok else "Hand: NOT DETECTED"
    cv2.putText(frame, hs_txt,
                (bar_x, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hs_col, 1, cv2.LINE_AA)

    # Status line
    cv2.putText(frame, status_msg,
                (bar_x, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1, cv2.LINE_AA)

    # Right sidebar word list
    sx = w - 165
    sy = 105
    cv2.putText(frame, "Dynamic Words:", (sx, sy - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GRAY, 1, cv2.LINE_AA)
    for i, wrd in enumerate(all_words):
        cnt = count_sequences(member_dir / wrd)
        done = cnt >= target
        cur = i == word_idx
        col = _GREEN if done else (_CYAN if cur else _GRAY)
        prefix = "▶ " if cur else "  "
        wlabel = wrd.replace("_", " ")
        if len(wlabel) > 10:
            wlabel = wlabel[:9] + "…"
        cv2.putText(frame, f"{prefix}{wlabel} {cnt}",
                    (sx, sy + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1, cv2.LINE_AA)

    # Countdown banner
    if countdown_val is not None and countdown_val > 0:
        cv2.rectangle(frame, (w // 2 - 120, h // 2 - 60), (w // 2 + 120, h // 2 + 50), _DARK, -1)
        cv2.rectangle(frame, (w // 2 - 120, h // 2 - 60), (w // 2 + 120, h // 2 + 50), _YELLOW, 3)
        cv2.putText(frame, "GET READY",
                    (w // 2 - 80, h // 2 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, _YELLOW, 2, cv2.LINE_AA)
        cv2.putText(frame, str(countdown_val),
                    (w // 2 - 16, h // 2 + 35), cv2.FONT_HERSHEY_DUPLEX, 1.4, _WHITE, 3, cv2.LINE_AA)

    # Recording banner & frame progress bar
    if recording:
        # Red pulsing dot + banner
        cv2.rectangle(frame, (bar_x, 205), (bar_x + bar_w, 255), (15, 15, 60), -1)
        cv2.rectangle(frame, (bar_x, 205), (bar_x + bar_w, 255), _RED, 2)
        cv2.circle(frame, (bar_x + 20, 230), 8, _RED, -1)
        cv2.putText(frame, f"RECORDING: Frame {rec_frame_idx + 1}/{SEQUENCE_LENGTH}",
                    (bar_x + 36, 236), cv2.FONT_HERSHEY_DUPLEX, 0.62, _WHITE, 1, cv2.LINE_AA)

        # Inner frame progress
        f_pct = (rec_frame_idx + 1) / SEQUENCE_LENGTH
        fb_y = 246
        cv2.rectangle(frame, (bar_x + 4, fb_y), (bar_x + int((bar_w - 8) * f_pct), fb_y + 4), _GREEN, -1)

    # Bottom hints
    cv2.putText(frame, "SPACE=Record 30-frame sequence | N=Next word | P=Prev word | Q=Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "TIP: Keep hand inside frame for all 30 frames (gesture takes ~1 sec)",
                (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _ORANGE, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="collect_data_dynamic.py",
        description="SANKETA — Dynamic ISL Sequence Hand Landmark Data Collector",
    )
    p.add_argument("--member", default="member_01",
                   help="Member ID (default: member_01)")
    p.add_argument("--samples", type=int, default=200,
                   help="Target sequences per word (default: 200)")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device ID (default: 0)")
    p.add_argument("--word", default=None,
                   help="Specific dynamic word to record (optional)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config()
    member_name, dynamic_words = get_dynamic_words_for_member(config, args.member)

    if args.word:
        clean_word = args.word.upper().strip()
        if clean_word in dynamic_words:
            dynamic_words = [clean_word]
        else:
            logger.warning("Word '%s' not in dynamic list (%s). Using '%s'.",
                           clean_word, dynamic_words, clean_word)
            dynamic_words = [clean_word]

    target_sequences = args.samples
    member_dir = get_dynamic_member_dir(args.member)
    member_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 65)
    print("  SANKETA — Dynamic ISL Sequence Collector (30 Frames)")
    print(f"  Member      : {member_name} ({args.member})")
    print(f"  Target Words: {', '.join(dynamic_words)}")
    print(f"  Sequences   : {target_sequences} per word")
    print(f"  Frames/Seq  : {SEQUENCE_LENGTH} frames (Shape: 30x63 float32)")
    print(f"  Output Dir  : {member_dir}")
    print("=" * 65)
    print()
    print("Controls:")
    print("  SPACE    — Record one complete 30-frame sequence")
    print("  N        — Next word")
    print("  P        — Previous word")
    print("  Q / ESC  — Quit")
    print()

    # Find starting word
    word_idx = 0
    for i, w in enumerate(dynamic_words):
        if count_sequences(member_dir / w) < target_sequences:
            word_idx = i
            break

    # Initialize camera
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

    # State variables
    status_msg = "Press SPACE to start recording sequence"
    status_color = _WHITE
    fps = 30.0
    prev_time = time.time()

    recording = False
    recorded_frames: list[np.ndarray | None] = []
    countdown_start = 0.0
    countdown_val = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera frame read failed.")
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now

            frame = cv2.flip(frame, 1)  # Mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    _mp_draw.draw_landmarks(
                        frame, hl, _mp_hands.HAND_CONNECTIONS,
                        _mp_draw_styles.get_default_hand_landmarks_style(),
                        _mp_draw_styles.get_default_hand_connections_style(),
                    )

            current_landmarks = extract_landmarks(results)
            hand_ok = is_valid_landmark(current_landmarks)

            cur_word = dynamic_words[word_idx]
            word_dir = member_dir / cur_word
            collected = count_sequences(word_dir)

            # Handle countdown
            if countdown_val is not None:
                elapsed = time.time() - countdown_start
                remaining = int(np.ceil(1.0 - elapsed))
                if remaining > 0:
                    countdown_val = remaining
                else:
                    # Countdown finished -> begin recording
                    countdown_val = None
                    recording = True
                    recorded_frames = []
                    status_msg = f"Recording '{cur_word}' sequence..."
                    status_color = _YELLOW

            # Handle sequence recording
            if recording:
                recorded_frames.append(current_landmarks if hand_ok else None)
                rec_idx = len(recorded_frames) - 1

                if len(recorded_frames) >= SEQUENCE_LENGTH:
                    # Sequence complete -> validate and repair
                    recording = False
                    repaired_seq = repair_sequence(recorded_frames)

                    if repaired_seq is not None:
                        out_path = next_sequence_path(word_dir)
                        np.save(str(out_path), repaired_seq)
                        collected = count_sequences(word_dir)
                        status_msg = f"Saved sequence {collected}/{target_sequences} ✓"
                        status_color = _GREEN
                        if collected >= target_sequences:
                            # Auto-advance to next incomplete dynamic word
                            for next_i in range(len(dynamic_words)):
                                check_w = dynamic_words[(word_idx + 1 + next_i) % len(dynamic_words)]
                                if count_sequences(member_dir / check_w) < target_sequences:
                                    word_idx = (word_idx + 1 + next_i) % len(dynamic_words)
                                    status_msg = f"Completed {cur_word}! Switched to {check_w}"
                                    status_color = _CYAN
                                    break
                            else:
                                status_msg = "All dynamic words completed! (Press Q to quit)"
                                status_color = _GREEN
                    else:
                        missed = sum(1 for f in recorded_frames if f is None)
                        status_msg = f"REJECTED: Hand lost in {missed}/{SEQUENCE_LENGTH} frames! Try again."
                        status_color = _RED

                    recorded_frames = []
            else:
                rec_idx = 0

            # Render HUD
            hud_frame = draw_hud(
                frame=frame,
                word=cur_word,
                word_idx=word_idx,
                word_total=len(dynamic_words),
                member_name=member_name,
                collected=collected,
                target=target_sequences,
                hand_ok=hand_ok,
                recording=recording,
                rec_frame_idx=rec_idx,
                countdown_val=countdown_val,
                status_msg=status_msg,
                status_color=status_color,
                fps=fps,
                all_words=dynamic_words,
                member_dir=member_dir,
            )

            cv2.imshow("SANKETA — Dynamic Sequence Collector", hud_frame)
            key = cv2.waitKey(1) & 0xFF

            # Key controls
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("\n  Exiting dynamic collector. Progress is safely saved.")
                break

            elif key == 32:  # SPACE
                if not recording and countdown_val is None:
                    if not hand_ok:
                        status_msg = "Please position your hand in camera frame first!"
                        status_color = _RED
                    else:
                        # Start 1-second readiness countdown
                        countdown_start = time.time()
                        countdown_val = 1
                        status_msg = "Get ready to perform sign motion..."
                        status_color = _YELLOW

            elif key in (ord('n'), ord('N')):  # Next word
                if not recording:
                    word_idx = (word_idx + 1) % len(dynamic_words)
                    status_msg = f"Switched to '{dynamic_words[word_idx]}'"
                    status_color = _CYAN

            elif key in (ord('p'), ord('P')):  # Prev word
                if not recording:
                    word_idx = (word_idx - 1) % len(dynamic_words)
                    status_msg = f"Switched to '{dynamic_words[word_idx]}'"
                    status_color = _CYAN

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
