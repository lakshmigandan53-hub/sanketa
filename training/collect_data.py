#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SANKETA - Local Multi-Person Static Sign Language Data Collector
===================================================================

Collects 21-point MediaPipe hand landmark samples per team member per sign.
Saves raw 63 numerical features (x, y, z for 21 joints) into:
    dataset/<member_id>/<sign_name>/sample_XXXX.npy

Features & Controls:
- Arbitrary Member ID (not hardcoded to 4 members)
- Configurable target samples per sign (default: 300)
- Quality controls:
    * Rejects frames without a detected hand
    * Minimum time interval between captures (e.g. 90ms)
    * Minimum hand movement filter to avoid capturing identical static duplicates
    * 63-feature shape and validity checks
- Keyboard Controls:
    SPACE : Toggle automatic recording (ON / PAUSED)
    C     : Capture single valid sample manually
    N     : Next sign
    P     : Previous sign
    R     : Reset / clear samples for current sign
    Q/ESC : Quit safely (all progress is preserved)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Suppress verbose TF/MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import mediapipe as mp
import numpy as np

# Add project root to sys.path to import training.preprocess
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from training.preprocess import STATIC_SIGNS, validate_landmark_vector
except ImportError:
    STATIC_SIGNS = [
        "HELLO", "YES", "NO", "WATER", "FOOD", "MILK", "TEA", "BOOK", "PEN", "PHONE", "HELP"
    ]
    def validate_landmark_vector(vec):
        return (vec.shape == (63,) and not np.isnan(vec).any()), "OK"


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DEFAULT_DATASET_DIR = _PROJECT_ROOT / "dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SANKETA Local Multi-Person Static Sign Data Collector"
    )
    parser.add_argument(
        "--member",
        type=str,
        default="",
        help="Member identifier or name (e.g., member_1, member_2, alice)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Target number of valid samples per sign (default: 300)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Output directory for dataset (default: ./dataset)",
    )
    return parser.parse_args()


def count_existing_samples(sign_dir: Path) -> int:
    if not sign_dir.exists():
        return 0
    return sum(1 for f in sign_dir.glob("sample_*.npy"))


def save_sample(sign_dir: Path, landmark_array: np.ndarray) -> int:
    sign_dir.mkdir(parents=True, exist_ok=True)
    existing = list(sign_dir.glob("sample_*.npy"))
    if existing:
        indices = []
        for p in existing:
            try:
                idx = int(p.stem.split("_")[-1])
                indices.append(idx)
            except ValueError:
                pass
        next_idx = max(indices, default=-1) + 1
    else:
        next_idx = 0

    target_path = sign_dir / f"sample_{next_idx:04d}.npy"
    np.save(target_path, landmark_array.astype(np.float32))
    return next_idx + 1


def main():
    args = parse_args()

    # Prompt for member ID if not provided via CLI
    member_id = args.member.strip()
    if not member_id:
        print("\n" + "=" * 65)
        print("SANKETA STATIC ISL DATA COLLECTOR")
        print("=" * 65)
        member_id = input("Enter Member ID or Name (e.g. member_1, member_2): ").strip()
        if not member_id:
            member_id = "member_1"
            print(f"Defaulting to: {member_id}")

    # Sanitize member ID for filesystem
    safe_member = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in member_id)
    dataset_root = Path(args.dataset_dir)
    member_dir = dataset_root / safe_member
    target_samples = args.samples

    print(f"\nTarget Dataset: {member_dir.resolve()}")
    print(f"Target Samples per Sign: {target_samples}")
    print(f"Supported Signs ({len(STATIC_SIGNS)}): {', '.join(STATIC_SIGNS)}\n")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Initialize Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}. Please check connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_sign_idx = 0
    auto_recording = False
    last_capture_time = 0.0
    min_capture_interval = 0.09  # 90ms between consecutive captures (~11 samples/sec max)
    last_saved_landmarks: Optional[np.ndarray] = None
    min_movement_threshold = 0.008  # Minimum displacement to avoid duplicate static frames

    fps_last_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    print("Controls:")
    print("  [SPACE]  : Toggle Automatic Capture")
    print("  [C]      : Single Capture")
    print("  [N]      : Next Sign")
    print("  [P]      : Previous Sign")
    print("  [R]      : Reset/Clear Current Sign")
    print("  [Q/ESC]  : Save & Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            time.sleep(0.05)
            continue

        # Mirror frame horizontally for intuitive selfie view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        fps_frame_count += 1
        now = time.time()
        if now - fps_last_time >= 1.0:
            current_fps = fps_frame_count / (now - fps_last_time)
            fps_frame_count = 0
            fps_last_time = now

        current_sign = STATIC_SIGNS[current_sign_idx]
        sign_dir = member_dir / current_sign
        sample_count = count_existing_samples(sign_dir)

        # Process MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False
        raw_landmarks: Optional[np.ndarray] = None

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_lms = results.multi_hand_landmarks[0]

            # Draw skeleton overlay
            mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract 63 raw coordinates [x0, y0, z0, x1, y1, z1, ...]
            coords = []
            for lm in hand_lms.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            raw_landmarks = np.array(coords, dtype=np.float32)

        # Auto capture logic
        captured_this_frame = False
        if auto_recording and hand_detected and raw_landmarks is not None:
            if sample_count < target_samples:
                if (now - last_capture_time) >= min_capture_interval:
                    # Check movement threshold against previous sample
                    movement_delta = 1.0
                    if last_saved_landmarks is not None:
                        movement_delta = float(np.mean(np.abs(raw_landmarks - last_saved_landmarks)))

                    if movement_delta >= min_movement_threshold:
                        is_valid, _ = validate_landmark_vector(raw_landmarks)
                        if is_valid:
                            save_sample(sign_dir, raw_landmarks)
                            last_saved_landmarks = raw_landmarks.copy()
                            last_capture_time = now
                            sample_count += 1
                            captured_this_frame = True
            else:
                # Reached target samples for this sign -> pause auto recording
                auto_recording = False

        # --- ON SCREEN HUD DISPLAY ---
        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Sign name & Member ID
        cv2.putText(
            frame,
            f"SIGN [{current_sign_idx + 1}/{len(STATIC_SIGNS)}]: {current_sign}",
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"MEMBER: {safe_member} | FPS: {current_fps:.1f}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (148, 163, 184),
            1,
        )

        # Status & Sample Count
        progress_pct = min(100, int((sample_count / target_samples) * 100))
        count_color = (34, 197, 94) if sample_count >= target_samples else (56, 189, 248)
        cv2.putText(
            frame,
            f"Samples: {sample_count}/{target_samples} ({progress_pct}%)",
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            count_color,
            1,
        )

        # Hand detected indicator (top right)
        if hand_detected:
            cv2.putText(frame, "HAND: DETECTED", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 2)
        else:
            cv2.putText(frame, "HAND: NO HAND", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (239, 68, 68), 2)

        # Recording state badge (top right)
        if auto_recording:
            rec_badge = "AUTO RECORDING [ON]"
            rec_color = (34, 197, 94) if not captured_this_frame else (255, 255, 0)
        else:
            rec_badge = "RECORDING [PAUSED]"
            rec_color = (203, 213, 225)
        cv2.putText(frame, rec_badge, (w - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2)

        # Bottom help footer
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 23, 42), -1)
        help_txt = "[SPACE] Record/Pause | [C] Single | [N] Next | [P] Prev | [R] Reset | [Q] Quit"
        cv2.putText(frame, help_txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1)

        cv2.imshow("SANKETA ISL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            print("\n[INFO] Exiting data collection. All captured samples are saved.")
            break
        elif key == ord(" "):  # SPACE: toggle auto capture
            auto_recording = not auto_recording
            last_saved_landmarks = None
            print(f"[{'ON' if auto_recording else 'PAUSED'}] Auto-recording for '{current_sign}'")
        elif key in (ord("c"), ord("C")):  # Manual single capture
            if hand_detected and raw_landmarks is not None:
                is_valid, _ = validate_landmark_vector(raw_landmarks)
                if is_valid:
                    save_sample(sign_dir, raw_landmarks)
                    sample_count += 1
                    print(f"Captured sample {sample_count}/{target_samples} for '{current_sign}'")
            else:
                print("[WARN] Cannot capture: No hand detected.")
        elif key in (ord("n"), ord("N")):  # Next sign
            current_sign_idx = (current_sign_idx + 1) % len(STATIC_SIGNS)
            auto_recording = False
            last_saved_landmarks = None
            print(f"Switched to sign [{current_sign_idx + 1}/{len(STATIC_SIGNS)}]: {STATIC_SIGNS[current_sign_idx]}")
        elif key in (ord("p"), ord("P")):  # Previous sign
            current_sign_idx = (current_sign_idx - 1) % len(STATIC_SIGNS)
            auto_recording = False
            last_saved_landmarks = None
            print(f"Switched to sign [{current_sign_idx + 1}/{len(STATIC_SIGNS)}]: {STATIC_SIGNS[current_sign_idx]}")
        elif key in (ord("r"), ord("R")):  # Reset sign
            print(f"Resetting samples for {current_sign}...")
            if sign_dir.exists():
                for f in sign_dir.glob("sample_*.npy"):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            sample_count = 0
            auto_recording = False

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
