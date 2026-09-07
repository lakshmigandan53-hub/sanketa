#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SANKETA - Starter Multi-Person Dataset Generator
Creates realistic physical variations (different hand sizes, slight angles, finger jitter)
for member_02, member_03, and member_04 based on member_01.
This enables immediate testing of person-based train/test splits.
Real team members can overwrite these anytime by running collect_data.py.
"""

from __future__ import annotations
import math
import shutil
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = _PROJECT_ROOT / "dataset"
MEMBER_01_DIR = DATASET_DIR / "member_01"


def rotate_landmarks_z(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates landmarks around the Z axis (roll in the 2D camera plane)."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    wrist = pts[0:1, :]
    rel = pts - wrist
    rotated = (rel @ R.T) + wrist
    return rotated.astype(np.float32)


def generate_member_variation(src_member_dir: Path, target_member_dir: Path, scale: float, rot_range: float, noise_std: float):
    print(f"Generating variation for {target_member_dir.name} (scale={scale:.2f}, rot=+/-{rot_range} deg)...")
    target_member_dir.mkdir(parents=True, exist_ok=True)

    sign_dirs = [d for d in src_member_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for s_dir in sign_dirs:
        sign_name = s_dir.name
        t_sign_dir = target_member_dir / sign_name
        t_sign_dir.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(s_dir.glob("*.npy"))
        for f in npy_files:
            try:
                arr = np.load(f).reshape((21, 3))
                # 1. Wrist center
                wrist = arr[0:1, :].copy()
                centered = arr - wrist

                # 2. Hand scale variation (different person's physical hand size/distance)
                scaled = centered * scale

                # 3. Angle variation (different wrist angle / natural pose slant)
                angle = np.random.uniform(-rot_range, rot_range)
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
                rotated = scaled @ R.T

                # 4. Subtle anatomical joint jitter (natural inter-person variance)
                jitter = np.random.normal(0, noise_std, rotated.shape).astype(np.float32)
                jitter[0, :] = 0.0  # Keep wrist anchored
                varied = rotated + jitter + wrist

                # Save varied sample
                out_path = t_sign_dir / f.name
                np.save(out_path, varied.flatten().astype(np.float32))
            except Exception as e:
                pass


def main():
    if not MEMBER_01_DIR.exists():
        print(f"[ERROR] Source {MEMBER_01_DIR} does not exist.")
        return

    # Member 2: Slightly larger hand (scale 1.10), slight tilt (+/- 8 deg)
    generate_member_variation(
        MEMBER_01_DIR,
        DATASET_DIR / "member_02",
        scale=1.10,
        rot_range=8.0,
        noise_std=0.003
    )

    # Member 3: Slightly smaller hand (scale 0.90), slight tilt (+/- 10 deg)
    generate_member_variation(
        MEMBER_01_DIR,
        DATASET_DIR / "member_03",
        scale=0.90,
        rot_range=10.0,
        noise_std=0.004
    )

    # Member 4: Medium hand (scale 1.03), different camera distance / tilt (+/- 12 deg)
    generate_member_variation(
        MEMBER_01_DIR,
        DATASET_DIR / "member_04",
        scale=1.03,
        rot_range=12.0,
        noise_std=0.005
    )

    print("\n[OK] Multi-person dataset initialized with 4 members (member_01, member_02, member_03, member_04).")


if __name__ == "__main__":
    main()
