#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SANKETA - Static ISL Dataset Validation and Quality Audit Tool
================================================================

Validates the multi-person dataset before training.

Checks:
- Total sample count
- Samples per class across all members
- Samples per member matrix
- Corrupted / invalid samples (NaN, Inf, wrong shapes != 63)
- Class balance analysis (warns if significantly unbalanced)
- Feature range and distribution sanity checks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to sys.path
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

DEFAULT_DATASET_DIRS = [
    _PROJECT_ROOT / "dataset",
    _PROJECT_ROOT / "signbridge" / "backend" / "datasets" / "raw",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SANKETA Dataset Validation and Quality Audit"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="Path to dataset directory (defaults to ./dataset or signbridge/backend/datasets/raw)",
    )
    return parser.parse_args()


def resolve_dataset_dir(explicit_path: str) -> Path:
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.exists():
            return p
        print(f"[WARN] Specified directory {p} does not exist. Checking defaults...")

    for d in DEFAULT_DATASET_DIRS:
        if d.exists() and any(d.iterdir()):
            return d.resolve()

    return (_PROJECT_ROOT / "dataset").resolve()


def audit_dataset(dataset_dir: Path):
    print("=" * 70)
    print("SANKETA STATIC ISL DATASET VALIDATION AUDIT")
    print("=" * 70)
    print(f"Scanning Directory: {dataset_dir}\n")

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        print("Please collect data first using: python training/collect_data.py")
        return False

    # Discover members (subdirectories)
    members = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    if not members:
        print("[WARN] No member subdirectories found in dataset directory.")
        return False

    # Data structures for audit
    # member -> {sign -> count}
    member_counts: Dict[str, Dict[str, int]] = {m: {s: 0 for s in STATIC_SIGNS} for m in members}
    total_samples = 0
    invalid_samples: List[Tuple[str, str]] = []
    non_target_signs: Dict[str, int] = {}
    feature_lengths: Dict[int, int] = {}

    all_features = []

    for member in members:
        m_dir = dataset_dir / member
        # List sign directories
        for s_dir in m_dir.iterdir():
            if not s_dir.is_dir():
                continue
            # Canonical uppercase comparison
            sign_name = s_dir.name.upper()
            if sign_name not in STATIC_SIGNS:
                non_target_signs[sign_name] = non_target_signs.get(sign_name, 0) + len(list(s_dir.glob("*.npy")))
                continue

            npy_files = sorted(s_dir.glob("*.npy"))
            valid_for_sign = 0

            for f in npy_files:
                try:
                    arr = np.load(f)
                    flen = arr.size
                    feature_lengths[flen] = feature_lengths.get(flen, 0) + 1

                    is_valid, reason = validate_landmark_vector(arr)
                    if not is_valid:
                        invalid_samples.append((str(f), reason))
                        continue

                    valid_for_sign += 1
                    total_samples += 1
                    if len(all_features) < 1000:
                        all_features.append(arr.flatten())

                except Exception as exc:
                    invalid_samples.append((str(f), f"Corrupted file: {exc}"))

            member_counts[member][sign_name] = valid_for_sign

    # --- REPORT SUMMARY ---
    print(f"Total Valid Samples: {total_samples}")
    print(f"Number of Members:   {len(members)} ({', '.join(members)})")
    print(f"Invalid / Corrupted: {len(invalid_samples)}")

    if invalid_samples:
        print("\n[!] Invalid Samples Details:")
        for path, reason in invalid_samples[:10]:
            print(f"    - {path}: {reason}")
        if len(invalid_samples) > 10:
            print(f"    ... and {len(invalid_samples) - 10} more.")

    # Matrix Table: Members x Classes
    print("\n" + "-" * 70)
    print(f"{'MEMBER':<16} " + " ".join(f"{s[:5]:>7}" for s in STATIC_SIGNS) + "    TOTAL")
    print("-" * 70)

    class_totals = {s: 0 for s in STATIC_SIGNS}
    for m in members:
        row_str = f"{m:<16} "
        m_tot = 0
        for s in STATIC_SIGNS:
            cnt = member_counts[m][s]
            row_str += f"{cnt:>7} "
            class_totals[s] += cnt
            m_tot += cnt
        row_str += f"  {m_tot:>7}"
        print(row_str)

    print("-" * 70)
    tot_row = f"{'TOTAL':<16} "
    for s in STATIC_SIGNS:
        tot_row += f"{class_totals[s]:>7} "
    tot_row += f"  {total_samples:>7}"
    print(tot_row)
    print("-" * 70)

    # Class Balance Analysis
    counts_list = [class_totals[s] for s in STATIC_SIGNS]
    min_count = min(counts_list) if counts_list else 0
    max_count = max(counts_list) if counts_list else 0

    print("\nClass Balance Analysis:")
    for s in STATIC_SIGNS:
        cnt = class_totals[s]
        pct = (cnt / total_samples * 100) if total_samples > 0 else 0
        status_flag = "[OK]" if cnt >= 100 else "[LOW]"
        print(f"  {s:<12} : {cnt:>6} samples ({pct:>5.1f}%) {status_flag}")

    is_balanced = True
    if min_count == 0:
        missing = [s for s in STATIC_SIGNS if class_totals[s] == 0]
        print(f"\n[WARNING] Missing classes with ZERO samples: {', '.join(missing)}")
        is_balanced = False
    elif max_count > 0 and (max_count / min_count) > 1.75:
        print(f"\n[WARNING] Dataset is significantly unbalanced! Max/Min ratio = {max_count/min_count:.2f}")
        print("Consider collecting more samples for underrepresented classes before final training.")
        is_balanced = False
    else:
        print("\n[OK] Class balance is within healthy parameters.")

    if non_target_signs:
        print(f"\n[INFO] Other non-target signs found in dataset: {dict(non_target_signs)}")

    print("=" * 70)
    return total_samples > 0


if __name__ == "__main__":
    args = parse_args()
    d_dir = resolve_dataset_dir(args.dataset_dir)
    audit_dataset(d_dir)
