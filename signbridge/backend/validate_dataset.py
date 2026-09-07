# -*- coding: utf-8 -*-
"""
SANKETA — Dataset Validation Tool
backend/validate_dataset.py

Validates the collected dataset and produces a readiness report before training.

Usage:
  python validate_dataset.py
  python validate_dataset.py --min-samples 50     # lower threshold for early check
  python validate_dataset.py --member member_01   # single member only

Output:
  Console report showing per-member / per-word sample counts, warnings,
  and a GO / NOT READY verdict for training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent
_CONFIG_PATH = _BACKEND_DIR / "config" / "words.json"
_DATASET_DIR = _BACKEND_DIR / "datasets" / "raw"
_MODEL_DIR   = _BACKEND_DIR / "app" / "models"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colours (works in Windows Terminal / PowerShell ≥7)
# ---------------------------------------------------------------------------
_GRN = "\033[92m"
_YEL = "\033[93m"
_RED = "\033[91m"
_CYN = "\033[96m"
_RST = "\033[0m"
_BLD = "\033[1m"


def _ok(s):   return f"{_GRN}{s}{_RST}"
def _warn(s): return f"{_YEL}{s}{_RST}"
def _err(s):  return f"{_RED}{s}{_RST}"
def _hdr(s):  return f"{_BLD}{_CYN}{s}{_RST}"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        print(_err(f"Config not found: {_CONFIG_PATH}"))
        sys.exit(1)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict) -> list[str]:
    """Validate the words.json structure. Returns list of error strings."""
    errors = []
    members = config.get("members", {})
    if not members:
        errors.append("No members defined in config.")
        return errors

    all_words: list[str] = []
    for mid, mcfg in members.items():
        words = mcfg.get("words", [])
        if not words:
            errors.append(f"{mid}: no words configured.")
        if len(words) != len(set(words)):
            dupes = [w for w in set(words) if words.count(w) > 1]
            errors.append(f"{mid}: duplicate words: {dupes}")
        all_words.extend(words)

    # Cross-member duplicate check
    seen: dict[str, str] = {}
    for mid, mcfg in members.items():
        for w in mcfg.get("words", []):
            if w in seen:
                errors.append(f"Word '{w}' appears in both {seen[w]} and {mid}.")
            else:
                seen[w] = mid

    total = len(set(all_words))
    if total > 48:
        errors.append(f"Total unique words {total} exceeds 48.")

    return errors


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def count_samples(word_dir: Path) -> int:
    if not word_dir.exists():
        return 0
    return sum(1 for f in word_dir.iterdir() if f.suffix == ".npy")


def check_sample(fpath: Path, expected_type: str = "static") -> str | None:
    """
    Load and validate one .npy sample.
    For static: expected shape (63,)
    For dynamic: expected shape (30, 63)
    Returns None if valid, or an error string if invalid.
    """
    try:
        arr = np.load(str(fpath))
    except Exception as e:
        return f"Cannot load: {e}"

    if expected_type == "dynamic":
        if arr.shape != (30, 63):
            return f"Wrong shape {arr.shape}, expected (30, 63)"
    else:
        if arr.shape == (21, 3):
            arr = arr.flatten()
        if arr.shape != (63,):
            return f"Wrong shape {arr.shape}, expected (63,)"

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return "Contains NaN/Inf"
    if np.allclose(arr, 0.0):
        return "All zeros (empty landmark)"
    return None


def check_near_duplicates(samples: list[np.ndarray], threshold: float = 0.001) -> int:
    """
    Count approximate near-duplicate pairs (L2 dist < threshold).
    O(n²) — only run on reasonably sized datasets.
    """
    if len(samples) < 2 or len(samples) > 5000:
        return 0
    arr = np.array(samples, dtype=np.float32)
    count = 0
    for i in range(len(arr) - 1):
        dists = np.linalg.norm(arr[i + 1:] - arr[i], axis=1)
        count += int(np.sum(dists < threshold))
    return count


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(config: dict, member_filter: str | None, min_samples: int) -> bool:
    """
    Run full validation. Returns True if dataset is training-ready.
    """
    members_cfg = config.get("members", {})
    target      = config.get("target_samples_per_word", 200)

    if member_filter:
        if member_filter not in members_cfg:
            print(_err(f"Member '{member_filter}' not in config."))
            sys.exit(1)
        members_to_check = {member_filter: members_cfg[member_filter]}
    else:
        members_to_check = members_cfg

    print()
    print(_hdr("=" * 62))
    print(_hdr("  SANKETA — Dataset Validation Report"))
    print(_hdr("=" * 62))

    grand_total_samples   = 0
    grand_total_words     = 0
    grand_warnings        = 0
    grand_errors          = 0
    all_class_counts: dict[str, int] = {}

    for member_id, mcfg in members_to_check.items():
        display = mcfg.get("display_name", member_id)
        words   = mcfg.get("words", [])

        print()
        print(_hdr(f"  -- {display} ({member_id}) --"))
        print(f"  {'Word':<22} {'Samples':>8}  {'Status'}")
        print(f"  {'-'*22} {'-'*8}  {'-'*20}")

        member_dir    = _DATASET_DIR / member_id
        member_total  = 0
        member_errors = 0
        member_warns  = 0

        for word in words:
            wtype = mcfg.get("word_types", {}).get(word, "static")
            if wtype == "dynamic":
                dyn_dir = _DATASET_DIR / f"{member_id}_dynamic" / word
                word_dir = dyn_dir if dyn_dir.exists() else (member_dir / word)
            else:
                word_dir = member_dir / word

            n = count_samples(word_dir)
            member_total += n
            all_class_counts[word] = all_class_counts.get(word, 0) + n

            # Validate individual files
            bad_files = 0
            all_vecs: list[np.ndarray] = []
            if word_dir.exists():
                for fpath in sorted(word_dir.iterdir()):
                    if fpath.suffix != ".npy":
                        continue
                    err = check_sample(fpath, expected_type=wtype)
                    if err:
                        bad_files += 1
                    else:
                        all_vecs.append(np.load(str(fpath)).flatten())

            near_dups = check_near_duplicates(all_vecs)

            if n == 0:
                status = _err("MISSING — no samples")
                member_errors += 1
            elif n < min_samples:
                status = _err(f"INSUFFICIENT — need ≥{min_samples}")
                member_errors += 1
            elif n < target:
                status = _warn(f"LOW — {n}/{target}")
                member_warns += 1
            else:
                status = _ok(f"OK  {n}/{target}")

            extras = []
            if bad_files:
                extras.append(_err(f"{bad_files} corrupt files"))
                member_errors += bad_files
            if near_dups:
                extras.append(_warn(f"~{near_dups} near-duplicates"))
                member_warns += near_dups

            extra_str = "  " + ", ".join(extras) if extras else ""
            wtype = mcfg.get("word_types", {}).get(word, "static")
            type_tag = f"[{wtype[0].upper()}]"  # [S] or [D]
            print(f"  {type_tag} {word:<20} {n:>6}    {status}{extra_str}")

        print(f"\n  Total: {member_total} samples across {len(words)} words")
        grand_total_samples += member_total
        grand_total_words   += len(words)
        grand_warnings      += member_warns
        grand_errors        += member_errors

    # ── Overall summary ──────────────────────────────────────────────────────
    print()
    print(_hdr("=" * 62))
    print(_hdr("  Overall Summary"))
    print(_hdr("=" * 62))

    unique_classes = len(all_class_counts)
    configured_total = sum(
        len(mcfg.get("words", [])) for mcfg in members_to_check.values()
    )

    print(f"\n  Members checked   : {len(members_to_check)}")
    print(f"  Words configured  : {configured_total}")
    print(f"  Unique classes    : {unique_classes}")
    print(f"  Total samples     : {grand_total_samples}")
    print(f"  Target per word   : {config.get('target_samples_per_word', 200)}")

    # Words with zero data
    missing = [w for w, c in all_class_counts.items() if c == 0]
    no_data = [w for w in
               [word for mcfg in members_to_check.values()
                for word in mcfg.get("words", [])]
               if all_class_counts.get(w, 0) == 0]

    if no_data:
        print(f"\n  {_err('Words with ZERO samples:')}")
        for w in no_data:
            print(f"    • {w}")

    # Class balance
    if all_class_counts:
        counts = list(all_class_counts.values())
        mn, mx = min(counts), max(counts)
        print(f"\n  Sample range      : {mn} – {mx} per class")
        if mx > 0 and mn < mx * 0.3:
            print(f"  {_warn('⚠ Imbalanced dataset (min < 30% of max). Consider collecting more for underrepresented words.')}")

    # Verdict
    print()
    print(_hdr("─" * 62))
    if grand_errors == 0 and grand_total_samples > 0:
        print(_ok(f"  ✓ TRAINING READY — {grand_total_samples} samples, {unique_classes} classes"))
        print(_ok("  Run: python train_model.py"))
        ready = True
    elif grand_total_samples > 0:
        print(_warn(f"  ⚠ WARNINGS ({grand_warnings}) / ERRORS ({grand_errors})"))
        print(_warn("  Fix issues above or use --min-samples to override."))
        ready = False
    else:
        print(_err("  ✗ NO DATA — Run collect_data.py --member member_01 first"))
        ready = False
    print(_hdr("─" * 62))
    print()
    return ready


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_dataset.py",
        description="SANKETA — Dataset validation tool",
    )
    p.add_argument("--member", default=None,
                   help="Validate a single member only (default: all members)")
    p.add_argument("--min-samples", type=int, default=20,
                   help="Minimum samples per word to pass validation (default: 20)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    # Config structural validation
    cfg_errors = validate_config(config)
    if cfg_errors:
        print(_err("Configuration errors in words.json:"))
        for e in cfg_errors:
            print(f"  • {e}")
        sys.exit(1)

    ready = validate(config, args.member, args.min_samples)
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
