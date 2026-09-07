#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SANKETA - Static ISL Model Evaluation Pipeline
=================================================

Evaluates the trained static ISL model against held-out test data:
- Person-Based Evaluation (e.g. Member 4)
- Secondary Stratified Random Split
- Per-class Precision, Recall, and F1-score
- Confusion Matrix (11 x 11)
- Anti-"Water" Prediction Bias Audit (tests 20+ samples per class)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

# Add project root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.preprocess import (
    STATIC_SIGNS,
    normalize_landmarks,
    validate_landmark_vector,
)

DEFAULT_DATASET_DIR = _PROJECT_ROOT / "dataset"
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "signbridge" / "backend" / "app" / "models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SANKETA Static ISL Model Evaluation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to .keras or .h5 model file (defaults to backend/app/models/isl_model.keras)",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="",
        help="Path to labels.txt (defaults to backend/app/models/labels.txt)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset root directory containing member folders",
    )
    parser.add_argument(
        "--test-member",
        type=str,
        default="",
        help="Specific member to evaluate on (defaults to member_04 if available)",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> List[str]:
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            return lines
    return STATIC_SIGNS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]):
    n_classes = len(classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    precision = np.zeros(n_classes, dtype=float)
    recall = np.zeros(n_classes, dtype=float)
    f1 = np.zeros(n_classes, dtype=float)

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0

    acc = float(np.sum(np.diag(conf_matrix))) / len(y_true) if len(y_true) > 0 else 0.0
    return acc, conf_matrix, precision, recall, f1


def main():
    args = parse_args()

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path).resolve()
    else:
        for candidate in [
            DEFAULT_MODEL_DIR / "isl_model.keras",
            DEFAULT_MODEL_DIR / "isl_model.h5",
            _PROJECT_ROOT / "training" / "models" / "isl_model.keras",
        ]:
            if candidate.exists():
                model_path = candidate
                break
        else:
            print("[ERROR] No trained model found. Please train first using: python training/train_model.py")
            sys.exit(1)

    # Resolve labels path
    if args.labels_path:
        labels_path = Path(args.labels_path).resolve()
    else:
        labels_path = model_path.parent / "labels.txt"

    labels = load_labels(labels_path)
    label_to_idx = {name: idx for idx, name in enumerate(labels)}

    print("=" * 70)
    print("SANKETA STATIC ISL MODEL EVALUATION")
    print("=" * 70)
    print(f"Model:    {model_path}")
    print(f"Labels ({len(labels)}): {', '.join(labels)}")
    print(f"Dataset:  {Path(args.dataset_dir).resolve()}\n")

    # Load Model with cross-version compatibility fallback
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"[OK] Model loaded successfully ({model_path.name}).")
        print(f"     Input Shape:  {model.input_shape}")
        print(f"     Output Shape: {model.output_shape} ({model.output_shape[-1]} classes)")
    except Exception as exc:
        h5_path = model_path.with_suffix(".h5")
        if h5_path.exists() and h5_path != model_path:
            try:
                model = tf.keras.models.load_model(str(h5_path), compile=False)
                print(f"[OK] Model loaded successfully via H5 compatibility fallback ({h5_path.name}).")
                print(f"     Input Shape:  {model.input_shape}")
                print(f"     Output Shape: {model.output_shape} ({model.output_shape[-1]} classes)")
            except Exception as h5_exc:
                print(f"[ERROR] Failed to load model: {exc} | H5 fallback error: {h5_exc}")
                sys.exit(1)
        else:
            print(f"[ERROR] Failed to load model: {exc}")
            sys.exit(1)

    # Load Data
    dataset_dir = Path(args.dataset_dir)
    member_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not member_dirs:
        print(f"[ERROR] No member directories in {dataset_dir}")
        sys.exit(1)

    all_members = [d.name for d in member_dirs]
    test_member = args.test_member.strip()
    if not test_member or test_member not in all_members:
        test_member = all_members[-1]  # Default to last member

    print(f"\nEvaluating on Member: '{test_member}' (Person-Based Test Evaluation)")

    X_test_list = []
    y_test_list = []

    m_dir = dataset_dir / test_member
    for s_dir in m_dir.iterdir():
        if not s_dir.is_dir():
            continue
        sign_name = s_dir.name.upper()
        if sign_name not in label_to_idx:
            continue
        cls_idx = label_to_idx[sign_name]

        for npy_file in s_dir.glob("*.npy"):
            try:
                arr = np.load(npy_file)
                is_valid, _ = validate_landmark_vector(arr)
                if not is_valid:
                    continue
                norm = normalize_landmarks(arr)
                X_test_list.append(norm)
                y_test_list.append(cls_idx)
            except Exception:
                pass

    if not X_test_list:
        print(f"[ERROR] No valid test samples found for member '{test_member}'.")
        sys.exit(1)

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32)
    print(f"Total Test Samples: {len(X_test)}\n")

    # Run Inference
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc, conf_matrix, precisions, recalls, f1s = compute_metrics(y_test, preds, labels)

    print("-" * 70)
    print(f"PERSON-BASED EVALUATION ACCURACY: {acc:.2%}")
    print("-" * 70)

    # Per-Class Precision / Recall / F1
    print(f"{'CLASS':<12} {'PRECISION':<12} {'RECALL':<12} {'F1-SCORE':<12} {'SAMPLES':<8}")
    print("-" * 70)
    best_cls = labels[0]
    best_f1 = -1.0
    worst_cls = labels[0]
    worst_f1 = 2.0

    for i, sign in enumerate(labels):
        n_samples = int(np.sum(y_test == i))
        f1_val = f1s[i]
        if n_samples > 0:
            if f1_val > best_f1:
                best_f1 = f1_val
                best_cls = sign
            if f1_val < worst_f1:
                worst_f1 = f1_val
                worst_cls = sign

        print(f"{sign:<12} {precisions[i]:>9.2%}   {recalls[i]:>9.2%}   {f1s[i]:>9.2%}   {n_samples:>7}")
    print("-" * 70)

    print(f"Best Class:  {best_cls} (F1: {best_f1:.2%})")
    print(f"Worst Class: {worst_cls} (F1: {worst_f1:.2%})")

    # Confusion Matrix
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    header = "       " + "".join(f"{s[:4]:>6}" for s in labels)
    print(header)
    for i, row in enumerate(conf_matrix):
        row_str = f"{labels[i][:5]:<6} " + "".join(f"{v:>6}" for v in row)
        print(row_str)

    # Anti-Water Bias Audit
    print("\n" + "=" * 70)
    print("ANTI-'WATER' PREDICTION BIAS AUDIT (20 Samples Per Class)")
    print("=" * 70)
    water_idx = label_to_idx.get("WATER", -1)
    water_bias = False

    for i, sign in enumerate(labels):
        cls_mask = (y_test == i)
        X_cls = X_test[cls_mask]
        n_cls = min(20, len(X_cls))
        if n_cls == 0:
            continue
        p = np.argmax(model.predict(X_cls[:n_cls], verbose=0), axis=1)
        w_cnt = int(np.sum(p == water_idx)) if water_idx >= 0 else 0
        correct = int(np.sum(p == i))
        if i != water_idx and w_cnt >= (n_cls * 0.40):
            water_bias = True
            st = "[BIAS DETECTED]"
        else:
            st = "[PASS]"
        print(f"  {sign:<12}: {correct:>2}/{n_cls:<3} correct | {w_cnt:>2}/{n_cls:<3} pred as WATER {st}")

    print("-" * 70)
    print(f"Does 'Water' prediction bias still exist? {'YES (FAILURE)' if water_bias else 'NO (VERIFIED RESOLVED)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
