#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SANKETA - Static ISL Model Local Training Pipeline
=====================================================

Trains an 11-class static sign classifier with:
1. Strict wrist-centering & scale-normalization (eliminates "Water" position bias)
2. Person-based evaluation split (Member 4 held out as unseen test person)
3. Lightweight Keras architecture for real-time edge inference
4. Class balance & Anti-"Water" bias verification
5. Model and label export to backend/app/models/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Add project root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.preprocess import (
    STATIC_SIGNS,
    SIGN_TO_INDEX,
    INDEX_TO_SIGN,
    normalize_landmarks,
    validate_landmark_vector,
    NUM_FEATURES,
)

DEFAULT_DATASET_DIR = _PROJECT_ROOT / "dataset"
DEFAULT_EXPORT_DIR = _PROJECT_ROOT / "signbridge" / "backend" / "app" / "models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SANKETA Static ISL Model Local Training"
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
        help="Member ID to hold out exclusively for person-based evaluation (default: last member)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(DEFAULT_EXPORT_DIR),
        help="Target directory for exported model and labels",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_all_samples(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads all .npy samples across all members, validates them,
    and applies wrist-centering and scale normalization.

    Returns:
        X: np.ndarray shape (N, 63) normalized features
        y: np.ndarray shape (N,) integer class indices
        members: List[str] length N indicating which member contributed each sample
    """
    print(f"[1/6] Loading dataset from: {dataset_dir.resolve()}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    member_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not member_dirs:
        raise ValueError(f"No member subdirectories found in {dataset_dir}")

    X_list = []
    y_list = []
    m_list = []

    total_files = 0
    valid_files = 0

    for m_dir in member_dirs:
        member_name = m_dir.name
        for s_dir in m_dir.iterdir():
            if not s_dir.is_dir():
                continue
            sign_name = s_dir.name.upper()
            if sign_name not in SIGN_TO_INDEX:
                continue

            class_idx = SIGN_TO_INDEX[sign_name]
            for npy_path in s_dir.glob("*.npy"):
                total_files += 1
                try:
                    arr = np.load(npy_path)
                    is_valid, _ = validate_landmark_vector(arr)
                    if not is_valid:
                        continue

                    # Apply CRITICAL wrist-relative and scale normalization
                    norm_feats = normalize_landmarks(arr)

                    X_list.append(norm_feats)
                    y_list.append(class_idx)
                    m_list.append(member_name)
                    valid_files += 1
                except Exception:
                    continue

    print(f"      Loaded {valid_files} valid normalized samples out of {total_files} files.")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), m_list


# ---------------------------------------------------------------------------
# Person-Based Evaluation Split
# ---------------------------------------------------------------------------

def split_person_based(
    X: np.ndarray,
    y: np.ndarray,
    members: List[str],
    test_member: str,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Splits data such that designated test member(s) are strictly held out.
    Evaluates generalization to a completely unseen person.
    """
    unique_members = sorted(list(set(members)))
    print(f"[2/6] Configuring Person-Based Train/Test Split...")
    print(f"      Available members: {unique_members}")

    if not test_member or test_member not in unique_members:
        # Default to the last member
        test_member = unique_members[-1]

    train_members = [m for m in unique_members if m != test_member]

    if not train_members:
        print(f"[WARN] Only 1 member exists ({unique_members[0]}). Using stratified random split.")
        # Fallback to stratified random split
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        test_size = int(len(y) * 0.20)
        val_size = int(len(y) * 0.15)

        test_idx = indices[:test_size]
        val_idx = indices[test_size : test_size + val_size]
        train_idx = indices[test_size + val_size :]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx], "Random (Single Person)"

    print(f"      Held-Out Test Member (PERSON-BASED): '{test_member}'")
    print(f"      Training Members:                   {train_members}")

    members_arr = np.array(members)
    test_mask = (members_arr == test_member)
    train_val_mask = ~test_mask

    X_test, y_test = X[test_mask], y[test_mask]
    X_tv, y_tv = X[train_val_mask], y[train_val_mask]

    # Stratified validation split from training members
    tv_indices = np.arange(len(y_tv))
    np.random.shuffle(tv_indices)

    # Simple stratified selection
    val_indices = []
    train_indices = []
    for cls in range(len(STATIC_SIGNS)):
        cls_idx = tv_indices[y_tv[tv_indices] == cls]
        n_val = max(1, int(len(cls_idx) * val_ratio))
        val_indices.extend(cls_idx[:n_val])
        train_indices.extend(cls_idx[n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    X_train, y_train = X_tv[train_indices], y_tv[train_indices]
    X_val, y_val = X_tv[val_indices], y_tv[val_indices]

    print(f"      Train Samples:      {len(X_train)} ({len(train_members)} members)")
    print(f"      Validation Samples: {len(X_val)}")
    print(f"      Held-Out Test:      {len(X_test)} (Member: {test_member})")

    return X_train, y_train, X_val, y_val, X_test, y_test, test_member


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def build_static_classifier(num_classes: int = len(STATIC_SIGNS)) -> tf.keras.Model:
    """
    Constructs a lightweight, robust neural network for 63 normalized landmark features.
    Uses BatchNormalization and Dropout for noise resilience and generalization.
    Uses input_shape on the first Dense layer to ensure 100% bidirectional
    compatibility across Keras 2 (TF 2.12) and Keras 3 (TF 2.16).
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(NUM_FEATURES,), name="input_landmarks"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
            tf.keras.layers.BatchNormalization(name="bn_1"),
            tf.keras.layers.Dropout(0.3, name="drop_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
            tf.keras.layers.BatchNormalization(name="bn_2"),
            tf.keras.layers.Dropout(0.2, name="drop_2"),
            tf.keras.layers.Dense(32, activation="relu", name="dense_3"),
            tf.keras.layers.Dropout(0.1, name="drop_3"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
        ],
        name="signbridge_static_isl",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Metrics & Confusion Matrix (NumPy native, zero scikit-learn dependency)
# ---------------------------------------------------------------------------

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]):
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


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    export_dir = Path(args.export_dir)

    print("=" * 70)
    print("SANKETA STATIC ISL LOCAL MODEL TRAINING")
    print("=" * 70)

    # 1. Load and normalize data
    X, y, members = load_all_samples(dataset_dir)
    if len(X) == 0:
        print("[ERROR] No samples loaded. Cannot train.")
        sys.exit(1)

    # 2. Person-based split
    X_train, y_train, X_val, y_val, X_test, y_test, test_person_name = split_person_based(
        X, y, members, args.test_member
    )

    # Compute class weights for imbalanced handling
    cls_counts = np.bincount(y_train, minlength=len(STATIC_SIGNS))
    total_train = len(y_train)
    class_weights = {}
    for i, count in enumerate(cls_counts):
        class_weights[i] = float(total_train / (len(STATIC_SIGNS) * max(count, 1)))

    # 3. Build Model
    print("\n[3/6] Building Lightweight Classifier...")
    model = build_static_classifier(len(STATIC_SIGNS))
    model.summary(print_fn=lambda x: print(f"      {x}"))

    # 4. Train Model
    print(f"\n[4/6] Training model for up to {args.epochs} epochs (EarlyStopping patience=15)...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])

    # 5. Evaluate on Held-Out Test Person
    print("\n[5/6] Evaluating on Held-Out Test Set (Person-Based)...")
    y_test_probs = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)

    test_acc, conf_mat, precisions, recalls, f1s = compute_classification_metrics(
        y_test, y_test_pred, STATIC_SIGNS
    )

    print("\n" + "-" * 70)
    print(f"PERSON-BASED TEST ACCURACY ({test_person_name}): {test_acc:.2%}")
    print(f"TRAINING ACCURACY:                       {train_acc:.2%}")
    print(f"VALIDATION ACCURACY:                     {val_acc:.2%}")
    print("-" * 70)

    # Per-class table
    print(f"{'CLASS':<12} {'PRECISION':<12} {'RECALL':<12} {'F1-SCORE':<12} {'SAMPLES':<8}")
    print("-" * 70)
    for i, sign in enumerate(STATIC_SIGNS):
        n_samples = int(np.sum(y_test == i))
        print(f"{sign:<12} {precisions[i]:>9.2%}   {recalls[i]:>9.2%}   {f1s[i]:>9.2%}   {n_samples:>7}")
    print("-" * 70)

    # Confusion matrix
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    header = "       " + "".join(f"{s[:4]:>6}" for s in STATIC_SIGNS)
    print(header)
    for i, row in enumerate(conf_mat):
        row_str = f"{STATIC_SIGNS[i][:5]:<6} " + "".join(f"{v:>6}" for v in row)
        print(row_str)

    # -----------------------------------------------------------------------
    # CRITICAL: ANTI-"WATER" PREDICTION BIAS VERIFICATION
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANTI-'WATER' PREDICTION BIAS VERIFICATION")
    print("=" * 70)
    water_idx = SIGN_TO_INDEX["WATER"]
    water_bias_detected = False

    test_per_class = 20
    print(f"Testing {test_per_class} samples from each class on held-out test data:\n")
    print(f"{'ACTUAL CLASS':<14} {'CORRECT':<10} {'PREDICTED AS WATER':<22} {'STATUS'}")
    print("-" * 65)

    all_predicted_classes = []
    for cls_idx, sign in enumerate(STATIC_SIGNS):
        cls_mask = (y_test == cls_idx)
        X_cls = X_test[cls_mask]
        n_test = min(test_per_class, len(X_cls))

        if n_test == 0:
            print(f"{sign:<14} [No test samples]")
            continue

        sample_subset = X_cls[:n_test]
        preds = np.argmax(model.predict(sample_subset, verbose=0), axis=1)
        all_predicted_classes.extend(preds)

        correct_count = int(np.sum(preds == cls_idx))
        pred_water_count = int(np.sum(preds == water_idx))

        # If it's NOT the WATER class, predicting WATER for majority is a bias failure
        is_water_class = (cls_idx == water_idx)
        if not is_water_class and pred_water_count >= (n_test * 0.40):
            water_bias_detected = True
            status = "[BIAS DETECTED]"
        else:
            status = "[PASS]"

        print(f"{sign:<14} {correct_count:>2}/{n_test:<7} {pred_water_count:>2}/{n_test:<19} {status}")

    pred_counts = np.bincount(all_predicted_classes, minlength=len(STATIC_SIGNS))
    water_pred_pct = pred_counts[water_idx] / len(all_predicted_classes) if all_predicted_classes else 0.0

    print("-" * 65)
    print(f"Total 'WATER' predictions across all test samples: {pred_counts[water_idx]}/{len(all_predicted_classes)} ({water_pred_pct:.1%})")

    if water_bias_detected or water_pred_pct > 0.35:
        print("\n[CRITICAL ERROR] 'Everything = Water' prediction bias detected!")
        print("Training cannot be declared successful. Halting export.")
        sys.exit(1)
    else:
        print("\n[VERIFIED] NO 'Water' prediction bias detected. Predictions are balanced across classes.")

    # 6. Export Model & Labels
    print("\n[6/6] Exporting Model & Labels...")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export isl_model.keras
    target_model_path = export_dir / "isl_model.keras"

    # Backup previous model if it exists
    if target_model_path.exists():
        backup_path = export_dir / "isl_model_backup.keras"
        try:
            shutil.copy(target_model_path, backup_path)
            print(f"      Backed up previous model to: {backup_path.name}")
        except Exception:
            pass

    # Save model in standard Keras format compatible with TF 2.12
    model.save(str(target_model_path))
    print(f"      Exported Model:  {target_model_path.resolve()} ({target_model_path.stat().st_size / 1024:.1f} KB)")

    # Also save .h5 format as extra safety fallback
    target_h5_path = export_dir / "isl_model.h5"
    model.save(str(target_h5_path))

    # Export labels.txt with EXACT class order
    labels_txt_path = export_dir / "labels.txt"
    with open(labels_txt_path, "w", encoding="utf-8") as f:
        for sign in STATIC_SIGNS:
            f.write(f"{sign}\n")
    print(f"      Exported Labels: {labels_txt_path.resolve()}")

    # Export labels.json
    labels_json_path = export_dir / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": STATIC_SIGNS,
                "input_shape": [63],
                "num_classes": len(STATIC_SIGNS),
                "model_version": "2.0_static_multi_person",
                "test_accuracy": round(test_acc, 4),
                "test_person": test_person_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    print(f"      Exported JSON:   {labels_json_path.resolve()}")

    # Export static_classes.json
    static_classes_path = export_dir / "static_classes.json"
    with open(static_classes_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": STATIC_SIGNS,
                "num_classes": len(STATIC_SIGNS),
            },
            f,
            indent=2,
        )
    print(f"      Exported Classes JSON: {static_classes_path.resolve()}")

    # Also export to training/models/ for safe local retention
    local_models_dir = _PROJECT_ROOT / "training" / "models"
    local_models_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(local_models_dir / "isl_model.keras"))
    shutil.copy(labels_txt_path, local_models_dir / "labels.txt")
    shutil.copy(labels_json_path, local_models_dir / "labels.json")

    print("\n" + "=" * 70)
    print("TRAINING & EXPORT COMPLETE")
    print("=" * 70)
    print(f"Model File:             {target_model_path.name}")
    print(f"Labels:                 {', '.join(STATIC_SIGNS)}")
    print(f"Person-Based Test Acc:  {test_acc:.2%}")
    print(f"Water Bias Exists:      NO")
    print("=" * 70)


if __name__ == "__main__":
    train()
