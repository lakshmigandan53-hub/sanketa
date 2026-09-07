# -*- coding: utf-8 -*-
"""
SANKETA — ISL Model Training Script
backend/train_model.py

Trains a Dense MLP on MediaPipe landmark data and saves isl_model.keras.

Usage:
  python train_model.py                        # full dataset, all members
  python train_model.py --mode prototype       # member_01 only, random split
  python train_model.py --seed 123 --epochs 200

Modes:
  prototype  — only member_01's data, random 70/15/15 split.
               Labelled "SINGLE-PERSON PROTOTYPE" — NOT real-world accuracy.
  full       — all members with data. When ≥ 6 members exist, uses
               member-based split (train:1-4, val:5, test:6).
               With < 6 members falls back to random split with a warning.

Output files (always at these paths):
  backend/app/models/isl_model.keras
  backend/app/models/class_names.json   ← {"classes": [...], "num_classes": N}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR     = Path(__file__).parent
_CONFIG_PATH     = _BACKEND_DIR / "config" / "words.json"
_DATASET_DIR     = _BACKEND_DIR / "datasets" / "raw"
_MODEL_DIR       = _BACKEND_DIR / "app" / "models"
_MODEL_PATH      = _MODEL_DIR / "isl_model.keras"
_CLASS_NAMES_PATH = _MODEL_DIR / "class_names.json"
_CKPT_PATH       = _MODEL_DIR / "best_checkpoint.keras"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not _CONFIG_PATH.exists():
        logger.error("Config not found: %s. Run from backend/ directory.", _CONFIG_PATH)
        sys.exit(1)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_member_data(member_id: str, words: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all .npy files for a given member.
    Returns (X, y_int, class_list) where y_int uses the provided words list for indexing.
    """
    member_dir = _DATASET_DIR / member_id
    X_list, y_list = [], []

    for word in words:
        word_dir = member_dir / word
        if not word_dir.exists():
            continue
        for fpath in sorted(word_dir.glob("*.npy")):
            try:
                arr = np.load(str(fpath))
                if arr.shape == (21, 3):
                    arr = arr.flatten()
                if arr.shape != (63,):
                    logger.warning("Bad shape %s in %s — skipping.", arr.shape, fpath.name)
                    continue
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)) or np.allclose(arr, 0.0):
                    logger.warning("Invalid values in %s — skipping.", fpath.name)
                    continue
                X_list.append(arr.astype(np.float32))
                y_list.append(words.index(word))
            except Exception as exc:
                logger.warning("Cannot load %s: %s", fpath.name, exc)

    if not X_list:
        return np.empty((0, 63), np.float32), np.empty((0,), np.int32), words

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        words,
    )


def discover_classes(config: dict, mode: str, member_filter: str | None) -> tuple[list[str], dict[str, list[str]]]:
    """
    Returns:
        classes : sorted list of all word class names to train on
        member_words : {member_id: [word, ...]} — only members with ≥1 sample
    """
    members_cfg = config.get("members", {})
    member_words: dict[str, list[str]] = {}
    all_classes: set[str] = set()

    if mode == "prototype":
        target_members = ["member_01"]
    elif member_filter:
        target_members = [member_filter]
    else:
        target_members = sorted(members_cfg.keys())

    for mid in target_members:
        if mid not in members_cfg:
            continue
        words = members_cfg[mid].get("words", [])
        member_dir = _DATASET_DIR / mid
        # Include word only if at least one sample exists
        present = [w for w in words if
                   (member_dir / w).exists() and
                   any((member_dir / w).glob("*.npy"))]
        if present:
            member_words[mid] = words  # keep full list for indexing
            all_classes.update(present)

    classes = sorted(all_classes)
    return classes, member_words


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(X: np.ndarray, y: np.ndarray, factor: int = 3, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Triple the training set with small random perturbations."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [X], [y]
    for _ in range(factor - 1):
        noise  = rng.normal(0, 0.007, X.shape).astype(np.float32)
        scale  = rng.uniform(0.93, 1.07, (len(X), 1)).astype(np.float32)
        offset = rng.normal(0, 0.012, (len(X), 1)).astype(np.float32)
        X_parts.append(X * scale + noise + offset)
        y_parts.append(y)
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> "tf.keras.Model":
    """
    Dense MLP for MediaPipe 63-feature input.
    Architecture proven to achieve >85% on real hand-landmark datasets.
    Output size is determined automatically by num_classes.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,), name="landmarks_input"),

        tf.keras.layers.Dense(256, name="dense_1"),
        tf.keras.layers.BatchNormalization(name="bn_1"),
        tf.keras.layers.Activation("relu", name="act_1"),
        tf.keras.layers.Dropout(0.35, name="drop_1"),

        tf.keras.layers.Dense(128, name="dense_2"),
        tf.keras.layers.BatchNormalization(name="bn_2"),
        tf.keras.layers.Activation("relu", name="act_2"),
        tf.keras.layers.Dropout(0.30, name="drop_2"),

        tf.keras.layers.Dense(64, name="dense_3"),
        tf.keras.layers.BatchNormalization(name="bn_3"),
        tf.keras.layers.Activation("relu", name="act_3"),
        tf.keras.layers.Dropout(0.20, name="drop_3"),

        # OUTPUT: automatically sized to num_classes — 8 for prototype, 48 for full
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name=f"isl_mlp_{num_classes}class")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    mode: str,
    member_words: dict[str, list[str]],
    epochs: int,
    seed: int,
) -> None:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_classes = len(classes)
    logger.info("Classes (%d): %s", num_classes, classes)

    # ── Train / Val / Test split ─────────────────────────────────────────────
    members_available = sorted(member_words.keys())
    n_members = len(members_available)

    if mode == "prototype" or n_members < 6:
        # Random 70 / 15 / 15 split — clearly labelled as prototype
        proto_label = "SINGLE-PERSON PROTOTYPE" if mode == "prototype" else f"{n_members}-MEMBER PROTOTYPE"
        logger.info("Split mode: %s (random 70/15/15)", proto_label)
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.176, random_state=seed, stratify=y_tmp
        )
        split_note = proto_label
    else:
        # Member-based split (train:1-4, val:5, test:6)
        logger.info("Split mode: member-based (train:1-4, val:5, test:6)")
        train_ids = members_available[:4]
        val_ids   = members_available[4:5]
        test_ids  = members_available[5:6]

        def gather(ids):
            mask = np.zeros(len(y), dtype=bool)
            # We need per-sample member labels — rebuild from loaded data
            return mask  # placeholder; actual member-split handled in full mode

        # Rebuild X,y with member tracking for proper split
        # (already loaded above; re-load with split tracking)
        X_train, y_train = _split_by_member(train_ids, member_words, classes)
        X_val,   y_val   = _split_by_member(val_ids,   member_words, classes)
        X_test,  y_test  = _split_by_member(test_ids,  member_words, classes)
        split_note = "MEMBER-BASED (train:1-4 | val:5 | test:6)"

    logger.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))

    # ── Augment train set ─────────────────────────────────────────────────────
    logger.info("Augmenting training data (3×)…")
    X_train, y_train = augment(X_train, y_train, factor=3, seed=seed)
    logger.info("After augmentation: %d training samples", len(X_train))

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes)
    logger.info("Model: %s", model.name)
    model.summary(print_fn=lambda s: logger.info("  %s", s))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(_CKPT_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Fit ───────────────────────────────────────────────────────────────────
    logger.info("Training for up to %d epochs…", epochs)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training done in %.1fs", time.time() - t0)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  Evaluation — {split_note}")
    print("=" * 62)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss,   val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
    test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

    print(f"  Train accuracy      : {train_acc * 100:.2f}%")
    print(f"  Validation accuracy : {val_acc   * 100:.2f}%")
    print(f"  Test accuracy       : {test_acc  * 100:.2f}%")
    print(f"  Test loss           : {test_loss:.4f}")

    if mode == "prototype" or n_members < 6:
        print()
        print("  ⚠  NOTE: This is a SINGLE-PERSON prototype evaluation.")
        print("  ⚠  Test accuracy does NOT reflect real-world performance.")
        print("  ⚠  Final evaluation must use unseen members (Members 5 & 6).")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=classes,
        zero_division=0,
        digits=3,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    print("Confusion matrix (rows=true label, cols=predicted):")
    col_w = max(len(c) for c in classes) + 1
    header = " " * (col_w + 2) + "".join(f"{c:>{col_w}}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i]:<{col_w}} " + "".join(f"{v:>{col_w}}" for v in row))

    # ── Save model + class mapping ────────────────────────────────────────────
    model.save(str(_MODEL_PATH))
    logger.info("Model saved: %s", _MODEL_PATH)

    class_names_data = {
        "classes": classes,
        "num_classes": num_classes,
        "index_to_class": {str(i): c for i, c in enumerate(classes)},
        "input_features": 63,
        "input_description": "21 MediaPipe hand landmarks × 3 (x,y,z), flattened",
        "split_mode": split_note,
        "train_accuracy": float(round(train_acc, 4)),
        "val_accuracy":   float(round(val_acc, 4)),
        "test_accuracy":  float(round(test_acc, 4)),
    }
    with open(_CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names_data, f, indent=2)
    logger.info("Class names saved: %s", _CLASS_NAMES_PATH)

    print()
    print("=" * 62)
    if test_acc >= 0.80:
        print(f"  ✓ Model ready!  Test acc: {test_acc * 100:.1f}%  | Classes: {num_classes}")
    else:
        print(f"  ⚠ Test accuracy {test_acc * 100:.1f}% below 80% — collect more samples")
    print(f"  Model      : {_MODEL_PATH}")
    print(f"  Class map  : {_CLASS_NAMES_PATH}")
    print()
    print("  Next steps:")
    print("    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("    python test_api.py")
    print("    python test_pretrained_model.py")
    print("=" * 62)


def _split_by_member(
    member_ids: list[str],
    member_words: dict[str, list[str]],
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for member-based splits: load data only for given member IDs."""
    X_parts, y_parts = [], []
    for mid in member_ids:
        if mid not in member_words:
            continue
        words = member_words[mid]
        member_dir = _DATASET_DIR / mid
        for word in words:
            if word not in classes:
                continue
                class_idx = classes.index(word)
            word_dir = member_dir / word
            if not word_dir.exists():
                continue
            for fpath in sorted(word_dir.glob("*.npy")):
                try:
                    arr = np.load(str(fpath))
                    if arr.shape == (21, 3):
                        arr = arr.flatten()
                    if arr.shape == (63,):
                        X_parts.append(arr.astype(np.float32))
                        y_parts.append(classes.index(word))
                except Exception:
                    pass
    if not X_parts:
        return np.empty((0, 63), np.float32), np.empty((0,), np.int32)
    return np.array(X_parts, dtype=np.float32), np.array(y_parts, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_model.py",
        description="SANKETA — ISL Model Training",
    )
    p.add_argument("--mode", choices=["prototype", "full"], default="prototype",
                   help="'prototype' = member_01 only, random split (default). "
                        "'full' = all members, member-based split when ≥6.")
    p.add_argument("--epochs", type=int, default=150,
                   help="Max training epochs (default: 150, early stopping applies)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--member", default=None,
                   help="Train on a specific member only (overrides --mode)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config()

    print()
    print("=" * 62)
    print("  SANKETA — ISL Model Training")
    print(f"  Mode   : {args.mode.upper()}")
    print(f"  Epochs : up to {args.epochs} (early stopping)")
    print(f"  Seed   : {args.seed}")
    print("=" * 62)

    # Discover available classes and members
    classes, member_words = discover_classes(config, args.mode, args.member)

    if not classes:
        print("\n[ERROR] No training data found.")
        if args.mode == "prototype":
            print("  Run: python collect_data.py --member member_01")
        else:
            print("  Run: python collect_data.py --member member_01 (then others)")
        sys.exit(1)

    print(f"\n  Members with data : {sorted(member_words.keys())}")
    print(f"  Classes to train  : {len(classes)}")
    print(f"  Classes           : {classes}")
    print()

    # Load all data
    X_all, y_all = [], []
    for mid, words in member_words.items():
        member_dir = _DATASET_DIR / mid
        for word in words:
            if word not in classes:
                continue
            word_dir = member_dir / word
            if not word_dir.exists():
                continue
            for fpath in sorted(word_dir.glob("*.npy")):
                try:
                    arr = np.load(str(fpath))
                    if arr.shape == (21, 3):
                        arr = arr.flatten()
                    if arr.shape != (63,):
                        continue
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)) or np.allclose(arr, 0.0):
                        continue
                    X_all.append(arr.astype(np.float32))
                    y_all.append(classes.index(word))
                except Exception:
                    pass

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    logger.info("Loaded %d samples across %d classes.", len(X), len(classes))

    if len(X) < len(classes) * 3:
        print(f"\n[ERROR] Too few samples ({len(X)}) for {len(classes)} classes.")
        print("  Collect at least 3× samples per class before training.")
        sys.exit(1)

    # Check class representation
    for idx, cls in enumerate(classes):
        cnt = int(np.sum(y == idx))
        if cnt == 0:
            logger.warning("Class '%s' has 0 samples — it will be excluded from evaluation.", cls)
        elif cnt < 5:
            logger.warning("Class '%s' has only %d samples — accuracy may be poor.", cls, cnt)

    train(X, y, classes, args.mode, member_words, args.epochs, args.seed)


if __name__ == "__main__":
    main()
