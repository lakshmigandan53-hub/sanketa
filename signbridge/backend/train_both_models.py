# -*- coding: utf-8 -*-
"""
SANKETA — Dual Model Training Script (Static + Dynamic)
backend/train_both_models.py

Trains:
1. Static Model:
   - Architecture: Dense MLP
   - Input: (63,) MediaPipe hand landmarks
   - Classes: HELLO, YES, NO, WATER (from datasets/raw/member_01)
   - Output: backend/app/models/static_model.keras, static_classes.json

2. Dynamic Model:
   - Architecture: Bidirectional LSTM / GRU sequence classifier
   - Input: (30, 63) MediaPipe sequence
   - Classes: THANK_YOU, PLEASE, SORRY, HELP (from datasets/raw/member_01_dynamic)
   - Output: backend/app/models/dynamic_model.keras, dynamic_classes.json
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf

def numpy_train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_val = max(1, int(len(c_idx) * test_size))
        val_idx.extend(c_idx[:n_val])
        train_idx.extend(c_idx[n_val:])
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent
DATASET_RAW_DIR = BACKEND_DIR / "datasets" / "raw"
MODELS_DIR = BACKEND_DIR / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_CLASSES = ["HELLO", "NO", "WATER", "YES"]
DYNAMIC_CLASSES = ["HELP", "PLEASE", "SORRY", "THANK_YOU"]


# ---------------------------------------------------------------------------
# Training Static Model
# ---------------------------------------------------------------------------
def train_static_model():
    logger.info("=" * 60)
    logger.info("TRAINING STATIC MODEL (63 features -> 4 classes)")
    logger.info("=" * 60)

    static_dir = DATASET_RAW_DIR / "member_01"
    X, y = [], []

    for label_idx, word in enumerate(STATIC_CLASSES):
        word_dir = static_dir / word
        files = sorted(list(word_dir.glob("*.npy")))
        logger.info(f"  Loading {word}: {len(files)} samples")
        for f in files:
            arr = np.load(str(f)).astype(np.float32)
            if arr.shape == (21, 3):
                arr = arr.flatten()
            if arr.shape == (63,) and not np.isnan(arr).any():
                X.append(arr)
                y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    logger.info(f"Total static samples: {X.shape[0]}, feature shape: {X.shape[1:]}")

    X_train, X_val, y_train, y_val = numpy_train_test_split(
        X, y, test_size=0.2, seed=42
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(STATIC_CLASSES), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Static Model Validation Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}")

    # Save model and classes
    static_model_path = MODELS_DIR / "static_model.keras"
    static_classes_path = MODELS_DIR / "static_classes.json"
    model.save(str(static_model_path))

    with open(static_classes_path, "w", encoding="utf-8") as f:
        json.dump({"classes": STATIC_CLASSES, "num_classes": len(STATIC_CLASSES), "val_accuracy": float(acc)}, f, indent=2)

    # Legacy fallback isl_model.keras
    legacy_model_path = MODELS_DIR / "isl_model.keras"
    model.save(str(legacy_model_path))

    return acc, str(static_model_path), str(static_classes_path)


# ---------------------------------------------------------------------------
# Training Dynamic Model
# ---------------------------------------------------------------------------
def train_dynamic_model():
    logger.info("=" * 60)
    logger.info("TRAINING DYNAMIC MODEL (30x63 sequence -> 4 classes)")
    logger.info("=" * 60)

    dyn_dir = DATASET_RAW_DIR / "member_01_dynamic"
    X, y = [], []

    for label_idx, word in enumerate(DYNAMIC_CLASSES):
        word_dir = dyn_dir / word
        files = sorted(list(word_dir.glob("*.npy")))
        logger.info(f"  Loading {word}: {len(files)} sequences")
        for f in files:
            arr = np.load(str(f)).astype(np.float32)
            if arr.shape == (30, 63) and not np.isnan(arr).any():
                X.append(arr)
                y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    logger.info(f"Total dynamic sequences: {X.shape[0]}, sequence shape: {X.shape[1:]}")

    X_train, X_val, y_train, y_val = numpy_train_test_split(
        X, y, test_size=0.2, seed=42
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 63)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(DYNAMIC_CLASSES), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Dynamic Model Validation Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}")

    # Save model and classes
    dyn_model_path = MODELS_DIR / "dynamic_model.keras"
    dyn_classes_path = MODELS_DIR / "dynamic_classes.json"
    model.save(str(dyn_model_path))

    with open(dyn_classes_path, "w", encoding="utf-8") as f:
        json.dump({"classes": DYNAMIC_CLASSES, "num_classes": len(DYNAMIC_CLASSES), "val_accuracy": float(acc)}, f, indent=2)

    return acc, str(dyn_model_path), str(dyn_classes_path)


# ---------------------------------------------------------------------------
# Combined Class Names Config
# ---------------------------------------------------------------------------
def save_combined_classes():
    all_classes = sorted(STATIC_CLASSES + DYNAMIC_CLASSES)
    combined_path = MODELS_DIR / "class_names.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": all_classes,
            "num_classes": len(all_classes),
            "static_classes": STATIC_CLASSES,
            "dynamic_classes": DYNAMIC_CLASSES,
        }, f, indent=2)
    logger.info(f"Saved all 8 classes to {combined_path}")


if __name__ == "__main__":
    static_acc, static_m_path, static_c_path = train_static_model()
    dynamic_acc, dyn_m_path, dyn_c_path = train_dynamic_model()
    save_combined_classes()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Static Model  : {static_acc * 100:.2f}% validation accuracy -> {static_m_path}")
    print(f"Dynamic Model : {dynamic_acc * 100:.2f}% validation accuracy -> {dyn_m_path}")
    print("=" * 60)
