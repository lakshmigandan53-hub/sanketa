# -*- coding: utf-8 -*-
"""
SANKETA Full AI Model Training & Evaluation Pipeline
backend/train_and_evaluate_v2.py

Fulfills Phase 1 through Phase 9:
1. Dataset inspection & integrity reporting
2. Strict leak-free train/val/test splitting (sequence-level for dynamic)
3. Realistic augmentation preserving hand physics & temporal trajectory
4. Deep neural network architectures (Dense with BatchNorm/Dropout for static, BiLSTM for dynamic)
5. Model checkpoints, early stopping, LR scheduling
6. Full held-out test evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
7. Generation of static_confusion_matrix.png, dynamic_confusion_matrix.png, training_report.json
8. Comparison of V1 vs V2 models
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import logging
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SANKETA_TRAINER")

BACKEND_DIR = Path(__file__).parent
DATASETS_RAW = BACKEND_DIR / "datasets" / "raw"
MODELS_DIR = BACKEND_DIR / "app" / "models"
REPORTS_DIR = BACKEND_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_CLASSES = ["HELLO", "NO", "WATER", "YES"]
DYNAMIC_CLASSES = ["HELP", "PLEASE", "SORRY", "THANK_YOU"]

# ---------------------------------------------------------------------------
# PHASE 1: INSPECT THE DATASET
# ---------------------------------------------------------------------------
def inspect_dataset():
    logger.info("=" * 70)
    logger.info("PHASE 1: INSPECTING DATASET INTEGRITY & SHAPES")
    logger.info("=" * 70)

    static_stats = {}
    dynamic_stats = {}

    # Inspect Static Classes
    static_base = DATASETS_RAW / "member_01"
    for word in STATIC_CLASSES:
        p = static_base / word
        files = sorted(list(p.glob("*.npy")))
        shapes = set()
        mins, maxs = [], []
        corrupt = 0
        for f in files:
            try:
                arr = np.load(f)
                if arr.shape == (21, 3):
                    arr = arr.flatten()
                shapes.add(arr.shape)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    corrupt += 1
                else:
                    mins.append(float(arr.min()))
                    maxs.append(float(arr.max()))
            except Exception:
                corrupt += 1

        static_stats[word] = {
            "count": len(files),
            "shapes": [list(s) for s in shapes],
            "min": min(mins) if mins else None,
            "max": max(maxs) if maxs else None,
            "corrupt": corrupt,
            "valid_shape": all(s == (63,) for s in shapes)
        }
        logger.info(f"  [Static]  {word:<10}: {len(files)} samples, shapes={shapes}, range=[{min(mins):.3f}, {max(maxs):.3f}], corrupt={corrupt}")

    # Inspect Dynamic Classes
    dyn_base = DATASETS_RAW / "member_01_dynamic"
    for word in DYNAMIC_CLASSES:
        p = dyn_base / word
        files = sorted(list(p.glob("*.npy")))
        shapes = set()
        mins, maxs = [], []
        corrupt = 0
        for f in files:
            try:
                arr = np.load(f)
                shapes.add(arr.shape)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    corrupt += 1
                else:
                    mins.append(float(arr.min()))
                    maxs.append(float(arr.max()))
            except Exception:
                corrupt += 1

        dynamic_stats[word] = {
            "count": len(files),
            "shapes": [list(s) for s in shapes],
            "min": min(mins) if mins else None,
            "max": max(maxs) if maxs else None,
            "corrupt": corrupt,
            "valid_shape": all(s == (30, 63) for s in shapes)
        }
        logger.info(f"  [Dynamic] {word:<10}: {len(files)} sequences, shapes={shapes}, range=[{min(mins):.3f}, {max(maxs):.3f}], corrupt={corrupt}")

    return static_stats, dynamic_stats


# ---------------------------------------------------------------------------
# PHASE 2 & 3: DATA PREPROCESSING & LEAK-FREE SPLITTING
# ---------------------------------------------------------------------------
def split_stratified_sequence(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits sequences (or static samples) strictly at sample/sequence level.
    Ensures zero inter-frame leakage.
    """
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n = len(c_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_idx.extend(c_idx[:n_train])
        val_idx.extend(c_idx[n_train:n_train + n_val])
        test_idx.extend(c_idx[n_train + n_val:])

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def augment_static_landmarks(lm_flat, angle_deg=6.0, scale_range=0.06, trans_range=0.02):
    """Physically plausible 2D rotation, scaling, translation, and Gaussian jitter."""
    lm = lm_flat.reshape(21, 3).copy()
    center = np.mean(lm[:, :2], axis=0)

    theta = np.radians(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    lm[:, :2] = (lm[:, :2] - center) @ R.T + center

    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    lm[:, :2] = (lm[:, :2] - center) * scale + center

    tx = np.random.uniform(-trans_range, trans_range)
    ty = np.random.uniform(-trans_range, trans_range)
    lm[:, 0] += tx
    lm[:, 1] += ty

    lm += np.random.normal(0, 0.001, lm.shape)
    return np.clip(lm.flatten(), -0.2, 1.2).astype(np.float32)


def augment_dynamic_sequence(seq, trans_range=0.015, scale_range=0.05, noise_std=0.001):
    """Augments dynamic sequences preserving temporal trajectory."""
    seq_aug = seq.copy()
    # Consistent trajectory translation
    tx = np.random.uniform(-trans_range, trans_range)
    ty = np.random.uniform(-trans_range, trans_range)
    seq_aug = seq_aug.reshape(30, 21, 3)
    seq_aug[:, :, 0] += tx
    seq_aug[:, :, 1] += ty
    
    # Scale
    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    seq_aug = seq_aug * scale

    # Temporal jitter
    seq_aug += np.random.normal(0, noise_std, seq_aug.shape)
    return np.clip(seq_aug.reshape(30, 63), -0.2, 1.2).astype(np.float32)


# ---------------------------------------------------------------------------
# PHASE 4: TRAIN STATIC MODEL
# ---------------------------------------------------------------------------
def train_static_model():
    logger.info("=" * 70)
    logger.info("PHASE 4: TRAINING STATIC MODEL V2")
    logger.info("=" * 70)

    X_raw, y_raw = [], []
    for label_idx, word in enumerate(STATIC_CLASSES):
        p = DATASETS_RAW / "member_01" / word
        for f in sorted(list(p.glob("*.npy"))):
            arr = np.load(f).astype(np.float32)
            if arr.shape == (21, 3):
                arr = arr.flatten()
            if arr.shape == (63,):
                X_raw.append(arr)
                y_raw.append(label_idx)

    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)

    X_train, y_train, X_val, y_val, X_test, y_test = split_stratified_sequence(
        X_raw, y_raw, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    # Augment training set (3x)
    X_train_aug, y_train_aug = [], []
    for x, y in zip(X_train, y_train):
        X_train_aug.append(x)
        y_train_aug.append(y)
        for _ in range(2):
            X_train_aug.append(augment_static_landmarks(x))
            y_train_aug.append(y)

    X_train_aug = np.array(X_train_aug, dtype=np.float32)
    y_train_aug = np.array(y_train_aug, dtype=np.int32)

    logger.info(f"Static Split: Train={len(X_train_aug)} (augmented from {len(X_train)}), Val={len(X_val)}, Test={len(X_test)}")

    # Neural Network Architecture requested:
    # Dense(256) -> BatchNorm -> ReLU -> Dropout
    # Dense(128) -> BatchNorm -> ReLU -> Dropout
    # Dense(64) -> BatchNorm -> ReLU -> Dropout
    # Dense(4) -> Softmax
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(len(STATIC_CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = str(MODELS_DIR / "static_model_v2.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Load best checkpoint for test evaluation
    best_model = tf.keras.models.load_model(checkpoint_path)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train_aug, y_train_aug, verbose=0)

    logger.info(f"Static Model V2 Results -> Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")

    return best_model, history, X_test, y_test, train_acc, val_acc, test_acc


# ---------------------------------------------------------------------------
# PHASE 5 & 6: TRAIN DYNAMIC MODEL
# ---------------------------------------------------------------------------
def train_dynamic_model():
    logger.info("=" * 70)
    logger.info("PHASE 5: TRAINING DYNAMIC MODEL V2 (BiLSTM Temporal Architecture)")
    logger.info("=" * 70)

    X_raw, y_raw = [], []
    for label_idx, word in enumerate(DYNAMIC_CLASSES):
        p = DATASETS_RAW / "member_01_dynamic" / word
        for f in sorted(list(p.glob("*.npy"))):
            arr = np.load(f).astype(np.float32)
            if arr.shape == (30, 63):
                X_raw.append(arr)
                y_raw.append(label_idx)

    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)

    X_train, y_train, X_val, y_val, X_test, y_test = split_stratified_sequence(
        X_raw, y_raw, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    # Augment dynamic sequences (2x)
    X_train_aug, y_train_aug = [], []
    for seq, y in zip(X_train, y_train):
        X_train_aug.append(seq)
        y_train_aug.append(y)
        X_train_aug.append(augment_dynamic_sequence(seq))
        y_train_aug.append(y)

    X_train_aug = np.array(X_train_aug, dtype=np.float32)
    y_train_aug = np.array(y_train_aug, dtype=np.int32)

    logger.info(f"Dynamic Split: Train={len(X_train_aug)} (augmented from {len(X_train)}), Val={len(X_val)}, Test={len(X_test)}")

    # Architecture requested:
    # Bidirectional LSTM(128, return_sequences=True) -> Dropout(0.25)
    # Bidirectional LSTM(64) -> Dropout(0.20)
    # Dense(64, ReLU) -> Dense(4, Softmax)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 63)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(DYNAMIC_CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = str(MODELS_DIR / "dynamic_model_v2.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train_aug, y_train_aug, verbose=0)

    logger.info(f"Dynamic Model V2 Results -> Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")

    return best_model, history, X_test, y_test, train_acc, val_acc, test_acc


# ---------------------------------------------------------------------------
# PHASE 8: EVALUATION & REPORT GENERATION
# ---------------------------------------------------------------------------
def plot_and_save_confusion_matrix(cm, classes, title, output_path):
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight="bold")

    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {output_path}")


def evaluate_models(s_model, s_X_test, s_y_test, d_model, d_X_test, d_y_test,
                    s_train_acc, s_val_acc, s_test_acc,
                    d_train_acc, d_val_acc, d_test_acc,
                    static_stats, dynamic_stats):

    logger.info("=" * 70)
    logger.info("PHASE 8: EVALUATING ON HELD-OUT TEST DATA")
    logger.info("=" * 70)

    # Static Evaluation
    s_preds = s_model.predict(s_X_test, verbose=0)
    s_pred_cls = np.argmax(s_preds, axis=1)
    s_cm = confusion_matrix(s_y_test, s_pred_cls)
    s_p, s_r, s_f1, _ = precision_recall_fscore_support(s_y_test, s_pred_cls, average="weighted")
    s_per_p, s_per_r, s_per_f1, _ = precision_recall_fscore_support(s_y_test, s_pred_cls, average=None)

    s_per_class = {}
    for i, c in enumerate(STATIC_CLASSES):
        s_per_class[c] = {
            "accuracy": float(s_cm[i, i] / s_cm[i].sum()),
            "precision": float(s_per_p[i]),
            "recall": float(s_per_r[i]),
            "f1_score": float(s_per_f1[i]),
            "test_samples": int(s_cm[i].sum())
        }

    plot_and_save_confusion_matrix(
        s_cm, STATIC_CLASSES,
        f"Static Model V2 Confusion Matrix (Test Acc: {s_test_acc*100:.1f}%)",
        REPORTS_DIR / "static_confusion_matrix.png"
    )

    # Dynamic Evaluation
    d_preds = d_model.predict(d_X_test, verbose=0)
    d_pred_cls = np.argmax(d_preds, axis=1)
    d_cm = confusion_matrix(d_y_test, d_pred_cls)
    d_p, d_r, d_f1, _ = precision_recall_fscore_support(d_y_test, d_pred_cls, average="weighted")
    d_per_p, d_per_r, d_per_f1, _ = precision_recall_fscore_support(d_y_test, d_pred_cls, average=None)

    d_per_class = {}
    for i, c in enumerate(DYNAMIC_CLASSES):
        d_per_class[c] = {
            "accuracy": float(d_cm[i, i] / d_cm[i].sum()),
            "precision": float(d_per_p[i]),
            "recall": float(d_per_r[i]),
            "f1_score": float(d_per_f1[i]),
            "test_samples": int(d_cm[i].sum())
        }

    plot_and_save_confusion_matrix(
        d_cm, DYNAMIC_CLASSES,
        f"Dynamic Model V2 Confusion Matrix (Test Acc: {d_test_acc*100:.1f}%)",
        REPORTS_DIR / "dynamic_confusion_matrix.png"
    )

    # Comparison with V1 (Backups)
    s_v1 = tf.keras.models.load_model(str(MODELS_DIR / "static_model_v1.keras"))
    d_v1 = tf.keras.models.load_model(str(MODELS_DIR / "dynamic_model_v1.keras"))

    s_v1_preds = s_v1.predict(s_X_test, verbose=0)
    s_v1_acc = float(np.mean(np.argmax(s_v1_preds, axis=1) == s_y_test))
    d_v1_preds = d_v1.predict(d_X_test, verbose=0)
    d_v1_acc = float(np.mean(np.argmax(d_v1_preds, axis=1) == d_y_test))

    # Save Class Metadata Files
    with open(MODELS_DIR / "static_classes_v2.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v2",
            "classes": STATIC_CLASSES,
            "test_accuracy": float(s_test_acc),
            "per_class_accuracy": {c: s_per_class[c]["accuracy"] for c in STATIC_CLASSES},
            "confusion_matrix": s_cm.tolist()
        }, f, indent=2)

    with open(MODELS_DIR / "dynamic_classes_v2.json", "w", encoding="utf-8") as f:
        json.dump({
            "version": "v2",
            "classes": DYNAMIC_CLASSES,
            "test_accuracy": float(d_test_acc),
            "per_class_accuracy": {c: d_per_class[c]["accuracy"] for c in DYNAMIC_CLASSES},
            "confusion_matrix": d_cm.tolist()
        }, f, indent=2)

    # Master Training Report JSON
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_summary": {
            "static": static_stats,
            "dynamic": dynamic_stats
        },
        "static_model_v2": {
            "architecture": "Dense(256)->BatchNorm->ReLU->Dropout(0.25)->Dense(128)->BatchNorm->ReLU->Dropout(0.2)->Dense(64)->BatchNorm->ReLU->Dropout(0.15)->Dense(4,Softmax)",
            "training_accuracy": float(s_train_acc),
            "validation_accuracy": float(s_val_acc),
            "test_accuracy": float(s_test_acc),
            "weighted_precision": float(s_p),
            "weighted_recall": float(s_r),
            "weighted_f1": float(s_f1),
            "per_class": s_per_class,
            "confusion_matrix": s_cm.tolist(),
            "classes": STATIC_CLASSES
        },
        "dynamic_model_v2": {
            "architecture": "Bidirectional(LSTM(128, return_seq=True))->Dropout(0.25)->Bidirectional(LSTM(64))->Dropout(0.2)->Dense(64,ReLU)->Dense(4,Softmax)",
            "training_accuracy": float(d_train_acc),
            "validation_accuracy": float(d_val_acc),
            "test_accuracy": float(d_test_acc),
            "weighted_precision": float(d_p),
            "weighted_recall": float(d_r),
            "weighted_f1": float(d_f1),
            "per_class": d_per_class,
            "confusion_matrix": d_cm.tolist(),
            "classes": DYNAMIC_CLASSES
        },
        "model_comparison": {
            "static_v1_accuracy": s_v1_acc,
            "static_v2_accuracy": float(s_test_acc),
            "dynamic_v1_accuracy": d_v1_acc,
            "dynamic_v2_accuracy": float(d_test_acc)
        }
    }

    report_path = REPORTS_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Saved full training report to: {report_path}")
    return full_report


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    static_stats, dynamic_stats = inspect_dataset()
    
    s_model, s_hist, s_X_test, s_y_test, s_tr_acc, s_val_acc, s_te_acc = train_static_model()
    d_model, d_hist, d_X_test, d_y_test, d_tr_acc, d_val_acc, d_te_acc = train_dynamic_model()

    report = evaluate_models(
        s_model, s_X_test, s_y_test,
        d_model, d_X_test, d_y_test,
        s_tr_acc, s_val_acc, s_te_acc,
        d_tr_acc, d_val_acc, d_te_acc,
        static_stats, dynamic_stats
    )

    t_elapsed = time.time() - t0
    logger.info(f"Training and Evaluation Completed in {t_elapsed:.1f} seconds.")
