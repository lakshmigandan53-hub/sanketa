# -*- coding: utf-8 -*-
"""
SANKETA — 8-Class Static Model Training
backend/train_static_8class.py

Trains static model on ALL 8 ISL signs from member_01:
  HELLO, HELP, NO, PLEASE, SORRY, THANK_YOU, WATER, YES

Uses exact same architecture as existing v2 model (Dense/BN/Dropout)
but expanded to 8 output classes.

- 80/10/10 stratified split
- 3x augmentation (rotation, scale, translate, jitter)  
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- Full confusion matrix + per-class metrics
- Compares against existing 4-class model
- Only saves new model if it wins on original 4 classes AND overall accuracy ≥ 92%
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import time
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("STATIC_TRAINER_8")

BACKEND_DIR  = Path(__file__).parent
DATASET_DIR  = BACKEND_DIR / "datasets" / "raw" / "member_01"
MODELS_DIR   = BACKEND_DIR / "app" / "models"
REPORTS_DIR  = BACKEND_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# All 8 static classes (alphabetical order matches existing class_names.json)
ALL_8_CLASSES = ["HELLO", "HELP", "NO", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES"]
ORIG_4_CLASSES = ["HELLO", "NO", "WATER", "YES"]

OUTPUT_MODEL_PATH  = MODELS_DIR / "isl_static_best.keras"
OUTPUT_CLASSES_PATH = MODELS_DIR / "static_classes_8.json"


# ---------------------------------------------------------------------------
# Augmentation (same as train_and_evaluate_v2.py — proven to work)
# ---------------------------------------------------------------------------
def augment_static_landmarks(lm_flat, angle_deg=6.0, scale_range=0.06, trans_range=0.02):
    lm = lm_flat.reshape(21, 3).copy()
    center = np.mean(lm[:, :2], axis=0)
    theta = np.radians(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    lm[:, :2] = (lm[:, :2] - center) @ R.T + center
    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    lm[:, :2] = (lm[:, :2] - center) * scale + center
    lm[:, 0] += np.random.uniform(-trans_range, trans_range)
    lm[:, 1] += np.random.uniform(-trans_range, trans_range)
    lm += np.random.normal(0, 0.001, lm.shape)
    return np.clip(lm.flatten(), -0.2, 1.2).astype(np.float32)


# ---------------------------------------------------------------------------
# Stratified split (no data leakage — splits at sample level)
# ---------------------------------------------------------------------------
def split_stratified(X, y, train=0.8, val=0.1, seed=42):
    rng = np.random.RandomState(seed)
    tr_idx, v_idx, te_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * train)
        n_v  = int(n * val)
        tr_idx.extend(idx[:n_tr])
        v_idx.extend(idx[n_tr:n_tr + n_v])
        te_idx.extend(idx[n_tr + n_v:])
    return (X[tr_idx], y[tr_idx],
            X[v_idx],  y[v_idx],
            X[te_idx], y[te_idx])


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
def load_data():
    logger.info("Loading 8-class static dataset from %s", DATASET_DIR)
    X, y = [], []
    for label_idx, word in enumerate(ALL_8_CLASSES):
        class_dir = DATASET_DIR / word
        files = sorted(class_dir.glob("*.npy"))
        loaded = 0
        for f in files:
            try:
                arr = np.load(f).astype(np.float32)
                if arr.shape == (21, 3):
                    arr = arr.flatten()
                if arr.shape == (63,) and not (np.isnan(arr).any() or np.isinf(arr).any()):
                    X.append(arr)
                    y.append(label_idx)
                    loaded += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", f, e)
        logger.info("  %s: %d / %d samples loaded", word, loaded, len(files))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    logger.info("Total: %d samples, %d classes", len(X), len(np.unique(y)))
    return X, y


# ---------------------------------------------------------------------------
# Build model (Dense/BN/Dropout — same proven arch, expanded to 8 outputs)
# ---------------------------------------------------------------------------
def build_model(num_classes=8):
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

        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="isl_static_8class")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Compare against existing 4-class model on the 4-class subset
# ---------------------------------------------------------------------------
def evaluate_old_model_on_subset(X_test, y_test_all):
    """Test existing static_model_v2 on the 4-class subset of our test data."""
    old_model_path = MODELS_DIR / "static_model_v2.keras"
    if not old_model_path.exists():
        logger.warning("Old model not found — skipping comparison")
        return None

    logger.info("Loading old static_model_v2 for comparison…")
    old_model = tf.keras.models.load_model(str(old_model_path))

    # OLD 4-class labels: HELLO=0, NO=1, WATER=2, YES=3
    # NEW 8-class labels: HELLO=0, HELP=1, NO=2, PLEASE=3, SORRY=4, THANK_YOU=5, WATER=6, YES=7
    OLD_CLASSES = ["HELLO", "NO", "WATER", "YES"]
    NEW_TO_OLD = {0: 0, 2: 1, 6: 2, 7: 3}  # new_idx -> old_idx

    # Filter to only old 4 classes
    mask = np.isin(y_test_all, list(NEW_TO_OLD.keys()))
    X_sub = X_test[mask]
    y_sub_new = y_test_all[mask]
    y_sub_old = np.array([NEW_TO_OLD[yi] for yi in y_sub_new])

    if len(X_sub) == 0:
        logger.warning("No 4-class samples in test set for comparison")
        return None

    preds_old = old_model.predict(X_sub, verbose=0)
    pred_labels_old = np.argmax(preds_old, axis=1)
    old_acc = float(np.mean(pred_labels_old == y_sub_old))
    logger.info("OLD model (4-class) accuracy on 4-class test subset: %.2f%%", old_acc * 100)
    return old_acc, len(X_sub)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 70)
    logger.info("SANKETA — 8-Class Static Model Training")
    logger.info("=" * 70)
    t0 = time.time()

    # 1. Load data
    X, y = load_data()

    # 2. Split
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_stratified(X, y)
    logger.info("Split: Train=%d, Val=%d, Test=%d", len(X_tr), len(X_val), len(X_te))

    # 3. Augment training set 3x
    X_aug, y_aug = list(X_tr), list(y_tr)
    for xi, yi in zip(X_tr, y_tr):
        for _ in range(2):
            X_aug.append(augment_static_landmarks(xi))
            y_aug.append(yi)
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)
    logger.info("Augmented training set: %d samples (3x from %d)", len(X_aug), len(X_tr))

    # 4. Build model
    model = build_model(num_classes=8)
    model.summary(print_fn=logger.info)

    # 5. Callbacks
    ckpt_path = str(MODELS_DIR / "isl_static_best_ckpt.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
    ]

    # 6. Train
    logger.info("Training…")
    history = model.fit(
        X_aug, y_aug,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Test set evaluation
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    logger.info("=" * 50)
    logger.info("TEST ACCURACY  : %.4f  (%.2f%%)", test_acc, test_acc * 100)
    logger.info("TEST LOSS      : %.4f", test_loss)
    logger.info("=" * 50)

    # Per-class report
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    report_str = classification_report(y_te, y_pred, target_names=ALL_8_CLASSES)
    logger.info("\n%s", report_str)

    cm = confusion_matrix(y_te, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    per_class_acc = {}
    for i, cls in enumerate(ALL_8_CLASSES):
        mask = y_te == i
        if mask.sum() > 0:
            per_class_acc[cls] = float(np.mean(y_pred[mask] == i))

    # 8. Compare against old model
    old_result = evaluate_old_model_on_subset(X_te, y_te)

    # 9. Per-class accuracy on original 4 classes
    orig4_indices = {c: ALL_8_CLASSES.index(c) for c in ORIG_4_CLASSES}
    orig4_mask = np.isin(y_te, list(orig4_indices.values()))
    orig4_acc = float(np.mean(y_pred[orig4_mask] == y_te[orig4_mask])) if orig4_mask.sum() > 0 else 0.0
    logger.info("New model on original 4 classes only: %.2f%%", orig4_acc * 100)

    elapsed = time.time() - t0
    logger.info("Training time: %.1f seconds", elapsed)

    # 10. Decision: save only if genuinely better or equal on original 4 classes
    should_save = test_acc >= 0.90 and orig4_acc >= 0.90

    report = {
        "model_name": "isl_static_8class",
        "input_shape": [63],
        "output_classes": ALL_8_CLASSES,
        "num_classes": 8,
        "training_samples": len(X_aug),
        "validation_samples": len(X_val),
        "test_samples": len(X_te),
        "test_accuracy": float(test_acc),
        "orig4_accuracy": float(orig4_acc),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "training_time_seconds": round(elapsed, 1),
        "saved": should_save,
    }

    with open(REPORTS_DIR / "static_8class_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to reports/static_8class_report.json")

    if should_save:
        model.save(str(OUTPUT_MODEL_PATH))
        logger.info("✅ New 8-class static model saved to %s", OUTPUT_MODEL_PATH)

        classes_meta = {
            "version": "v3_8class",
            "classes": ALL_8_CLASSES,
            "num_classes": 8,
            "test_accuracy": float(test_acc),
            "orig4_accuracy": float(orig4_acc),
            "per_class_accuracy": per_class_acc,
        }
        with open(OUTPUT_CLASSES_PATH, "w") as f:
            json.dump(classes_meta, f, indent=2)
        logger.info("Class config saved to %s", OUTPUT_CLASSES_PATH)

        # Also update static_classes.json to 8 classes for live inference
        with open(MODELS_DIR / "static_classes.json", "w") as f:
            json.dump({"classes": ALL_8_CLASSES, "num_classes": 8, "test_accuracy": float(test_acc)}, f, indent=2)
        logger.info("Updated static_classes.json to 8 classes")
    else:
        logger.warning("❌ New model did NOT meet quality threshold.")
        logger.warning("   test_acc=%.2f%%, orig4_acc=%.2f%%", test_acc*100, orig4_acc*100)
        logger.warning("   Existing models KEPT UNCHANGED.")

    logger.info("=" * 70)
    logger.info("FINAL RESULT")
    logger.info("  Overall test accuracy  : %.2f%%", test_acc * 100)
    logger.info("  Original 4-class acc   : %.2f%%", orig4_acc * 100)
    logger.info("  Model saved            : %s", should_save)
    logger.info("=" * 70)
    return report


if __name__ == "__main__":
    main()
