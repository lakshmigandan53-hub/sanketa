# -*- coding: utf-8 -*-
"""
SANKETA — 20-Class Static Model Training
backend/train_static_20class.py

Trains static model on all 20 ISL signs from collected datasets:
  HELLO, YES, NO, WATER, HELP,       (5 carry-over from 8-class)
  FOOD, MILK, TEA, BOOK, PEN,        (5 new)
  PHONE, COMPUTER, HOME, SCHOOL,     (4 new)
  COLLEGE, DOCTOR, HOSPITAL,         (3 new)
  MONEY, HOUSE, CAR                  (3 new)

Uses exact same architecture as 8-class model (Dense/BN/Dropout)
expanded to 20 output classes.

DATA SOURCES:
  - member_01: 5 existing classes (HELLO, YES, NO, WATER, HELP)
  - member_01: 15 new classes (collected after running collect_data.py)
  - member_02..N: additional members' data (if available)

USAGE:
  python train_static_20class.py
  python train_static_20class.py --dataset datasets/raw/member_01
  python train_static_20class.py --deploy   # auto-deploy if threshold met
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import shutil
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
logger = logging.getLogger("STATIC_TRAINER_20")

BACKEND_DIR  = Path(__file__).parent
DATASET_ROOT = BACKEND_DIR / "datasets" / "raw"
MODELS_DIR   = BACKEND_DIR / "app" / "models"
REPORTS_DIR  = BACKEND_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 20 target classes (alphabetical for reproducibility)
ALL_20_CLASSES = [
    "BOOK", "CAR", "COLLEGE", "COMPUTER", "DOCTOR",
    "FOOD", "HELLO", "HELP", "HOME", "HOSPITAL",
    "HOUSE", "MILK", "MONEY", "NO", "PEN",
    "PHONE", "SCHOOL", "TEA", "WATER", "YES",
]

# Classes that had data from the original 8-class system
EXISTING_CLASSES = {"HELLO", "YES", "NO", "WATER", "HELP"}

# Thresholds for auto-deploy
MIN_TEST_ACC  = 0.88   # Lower threshold since 20 classes is harder
MIN_CLASS_ACC = 0.75   # Minimum per-class accuracy before we flag

OUTPUT_MODEL_PATH   = MODELS_DIR / "isl_static_20class.keras"
OUTPUT_CLASSES_PATH = MODELS_DIR / "static_classes_20.json"
DEPLOY_MODEL_PATH   = MODELS_DIR / "isl_model.keras"
BACKUP_MODEL_PATH   = MODELS_DIR / "isl_model_backup.keras"


# ---------------------------------------------------------------------------
# Normalization — must match collect_data.py and sign_recognition.py exactly
# ---------------------------------------------------------------------------
def normalize_landmarks(raw_63: np.ndarray) -> np.ndarray:
    """
    Wrist-centered, scale-normalized landmark vector.
    Input:  (63,) raw MediaPipe x,y,z landmarks
    Output: (63,) normalized float32
    """
    pts = raw_63.reshape(21, 3).copy()
    wrist = pts[0].copy()
    pts -= wrist
    max_dist = float(np.max(np.linalg.norm(pts, axis=1)))
    if max_dist < 1e-6:
        max_dist = 1.0
    return (pts / max_dist).flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation (same as 8-class trainer — proven to work)
# ---------------------------------------------------------------------------
def augment_static(lm_flat: np.ndarray,
                   angle_deg: float = 6.0,
                   scale_range: float = 0.06,
                   trans_range: float = 0.02) -> np.ndarray:
    """Apply random rotation + scale + translate + jitter to a normalized sample."""
    lm = lm_flat.reshape(21, 3).copy()
    # Rotation in XY plane
    center = np.mean(lm[:, :2], axis=0)
    theta = np.radians(np.random.uniform(-angle_deg, angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    lm[:, :2] = (lm[:, :2] - center) @ R.T + center
    # Scale
    scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    lm[:, :2] = (lm[:, :2] - center) * scale + center
    # Translate
    lm[:, 0] += np.random.uniform(-trans_range, trans_range)
    lm[:, 1] += np.random.uniform(-trans_range, trans_range)
    # Gaussian jitter
    lm += np.random.normal(0, 0.001, lm.shape)
    return np.clip(lm.flatten(), -0.2, 1.2).astype(np.float32)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def split_stratified(X: np.ndarray, y: np.ndarray,
                     train: float = 0.80, val: float = 0.10,
                     seed: int = 42):
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
def load_data(members: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Scan DATASET_ROOT for member directories.
    Load only classes present in ALL_20_CLASSES.
    Returns X (N, 63), y (N,), and per-class/member stats.
    """
    if members is None:
        # Auto-discover members
        members = sorted([d.name for d in DATASET_ROOT.iterdir() if d.is_dir()])

    logger.info("Loading 20-class static dataset from %s", DATASET_ROOT)
    logger.info("Member directories found: %s", members)

    X, y = [], []
    stats = {cls: {"members": {}, "total": 0} for cls in ALL_20_CLASSES}
    class_to_idx = {cls: i for i, cls in enumerate(ALL_20_CLASSES)}

    # Only process member_XX directories (not member_01_dynamic etc.)
    static_members = [m for m in members if not m.endswith("_dynamic")]

    for member_id in static_members:
        member_dir = DATASET_ROOT / member_id
        if not member_dir.is_dir():
            continue

        for word in ALL_20_CLASSES:
            word_dir = member_dir / word
            if not word_dir.is_dir():
                continue

            files = sorted(word_dir.glob("*.npy"))
            loaded = 0
            for f in files:
                try:
                    arr = np.load(f).astype(np.float32)
                    if arr.shape == (21, 3):
                        arr = arr.flatten()
                    if arr.shape != (63,):
                        continue
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        continue
                    # Apply normalization at load time
                    arr = normalize_landmarks(arr)
                    X.append(arr)
                    y.append(class_to_idx[word])
                    loaded += 1
                except Exception as e:
                    logger.warning("Failed to load %s: %s", f, e)

            stats[word]["members"][member_id] = loaded
            stats[word]["total"] += loaded

    if len(X) == 0:
        raise RuntimeError(
            "No data found! Run collect_data.py first for the 20 target classes.\n"
            f"Expected data in: {DATASET_ROOT}"
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Report
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    missing_classes = []
    for cls in ALL_20_CLASSES:
        total = stats[cls]["total"]
        status = "[OK]" if total >= 50 else ("[WARN: LOW]" if total > 0 else "[MISSING]")
        if total == 0:
            missing_classes.append(cls)
        logger.info("  %s %-12s: %d samples", status, cls, total)

    logger.info("-" * 60)
    logger.info("Total samples  : %d", len(X))
    logger.info("Total classes  : %d", len(np.unique(y)))

    if missing_classes:
        logger.warning("MISSING DATA for classes: %s", missing_classes)
        logger.warning("Run: python collect_data.py --member member_01")
        logger.warning("These classes will be SKIPPED in training.")

    return X, y, stats


# ---------------------------------------------------------------------------
# Build model (Dense/BN/Dropout — expanded to 20 outputs)
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 20) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,), name="landmarks"),

        tf.keras.layers.Dense(512, name="dense_1"),
        tf.keras.layers.BatchNormalization(name="bn_1"),
        tf.keras.layers.ReLU(name="relu_1"),
        tf.keras.layers.Dropout(0.30, name="drop_1"),

        tf.keras.layers.Dense(256, name="dense_2"),
        tf.keras.layers.BatchNormalization(name="bn_2"),
        tf.keras.layers.ReLU(name="relu_2"),
        tf.keras.layers.Dropout(0.25, name="drop_2"),

        tf.keras.layers.Dense(128, name="dense_3"),
        tf.keras.layers.BatchNormalization(name="bn_3"),
        tf.keras.layers.ReLU(name="relu_3"),
        tf.keras.layers.Dropout(0.20, name="drop_3"),

        tf.keras.layers.Dense(64, name="dense_4"),
        tf.keras.layers.BatchNormalization(name="bn_4"),
        tf.keras.layers.ReLU(name="relu_4"),
        tf.keras.layers.Dropout(0.15, name="drop_4"),

        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="isl_static_20class")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Water bias check — verify no single class dominates predictions
# ---------------------------------------------------------------------------
def check_prediction_bias(model, X_test, y_test, class_names):
    """Check that predictions are distributed across all classes."""
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    counts = {cls: int(np.sum(preds == i)) for i, cls in enumerate(class_names)}
    total = len(preds)

    logger.info("=" * 60)
    logger.info("PREDICTION DISTRIBUTION CHECK (bias test)")
    logger.info("=" * 60)
    bias_detected = False
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / total if total > 0 else 0
        flag = " <<< BIAS!" if pct > 30.0 else ""
        if pct > 30.0:
            bias_detected = True
        logger.info("  %-12s : %3d  (%.1f%%)%s", cls, cnt, pct, flag)

    if bias_detected:
        logger.warning("PREDICTION BIAS DETECTED! One class dominates predictions.")
        logger.warning("Check normalization and class balance before deploying.")
    else:
        logger.info("No bias detected. Predictions distributed across all classes.")

    return bias_detected, counts


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train 20-class ISL static model")
    parser.add_argument("--deploy", action="store_true",
                        help="Auto-deploy model if thresholds met")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max training epochs (default: 150)")
    parser.add_argument("--augment-factor", type=int, default=3,
                        help="Augmentation multiplier (default: 3x)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SANKETA -- 20-Class Static Model Training")
    logger.info("=" * 70)
    t0 = time.time()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load data
    X, y, stats = load_data()

    # Filter to classes that actually have data
    present_labels = sorted(np.unique(y))
    present_classes = [ALL_20_CLASSES[i] for i in present_labels]
    missing_classes = [cls for cls in ALL_20_CLASSES if cls not in present_classes]

    if missing_classes:
        logger.warning("Training on %d/%d classes (missing: %s)",
                       len(present_classes), len(ALL_20_CLASSES), missing_classes)
        # Re-map labels to be contiguous
        old_to_new = {old: new for new, old in enumerate(present_labels)}
        y = np.array([old_to_new[yi] for yi in y], dtype=np.int32)
        class_names = present_classes
        num_classes = len(class_names)
    else:
        class_names = ALL_20_CLASSES
        num_classes = 20
        logger.info("All 20 classes present. Full model training.")

    # 2. Stratified split
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_stratified(X, y)
    logger.info("Split: Train=%d, Val=%d, Test=%d", len(X_tr), len(X_val), len(X_te))

    # 3. Augment training set
    logger.info("Augmenting training set %dx...", args.augment_factor)
    X_aug, y_aug = list(X_tr), list(y_tr)
    for xi, yi in zip(X_tr, y_tr):
        for _ in range(args.augment_factor - 1):
            X_aug.append(augment_static(xi))
            y_aug.append(yi)
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)
    logger.info("Augmented training set: %d samples (%dx from %d)",
                len(X_aug), args.augment_factor, len(X_tr))

    # 4. Compute class weights for imbalanced classes
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_aug), y=y_aug)
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    logger.info("Class weights: %s",
                {class_names[i]: round(w, 3) for i, w in class_weights.items()})

    # 5. Build model
    model = build_model(num_classes=num_classes)
    model.summary(print_fn=logger.info)

    # 6. Callbacks
    ckpt_path = str(MODELS_DIR / "isl_static_20class_ckpt.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
    ]

    # 7. Train
    logger.info("Training... (max %d epochs)", args.epochs)
    history = model.fit(
        X_aug, y_aug,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Test evaluation
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    logger.info("=" * 60)
    logger.info("TEST ACCURACY  : %.4f  (%.2f%%)", test_acc, test_acc * 100)
    logger.info("TEST LOSS      : %.4f", test_loss)
    logger.info("=" * 60)

    # Per-class report
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    report_str = classification_report(y_te, y_pred, target_names=class_names,
                                       zero_division=0)
    logger.info("\n%s", report_str)

    cm = confusion_matrix(y_te, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    per_class_acc = {}
    worst_class, worst_acc = "", 1.0
    best_class, best_acc = "", 0.0
    for i, cls in enumerate(class_names):
        mask = y_te == i
        if mask.sum() > 0:
            acc = float(np.mean(y_pred[mask] == i))
            per_class_acc[cls] = round(acc, 4)
            if acc < worst_acc:
                worst_acc = acc
                worst_class = cls
            if acc > best_acc:
                best_acc = acc
                best_class = cls

    logger.info("Best class  : %s (%.2f%%)", best_class, best_acc * 100)
    logger.info("Worst class : %s (%.2f%%)", worst_class, worst_acc * 100)

    # 9. Bias check
    bias_detected, pred_dist = check_prediction_bias(model, X_te, y_te, class_names)

    elapsed = time.time() - t0
    logger.info("Training time: %.1f seconds", elapsed)

    # 10. Save new model (always save to staging path)
    model.save(str(OUTPUT_MODEL_PATH))
    logger.info("Staged model saved to: %s", OUTPUT_MODEL_PATH)

    classes_meta = {
        "version": "v1_20class",
        "classes": class_names,
        "num_classes": num_classes,
        "missing_classes": missing_classes,
        "test_accuracy": float(test_acc),
        "per_class_accuracy": per_class_acc,
        "best_class": best_class,
        "worst_class": worst_class,
        "bias_detected": bias_detected,
    }
    with open(OUTPUT_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes_meta, f, indent=2)
    logger.info("Class config saved to: %s", OUTPUT_CLASSES_PATH)

    # Save labels.txt (one class per line, in order)
    labels_txt = MODELS_DIR / "labels.txt"
    with open(labels_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    logger.info("Labels saved to: %s", labels_txt)

    # Save labels.json
    labels_json = MODELS_DIR / "labels.json"
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "num_classes": num_classes}, f, indent=2)

    # 11. Report
    report = {
        "model_name": "isl_static_20class",
        "input_shape": [63],
        "output_classes": class_names,
        "num_classes": num_classes,
        "missing_classes": missing_classes,
        "training_samples": len(X_aug),
        "validation_samples": len(X_val),
        "test_samples": len(X_te),
        "test_accuracy": float(test_acc),
        "per_class_accuracy": per_class_acc,
        "best_class": best_class,
        "worst_class": worst_class,
        "bias_detected": bias_detected,
        "prediction_distribution": pred_dist,
        "confusion_matrix": cm.tolist(),
        "training_time_seconds": round(elapsed, 1),
    }
    report_path = REPORTS_DIR / "static_20class_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Full report saved to: %s", report_path)

    # 12. Decision: deploy?
    should_deploy = (
        test_acc >= MIN_TEST_ACC and
        not bias_detected and
        num_classes == 20  # Only deploy if all 20 classes are trained
    )

    logger.info("=" * 70)
    logger.info("DEPLOYMENT DECISION")
    logger.info("  Test accuracy : %.2f%% (need >= %.0f%%)", test_acc*100, MIN_TEST_ACC*100)
    logger.info("  Bias detected : %s", bias_detected)
    logger.info("  All 20 classes: %s", num_classes == 20)
    logger.info("  Ready to deploy: %s", should_deploy)
    logger.info("=" * 70)

    if args.deploy and should_deploy:
        # Backup current model
        if DEPLOY_MODEL_PATH.exists():
            shutil.copy2(str(DEPLOY_MODEL_PATH), str(BACKUP_MODEL_PATH))
            logger.info("Backed up existing model to: %s", BACKUP_MODEL_PATH)

        # Deploy new model
        shutil.copy2(str(OUTPUT_MODEL_PATH), str(DEPLOY_MODEL_PATH))
        logger.info("[DEPLOYED] New 20-class model deployed to: %s", DEPLOY_MODEL_PATH)

        # Update static_classes.json for live inference
        with open(MODELS_DIR / "static_classes.json", "w", encoding="utf-8") as f:
            json.dump({"classes": class_names, "num_classes": num_classes,
                       "test_accuracy": float(test_acc)}, f, indent=2)
        logger.info("Updated static_classes.json for live inference")

    elif args.deploy and not should_deploy:
        logger.warning("Model did NOT meet deployment thresholds.")
        logger.warning("  Existing model KEPT UNCHANGED.")
        logger.warning("  Staged model available at: %s", OUTPUT_MODEL_PATH)
        logger.warning("  Collect more data and retrain, or lower thresholds if needed.")
    else:
        logger.info("Run with --deploy flag to auto-deploy after verifying results.")
        logger.info("Or manually copy %s -> %s", OUTPUT_MODEL_PATH, DEPLOY_MODEL_PATH)

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL REPORT")
    logger.info("")
    logger.info("DATASET")
    logger.info("  Total samples   : %d", len(X))
    for cls in class_names:
        logger.info("  %-12s  : %d", cls, stats.get(cls, {}).get("total", 0))
    logger.info("")
    logger.info("MODEL")
    logger.info("  Input shape     : (63,)")
    logger.info("  Output classes  : %d", num_classes)
    logger.info("  Model file      : %s", OUTPUT_MODEL_PATH)
    logger.info("")
    logger.info("RESULTS")
    logger.info("  Test accuracy   : %.2f%%", test_acc * 100)
    logger.info("  Best class      : %s (%.2f%%)", best_class, best_acc * 100)
    logger.info("  Worst class     : %s (%.2f%%)", worst_class, worst_acc * 100)
    logger.info("  Water bias      : %s", "YES" if bias_detected else "NO")
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    main()
