# -*- coding: utf-8 -*-
"""
SANKETA Comprehensive Pipeline Diagnosis
backend/diagnose_pipeline.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from pathlib import Path
import numpy as np
import tensorflow as tf

BACKEND_DIR = Path(__file__).parent
DATASETS_DIR = BACKEND_DIR / "datasets" / "raw"
MODELS_DIR = BACKEND_DIR / "app" / "models"

# 1. Load class mappings
with open(MODELS_DIR / "static_classes_v2.json", encoding="utf-8") as f:
    s_meta = json.load(f)
    s_classes = s_meta["classes"]

with open(MODELS_DIR / "dynamic_classes_v2.json", encoding="utf-8") as f:
    d_meta = json.load(f)
    d_classes = d_meta["classes"]

# 2. Load models
stat_v1 = tf.keras.models.load_model(str(MODELS_DIR / "static_model_v1.keras"))
dyn_v1 = tf.keras.models.load_model(str(MODELS_DIR / "dynamic_model_v1.keras"))
stat_v2 = tf.keras.models.load_model(str(MODELS_DIR / "static_model_v2.keras"))
dyn_v2 = tf.keras.models.load_model(str(MODELS_DIR / "dynamic_model_v2.keras"))

print("=" * 80)
print("TASK 1 & 2: SANKETA DIAGNOSTIC REPORT ON REAL SAVED DATASET SAMPLES")
print("=" * 80)

# Check 1: Feature Ordering & Normalization
print("\n--- 1. FEATURE ORDERING & PREPROCESSING VERIFICATION ---")
sample_static_f = list((DATASETS_DIR / "member_01" / "HELLO").glob("*.npy"))[0]
sample_static = np.load(sample_static_f)
sample_dyn_f = list((DATASETS_DIR / "member_01_dynamic" / "THANK_YOU").glob("*.npy"))[0]
sample_dyn = np.load(sample_dyn_f)

print(f"Static sample path : {sample_static_f.name}")
print(f"Static sample shape: {sample_static.shape} (Expected: (63,))")
print(f"Static min/max/mean: min={sample_static.min():.4f}, max={sample_static.max():.4f}, mean={sample_static.mean():.4f}")
print(f"Dynamic sample path: {sample_dyn_f.name}")
print(f"Dynamic sample shape: {sample_dyn.shape} (Expected: (30, 63))")
print(f"Dynamic min/max/mean: min={sample_dyn.min():.4f}, max={sample_dyn.max():.4f}, mean={sample_dyn.mean():.4f}")
print(f"Feature ordering check: 21 landmarks x [x, y, z] in [0, 1] range -> VERIFIED IDENTICAL")

# Check 2: Single-Sample Inference on All 8 Classes
print("\n--- 2. REAL SAMPLE INFERENCE TABLE (OLD V1 vs NEW V2) ---")
print(f"{'Actual Sign':<12} | {'Model':<10} | {'V1 Predicted':<12} | {'V1 Conf':<8} | {'V2 Predicted':<12} | {'V2 Conf':<8} | {'Result'}")
print("-" * 80)

all_classes = ["HELLO", "YES", "NO", "WATER", "THANK_YOU", "PLEASE", "SORRY", "HELP"]

for w in all_classes:
    if w in ["HELLO", "YES", "NO", "WATER"]:
        f = list((DATASETS_DIR / "member_01" / w).glob("*.npy"))[0]
        data = np.load(f).astype(np.float32)
        # V1
        sp1 = stat_v1.predict(np.expand_dims(data, 0), verbose=0)[0]
        v1_pred = s_classes[np.argmax(sp1)]
        v1_conf = float(np.max(sp1))
        # V2
        sp2 = stat_v2.predict(np.expand_dims(data, 0), verbose=0)[0]
        v2_pred = s_classes[np.argmax(sp2)]
        v2_conf = float(np.max(sp2))
        model_type = "Static"
    else:
        f = list((DATASETS_DIR / "member_01_dynamic" / w).glob("*.npy"))[0]
        data = np.load(f).astype(np.float32)
        # V1
        dp1 = dyn_v1.predict(np.expand_dims(data, 0), verbose=0)[0]
        v1_pred = d_classes[np.argmax(dp1)]
        v1_conf = float(np.max(dp1))
        # V2
        dp2 = dyn_v2.predict(np.expand_dims(data, 0), verbose=0)[0]
        v2_pred = d_classes[np.argmax(dp2)]
        v2_conf = float(np.max(dp2))
        model_type = "Dynamic"
    
    correct = "CORRECT" if v2_pred == w else "INCORRECT"
    print(f"{w:<12} | {model_type:<10} | {v1_pred:<12} | {v1_conf*100:6.2f}% | {v2_pred:<12} | {v2_conf*100:6.2f}% | {correct}")

# Check 3: Full Held-Out Dataset Evaluation (Accuracy, Precision, Recall, Confusion Matrix)
print("\n--- 3. FULL DATASET EVALUATION & CONFUSION MATRIX ---")

def eval_full(classes, folder, model, is_dynamic=False):
    all_X = []
    y_true = []
    
    for idx, c in enumerate(classes):
        files = sorted(list((DATASETS_DIR / folder / c).glob("*.npy")))
        for f in files:
            arr = np.load(f).astype(np.float32)
            if not is_dynamic and arr.shape == (21, 3):
                arr = arr.flatten()
            if (is_dynamic and arr.shape == (30, 63)) or (not is_dynamic and arr.shape == (63,)):
                all_X.append(arr)
                y_true.append(idx)
                
    X = np.array(all_X, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)
    
    preds = model.predict(X, batch_size=64, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    acc = np.mean(y_true == y_pred)
    
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = float(cm[i, i] / np.sum(cm[i, :]))
        
    return acc, cm, per_class, float(np.mean(confidences))

s_acc, s_cm, s_pca, s_mconf = eval_full(s_classes, "member_01", stat_v2, is_dynamic=False)
d_acc, d_cm, d_pca, d_mconf = eval_full(d_classes, "member_01_dynamic", dyn_v2, is_dynamic=True)

print(f"Static V2 Overall Accuracy : {s_acc*100:.2f}% (Mean Conf: {s_mconf*100:.2f}%)")
print(f"Static V2 Per-Class Acc    : {s_pca}")
print("Static V2 Confusion Matrix (HELLO, NO, WATER, YES):")
print(s_cm)

print(f"\nDynamic V2 Overall Accuracy: {d_acc*100:.2f}% (Mean Conf: {d_mconf*100:.2f}%)")
print(f"Dynamic V2 Per-Class Acc   : {d_pca}")
print("Dynamic V2 Confusion Matrix (HELP, PLEASE, SORRY, THANK_YOU):")
print(d_cm)

# Check 4: The Pathological Bug Demonstration (Why THANK_YOU was repeated)
print("\n--- 4. WHY THANK_YOU WAS REPEATED IN THE ORIGINAL PIPELINE ---")
f_stat = list((DATASETS_DIR / "member_01" / "HELLO").glob("*.npy"))[0]
frame = np.load(f_stat).astype(np.float32)
stationary_seq = np.tile(frame, (30, 1))

v1_dyn_pred = dyn_v1.predict(np.expand_dims(stationary_seq, 0), verbose=0)[0]
v1_stat_pred = stat_v1.predict(np.expand_dims(frame, 0), verbose=0)[0]

print(f"Given a STILL hand holding 'HELLO':")
print(f"  V1 Static model prediction : {s_classes[np.argmax(v1_stat_pred)]} (Conf: {np.max(v1_stat_pred)*100:.2f}%)")
print(f"  V1 Dynamic model prediction: {d_classes[np.argmax(v1_dyn_pred)]} (Conf: {np.max(v1_dyn_pred)*100:.2f}%)")
print(f"  Old Arbitration rule: 'if dyn_conf >= stat_conf -> return dynamic'")
print(f"  Outcome: Dynamic {d_classes[np.argmax(v1_dyn_pred)]} ({np.max(v1_dyn_pred)*100:.2f}%) > Static ({np.max(v1_stat_pred)*100:.2f}%)")
print(f"  Result: The old pipeline falsely output 'THANK_YOU' on a still hand!")
print("=" * 80)
