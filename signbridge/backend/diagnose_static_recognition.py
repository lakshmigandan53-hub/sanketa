"""
SANKETA - Diagnostic Script for Static Sign Recognition
Tests:
1. Model loading & input shape
2. Training preprocessing vs Live preprocessing
3. Training sample vs live vector inspection
4. Label ordering (0: HELLO -> 10: HELP)
5. Accuracy across all 11 classes (all 4 members)
6. Scale and translation invariance (center, left, right, near, far)
"""

import json
import sys
from pathlib import Path
import numpy as np

# Setup paths
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent.parent
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from app.services.sign_recognition import SignRecognitionService, normalize_landmarks
from training.preprocess import (
    STATIC_SIGNS,
    SIGN_TO_INDEX,
    INDEX_TO_SIGN,
    normalize_landmarks as train_normalize_landmarks,
    validate_landmark_vector,
)

print("=" * 70)
print("SANKETA STATIC SIGN RECOGNITION DIAGNOSTICS")
print("=" * 70)

# 1. Model & Label Verification
svc = SignRecognitionService()
print(f"\n1. MODEL LOADING & INPUT SHAPE:")
print(f"   Static Model Loaded: {svc.static_model_loaded}")
print(f"   Model Path: {svc._static_model_path_used}")
if svc._static_model:
    print(f"   Model Input Shape: {svc._static_model.input_shape}")
    print(f"   Model Output Shape: {svc._static_model.output_shape}")

print(f"\n2. LABEL ORDERING CHECK:")
expected_order = [
    "HELLO", "YES", "NO", "WATER", "FOOD",
    "MILK", "TEA", "BOOK", "PEN", "PHONE", "HELP"
]
actual_static_classes = svc._static_classes
print(f"   Expected ({len(expected_order)}): {expected_order}")
print(f"   Actual   ({len(actual_static_classes)}): {actual_static_classes}")
labels_match = (expected_order == actual_static_classes)
print(f"   Labels Match Expected Order Exactly: {'[PASS]' if labels_match else '[FAIL]'}")

# Check files on disk
labels_txt = (_BACKEND_DIR / "app" / "models" / "labels.txt").read_text().splitlines()
print(f"   labels.txt ({len(labels_txt)} items): {labels_txt}")
labels_txt_match = (expected_order == labels_txt)
print(f"   labels.txt Match: {'[PASS]' if labels_txt_match else '[FAIL]'}")

# 3. Preprocessing Parity Check
print(f"\n3. PREPROCESSING PARITY CHECK (Training vs Live Inference):")
# Generate synthetic random landmark array (21 x 3)
np.random.seed(42)
test_raw = np.random.uniform(0.1, 0.9, size=(63,)).astype(np.float32)

train_norm = train_normalize_landmarks(test_raw)
live_norm = normalize_landmarks(test_raw)
diff = np.max(np.abs(train_norm - live_norm))
print(f"   Max absolute difference between train_normalize and live_normalize: {diff:.8e}")
print(f"   Preprocessing Parity: {'[PASS]' if diff < 1e-7 else '[FAIL]'}")

# 4. Sample Inspection from Dataset
print(f"\n4. DATASET SAMPLE INSPECTION:")
dataset_dir = _ROOT_DIR / "dataset"
sample_file = dataset_dir / "member_01" / "HELLO" / "sample_0000.npy"
if sample_file.exists():
    raw_sample = np.load(sample_file)
    valid, msg = validate_landmark_vector(raw_sample)
    print(f"   Sample: member_01/HELLO/sample_0000.npy")
    print(f"   Raw Shape: {raw_sample.shape}, Dtype: {raw_sample.dtype}")
    print(f"   Raw Range: min={raw_sample.min():.4f}, max={raw_sample.max():.4f}, mean={raw_sample.mean():.4f}")
    print(f"   First 6 raw values: {raw_sample[:6].tolist()}")
    print(f"   NaN / Inf check: has_nan={np.isnan(raw_sample).any()}, has_inf={np.isinf(raw_sample).any()}")
    
    norm_sample = normalize_landmarks(raw_sample)
    print(f"   Normalized Shape: {norm_sample.shape}")
    print(f"   Normalized Range: min={norm_sample.min():.4f}, max={norm_sample.max():.4f}, mean={norm_sample.mean():.4f}")
    print(f"   First 6 normalized values: {norm_sample[:6].tolist()}")
    
    # Run prediction through service
    res = svc.predict(raw_sample)
    print(f"   Prediction Output: {res}")
else:
    print(f"   [WARNING] Sample file {sample_file} not found!")

# 5. Full Evaluation Across All 11 Classes
print(f"\n5. RECOGNITION EVALUATION ACROSS ALL 11 CLASSES (40 samples per class across members):")
print(f"{'Class':<10} | {'Correct':<8} | {'Total':<6} | {'Accuracy':<10} | {'Avg Conf':<10} | {'Status'}")
print("-" * 65)

all_correct = 0
all_total = 0

for cls in expected_order:
    cls_correct = 0
    cls_total = 0
    cls_confs = []
    
    # Sample from each member
    for mem in ["member_01", "member_02", "member_03", "member_04"]:
        cls_dir = dataset_dir / mem / cls
        if not cls_dir.exists():
            continue
        files = sorted(list(cls_dir.glob("*.npy")))[:10]  # 10 per member = 40 total
        for f in files:
            raw = np.load(f)
            res = svc.predict(raw)
            pred_word = res.get("word")
            conf = res.get("confidence", 0.0)
            cls_confs.append(conf)
            if pred_word == cls:
                cls_correct += 1
            cls_total += 1
            
    acc = (cls_correct / cls_total * 100) if cls_total > 0 else 0
    avg_c = (sum(cls_confs) / len(cls_confs) * 100) if cls_confs else 0
    all_correct += cls_correct
    all_total += cls_total
    status = "[PASS]" if acc >= 90 else "[WARN]"
    print(f"{cls:<10} | {cls_correct:<8} | {cls_total:<6} | {acc:>8.1f}% | {avg_c:>8.1f}% | {status}")

total_acc = (all_correct / all_total * 100) if all_total > 0 else 0
print("-" * 65)
print(f"{'OVERALL':<10} | {all_correct:<8} | {all_total:<6} | {total_acc:>8.1f}% |")

# 6. Hand Position & Scale Invariance Tests (Center, Left, Right, Near, Far)
print(f"\n6. HAND POSITION & SCALE INVARIANCE TESTS (Class: HELLO):")
base_raw = np.load(dataset_dir / "member_01" / "HELLO" / "sample_0000.npy").reshape(21, 3)

positions = {
    "Original (Center)": base_raw.copy(),
    "Shift Left (-0.3 x)": base_raw + np.array([-0.3, 0.0, 0.0], dtype=np.float32),
    "Shift Right (+0.35 x)": base_raw + np.array([0.35, 0.0, 0.0], dtype=np.float32),
    "Shift Up (-0.25 y)": base_raw + np.array([0.0, -0.25, 0.0], dtype=np.float32),
    "Shift Down (+0.25 y)": base_raw + np.array([0.0, 0.25, 0.0], dtype=np.float32),
    "Far (Scaled 0.5x)": base_raw * 0.5,
    "Near (Scaled 2.0x)": base_raw * 2.0,
    "Diagonal + Far": (base_raw + np.array([0.2, -0.2, 0.0], dtype=np.float32)) * 0.6,
    "Diagonal + Near": (base_raw + np.array([-0.2, 0.2, 0.0], dtype=np.float32)) * 1.8,
}

for name, pos_coords in positions.items():
    flat_coords = pos_coords.flatten()
    res = svc.predict(flat_coords)
    p_word = res.get("word")
    p_conf = res.get("confidence", 0.0)
    match = (p_word == "HELLO")
    status = "[PASS]" if match and p_conf >= 0.90 else "[FAIL]"
    print(f"   {name:<22}: Predicted={p_word:<8} Confidence={p_conf * 100:.1f}%  {status}")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
