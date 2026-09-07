# -*- coding: utf-8 -*-
"""
SANKETA — Real Model Automated Inference Test
backend/test_models_real.py

Tests both isl_static_best and isl_dynamic_best through SignRecognitionService
using real recorded ISL frames from member_01 and member_01_dynamic.
"""

import sys, os
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.sign_recognition import SignRecognitionService

STATIC_DIR = Path(__file__).parent / "datasets" / "raw" / "member_01"
DYNAMIC_DIR = Path(__file__).parent / "datasets" / "raw" / "member_01_dynamic"


def test_models():
    print("=" * 70)
    print("SANKETA - AUTOMATED LOCAL MODEL EVALUATION & INFERENCE TEST")
    print("=" * 70)

    svc = SignRecognitionService(confidence_threshold=0.60)
    print(f"Static Model Loaded : {svc.static_model_loaded}")
    print(f"Dynamic Model Loaded: {svc.dynamic_model_loaded}")
    print(f"Static Info : {svc.static_info}")
    print(f"Dynamic Info: {svc.dynamic_info}")
    print("=" * 70)

    # 1. TEST STATIC SIGNS
    static_classes = ["HELLO", "YES", "NO", "WATER"]
    static_correct = 0
    static_total = 0

    print("\n--- [PART F1] TESTING STATIC SIGNS ---")
    for cls_name in static_classes:
        cls_dir = STATIC_DIR / cls_name
        files = sorted(cls_dir.glob("*.npy"))
        # Test 15 distinct samples across the dataset
        step = max(1, len(files) // 15)
        test_files = files[::step][:15]

        for f in test_files:
            arr = np.load(f).astype(np.float32)
            if arr.shape == (21, 3):
                arr = arr.flatten()
            if arr.shape != (63,) or np.isnan(arr).any():
                continue

            svc.reset_buffer()
            res = svc.predict(arr)
            pred = res.get("word")
            conf = res.get("confidence", 0.0)
            is_correct = (pred == cls_name)
            if is_correct:
                static_correct += 1
            static_total += 1

            status = "[CORRECT]" if is_correct else "[INCORRECT]"
            print(f"  Expected: {cls_name:<10} | Predicted: {pred:<10} | Conf: {conf*100:5.1f}% | {status}")

    static_acc = (static_correct / static_total * 100) if static_total > 0 else 0.0
    print(f"\n>> Static Accuracy: {static_correct}/{static_total} ({static_acc:.2f}%)")

    # 2. TEST DYNAMIC SIGNS
    dynamic_classes = ["THANK_YOU", "PLEASE", "SORRY", "HELP"]
    dynamic_correct = 0
    dynamic_total = 0

    print("\n--- [PART F2] TESTING DYNAMIC SIGNS ---")
    for cls_name in dynamic_classes:
        cls_dir = DYNAMIC_DIR / cls_name
        files = sorted(cls_dir.glob("*.npy"))
        step = max(1, len(files) // 10)
        test_files = files[::step][:10]

        for f in test_files:
            seq = np.load(f).astype(np.float32)
            if seq.shape != (30, 63) or np.isnan(seq).any():
                continue

            res = svc.predict_sequence(seq)
            pred = res.get("word")
            conf = res.get("confidence", 0.0)
            is_correct = (pred == cls_name)
            if is_correct:
                dynamic_correct += 1
            dynamic_total += 1

            status = "[CORRECT]" if is_correct else "[INCORRECT]"
            print(f"  Expected: {cls_name:<10} | Predicted: {pred:<10} | Conf: {conf*100:5.1f}% | {status}")

    dynamic_acc = (dynamic_correct / dynamic_total * 100) if dynamic_total > 0 else 0.0
    print(f"\n>> Dynamic Accuracy: {dynamic_correct}/{dynamic_total} ({dynamic_acc:.2f}%)")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY:")
    print(f"  Static Real-Time Accuracy : {static_acc:.2f}% ({static_correct}/{static_total})")
    print(f"  Dynamic Real-Time Accuracy: {dynamic_acc:.2f}% ({dynamic_correct}/{dynamic_total})")
    print("=" * 70)

    assert static_acc >= 90.0, f"Static accuracy too low: {static_acc}%"
    assert dynamic_acc >= 90.0, f"Dynamic accuracy too low: {dynamic_acc}%"
    print("ALL ACCURACY CHECKS PASSED!")


if __name__ == "__main__":
    test_models()
