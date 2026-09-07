"""
SANKETA Recognition Pipeline Diagnostics
Checks: model files, loading, shapes, landmark pipeline, inference, confidence threshold
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "app" / "models"

print("=" * 65)
print("SANKETA RECOGNITION PIPELINE DIAGNOSTICS")
print("=" * 65)

# ── 1. MODEL FILES ──────────────────────────────────────────────────
print("\n[1] MODEL FILES")
model_files = {
    "isl_static_best.keras":  MODELS_DIR / "isl_static_best.keras",
    "isl_static_best.h5":     MODELS_DIR / "isl_static_best.h5",
    "static_model_v2.keras":  MODELS_DIR / "static_model_v2.keras",
    "static_model.keras":     MODELS_DIR / "static_model.keras",
    "isl_model.keras":        MODELS_DIR / "isl_model.keras",
    "isl_dynamic_best.keras": MODELS_DIR / "isl_dynamic_best.keras",
    "dynamic_model_v2.keras": MODELS_DIR / "dynamic_model_v2.keras",
}
for name, p in model_files.items():
    size = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "MISSING"
    print(f"  {'[OK]' if p.exists() else '[X]'} {name}: {size}")

# ── 2. CLASSES FILES ────────────────────────────────────────────────
print("\n[2] CLASS LABEL FILES")
class_files = {
    "static_classes_best.json":   MODELS_DIR / "static_classes_best.json",
    "static_classes_v2.json":     MODELS_DIR / "static_classes_v2.json",
    "static_classes.json":        MODELS_DIR / "static_classes.json",
    "dynamic_classes_best.json":  MODELS_DIR / "dynamic_classes_best.json",
}
for name, p in class_files.items():
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        classes = d.get("classes", [])
        print(f"  [OK] {name}: {classes}")
    else:
        print(f"  [X] {name}: MISSING")

# ── 3. LOAD STATIC MODEL ────────────────────────────────────────────
print("\n[3] LOAD STATIC MODEL")
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Mirror the service's priority logic
    static_path = None
    for candidate in [
        MODELS_DIR / "isl_static_best.keras",
        MODELS_DIR / "static_model_v2.keras",
        MODELS_DIR / "static_model.keras",
        MODELS_DIR / "isl_model.keras",
    ]:
        if candidate.exists():
            static_path = candidate
            break

    if static_path is None:
        print("  ✗ NO STATIC MODEL FOUND")
        static_model = None
    else:
        print(f"  Loading: {static_path.name} ...")
        static_model = tf.keras.models.load_model(str(static_path), compile=False)
        print(f"  ✓ Loaded OK")
        print(f"  Input shape:  {static_model.input_shape}")
        print(f"  Output shape: {static_model.output_shape}")
        n_classes = static_model.output_shape[-1]
        print(f"  Output classes: {n_classes}")

        # Read the class list that service will use
        sc_path = MODELS_DIR / "static_classes_best.json"
        if sc_path.exists():
            with open(sc_path) as f:
                sc = json.load(f)
            sc_labels = sc.get("classes", [])
        else:
            sc_labels = ["HELLO", "NO", "WATER", "YES"]
        print(f"  Label list ({len(sc_labels)}): {sc_labels}")
        if n_classes != len(sc_labels):
            print(f"  ⚠ MISMATCH: model outputs {n_classes} but labels has {len(sc_labels)}")
        else:
            print(f"  ✓ Output dimension matches label count")

except Exception as e:
    print(f"  ✗ FAILED TO LOAD: {e}")
    static_model = None

# ── 4. LOAD DYNAMIC MODEL ───────────────────────────────────────────
print("\n[4] LOAD DYNAMIC MODEL")
try:
    dyn_path = None
    for candidate in [
        MODELS_DIR / "isl_dynamic_best.keras",
        MODELS_DIR / "dynamic_model_v2.keras",
        MODELS_DIR / "dynamic_model.keras",
    ]:
        if candidate.exists():
            dyn_path = candidate
            break

    if dyn_path is None:
        print("  ✗ NO DYNAMIC MODEL FOUND")
        dynamic_model = None
    else:
        print(f"  Loading: {dyn_path.name} ...")
        dynamic_model = tf.keras.models.load_model(str(dyn_path), compile=False)
        print(f"  ✓ Loaded OK")
        print(f"  Input shape:  {dynamic_model.input_shape}")
        print(f"  Output shape: {dynamic_model.output_shape}")

except Exception as e:
    print(f"  ✗ FAILED TO LOAD: {e}")
    dynamic_model = None

# ── 5. STATIC INFERENCE TEST ─────────────────────────────────────────
print("\n[5] STATIC INFERENCE TEST")
if static_model is not None:
    # Simulate a hand landmark vector (all zeros = neutral pose)
    dummy_zeros = np.zeros((1, 63), dtype=np.float32)
    dummy_rand  = np.random.rand(1, 63).astype(np.float32)

    try:
        preds_z = static_model.predict(dummy_zeros, verbose=0)[0]
        preds_r = static_model.predict(dummy_rand,  verbose=0)[0]
        idx_z = int(np.argmax(preds_z))
        idx_r = int(np.argmax(preds_r))
        conf_z = float(preds_z[idx_z])
        conf_r = float(preds_r[idx_r])
        print(f"  Zeros input  → class {idx_z} ({sc_labels[idx_z] if idx_z < len(sc_labels) else '?'}), conf={conf_z:.4f}")
        print(f"  Random input → class {idx_r} ({sc_labels[idx_r] if idx_r < len(sc_labels) else '?'}), conf={conf_r:.4f}")
        print(f"  Full softmax (zeros): {np.round(preds_z, 4).tolist()}")
        print(f"  Full softmax (rand):  {np.round(preds_r, 4).tolist()}")
        print(f"\n  Confidence threshold in service: 0.65")
        if conf_z >= 0.65:
            print(f"  ✓ zeros input WOULD pass threshold → {sc_labels[idx_z]}")
        else:
            print(f"  ⚠ zeros input conf={conf_z:.2%} < 0.65 → returned as 'uncertain'")
        if conf_r >= 0.65:
            print(f"  ✓ random input WOULD pass threshold → {sc_labels[idx_r]}")
        else:
            print(f"  ⚠ random input conf={conf_r:.2%} < 0.65 → returned as 'uncertain'")
    except Exception as e:
        print(f"  ✗ INFERENCE FAILED: {e}")
else:
    print("  SKIPPED — model not loaded")

# ── 6. MEDIAPIPE LANDMARK TEST ───────────────────────────────────────
print("\n[6] MEDIAPIPE LANDMARK EXTRACTION TEST")
try:
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    # Create a simple synthetic test frame (gray, not a real hand)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        arr = np.array([[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32).flatten()
        print(f"  ✓ Landmarks extracted on synthetic frame: shape={arr.shape}")
    else:
        print(f"  ✓ MediaPipe initialized OK (no hand in synthetic blank frame — expected)")
    hands.close()

except Exception as e:
    print(f"  ✗ MEDIAPIPE FAILED: {e}")

# ── 7. CONFIDENCE THRESHOLD ANALYSIS ─────────────────────────────────
print("\n[7] CONFIDENCE THRESHOLD ANALYSIS")
print("  Current threshold in service: 0.65 (env ISL_CONFIDENCE_THRESHOLD)")
print("  The static model has 4 classes: HELLO, NO, WATER, YES")
print("  For a random real hand pose, if model is well-trained,")
print("  at least one class should get >0.65 softmax probability.")
print()
if static_model is not None:
    # Test with 20 random inputs and check how many pass threshold
    passed = 0
    for _ in range(20):
        r = np.random.rand(1, 63).astype(np.float32)
        p = static_model.predict(r, verbose=0)[0]
        if float(np.max(p)) >= 0.65:
            passed += 1
    print(f"  Random vectors passing threshold: {passed}/20")
    if passed < 5:
        print("  ⚠ Model likely requires REAL hand landmark data (not random noise)")
        print("  ⚠ With 4 balanced classes and good training, real hands should score >0.65")
    else:
        print("  ✓ Model is producing high-confidence outputs on random inputs too")

# ── 8. FRONTEND FLOW SUMMARY ──────────────────────────────────────────
print("\n[8] FRONTEND RECOGNITION FLOW")
print("  Camera.jsx calls apiTranslateLandmarks(lms) every ~160ms")
print("  lms = array of 63 floats from client-side MediaPipe")
print("  POST /translate/landmarks → sign_recognition.predict(arr)")
print("  Returns: { success, word, confidence, type, message }")
print()
print("  DISPLAY LOGIC in Camera.jsx:")
print("  - data.success && data.word && data.word !== 'uncertain' → setCurrentPrediction")
print("  - stabilityState must reach 'STABLE' (5/8 frames consistent)")
print("  - lastAcceptedWord cooldown: 3 seconds")
print("  - ONLY words with conf >= 0.60 shown in final signed word display")

# ── 9. SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Static model found:  {'YES — ' + static_path.name if static_path and static_path.exists() else 'NO'}")
print(f"  Static model loaded: {'YES' if static_model else 'NO'}")
if static_model:
    print(f"  Static input shape:  {static_model.input_shape}")
    print(f"  Static output shape: {static_model.output_shape}")
    print(f"  Static classes:      {sc_labels}")
print(f"  Dynamic model found: {'YES — ' + dyn_path.name if dyn_path and dyn_path.exists() else 'NO'}")
print(f"  Dynamic model loaded:{'YES' if dynamic_model else 'NO'}")
print("=" * 65)
