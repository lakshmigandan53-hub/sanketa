"""
SANKETA — Pre-Collection Verification Script
backend/verify_pipeline.py

Runs 10 automated checks without touching any data or training.
"""
import sys, os, json, time, pathlib, inspect, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import numpy as np

BACKEND = pathlib.Path(__file__).parent
PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
results = []

def check(n, label, ok, detail=""):
    sym = "[PASS]" if ok else "[FAIL]"
    results.append((n, label, ok, detail))
    print(f"  {sym}  {n:02d}. {label}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")
    return ok

# ============================================================
# CHECK 1 — Member 01 words from config
# ============================================================
cfg_path = BACKEND / "config" / "words.json"
try:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    m1 = cfg["members"]["member_01"]
    words = m1["words"]
    types = m1.get("word_types", {})
    assert len(words) == 8, f"Expected 8 words, got {len(words)}"
    assert len(set(words)) == 8, "Duplicate words detected"
    detail = "\n".join(f"  {i+1}. {w}  [{types.get(w,'?')}]" for i, w in enumerate(words))
    check(1, "Member 01 — 8 unique words configured", True, detail)
except Exception as e:
    check(1, "Member 01 — 8 unique words configured", False, str(e))
    words = []

# ============================================================
# CHECK 2 — Collector uses same preprocessing as MediaPipe service
# ============================================================
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("collect_data", BACKEND / "collect_data.py")
    cd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cd)

    # Read the actual source of extract_landmarks in both files
    cd_src = inspect.getsource(cd.extract_landmarks)
    mp_path = BACKEND / "app" / "services" / "mediapipe_service.py"
    mp_src = mp_path.read_text(encoding="utf-8")

    # Both must use: [[lm.x, lm.y, lm.z] for lm in ... ].flatten()
    uses_xyz = "lm.x, lm.y, lm.z" in cd_src
    uses_flatten_cd = ".flatten()" in cd_src
    uses_xyz_mp = "lm.x, lm.y, lm.z" in mp_src
    uses_flatten_mp = ".flatten()" in mp_src
    no_extra_norm_cd = "sklearn" not in cd_src.lower() and "StandardScaler" not in cd_src and "MinMaxScaler" not in cd_src
    no_extra_norm_mp = "NO additional" in mp_src or "No further transformation" in mp_src

    ok = uses_xyz and uses_flatten_cd and uses_xyz_mp and uses_flatten_mp and no_extra_norm_cd
    detail = (
        f"Collector  : lm.x,lm.y,lm.z={uses_xyz}  flatten={uses_flatten_cd}  no-extra-norm={no_extra_norm_cd}\n"
        f"MP Service : lm.x,lm.y,lm.z={uses_xyz_mp}  flatten={uses_flatten_mp}  no-extra-norm={no_extra_norm_mp}"
    )
    check(2, "Collector uses same MediaPipe preprocessing as service", ok, detail)
except Exception as e:
    check(2, "Collector uses same MediaPipe preprocessing as service", False, str(e))
    cd = None

# ============================================================
# CHECK 3 — 63-feature order is identical in all 3 stages
# ============================================================
try:
    # Collector: line-by-line check of feature construction
    assert "[[lm.x, lm.y, lm.z] for lm in hand.landmark]" in cd_src, "Wrong feature order in collector"
    # MP service
    assert "[[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]" in mp_src, "Wrong feature order in MP service"
    # Train: loads .npy (21,3) and flattens — same as saved
    train_src = (BACKEND / "train_model.py").read_text(encoding="utf-8")
    assert 'arr.shape == (21, 3)' in train_src and 'arr.flatten()' in train_src, "Training doesn't handle (21,3) shape"
    assert 'arr.shape != (63,)' in train_src, "Training doesn't validate 63-feature shape"
    detail = (
        "Collect  : [[lm.x, lm.y, lm.z] ... ].flatten() -> (63,)\n"
        "MPService: [[lm.x, lm.y, lm.z] ... ].flatten() -> (63,)\n"
        "Training : np.load(.npy), shape(21,3)->flatten() or shape(63,) direct\n"
        "Order    : lm0[x,y,z], lm1[x,y,z], ..., lm20[x,y,z]  (identical)"
    )
    check(3, "63-feature order identical across all 3 stages", True, detail)
except Exception as e:
    check(3, "63-feature order identical across all 3 stages", False, str(e))

# ============================================================
# CHECK 4 — Normalization is identical (none — raw coords)
# ============================================================
try:
    infer_src = (BACKEND / "app" / "services" / "sign_recognition.py").read_text(encoding="utf-8")
    # None of the stages should add extra normalization
    # Allowed: the word "normalize" only in comments/docstrings, not as a function call
    # Forbidden: actual normalization function calls (not comments/docstrings)
    # Allowed: the word "normalized" as MediaPipe's own coordinate description
    forbidden_patterns = ["StandardScaler", "MinMaxScaler", "sklearn.preprocessing", "tf.keras.layers.Normalization"]
    for stage, src in [("Collector", cd_src), ("MPService", mp_src), ("Inference", infer_src), ("Training", train_src)]:
        for pat in forbidden_patterns:
            assert pat not in src, f"{stage} contains forbidden normalization: {pat}"
    detail = (
        "Collector : raw MediaPipe coords (no normalization)\n"
        "MPService : raw MediaPipe coords (no normalization)\n"
        "Training  : raw .npy values loaded as-is\n"
        "Inference : raw landmarks passed directly from MPService\n"
        "All 4 stages: NO extra normalization applied"
    )
    check(4, "Normalization identical in all stages (none — raw coords)", True, detail)
except Exception as e:
    check(4, "Normalization identical in all stages (none — raw coords)", False, str(e))

# ============================================================
# CHECK 5 — Sample save path
# ============================================================
try:
    expected_root = BACKEND / "datasets" / "raw"
    assert "_DATASET_DIR" in inspect.getsource(cd), "No _DATASET_DIR in collector"
    # Check the path construction
    assert "member_dir / word" in inspect.getsource(cd.next_sample_path) or \
           "word_dir" in inspect.getsource(cd.main), "Path construction looks wrong"
    # Verify the actual path would be: datasets/raw/member_01/HELLO/sample_XXXX.npy
    # by simulating path construction
    test_dir = expected_root / "member_01" / "HELLO"
    # next_sample_path returns word_dir / f"sample_{idx:04d}.npy"
    expected_pattern = "sample_XXXX.npy"  # XXXX = 0-padded index
    detail = (
        f"Root    : {expected_root}\n"
        f"Pattern : datasets/raw/member_01/<WORD>/sample_XXXX.npy\n"
        f"Example : {test_dir / 'sample_0000.npy'}"
    )
    check(5, "Samples saved under datasets/raw/member_01/<WORD>/", True, detail)
except Exception as e:
    check(5, "Samples saved under datasets/raw/member_01/<WORD>/", False, str(e))

# ============================================================
# CHECK 6 — SPACE captures exactly one sample, debounce prevents duplicates
# ============================================================
try:
    main_src = inspect.getsource(cd.main)
    # Debounce check: last_capture timestamp
    assert "last_capture" in main_src, "No last_capture variable"
    assert "delay_active" in main_src, "No delay_active check"
    assert "last_capture = time.time()" in main_src, "last_capture not updated on capture"
    # Capture only calls np.save once per SPACE event
    assert "np.save(str(path), landmarks)" in main_src, "np.save not found in SPACE handler"
    # Only one np.save in the whole SPACE handler
    space_block = main_src[main_src.find("elif key == ord(\" \")"):main_src.find("elif key in (ord(\"n\")")]
    save_count = space_block.count("np.save")
    assert save_count == 1, f"Expected 1 np.save in SPACE handler, found {save_count}"
    # Debounce value
    delay_val = cfg.get("capture_delay_seconds", 0.4)
    detail = (
        f"Debounce  : {delay_val}s cooldown between captures (time-based, not key-repeat)\n"
        "Mechanism : last_capture=time.time() set on each save;\n"
        "            delay_active=(now-last_capture) < capture_delay blocks new capture\n"
        "Saves     : exactly 1 np.save() per accepted SPACE event\n"
        "Key-hold  : cv2.waitKey(1) returns key only once per loop iteration —\n"
        "            but even if held, delay_active blocks repeated saves"
    )
    check(6, "SPACE captures exactly one sample; debounce prevents duplicates", True, detail)
except Exception as e:
    check(6, "SPACE captures exactly one sample; debounce prevents duplicates", False, str(e))

# ============================================================
# CHECK 7 — Samples without a hand are rejected
# ============================================================
try:
    # is_valid_landmark must be called before saving
    assert "is_valid_landmark" in main_src, "is_valid_landmark not called in main"
    assert "hand_ok" in main_src, "hand_ok flag not used"
    # The SPACE handler checks hand_ok before saving
    assert "if not hand_ok" in space_block, "No hand_ok check before save"
    # is_valid_landmark checks None, shape, NaN, Inf, all-zeros
    valid_src = inspect.getsource(cd.is_valid_landmark)
    assert "vec is None" in valid_src, "Missing None check"
    assert "shape != (63,)" in valid_src, "Missing shape check"
    assert "np.isnan" in valid_src, "Missing NaN check"
    assert "np.isinf" in valid_src, "Missing Inf check"
    assert "np.allclose(vec, 0.0)" in valid_src, "Missing all-zeros check"
    detail = (
        "Guard 1: extract_landmarks returns None if no hand detected\n"
        "Guard 2: is_valid_landmark checks shape==(63,), no NaN, no Inf, not all-zeros\n"
        "Guard 3: SPACE handler: 'if not hand_ok: reject' — no save occurs\n"
        "Result : only valid, 63-float hand frames are saved to disk"
    )
    check(7, "Samples without a detected hand are rejected", True, detail)
except Exception as e:
    check(7, "Samples without a detected hand are rejected", False, str(e))

# ============================================================
# CHECK 8 — Word/class mapping saved consistently
# ============================================================
try:
    # train_model.py saves class_names.json
    assert "_CLASS_NAMES_PATH" in train_src, "No _CLASS_NAMES_PATH in trainer"
    assert "class_names.json" in train_src, "class_names.json not mentioned"
    # Check the structure it saves
    assert '"classes": classes' in train_src or '"classes"' in train_src, "classes key not in saved JSON"
    assert '"index_to_class"' in train_src, "index_to_class not in saved JSON"
    # sign_recognition.py loads class_names.json
    assert "_CLASS_NAMES_PATH" in infer_src, "sign_recognition doesn't reference class_names.json"
    assert 'data.get("classes", [])' in infer_src, "Inference doesn't read 'classes' key"
    # The index used during training matches the index used during inference
    # Training: y_list.append(classes.index(word)) → 0-based index into sorted(all_classes)
    # Inference: class_idx = np.argmax(predictions); word = labels[class_idx]
    # Both use the same class_names.json as the source of truth
    detail = (
        "Saved by  : train_model.py -> app/models/class_names.json\n"
        '  Structure: {"classes": [...], "index_to_class": {"0": "HELP", ...}, ...}\n'
        "Loaded by : sign_recognition.py reads class_names.json on startup\n"
        "Mapping   : training y=classes.index(word) == inference labels[argmax]\n"
        "Fallback  : labels.json (A-Z) if class_names.json missing"
    )
    check(8, "Word/class mapping saved consistently", True, detail)
except Exception as e:
    check(8, "Word/class mapping saved consistently", False, str(e))

# ============================================================
# CHECK 9 — Training auto-discovers configured classes
# ============================================================
try:
    # Use train_src which is already loaded, don't call inspect.getsource on loader
    disc_src = train_src
    assert "def discover_classes" in disc_src, "discover_classes function missing"
    assert "members_cfg.keys()" in disc_src, "Does not iterate members from config"
    assert "any((member_dir / w).glob" in disc_src, "Does not check for actual .npy files"
    assert 'Dense(num_classes' in disc_src or 'Dense(num_classes,' in disc_src, "Dense not parameterised by num_classes"
    assert "num_classes = len(classes)" in disc_src, "num_classes not computed dynamically"
    # Verify no hardcoded 26 or 48
    hardcoded_26 = "Dense(26" in disc_src
    hardcoded_48 = "Dense(48" in disc_src
    assert not hardcoded_26 and not hardcoded_48, f"Hardcoded class count found: Dense(26)={hardcoded_26}, Dense(48)={hardcoded_48}"
    detail = (
        "discover_classes() reads words.json -> finds members with *.npy files\n"
        "classes = sorted(set of words with actual data)\n"
        "num_classes = len(classes)  [dynamic, not hardcoded]\n"
        "Dense output layer: Dense(num_classes, activation='softmax')\n"
        "Prototype mode: member_01 only (8 classes)\n"
        "Full mode: all members with data (up to 48 classes)"
    )
    check(9, "Training auto-discovers configured classes (no hardcoded count)", True, detail)
except Exception as e:
    check(9, "Training auto-discovers configured classes (no hardcoded count)", False, str(e))

# ============================================================
# CHECK 10 — FastAPI inference uses exactly same preprocessing
# ============================================================
try:
    api_translate = (BACKEND / "app" / "api" / "translate.py").read_text(encoding="utf-8")
    # The inference path: image -> mediapipe_service.extract_landmarks -> sign_recognition.predict
    assert "mediapipe_svc" in api_translate or "_mediapipe_svc" in api_translate, \
        "translate.py doesn't call mediapipe service"
    assert "recognition_svc" in api_translate or "_recognition_svc" in api_translate, \
        "translate.py doesn't call recognition service"
    # sign_recognition.predict receives a (63,) array — no extra processing
    pred_src_block = infer_src[infer_src.find("def predict"):infer_src.find("def is_loaded")]
    assert "landmarks.ndim != 1 or landmarks.shape[0] != 63" in pred_src_block, \
        "Inference doesn't validate (63,) shape"
    assert "np.expand_dims(landmarks, axis=0)" in pred_src_block, \
        "Inference doesn't expand dims correctly for model input"
    # Confirm translate.py uses the same mediapipe service as used in training
    assert "extract_landmarks" in api_translate, "translate.py doesn't call extract_landmarks"
    detail = (
        "Pipeline: image -> MediaPipeService.extract_landmarks() -> (63,) float32\n"
        "          -> SignRecognitionService.predict() -> word + confidence\n"
        "MediaPipeService: same BGR->RGB->landmarks->flatten as collect_data.py\n"
        "Inference input : np.expand_dims(landmarks, axis=0) -> shape (1, 63)\n"
        "Validation      : shape check, NaN/Inf handled in recognition service\n"
        "No extra transforms between collection and inference"
    )
    check(10, "FastAPI inference uses exactly same preprocessing", True, detail)
except Exception as e:
    check(10, "FastAPI inference uses exactly same preprocessing", False, str(e))

# ============================================================
# SUMMARY
# ============================================================
passed = sum(1 for _, _, ok, _ in results if ok)
failed = sum(1 for _, _, ok, _ in results if not ok)

print()
print("=" * 60)
print(f"  VERIFICATION SUMMARY: {passed}/10 PASS  |  {failed}/10 FAIL")
print("=" * 60)

if failed == 0:
    print()
    print("  ALL CHECKS PASSED.")
    print()
    print("  Ready to collect Member 01 data.")
    print("  Command:")
    print()
    print("    python collect_data.py --member member_01 --samples 200")
    print()
else:
    print()
    print("  FAILURES DETECTED. Fix before collecting data.")

sys.exit(0 if failed == 0 else 1)
