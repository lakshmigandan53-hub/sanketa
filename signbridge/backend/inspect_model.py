"""
Re-save isl_model.keras (Keras 3.x) as TF 2.12-compatible H5.
Uses exact name-to-name mapping between Keras 3.x h5 paths and TF 2.12 weight names.
"""
import zipfile, json, os, shutil, tempfile
import numpy as np
import h5py

MODEL_PATH = "app/models/isl_model.keras"
OUT_H5 = "app/models/isl_model_tf212.h5"

# ── Step 1: Extract weights from Keras 3.x .keras zip ─────────────────────────
tmpdir = tempfile.mkdtemp()
weights_h5_path = os.path.join(tmpdir, 'model.weights.h5')
try:
    with zipfile.ZipFile(MODEL_PATH, 'r') as z:
        z.extract('model.weights.h5', tmpdir)

    # Read all layer weights by path
    keras3_weights = {}
    with h5py.File(weights_h5_path, 'r') as f:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and name.startswith('layers/'):
                keras3_weights[name] = np.array(obj)
        f.visititems(collect)

    print("Keras 3.x layer weights found:")
    for k, v in keras3_weights.items():
        print(f"  {k}: shape={v.shape}")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ── Step 2: Build model with TF 2.12 API ──────────────────────────────────────
import tensorflow as tf
print("\nTF version:", tf.__version__)

model = tf.keras.Sequential(name="sanketa_isl_static")
model.add(tf.keras.Input(shape=(63,), name="input_landmarks"))
model.add(tf.keras.layers.Dense(128, activation='relu', name='dense'))
model.add(tf.keras.layers.BatchNormalization(name='batch_normalization'))
model.add(tf.keras.layers.Dropout(0.3, name='dropout'))
model.add(tf.keras.layers.Dense(64, activation='relu', name='dense_1'))
model.add(tf.keras.layers.BatchNormalization(name='batch_normalization_1'))
model.add(tf.keras.layers.Dropout(0.2, name='dropout_1'))
model.add(tf.keras.layers.Dense(32, activation='relu', name='dense_2'))
model.add(tf.keras.layers.Dropout(0.1, name='dropout_2'))
model.add(tf.keras.layers.Dense(11, activation='softmax', name='dense_3'))

# Build the model so weights exist
model.build((None, 63))
model.summary()

# ── Step 3: Manual weight assignment with explicit name mapping ────────────────
# Keras 3.x BN var order: [gamma, beta, moving_mean, moving_variance] (vars/0..3)
# TF 2.12 BN weight order: gamma, beta, moving_mean, moving_variance
# Dense: vars/0=kernel, vars/1=bias

assignments = {
    # Dense layer 1
    'dense/kernel:0':                              keras3_weights['layers/dense/vars/0'],
    'dense/bias:0':                                keras3_weights['layers/dense/vars/1'],
    # BN 1: Keras3 vars/0=gamma, vars/1=beta, vars/2=moving_mean, vars/3=moving_var
    'batch_normalization/gamma:0':                 keras3_weights['layers/batch_normalization/vars/0'],
    'batch_normalization/beta:0':                  keras3_weights['layers/batch_normalization/vars/1'],
    'batch_normalization/moving_mean:0':           keras3_weights['layers/batch_normalization/vars/2'],
    'batch_normalization/moving_variance:0':       keras3_weights['layers/batch_normalization/vars/3'],
    # Dense layer 2
    'dense_1/kernel:0':                            keras3_weights['layers/dense_1/vars/0'],
    'dense_1/bias:0':                              keras3_weights['layers/dense_1/vars/1'],
    # BN 2
    'batch_normalization_1/gamma:0':               keras3_weights['layers/batch_normalization_1/vars/0'],
    'batch_normalization_1/beta:0':                keras3_weights['layers/batch_normalization_1/vars/1'],
    'batch_normalization_1/moving_mean:0':         keras3_weights['layers/batch_normalization_1/vars/2'],
    'batch_normalization_1/moving_variance:0':     keras3_weights['layers/batch_normalization_1/vars/3'],
    # Dense layer 3
    'dense_2/kernel:0':                            keras3_weights['layers/dense_2/vars/0'],
    'dense_2/bias:0':                              keras3_weights['layers/dense_2/vars/1'],
    # Dense layer 4 (output)
    'dense_3/kernel:0':                            keras3_weights['layers/dense_3/vars/0'],
    'dense_3/bias:0':                              keras3_weights['layers/dense_3/vars/1'],
}

print("\nModel weights to assign:")
model_weights_list = []
for w in model.weights:
    name = w.name
    if name in assignments:
        val = assignments[name]
        assert val.shape == tuple(w.shape), f"Shape mismatch for {name}: got {val.shape}, expected {w.shape}"
        model_weights_list.append(val)
        print(f"  [OK] {name}: {val.shape}")
    else:
        print(f"  [WARN] No mapping for {name}, using zeros")
        model_weights_list.append(np.zeros(w.shape.as_list(), dtype=np.float32))

model.set_weights(model_weights_list)
print("All weights assigned!")

# ── Step 4: Verify predictions look sane ──────────────────────────────────────
dummy = np.zeros((1, 63), dtype=np.float32)
preds = model.predict(dummy, verbose=0)
print(f"\nDummy prediction sum: {preds.sum():.6f} (should be ~1.0)")
if np.isnan(preds).any():
    print("ERROR: NaN in predictions — weight assignment failed")
    raise RuntimeError("NaN predictions after weight assignment")

print(f"Prediction: {preds[0]}")
print(f"Argmax: {np.argmax(preds[0])} (class: {np.argmax(preds[0])})")

# Test with non-zero input
np.random.seed(42)
test_lms = np.random.rand(1, 63).astype(np.float32)
preds2 = model.predict(test_lms, verbose=0)
print(f"\nRandom input prediction sum: {preds2.sum():.6f}")
print(f"Random input prediction: {preds2[0]}")
print(f"Predicted class: {np.argmax(preds2[0])}")

# ── Step 5: Save in TF 2.12 H5 format ────────────────────────────────────────
model.save(OUT_H5)
print(f"\nSaved TF 2.12-compatible model to: {OUT_H5}")

# ── Step 6: Verify reloaded model ────────────────────────────────────────────
loaded = tf.keras.models.load_model(OUT_H5)
preds3 = loaded.predict(test_lms, verbose=0)
max_diff = float(np.max(np.abs(preds2 - preds3)))
print(f"Reload max diff: {max_diff:.2e}")
print("DONE! Model is ready for TF 2.12.")
