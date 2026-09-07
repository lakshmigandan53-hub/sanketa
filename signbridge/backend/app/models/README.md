# ISL Model Directory

Place your trained TensorFlow/Keras model here as:

```
app/models/isl_model.keras
```

## Expected model specification

| Property | Value |
|---|---|
| Input shape | `(None, 63)` — 21 MediaPipe landmarks × 3 coords (x, y, z) |
| Output shape | `(None, N)` — one probability per ISL sign class |
| Format | Keras native format (`.keras`) |

## Suggested class label order

The default label list in `app/services/sign_recognition.py` is:

```
A B C D E F G H I K L M N O P Q R S T U V W X Y   (24 static letters)
0 1 2 3 4 5 6 7 8 9                                (10 digits)
```

> **J and Z** are excluded because they require motion (dynamic gestures).  
> Adjust `ISL_CLASS_LABELS` in `sign_recognition.py` to match your actual training labels.

## Training tip

Use `MediaPipeService.extract_landmarks()` to collect training data — the same
preprocessing pipeline guarantees that training features match inference features.
