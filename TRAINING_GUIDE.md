# SANKETA: Multi-Person Static ISL Training Guide

This guide explains how each team member collects high-quality Indian Sign Language (ISL) hand landmark data to train the SANKETA static recognition model locally.

---

## 1. Supported Static Signs (8 Classes)

The static sign classifier recognizes the following 8 signs:

1. **HELLO** (Open palm facing camera, upright hand, fingers spread naturally)
2. **YES** (Fist nodding slightly or upright fist with thumb folded)
3. **NO** (Index and middle finger extended side-by-side or wagging slightly)
4. **WATER** ("W" hand shape — index, middle, and ring fingers extended upward, thumb holding pinky)
5. **THANK_YOU** (Open flat hand touching chin/lips and gesturing slightly forward)
6. **PLEASE** (Open flat palm held flat over chest / in front of body)
7. **SORRY** (Closed fist placed over chest / heart area with gentle circular motion)
8. **HELP** (Closed fist with thumb upright resting upon a flat open palm)

> [!IMPORTANT]
> Keep the hand shape steady and consistent while capturing each static sign. Do not perform large full-arm motions during static collection.

---

## 2. Multi-Person Protocol

Each team member collects all 8 signs to ensure the trained model generalizes across different hand sizes, skin tones, and subtle variations:

- **Member 1 (`member_1`):** Collect all 8 signs (300 samples per sign).
- **Member 2 (`member_2`):** Collect all 8 signs (300 samples per sign).
- **Member 3 (`member_3`):** Collect all 8 signs (300 samples per sign).
- **Member 4 (`member_4`):** Collect all 8 signs (300 samples per sign).

Total dataset size for 4 members = **$8 \times 300 \times 4 = 9,600$ valid samples**.

### Team Member Guidelines During Recording:
- **Vary Hand Positions:** Move your hand slightly around the camera frame (center, slightly left, slightly right, upper half).
- **Vary Camera Distance:** Sit slightly closer (30–40 cm) for some samples, and normal distance (60–80 cm) for others.
- **Vary Orientation:** Slightly tilt your hand ($[-10^\circ, +10^\circ]$) to ensure rotational robustness.
- **Lighting Conditions:** Use normal room lighting (avoid extreme backlighting or deep shadows).
- **Avoid Background Clutter:** Avoid having faces or other hands overlapping with the active signing hand.
- **Do NOT Intentionally Change the Sign Shape:** Maintain the canonical hand posture for that sign.

---

## 3. Data Collection Steps

Open a terminal in the project root:

```bash
# Member 1
python training/collect_data.py --member member_1

# Member 2
python training/collect_data.py --member member_2

# Member 3
python training/collect_data.py --member member_3

# Member 4
python training/collect_data.py --member member_4
```

### Keyboard Controls:
| Key | Action |
|---|---|
| `SPACE` | **Toggle Auto-Recording** (starts/pauses collecting valid samples automatically) |
| `C` | **Capture Single Sample** (manual capture) |
| `N` | **Next Sign** (advances to next ISL word) |
| `P` | **Previous Sign** |
| `R` | **Reset** (clears samples for the current sign if you made a mistake) |
| `Q` or `ESC` | **Save & Quit** |

---

## 4. Dataset Validation

Before training, verify that all samples are valid, complete, and balanced:

```bash
python training/validate_dataset.py
```

This will print:
- Total valid samples
- Member-by-class distribution matrix
- Verification of zero corrupted/NaN samples
- Class balance confirmation

---

## 5. Local Model Training

Train the TensorFlow/Keras model locally using:

```bash
python training/train_model.py
```

Optional arguments:
- `--test-member member_4`: Choose which member to hold out as the unseen test person (default: last member).
- `--epochs 50`: Maximum training epochs (default: 50, with EarlyStopping).

During training, the script will:
1. Apply wrist-centering and scale normalization to all 63 coordinates.
2. Train on Members 1, 2, 3 and validate on Member 4 (true person-based generalization).
3. Test 20 samples from each class and assert that the **"Everything = Water"** prediction bias is eliminated.
4. Export the trained model to:
   - `signbridge/backend/app/models/isl_model.keras`
   - `signbridge/backend/app/models/labels.txt`
   - `signbridge/backend/app/models/labels.json`

---

## 6. Model Evaluation

Run a standalone evaluation on the held-out test person:

```bash
python training/evaluate_model.py
```

This displays:
- Overall accuracy & person-based test accuracy
- Per-class Precision, Recall, and F1-score
- $8 \times 8$ Confusion Matrix
- Anti-"Water" bias verification report

---

## 7. Live Testing in SANKETA

1. Start the SANKETA backend:
   ```bash
   cd signbridge/backend
   uvicorn app.main:app --reload --port 8000
   ```
2. Start the web frontend:
   ```bash
   cd signbridge/web
   npm run dev
   ```
3. Open `http://localhost:5173/camera` in Google Chrome or Edge.
4. Verify that:
   - Webcam feed and green skeleton overlay are active.
   - Hand signs display `SIGN DETECTED` with the recognized word and confidence.
   - Stable signs trigger speech output or are ready for sentence accumulation.
