# SANKETA — 20-Class ISL Static Sign Reference Guide

This guide describes the standard ISL (Indian Sign Language) static handshapes
for all 20 target classes. Use this as reference while recording data.

---

## Quick Commands

```bash
# Step 1: Collect data for member_01 (15 new signs + skip already-done ones)
cd signbridge/backend
python collect_data.py --member member_01

# Step 2: Validate the dataset
python validate_dataset.py

# Step 3: Train the 20-class model
python train_static_20class.py

# Step 4: If accuracy >= 88%, deploy with:
python train_static_20class.py --deploy
```

---

## Data Collection Tips

- **Target**: 200 samples per sign per member
- **Vary**: distance (30-80cm), slight hand angles, lighting conditions
- **Hold static**: For static signs, hold the handshape still before pressing SPACE
- **Controls**: `SPACE` = capture, `N` = next sign, `P` = previous, `Q` = quit
- **Quality**: Ensure your hand is clearly visible with good lighting

---

## ISL Handshapes — All 20 Target Signs

> **IMPORTANT**: Hold the handshape **still** (static pose) before capturing.
> These are ISL-standard handshapes. Do not invent your own.

---

### Signs Carried Over from 8-Class Model (200 samples already collected)

| Sign | ISL Handshape Description |
|------|---------------------------|
| **HELLO** | Open palm, fingers together, hand raised at forehead level (wave/salute) |
| **YES** | Closed fist (thumb up or nodding fist motion -- capture the static fist pose) |
| **NO** | Index and middle fingers extended + spread (or index finger wagging -- capture static X-hand) |
| **WATER** | W-handshape: index + middle + ring fingers extended, touching lips |
| **HELP** | Flat open hand palm-up, placed on top of closed fist of other hand, both lifted upward |

---

### 15 New Signs to Collect

| Sign | ISL Handshape Description |
|------|---------------------------|
| **FOOD** | Flat O-hand (all fingertips pinched to thumb) brought toward mouth |
| **MILK** | Open-close fist motion (squeezing) -- capture one of the static states (closed fist or open claw) |
| **TEA** | F-hand (index + thumb pinching, others extended) at lip level, mimicking holding a cup |
| **BOOK** | Both palms flat together, then opening like a book -- capture the open-book pose (both palms face up) |
| **PEN** | Index finger and thumb pinched together (writing grip pose) near palm |
| **PHONE** | Y-hand (thumb + pinky extended, others closed) held at ear |
| **COMPUTER** | Dominant hand C-shape tapping on non-dominant flat palm |
| **HOME** | Flat hand, fingertips touch thumb (O/flat-O) moved from chin upward to cheek |
| **SCHOOL** | Both flat hands clap twice (static: capture both palms parallel, facing each other) |
| **COLLEGE** | Dominant C-hand or flat hand circling over palm -- capture the C-hand static pose |
| **DOCTOR** | D-hand (index finger extended, others curved) tapping wrist like taking pulse |
| **HOSPITAL** | H-hand (index + middle finger extended together) drawing a small cross on upper arm |
| **MONEY** | Flat O-hand tapping into opposite open palm (palm-up) |
| **HOUSE** | Both flat open palms forming a roof shape (inverted V / triangle overhead) |
| **CAR** | Both fists held out (like gripping a steering wheel) -- S-hands or A-hands facing each other |

---

## Notes on ISL Ambiguity

- **MILK**: Some ISL references show a milking motion. For static capture, use the closed-fist (S-hand) static pose and be consistent.
- **BOOK**: Capture with palms open and facing up (the "open book" static pose).
- **SCHOOL / COLLEGE**: If your team's local ISL variant differs, use the same handshape consistently across all members.
- **HOSPITAL**: The cross drawn on the arm is the standard ISL form. Capture the H-hand touching the upper arm.

---

## Per-Member Data Collection Order

Each member should collect all 20 signs. Signs 1-5 (HELLO, YES, NO, WATER, HELP) already have 200
samples from member_01. collect_data.py will automatically skip completed signs.

---

## Expected Dataset After Full Collection

| Member | Signs | Samples/Sign | Total |
|--------|-------|--------------|-------|
| member_01 | 20 | 200 | 4,000 |
| member_02 | 20 | 200 | 4,000 |
| member_03 | 20 | 200 | 4,000 |
| **Target** | 20 | 200 | **12,000+** |

Minimum for training: **50 samples per sign**.
Recommended: **200 samples per sign per member**.

---

## Training Output Files

After running `train_static_20class.py`, these files are produced:

| File | Purpose |
|------|---------|
| `app/models/isl_static_20class.keras` | Staged 20-class model (always saved) |
| `app/models/static_classes_20.json` | Class metadata + per-class accuracy |
| `app/models/labels.txt` | One class per line (used by live inference) |
| `app/models/labels.json` | Class list in JSON format |
| `reports/static_20class_report.json` | Full training report with confusion matrix |

After verifying results:
- Run `python train_static_20class.py --deploy` to replace `isl_model.keras`
- The previous model is backed up as `isl_model_backup.keras`

---

## Do NOT

- Do NOT invent or use arbitrary handshapes -- use ISL-standard forms
- Do NOT use the same hand position for every sample (vary slightly)
- Do NOT deploy the model until person-based test accuracy >= 88%
- Do NOT modify Camera.jsx, mediapipe_service.py, or the webcam pipeline
