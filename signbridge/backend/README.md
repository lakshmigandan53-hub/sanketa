# SANKETA Backend — Setup & Usage Guide

Indian Sign Language (ISL) Recognition API  
Built with FastAPI · MediaPipe · TensorFlow/Keras

---

## Architecture

```
signbridge/backend/
├── app/                       ← FastAPI application
│   ├── api/health.py          ← GET  /health
│   ├── api/translate.py       ← POST /translate/image
│   ├── api/model_info.py      ← GET  /model/info
│   ├── services/
│   │   ├── mediapipe_service.py   ← MediaPipe hand landmark extraction
│   │   └── sign_recognition.py   ← TF/Keras model inference + confidence threshold
│   ├── schemas/response.py    ← Pydantic response schemas
│   └── models/
│       ├── isl_model.keras    ← trained model (created by train_model.py)
│       └── class_names.json   ← word labels  (created by train_model.py)
│
├── config/
│   └── words.json             ← ALL word/member configuration (edit this)
│
├── datasets/
│   └── raw/
│       ├── member_01/
│       │   ├── HELLO/         ← .npy landmark files (one per captured frame)
│       │   ├── THANK_YOU/
│       │   └── ...
│       └── member_02/ ...
│
├── collect_data.py            ← webcam data collection
├── validate_dataset.py        ← dataset quality report
├── train_model.py             ← model training
├── test_api.py                ← API integration tests
└── test_pretrained_model.py   ← live webcam inference test
```

---

## Quick Start (Windows PowerShell)

### 1 — Create and activate virtual environment

```powershell
cd "c:\lakshmi gandan\sih\new repo\new\signbridge\backend"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Configure Your Words

Open `config\words.json` and set the **actual ISL words** for each member.

```json
"member_01": {
    "words": ["HELLO", "THANK_YOU", "YES", "NO", "PLEASE", "SORRY", "HELP", "WATER"],
    "word_types": {
        "HELLO": "static",
        "THANK_YOU": "dynamic"
    }
}
```

- Use `UPPER_SNAKE_CASE` (spaces → underscore)
- `"static"` = single hand position (e.g. YES, NO)
- `"dynamic"` = moving gesture (e.g. THANK_YOU, PLEASE)
- Each member must have exactly 8 unique words
- Total = 48 classes across all 6 members

> **Current Member 01 words (placeholders):**  
> HELLO · THANK_YOU · YES · NO · PLEASE · SORRY · HELP · WATER  
> Replace these with your actual assigned ISL words before collecting data.

---

## Step-by-Step: Member 01 Data Collection

### 3 — Start data collection

```powershell
python collect_data.py --member member_01 --samples 200
```

**On-screen controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Capture current frame |
| `N` | Next word |
| `P` | Previous word |
| `R` | Stay on current word |
| `Q` / `ESC` | Quit (progress saved) |

**Tips for good data quality:**
- Vary your **hand angle** slightly between captures
- Vary your **distance** from the camera (closer / farther)
- Capture in **different lighting** conditions
- For `[S]` static signs: hold the pose but vary your wrist angle
- For `[D]` dynamic signs: capture at different points in the motion
- Do NOT hold SPACE down — a 0.4 s cooldown prevents duplicate frames
- Target: **200 samples per word** (minimum 50 to start experimenting)

**Progress is saved automatically.** If you quit, re-running the command resumes from where you stopped.

---

### 4 — Validate the dataset

```powershell
python validate_dataset.py --member member_01
```

Example output:
```
── Member 01 (member_01) ──
  Word                 Samples  Status
  [S] HELLO               200   OK  200/200
  [D] THANK_YOU           198   LOW — 198/200
  [S] YES                 200   OK  200/200
  ...

TRAINING READY — 1596 samples, 8 classes
```

---

### 5 — Train the model (Member 01 prototype)

```powershell
python train_model.py --mode prototype
```

This trains on **Member 01 data only** using a random 70/15/15 split.

> ⚠ This is a **SINGLE-PERSON PROTOTYPE** result.  
> Test accuracy shown here is **not** real-world accuracy.  
> Final evaluation must use unseen members (Members 5 & 6).

Output:
- `app/models/isl_model.keras`
- `app/models/class_names.json`

---

### 6 — Start the FastAPI server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 7 — Open Swagger UI

Visit: **http://localhost:8000/docs**

---

### 8 — Run automated API tests

In a separate terminal (with venv active):

```powershell
python test_api.py
```

---

### 9 — Test with live webcam

```powershell
python test_pretrained_model.py
```

---

## API Reference

### `GET /health`
```json
{"status": "healthy", "service": "SANKETA AI Backend"}
```

### `GET /model/info`
```json
{
  "model_loaded": true,
  "model_path": "backend/app/models/isl_model.keras",
  "input_features": 63,
  "supported_classes": 8,
  "classes": ["HELLO", "HELP", "NO", "PLEASE", "SORRY", "THANK_YOU", "WATER", "YES"]
}
```

### `POST /translate/image`

Upload a hand image (JPEG / PNG / WebP):

```powershell
# PowerShell example with curl
curl -X POST "http://localhost:8000/translate/image" `
  -F "file=@hand_photo.jpg"
```

**Success response:**
```json
{
  "success": true,
  "word": "HELLO",
  "sign": "HELLO",
  "confidence": 0.96,
  "message": "ISL sign recognized successfully"
}
```

**Low confidence:**
```json
{
  "success": true,
  "word": "uncertain",
  "sign": "uncertain",
  "confidence": 0.42,
  "message": "Low confidence (42%). Best guess: 'HELLO' but below threshold (70%)."
}
```

**No hand detected:**
```json
{"success": false, "message": "No hand detected"}
```

---

## Confidence Threshold

Default: **70%**. Predictions below this return `"word": "uncertain"`.

Override via environment variable:
```powershell
$env:ISL_CONFIDENCE_THRESHOLD = "0.60"   # lower threshold (more permissive)
uvicorn app.main:app --reload --port 8000
```

---

## Multi-Member Expansion (Later)

When Members 2–6 are ready:

```powershell
# Member 2 collects their words
python collect_data.py --member member_02 --samples 200

# Validate all members
python validate_dataset.py

# Train full 48-class model with member-based split
python train_model.py --mode full
```

The full-mode split uses:
- **Train**: Members 1–4
- **Validation**: Member 5
- **Test**: Member 6 (unseen — real generalisation metric)

Existing Member 01 data is **never touched** when adding other members.

---

## Important Notes

| Claim | Reality |
|-------|---------|
| 200 samples is enough | It's a starting point. More variation = better accuracy. |
| Single-person test accuracy | NOT representative of real-world performance. |
| 100% accuracy | Not guaranteed. Target: >80% on unseen members. |
| One-time setup | Data collection is a one-time effort per member per word set. |

---

## Current Status

| Phase | Status |
|-------|--------|
| Backend API (FastAPI) | ✅ Complete |
| Data collection tool | ✅ Ready |
| Dataset validation | ✅ Ready |
| Model training pipeline | ✅ Ready |
| Model data | ⏳ **Requires Member 01 data collection** |
| React frontend | 🔜 Phase 2 |
| Flutter mobile app | 🔜 Phase 3 |
