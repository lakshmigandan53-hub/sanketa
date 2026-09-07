# SANKETA — Indian Sign Language Translation Platform

A full-stack platform for real-time Indian Sign Language (ISL) recognition,
built for the Smart India Hackathon (SIH).

---

## Architecture Overview

```
signbridge/
├── backend/        ← Phase 1 (this release) — FastAPI + MediaPipe + TensorFlow
├── frontend/       ← Phase 2 (planned)      — React web application
└── mobile/         ← Phase 3 (planned)      — Flutter Android / iOS application
```

---

## Phase 1 — Backend (Current)

A production-style Python REST API that:

- Accepts a camera image via HTTP upload
- Detects hand landmarks using **MediaPipe**
- Classifies the ISL sign using a **TensorFlow/Keras** model
- Returns the predicted sign and confidence score as JSON

👉 See [`backend/README.md`](backend/README.md) for full setup and usage instructions.

---

## Quick Start

```bash
cd signbridge/backend
python -m venv venv && .\venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

---

## Roadmap

| Phase | Component | Status |
|---|---|---|
| 1 | FastAPI backend + MediaPipe + TensorFlow | ✅ Complete |
| 2 | React web application | 🔜 Planned |
| 3 | Flutter mobile application | 🔜 Planned |
| 4 | Real-time video streaming | 🔜 Planned |
| 5 | ISL-to-speech / text output | 🔜 Planned |

---

## License

MIT — see `LICENSE` for details.
