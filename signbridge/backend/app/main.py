"""
SANKETA FastAPI Application Entry Point
Indian Sign Language Translation Backend
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, translate
from app.api import model_info

# ---------------------------------------------------------------------------
# Logging — suppress noisy TF / MediaPipe messages below WARNING level
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Quieten TensorFlow C++ kernel logs (oneDNN, XLA, etc.)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SANKETA AI Backend",
    description=(
        "Indian Sign Language (ISL) recognition backend. "
        "Accepts camera images, detects hand gestures via MediaPipe Hands "
        "(21 landmarks → 63 features), and classifies them using a TensorFlow/Keras "
        "Dense MLP model. Supports configurable ISL word classes per team member. "
        "Train with collect_data.py + train_model.py. "
        "See config/words.json to configure words."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "SANKETA Team"},
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# CORS — configured for React (web, Vercel), Flutter, and local development
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS: list[str] = [
    # React dev servers (Create React App / Vite)
    "http://localhost:3000",
    "http://localhost:5173",
    # Flutter web dev server
    "http://localhost:8080",
    # Flutter Android emulator — host machine is reachable at 10.0.2.2
    "http://10.0.2.2:8000",
    # Flutter iOS simulator
    "http://127.0.0.1:8000",
    # Generic localhost (covers any port dev tools may use)
    "http://localhost",
]

# Support additional custom origins via environment variable
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
if _env_origins:
    for _o in _env_origins.split(","):
        _o_clean = _o.strip()
        if _o_clean and _o_clean not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(_o_clean)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, tags=["Health"])
app.include_router(model_info.router, tags=["Model"])
app.include_router(translate.router, prefix="/translate", tags=["Translation"])
