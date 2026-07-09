"""
api/main.py

BioIQ FastAPI application entry point.

Registers CORS middleware and mounts all route handlers.

Run with:
    cd /path/to/bioiq
    source venv/bin/activate
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Ensure the project root is on sys.path so graph/ and synthesis/ imports
# resolve correctly regardless of how uvicorn is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from api.routes import router, limiter  # noqa: E402 — must come after sys.path fix

app = FastAPI(
    title="BioIQ",
    description="Rare and chronic disease research intelligence API",
    version="0.1.0",
)

# CORS — open for local dev; tighten CORS_ORIGINS in .env before public deploy
_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(router)
