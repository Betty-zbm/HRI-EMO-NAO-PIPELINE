"""
HRI-EMO emotion recognition server for NAO robot integration.

Run from repo root:
    uvicorn server.main:app --host 0.0.0.0 --port 8000

Development (auto-reload, watches server/ only — avoids venv reload loops):
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir server

Test page: http://localhost:8000/localtestpage
API docs:   http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``models.*`` imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from server.config import DEFAULT_BENCHMARK
from server.emotion_service import EmotionService

SERVER_DIR = Path(__file__).resolve().parent
LOCAL_TEST_PAGE = SERVER_DIR / "localtestpage.html"

app = FastAPI(
    title="HRI-EMO Emotion Server",
    description=(
        "Multimodal emotion recognition API for NAO. "
        "Accepts raw audio, transcribes with Whisper, extracts online features, "
        "and runs the fusion model."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

emotion_service = EmotionService()


@app.get("/")
def root():
    return {
        "message": "HRI-EMO emotion server running.",
        "docs": "/docs",
        "local_test_page": "/localtestpage",
    }


@app.get("/localtestpage")
def local_test_page():
    if LOCAL_TEST_PAGE.is_file():
        return FileResponse(LOCAL_TEST_PAGE)
    raise HTTPException(status_code=404, detail="localtestpage.html not found")


@app.get("/health")
def health():
    return emotion_service.health()


@app.post("/predict")
async def predict(
    audio: UploadFile = File(..., description="PCM WAV audio from NAO (mono, 16 kHz recommended)"),
    benchmark: str = Form(default=DEFAULT_BENCHMARK),
):
    """
    Predict emotion from uploaded audio.

    Pipeline:
      1. Truncate audio to first 10 seconds
      2. Whisper ASR transcription
      3. Online feature extraction (benchmark-specific)
      4. Fusion model inference

    **benchmark**: ``iemocap`` (single-label, WavLM+BERT) or ``mosei`` (multi-label, placeholder features)
    """
    benchmark = (benchmark or DEFAULT_BENCHMARK).strip().lower()
    if benchmark not in ("iemocap", "mosei"):
        raise HTTPException(
            status_code=400,
            detail="benchmark must be 'iemocap' or 'mosei'",
        )

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        result = emotion_service.run(
            data,
            benchmark=benchmark,  # type: ignore[arg-type]
            filename_hint=audio.filename or "upload.wav",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
    )
