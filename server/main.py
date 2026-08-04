"""
HRI-EMO emotion recognition server for NAO robot integration.

Run from repo root:
    uvicorn server.main:app --host 0.0.0.0 --port 8000

Development (auto-reload, watches server/ only — avoids venv reload loops):
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir server

Configuration platform: http://localhost:8000/platform
Test page:              http://localhost:8000/localtestpage
API docs:               http://localhost:8000/docs
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is on sys.path so ``models.*`` imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from server import checkpoint_registry, runtime_settings
from server.emotion_service import EmotionService

SERVER_DIR = Path(__file__).resolve().parent
LOCAL_TEST_PAGE = SERVER_DIR / "localtestpage.html"
PLATFORM_PAGE = SERVER_DIR / "platform.html"
PLATFORM_LOGO_SVG = SERVER_DIR / "platform_logo.svg"
PLATFORM_LOGO_PNG = SERVER_DIR / "platform_logo.png"
NAO_CLIENT_TEMPLATE = SERVER_DIR / "nao_scripts" / "emotion_client.py"

app = FastAPI(
    title="HRI-EMO Emotion Server",
    description=(
        "Multimodal emotion recognition API for NAO. "
        "Accepts raw audio, transcribes with Whisper, extracts online features, "
        "and runs the fusion model with the checkpoint selected on the "
        "configuration platform (/platform)."
    ),
    version="0.2.0",
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
        "platform": "/platform",
        "docs": "/docs",
        "local_test_page": "/localtestpage",
    }


@app.get("/localtestpage")
def local_test_page():
    if LOCAL_TEST_PAGE.is_file():
        return FileResponse(LOCAL_TEST_PAGE)
    raise HTTPException(status_code=404, detail="localtestpage.html not found")


@app.get("/platform")
def platform_page():
    if PLATFORM_PAGE.is_file():
        return FileResponse(PLATFORM_PAGE)
    raise HTTPException(status_code=404, detail="platform.html not found")


@app.get("/platform/logo")
def platform_logo():
    """Lab logo shown in the platform header (server/platform_logo.svg or .png)."""
    if PLATFORM_LOGO_SVG.is_file():
        return FileResponse(PLATFORM_LOGO_SVG, media_type="image/svg+xml")
    if PLATFORM_LOGO_PNG.is_file():
        return FileResponse(PLATFORM_LOGO_PNG)
    raise HTTPException(status_code=404, detail="platform logo not provided")


@app.get("/platform/demo-audio")
def platform_demo_audio():
    """Built-in example clip for the Playground's demo mode (?demo=1)."""
    demo = SERVER_DIR / "demo_audio.wav"
    if demo.is_file():
        return FileResponse(demo, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="demo audio not provided")


@app.get("/health")
def health():
    return emotion_service.health()


# ─── Configuration platform API ────────────────────────────────────────────────

@app.get("/api/checkpoints")
def list_checkpoints():
    """All selectable checkpoints with metadata, metrics, and availability."""
    return {"checkpoints": checkpoint_registry.describe_all()}


@app.get("/api/config")
def get_config():
    """Current active checkpoint + output preference (used by /predict)."""
    settings = runtime_settings.get_settings()
    info = checkpoint_registry.get(settings["checkpoint_id"])
    ok, reason = checkpoint_registry.availability(info)
    return {
        **settings,
        "checkpoint_name": info.display_name,
        "checkpoint_available": ok,
        "checkpoint_unavailable_reason": reason,
    }


@app.put("/api/config")
def put_config(payload: dict[str, Any] = Body(...)):
    """
    Update platform settings. Body:
    ``{"checkpoint_id": "...", "output": {"mode": "...", "top_n": 2, "threshold": 0.5}}``
    Takes effect immediately for all subsequent /predict calls (including NAO).
    """
    try:
        saved = runtime_settings.update_settings(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    info = checkpoint_registry.get(saved["checkpoint_id"])
    ok, reason = checkpoint_registry.availability(info)
    return {
        **saved,
        "checkpoint_name": info.display_name,
        "checkpoint_available": ok,
        "checkpoint_unavailable_reason": reason,
    }


_IP_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


@app.get("/api/nao-script")
def nao_script(
    host: str = Query(..., description="LAN IP of the machine running this server"),
    port: int = Query(8000, ge=1, le=65535),
):
    """Generate the Choregraphe Python-box client with the server address baked in."""
    host = host.strip()
    if not host or not _IP_RE.match(host):
        raise HTTPException(status_code=400, detail="Invalid host / IP address")
    if not NAO_CLIENT_TEMPLATE.is_file():
        raise HTTPException(status_code=500, detail="emotion_client.py template not found")

    script = NAO_CLIENT_TEMPLATE.read_text(encoding="utf-8")
    script = re.sub(
        r'^SERVER_HOST = ".*?"',
        'SERVER_HOST = "%s"' % host,
        script,
        count=1,
        flags=re.MULTILINE,
    )
    script = re.sub(
        r"^SERVER_PORT = \d+",
        "SERVER_PORT = %d" % port,
        script,
        count=1,
        flags=re.MULTILINE,
    )

    settings = runtime_settings.get_settings()
    info = checkpoint_registry.get(settings["checkpoint_id"])
    return {
        "script": script,
        "host": host,
        "port": port,
        "active_checkpoint": info.display_name,
        "note": (
            "The script always uses the checkpoint + output settings saved on "
            "this platform; update them here and NAO follows automatically."
        ),
    }


# ─── Prediction ────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(
    audio: UploadFile = File(..., description="PCM WAV audio from NAO (mono, 16 kHz recommended)"),
    benchmark: Optional[str] = Form(default=None),
    checkpoint: Optional[str] = Form(default=None),
):
    """
    Predict emotion from uploaded audio.

    Pipeline:
      1. Truncate audio to first 10 seconds
      2. Whisper ASR transcription
      3. Online feature extraction (checkpoint-specific)
      4. Fusion model inference + configured output formatting

    Checkpoint resolution: ``checkpoint`` form field (registry id) >
    legacy ``benchmark`` (``iemocap``/``mosei``) > platform settings.
    """
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        result = emotion_service.run(
            data,
            checkpoint_id=(checkpoint or "").strip() or None,
            benchmark=(benchmark or "").strip() or None,
            filename_hint=audio.filename or "upload.wav",
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
