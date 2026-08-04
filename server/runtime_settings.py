"""
Persisted runtime settings for the configuration platform.

The platform UI writes the active checkpoint + output preference here; the
``/predict`` endpoint reads it on every request, so NAO immediately runs with
whatever was last saved — no server restart, no changes on the robot.

Stored as JSON in ``server/runtime_settings.json`` (gitignored).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from server import checkpoint_registry

SETTINGS_PATH = Path(__file__).resolve().parent / "runtime_settings.json"

OUTPUT_MODES = ("single", "top_n", "threshold", "calibrated")

DEFAULT_SETTINGS: dict[str, Any] = {
    "checkpoint_id": "meld_4cls",
    "output": {
        "mode": "single",   # single | top_n | threshold | calibrated
        "top_n": 2,          # used when mode == top_n
        "threshold": 0.5,    # used when mode == threshold (global sigmoid cutoff)
    },
}

_lock = threading.Lock()


def _validate(settings: dict[str, Any]) -> dict[str, Any]:
    """Normalize + validate; raises ValueError with a user-readable message."""
    ckpt_id = settings.get("checkpoint_id", DEFAULT_SETTINGS["checkpoint_id"])
    try:
        info = checkpoint_registry.get(str(ckpt_id))
    except KeyError as exc:
        raise ValueError(str(exc)) from None

    raw_out = settings.get("output") or {}
    mode = str(raw_out.get("mode", "single")).lower()
    if mode not in OUTPUT_MODES:
        raise ValueError(f"output.mode must be one of {OUTPUT_MODES}, got '{mode}'")
    # Single-label checkpoints predict mutually exclusive classes — multiple
    # tags are not meaningful, so output preference only applies to multi-label.
    if info.task != "multi_label":
        mode = "single"
    elif mode == "calibrated" and not info.has_calibrated_thresholds:
        raise ValueError(
            f"Checkpoint '{info.id}' has no calibrated per-class thresholds; "
            "choose single / top_n / threshold instead."
        )

    try:
        top_n = int(raw_out.get("top_n", DEFAULT_SETTINGS["output"]["top_n"]))
    except (TypeError, ValueError):
        raise ValueError("output.top_n must be an integer") from None
    top_n = max(1, min(top_n, info.num_classes))

    try:
        threshold = float(raw_out.get("threshold", DEFAULT_SETTINGS["output"]["threshold"]))
    except (TypeError, ValueError):
        raise ValueError("output.threshold must be a number") from None
    if not (0.0 < threshold < 1.0):
        raise ValueError("output.threshold must be between 0 and 1 (exclusive)")

    return {
        "checkpoint_id": info.id,
        "output": {"mode": mode, "top_n": top_n, "threshold": threshold},
    }


def get_settings() -> dict[str, Any]:
    """Current settings; falls back to defaults if the file is missing/corrupt."""
    with _lock:
        if SETTINGS_PATH.is_file():
            try:
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                return _validate(raw)
            except (ValueError, OSError, json.JSONDecodeError):
                pass
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` into current settings, validate, persist, and return."""
    current = get_settings()
    merged = {
        "checkpoint_id": patch.get("checkpoint_id", current["checkpoint_id"]),
        "output": {**current["output"], **(patch.get("output") or {})},
    }
    validated = _validate(merged)
    with _lock:
        SETTINGS_PATH.write_text(
            json.dumps(validated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return validated
