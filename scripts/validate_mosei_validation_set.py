#!/usr/bin/env python3
"""
Batch-compare online MOSEI features vs offline CSD .pt for validation clips.

Reads ``data/online_mosei_validation/manifest.csv``, runs the server pipeline
(COVAREP + Whisper word timestamps + GloVe) on each ``test_clips/*.wav``,
and compares against ``features/mosei/seq_level/{audio,text}/{uid}.pt``.

Run from repo root:
  PYTHONPATH=. python scripts/validate_mosei_validation_set.py

  # Audio only (no GloVe file needed):
  PYTHONPATH=. python scripts/validate_mosei_validation_set.py --skip-text

  # Text only (skip COVAREP; reuse prior audio results in CSV if needed):
  PYTHONPATH=. python scripts/validate_mosei_validation_set.py --skip-audio

  # Full run (uses GLOVE_MODEL_PATH / COVAREP_RUNNER_BIN from server/config.py):
  PYTHONPATH=. python scripts/validate_mosei_validation_set.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from server.config import (
    COVAREP_RUNNER_BIN,
    GLOVE_MODEL_PATH,
    MAX_AUDIO_SECONDS,
    MOSEI_MAX_LEN_AUDIO,
    MOSEI_MAX_LEN_TEXT,
    ROOT_DIR,
    TARGET_SAMPLE_RATE,
    WHISPER_MODEL_SIZE,
    WHISPER_WORD_TIMESTAMPS,
)
from server.feature_extraction.covarep_runner import CovarepRunner
from server.feature_extraction.glove_encoder import GloveEncoder
from server.feature_extraction.whisper import WhisperTranscriber, decode_wav_bytes


@dataclass
class CompareMetrics:
    online_shape: tuple[int, ...]
    offline_shape: tuple[int, ...]
    aligned_shape: tuple[int, int]
    mae: float
    rmse: float
    mean_corr: float | None


def _resolve_repo_path(path: str | None, fallback: str = "") -> Path | None:
    raw = (path or fallback or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=str,
        default="data/online_mosei_validation/manifest.csv",
    )
    ap.add_argument(
        "--features-root",
        type=str,
        default="features/mosei/seq_level",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="data/online_mosei_validation/validation_results.csv",
    )
    ap.add_argument(
        "--glove-path",
        type=str,
        default="",
        help="Path to glove.840B.300d.txt (overrides server/config.py)",
    )
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument(
        "--server-mode",
        action="store_true",
        help="Apply server crop/truncate (10s audio, 300 COVAREP frames, 128 words)",
    )
    ap.add_argument(
        "--covarep-runner",
        type=str,
        default=None,
        help="MATLAB/Octave binary (default: COVAREP_RUNNER_BIN from config)",
    )
    ap.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Whisper size override (default from config)",
    )
    ap.add_argument(
        "--uids",
        nargs="*",
        default=None,
        help="Optional subset of uids to run",
    )
    return ap.parse_args()


def _load_offline_pt(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    h = obj["hidden"].float().numpy()
    if h.ndim == 1:
        h = h[None, :]
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


def compare_arrays(online: np.ndarray, offline: np.ndarray) -> CompareMetrics:
    min_len = min(online.shape[0], offline.shape[0])
    min_dim = min(online.shape[1], offline.shape[1])
    a = online[:min_len, :min_dim]
    b = offline[:min_len, :min_dim]
    diff = a - b

    corr_vals: list[float] = []
    for d in range(min_dim):
        if np.std(a[:, d]) > 1e-8 and np.std(b[:, d]) > 1e-8:
            corr_vals.append(float(np.corrcoef(a[:, d], b[:, d])[0, 1]))

    return CompareMetrics(
        online_shape=tuple(online.shape),
        offline_shape=tuple(offline.shape),
        aligned_shape=(min_len, min_dim),
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),
        mean_corr=float(np.mean(corr_vals)) if corr_vals else None,
    )


def _fmt_corr(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    features_root = Path(args.features_root)
    out_csv = Path(args.out_csv)

    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = [r for r in csv.DictReader(manifest_path.open()) if r.get("status") == "ok"]
    if args.uids:
        wanted = set(args.uids)
        rows = [r for r in rows if r["uid"] in wanted]
    if not rows:
        raise SystemExit("No validation rows to process.")

    max_seconds = MAX_AUDIO_SECONDS if args.server_mode else None
    max_audio_len = MOSEI_MAX_LEN_AUDIO if args.server_mode else None
    max_text_len = MOSEI_MAX_LEN_TEXT if args.server_mode else None

    covarep = None if args.skip_audio else CovarepRunner(
        runner_bin=args.covarep_runner or COVAREP_RUNNER_BIN
    )
    transcriber: WhisperTranscriber | None = None
    glove: GloveEncoder | None = None
    if not args.skip_text:
        glove_path = _resolve_repo_path(args.glove_path, GLOVE_MODEL_PATH)
        glove = GloveEncoder(model_path=glove_path)
        if not glove.is_configured:
            print(
                "[warn] GloVe not configured — skipping text. "
                "Extract glove.840B.300d.txt and pass --glove-path or set GLOVE_MODEL_PATH.",
                file=sys.stderr,
            )
            args.skip_text = True
        else:
            transcriber = WhisperTranscriber(
                model_size=args.whisper_model or WHISPER_MODEL_SIZE
            )

    mode_label = "server-mode" if args.server_mode else "full-segment"
    print(f"[info] mode={mode_label}  clips={len(rows)}")
    if not args.skip_audio:
        print(f"[info] COVAREP runner={covarep.runner_bin}  root={covarep.covarep_root}")
    if not args.skip_text:
        print(f"[info] Whisper model={args.whisper_model or WHISPER_MODEL_SIZE} "
              f"word_timestamps={WHISPER_WORD_TIMESTAMPS}")
        if glove is not None and glove.is_configured:
            print(f"[info] GloVe={glove.model_path}")

    result_rows: list[dict[str, object]] = []

    for idx, row in enumerate(rows, start=1):
        uid = row["uid"]
        clip_path = manifest_path.parent / row["clip_path"]
        if not clip_path.is_file():
            clip_path = ROOT_DIR / row["clip_path"]
        print(f"\n[{idx}/{len(rows)}] {uid}")

        out: dict[str, object] = {
            "uid": uid,
            "clip_path": str(clip_path.relative_to(ROOT_DIR) if clip_path.is_relative_to(ROOT_DIR) else clip_path),
            "duration_sec": row.get("duration_sec", ""),
            "mode": mode_label,
        }

        wav_bytes = clip_path.read_bytes()
        wav_np, sr = decode_wav_bytes(
            wav_bytes,
            filename_hint=clip_path.name,
            max_seconds=max_seconds,
        )

        if not args.skip_audio:
            offline_audio_pt = features_root / "audio" / f"{uid}.pt"
            if not offline_audio_pt.is_file():
                print(f"  [skip audio] missing {offline_audio_pt}")
                out["audio_status"] = "missing_offline"
            else:
                try:
                    online_audio = covarep.extract_frames(
                        wav_np,
                        sample_rate=sr,
                        max_len=max_audio_len,
                    )
                    offline_audio = _load_offline_pt(offline_audio_pt)
                    metrics = compare_arrays(online_audio, offline_audio)
                    out.update(
                        {
                            "audio_status": "ok",
                            "audio_online_shape": str(metrics.online_shape),
                            "audio_offline_shape": str(metrics.offline_shape),
                            "audio_aligned": str(metrics.aligned_shape),
                            "audio_mae": f"{metrics.mae:.6f}",
                            "audio_rmse": f"{metrics.rmse:.6f}",
                            "audio_mean_corr": _fmt_corr(metrics.mean_corr),
                        }
                    )
                    print(
                        f"  audio  online={metrics.online_shape} offline={metrics.offline_shape} "
                        f"MAE={metrics.mae:.4f} corr={_fmt_corr(metrics.mean_corr)}"
                    )
                except Exception as exc:
                    out["audio_status"] = "error"
                    out["audio_error"] = str(exc)
                    print(f"  [audio error] {exc}", file=sys.stderr)

        if not args.skip_text and transcriber is not None and glove is not None:
            offline_text_pt = features_root / "text" / f"{uid}.pt"
            if not offline_text_pt.is_file():
                print(f"  [skip text] missing {offline_text_pt}")
                out["text_status"] = "missing_offline"
            else:
                try:
                    asr = transcriber.transcribe(
                        wav_np,
                        sample_rate=sr,
                        word_timestamps=WHISPER_WORD_TIMESTAMPS,
                    )
                    duration = len(wav_np) / float(TARGET_SAMPLE_RATE)
                    if WHISPER_WORD_TIMESTAMPS and hasattr(asr, "words") and asr.words:
                        online_text = glove.encode_timed_words(
                            asr.words,
                            max_duration=duration,
                            max_len=max_text_len,
                        )
                        text_mode = "whisper_word_timestamps"
                    else:
                        text = asr.text if hasattr(asr, "text") else str(asr)
                        online_text = glove.encode_words(
                            glove.tokenize(text),
                            max_len=max_text_len,
                        )
                        text_mode = "whisper_plain"

                    offline_text = _load_offline_pt(offline_text_pt)
                    metrics = compare_arrays(online_text, offline_text)
                    out.update(
                        {
                            "text_status": "ok",
                            "text_mode": text_mode,
                            "whisper_transcript": (
                                asr.text[:120] if hasattr(asr, "text") else str(asr)[:120]
                            ),
                            "text_online_shape": str(metrics.online_shape),
                            "text_offline_shape": str(metrics.offline_shape),
                            "text_aligned": str(metrics.aligned_shape),
                            "text_mae": f"{metrics.mae:.6f}",
                            "text_rmse": f"{metrics.rmse:.6f}",
                            "text_mean_corr": _fmt_corr(metrics.mean_corr),
                        }
                    )
                    print(
                        f"  text   online={metrics.online_shape} offline={metrics.offline_shape} "
                        f"MAE={metrics.mae:.4f} corr={_fmt_corr(metrics.mean_corr)} "
                        f"({text_mode})"
                    )
                except Exception as exc:
                    out["text_status"] = "error"
                    out["text_error"] = str(exc)
                    print(f"  [text error] {exc}", file=sys.stderr)

        result_rows.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in result_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"\n[OK] wrote {out_csv}")

    audio_ok = [r for r in result_rows if r.get("audio_status") == "ok"]
    if audio_ok:
        mae_vals = [float(r["audio_mae"]) for r in audio_ok]
        corr_vals = [
            float(r["audio_mean_corr"])
            for r in audio_ok
            if r.get("audio_mean_corr") not in (None, "", "n/a")
        ]
        print(
            f"audio summary ({len(audio_ok)} clips): "
            f"MAE mean={np.mean(mae_vals):.4f}  "
            f"corr mean={np.mean(corr_vals):.4f}" if corr_vals else
            f"audio summary ({len(audio_ok)} clips): MAE mean={np.mean(mae_vals):.4f}"
        )

    text_ok = [r for r in result_rows if r.get("text_status") == "ok"]
    if text_ok:
        mae_vals = [float(r["text_mae"]) for r in text_ok]
        corr_vals = [
            float(r["text_mean_corr"])
            for r in text_ok
            if r.get("text_mean_corr") not in (None, "", "n/a")
        ]
        print(
            f"text summary ({len(text_ok)} clips): "
            f"MAE mean={np.mean(mae_vals):.4f}  "
            f"corr mean={np.mean(corr_vals):.4f}" if corr_vals else
            f"text summary ({len(text_ok)} clips): MAE mean={np.mean(mae_vals):.4f}"
        )


if __name__ == "__main__":
    main()
