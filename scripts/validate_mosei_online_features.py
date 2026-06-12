#!/usr/bin/env python3
"""
Compare online MOSEI features (COVAREP + GloVe) against offline CSD exports.

Examples (from repo root):

  # Sanity: COVAREP on a WAV file (shape + basic stats)
  PYTHONPATH=. python scripts/validate_mosei_online_features.py \\
    --wav path/to/clip.wav

  # Compare online COVAREP vs offline CSD-derived .pt (needs matching clip)
  PYTHONPATH=. python scripts/validate_mosei_online_features.py \\
    --wav path/to/clip.wav \\
    --offline-audio-pt features/mosei/seq_level/audio/VIDEO_ID_0.pt

  # GloVe text sanity
  PYTHONPATH=. python scripts/validate_mosei_online_features.py \\
    --text "I am so happy today"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from server.feature_extraction.covarep_runner import CovarepRunner
from server.feature_extraction.glove_encoder import GloveEncoder
from server.feature_extraction.whisper import decode_wav_bytes


def _load_offline_pt(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    h = obj["hidden"].float().numpy()
    if h.ndim == 1:
        h = h[None, :]
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


def _compare_arrays(name: str, online: np.ndarray, offline: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print(f"online shape:  {online.shape}")
    print(f"offline shape: {offline.shape}")

    min_len = min(online.shape[0], offline.shape[0])
    min_dim = min(online.shape[1], offline.shape[1])
    a = online[:min_len, :min_dim]
    b = offline[:min_len, :min_dim]

    diff = a - b
    print(f"aligned slice: [{min_len}, {min_dim}]")
    print(f"MAE:  {float(np.mean(np.abs(diff))):.6f}")
    print(f"RMSE: {float(np.sqrt(np.mean(diff ** 2))):.6f}")

    corr_vals = []
    for d in range(min_dim):
        if np.std(a[:, d]) > 1e-8 and np.std(b[:, d]) > 1e-8:
            corr_vals.append(float(np.corrcoef(a[:, d], b[:, d])[0, 1]))
    if corr_vals:
        print(f"mean per-dim corr: {float(np.mean(corr_vals)):.4f}")
    else:
        print("mean per-dim corr: n/a (constant columns)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default=None, help="WAV bytes path for COVAREP test")
    ap.add_argument(
        "--offline-audio-pt",
        type=str,
        default=None,
        help="Offline MOSEI COVAREP .pt to compare against",
    )
    ap.add_argument("--text", type=str, default=None, help="Transcript for GloVe test")
    ap.add_argument(
        "--offline-text-pt",
        type=str,
        default=None,
        help="Offline TimestampedWordVectors .pt (optional comparison)",
    )
    args = ap.parse_args()

    if not args.wav and not args.text:
        ap.error("Provide --wav and/or --text")

    if args.wav:
        wav_bytes = Path(args.wav).read_bytes()
        wav_np, sr = decode_wav_bytes(wav_bytes, filename_hint=Path(args.wav).name)
        runner = CovarepRunner()
        online_audio = runner.extract_frames(wav_np, sample_rate=sr)
        print("=== Online COVAREP ===")
        print(f"shape: {online_audio.shape}")
        print(f"mean:  {float(online_audio.mean()):.6f}")
        print(f"std:   {float(online_audio.std()):.6f}")

        if args.offline_audio_pt:
            offline_audio = _load_offline_pt(Path(args.offline_audio_pt))
            _compare_arrays("COVAREP online vs offline CSD .pt", online_audio, offline_audio)

    if args.text:
        encoder = GloveEncoder()
        online_text = encoder.encode_text(args.text)
        print("\n=== Online GloVe ===")
        print(f"words: {encoder.tokenize(args.text)}")
        print(f"shape: {online_text.shape}")
        print(f"mean:  {float(online_text.mean()):.6f}")
        print(f"std:   {float(online_text.std()):.6f}")

        if args.offline_text_pt:
            offline_text = _load_offline_pt(Path(args.offline_text_pt))
            _compare_arrays("GloVe online vs offline CSD .pt", online_text, offline_text)


if __name__ == "__main__":
    main()
