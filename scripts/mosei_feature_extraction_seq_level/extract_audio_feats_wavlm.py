#!/usr/bin/env python3
"""
Extract WavLM seq-level audio features for MOSEI segments.

Prerequisites:
  pip install transformers torchaudio tqdm

Expects per-segment WAV files named {uid}.wav produced by
download_and_cut_mosei_audio.py.

Run from repo root:
  python scripts/mosei_feature_extraction_seq_level/extract_audio_feats_wavlm.py \\
      --index_csv data/mosei_index_splits.csv \\
      --seg_dir   data/mosei_wavlm_bert/raw_audio/segments \\
      --out_dir   data/mosei_wavlm_bert/features/audio \\
      --model_name microsoft/wavlm-base-plus
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index_csv",   default="data/mosei_index_splits.csv")
    ap.add_argument("--seg_dir",     default="data/mosei_wavlm_bert/raw_audio/segments",
                    help="Directory with {uid}.wav files")
    ap.add_argument("--out_dir",     default="data/mosei_wavlm_bert/features/audio")
    ap.add_argument("--model_name",  default="microsoft/wavlm-base-plus")
    ap.add_argument("--target_sr",   type=int, default=16000)
    ap.add_argument("--max_seconds", type=float, default=15.0,
                    help="Truncate audio to this length (MOSEI segments can be ~10-15s)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch_size",  type=int, default=1,
                    help="Samples per forward pass (1 is safe for variable lengths)")
    return ap.parse_args()


def downsample_mask(mask: torch.Tensor, T_prime: int) -> torch.Tensor:
    """Linearly downsample attention mask from raw-frame length to WavLM output length."""
    L = mask.shape[-1]
    idx = torch.linspace(0, L - 1, steps=T_prime).round().long().clamp_(0, L - 1)
    return mask[..., idx]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    print(f"Rows: {len(df)}  (splits={args.splits})")

    print(f"Loading model {args.model_name} ...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    max_samples = int(args.target_sr * args.max_seconds)
    hidden_dim: int | None = None
    saved = skipped = missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="WavLM"):
        uid = str(row["uid"])
        out_path = out_dir / f"{uid}.pt"

        if out_path.exists():
            skipped += 1
            continue

        wav_path = seg_dir / f"{uid}.wav"
        if not wav_path.is_file():
            missing += 1
            continue

        # Load and preprocess (soundfile avoids torchcodec DLL issues on Windows)
        wav_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)  # stereo → mono
        wav = torch.from_numpy(wav_np)  # [T]

        if sr != args.target_sr:
            # ffmpeg cuts all segments to target_sr — this path should not be reached
            import warnings
            warnings.warn(f"{uid}: sr={sr}, expected {args.target_sr}; skipping")
            missing += 1
            continue

        # Normalise
        mx = float(wav.abs().max())
        if mx > 0:
            wav = wav / mx

        # Truncate (don't pad — feature_extractor handles padding per batch)
        if wav.numel() > max_samples:
            wav = wav[:max_samples]

        wav_np = wav.cpu().numpy().astype(np.float32)

        inputs = feature_extractor(
            [wav_np],
            sampling_rate=args.target_sr,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs)
        hs = out.last_hidden_state  # [1, T', H]
        T_prime, H = hs.shape[1], hs.shape[2]

        attn_down = downsample_mask(inputs["attention_mask"], T_prime)  # [1, T']
        hs = hs.squeeze(0).cpu()          # [T', H]
        attn_down = attn_down.squeeze(0).cpu().long()  # [T']

        if hidden_dim is None:
            hidden_dim = H

        torch.save({"hidden": hs, "attention_mask": attn_down}, out_path)
        saved += 1

    meta = {
        "source": "WavLM",
        "model": args.model_name,
        "hidden_dim": hidden_dim,
        "target_sr": args.target_sr,
        "max_seconds": args.max_seconds,
        "num_segments": saved + skipped,
        "note": "seq-level WavLM: hidden[T',H] + attention_mask[T']",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} | Skipped {skipped} (already done) | Missing WAV {missing}")
    print(f"Output: {out_dir}")
    if missing:
        print(f"  {missing} segments had no WAV file — re-run download_and_cut_mosei_audio.py")


if __name__ == "__main__":
    main()
