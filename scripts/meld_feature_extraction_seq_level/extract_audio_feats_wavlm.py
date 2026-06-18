#!/usr/bin/env python3
"""Extract seq-level WavLM audio features for MELD utterances."""
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/meld_index_full.csv")
    ap.add_argument("--wav_dir", default="data/meld_raw/wav")
    ap.add_argument("--model_name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--max_seconds", type=float, default=10.0)
    ap.add_argument("--out_dir", default="data/meld_features/seq_level/audio")
    return ap.parse_args()


def downsample_mask(mask: torch.Tensor, T_prime: int) -> torch.Tensor:
    L = mask.shape[-1]
    idx = torch.linspace(0, L - 1, steps=T_prime).round().long().clamp_(0, L - 1)
    return mask[..., idx]


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Rows: {len(df)}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    max_samples = int(args.target_sr * args.max_seconds)
    hidden_dim = None
    saved = skipped = missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="WavLM"):
        uid = str(row["uid"])
        out_path = out_dir / f"{uid}.pt"
        if out_path.exists():
            skipped += 1
            continue

        wav_path = wav_dir / f"{uid}.wav"
        if not wav_path.is_file():
            missing += 1
            continue

        wav_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)
        wav = torch.from_numpy(wav_np)

        if sr != args.target_sr:
            import warnings
            warnings.warn(f"{uid}: sr={sr}, expected {args.target_sr}; skipping")
            missing += 1
            continue

        mx = float(wav.abs().max())
        if mx > 0:
            wav = wav / mx
        if wav.numel() == 0:
            missing += 1
            continue
        if wav.numel() > max_samples:
            wav = wav[:max_samples]

        wav_np = wav.cpu().numpy().astype(np.float32)
        inputs = feature_extractor(
            [wav_np], sampling_rate=args.target_sr, return_tensors="pt",
            padding="longest", return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs)
        hs = out.last_hidden_state
        T_prime, H = hs.shape[1], hs.shape[2]
        attn_down = downsample_mask(inputs["attention_mask"], T_prime)
        hs = hs.squeeze(0).cpu()
        attn_down = attn_down.squeeze(0).cpu().long()

        if hidden_dim is None:
            hidden_dim = H

        torch.save({"hidden": hs, "attention_mask": attn_down}, out_path)
        saved += 1

    meta = {
        "source": "WavLM", "model": args.model_name, "hidden_dim": hidden_dim,
        "target_sr": args.target_sr, "max_seconds": args.max_seconds,
        "num_segments": saved + skipped,
        "note": "MELD seq-level WavLM: hidden[T',H] + attention_mask[T']",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} | Skipped {skipped} | Missing WAV {missing}")


if __name__ == "__main__":
    main()
