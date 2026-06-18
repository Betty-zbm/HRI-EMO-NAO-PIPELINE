#!/usr/bin/env python3
"""
Extract BERT seq-level text features for MOSEI using Whisper ASR.

Pipeline per segment:
  {uid}.wav  ->  Whisper (transcribe)  ->  BERT (encode)  ->  {uid}.pt

Prerequisites:
  pip install openai-whisper transformers tqdm

Run from repo root:
  python scripts/mosei_feature_extraction_seq_level/extract_text_feats_bert_whisper.py \\
      --index_csv data/mosei_index_splits.csv \\
      --seg_dir   data/mosei_wavlm_bert/raw_audio/segments \\
      --out_dir   data/mosei_wavlm_bert/features/text

Optional flags:
  --whisper_model  large-v2        (default: base.en — faster; use large-v2 for best accuracy)
  --bert_model     bert-base-uncased
  --max_len        128
  --batch_size     32              (BERT batch size; Whisper always runs 1-at-a-time)
  --save_transcripts               (also write a transcripts.csv next to meta.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index_csv",        default="data/mosei_index_splits.csv")
    ap.add_argument("--seg_dir",          default="data/mosei_wavlm_bert/raw_audio/segments",
                    help="Directory with {uid}.wav files from download_and_cut_mosei_audio.py")
    ap.add_argument("--out_dir",          default="data/mosei_wavlm_bert/features/text")
    ap.add_argument("--whisper_model",    default="base.en",
                    help="Whisper model size: tiny.en / base.en / small.en / medium.en / large-v2")
    ap.add_argument("--bert_model",       default="bert-base-uncased")
    ap.add_argument("--max_len",          type=int, default=128)
    ap.add_argument("--bert_batch_size",  type=int, default=32,
                    help="Number of transcripts to encode with BERT at once")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--save_transcripts", action="store_true",
                    help="Write transcripts.csv alongside meta.json")
    return ap.parse_args()


# ── Whisper transcription ──────────────────────────────────────────────────────

def load_whisper(model_name: str, device: str):
    try:
        import whisper
    except ImportError:
        raise SystemExit(
            "openai-whisper is not installed.\n"
            "Install with: pip install openai-whisper"
        )
    print(f"Loading Whisper {model_name} ...")
    model = whisper.load_model(model_name, device=device)
    return model


def transcribe(whisper_model, wav_path: Path) -> str:
    import whisper
    audio = whisper.load_audio(str(wav_path))
    result = whisper_model.transcribe(audio, language="en", fp16=torch.cuda.is_available())
    return result["text"].strip()


# ── BERT encoding ──────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(texts: List[str], tokenizer, bert_model,
                 max_len: int, device: torch.device
                 ) -> List[tuple[torch.Tensor, torch.Tensor]]:
    """Encode a batch of texts with BERT. Returns list of (hidden[L,H], mask[L])."""
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = bert_model(**enc)
    hiddens = out.last_hidden_state.cpu()      # [B, L, H]
    masks   = enc["attention_mask"].cpu()      # [B, L]
    return [(hiddens[i], masks[i]) for i in range(len(texts))]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    print(f"Segments: {len(df)}  (splits={args.splits})")

    # Load models
    whisper_model = load_whisper(args.whisper_model, str(device))

    print(f"Loading BERT {args.bert_model} ...")
    tokenizer  = AutoTokenizer.from_pretrained(args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model).to(device).eval()

    hidden_dim: Optional[int] = None
    saved = skipped = missing_wav = empty_text = 0

    # Collect rows that still need processing
    pending_uids:  List[str] = []
    pending_wavs:  List[Path] = []
    for _, row in df.iterrows():
        uid = str(row["uid"])
        out_path = out_dir / f"{uid}.pt"
        if out_path.exists():
            skipped += 1
            continue
        wav_path = seg_dir / f"{uid}.wav"
        if not wav_path.is_file():
            missing_wav += 1
            continue
        pending_uids.append(uid)
        pending_wavs.append(wav_path)

    print(f"To process: {len(pending_uids)} | Already done: {skipped} | Missing WAV: {missing_wav}")

    if not pending_uids:
        print("Nothing to do.")
        return

    # -----------------------------------------------------------------
    # Pass 1: Whisper transcription (sequential — Whisper is stateful)
    # -----------------------------------------------------------------
    print(f"\nStep 1/2: Whisper transcription ({args.whisper_model}) ...")
    uid_to_text: dict[str, str] = {}

    for uid, wav_path in tqdm(zip(pending_uids, pending_wavs),
                               total=len(pending_uids), desc="Whisper"):
        text = transcribe(whisper_model, wav_path)
        if not text:
            empty_text += 1
            text = "[UNK]"
        uid_to_text[uid] = text

    # Free Whisper GPU memory before BERT
    del whisper_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally save transcripts
    if args.save_transcripts:
        transcript_path = out_dir / "transcripts.csv"
        with transcript_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["uid", "text"])
            writer.writeheader()
            for uid in pending_uids:
                writer.writerow({"uid": uid, "text": uid_to_text[uid]})
        print(f"Transcripts saved to {transcript_path}")

    # -----------------------------------------------------------------
    # Pass 2: BERT encoding (batched)
    # -----------------------------------------------------------------
    print(f"\nStep 2/2: BERT encoding ({args.bert_model}, batch={args.bert_batch_size}) ...")
    B = args.bert_batch_size

    for start in tqdm(range(0, len(pending_uids), B), desc="BERT"):
        batch_uids  = pending_uids[start:start + B]
        batch_texts = [uid_to_text[u] for u in batch_uids]
        results     = encode_batch(batch_texts, tokenizer, bert_model,
                                   args.max_len, device)
        for uid, (hidden, mask) in zip(batch_uids, results):
            if hidden_dim is None:
                hidden_dim = int(hidden.size(-1))
            torch.save({"hidden": hidden, "attention_mask": mask},
                       out_dir / f"{uid}.pt")
            saved += 1

    meta = {
        "source": f"Whisper-{args.whisper_model} + {args.bert_model}",
        "whisper_model": args.whisper_model,
        "bert_model": args.bert_model,
        "hidden_dim": hidden_dim,
        "max_len": args.max_len,
        "num_segments": saved + skipped,
        "note": "seq-level BERT on Whisper transcripts: hidden[L,H] + attention_mask[L]",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} | Skipped {skipped} | Missing WAV {missing_wav} | Empty text {empty_text}")
    print(f"Output: {out_dir}")
    if missing_wav:
        print(f"  {missing_wav} segments had no WAV — re-run download_and_cut_mosei_audio.py first")


if __name__ == "__main__":
    main()
