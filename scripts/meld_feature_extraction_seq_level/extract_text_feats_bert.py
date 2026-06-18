#!/usr/bin/env python3
"""Extract seq-level BERT text features for MELD utterances (gold transcripts)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/meld_index_full.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="data/meld_features/seq_level/text")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Rows: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    pending_uids, pending_texts = [], []
    skipped = 0
    for _, row in df.iterrows():
        uid = str(row["uid"])
        if (out_dir / f"{uid}.pt").exists():
            skipped += 1
            continue
        text = str(row[args.text_col]) if pd.notna(row[args.text_col]) else ""
        if not text.strip():
            text = "[UNK]"
        pending_uids.append(uid)
        pending_texts.append(text)

    print(f"To process: {len(pending_uids)} | Already done: {skipped}")

    hidden_dim = None
    saved = 0
    B = args.batch_size
    for start in tqdm(range(0, len(pending_uids), B), desc="BERT"):
        batch_uids = pending_uids[start:start + B]
        batch_texts = pending_texts[start:start + B]
        enc = tokenizer(batch_texts, truncation=True, max_length=args.max_len,
                        padding="max_length", return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        hiddens = out.last_hidden_state.cpu()
        masks = enc["attention_mask"].cpu()
        for i, uid in enumerate(batch_uids):
            if hidden_dim is None:
                hidden_dim = int(hiddens[i].size(-1))
            torch.save({"hidden": hiddens[i], "attention_mask": masks[i]},
                      out_dir / f"{uid}.pt")
            saved += 1

    meta = {
        "source": "BERT (gold transcript)", "model": args.model_name,
        "hidden_dim": hidden_dim, "max_len": args.max_len,
        "num_segments": saved + skipped,
        "note": "MELD seq-level BERT on gold Utterance text: hidden[L,H] + attention_mask[L]",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} | Skipped {skipped}")


if __name__ == "__main__":
    main()
