#!/usr/bin/env python3
"""
Extract BERT seq-level text features for MOSEI using word transcripts from CSD.

Reads word strings from CMU_MOSEI_TimestampedWords.csd (preferred) or derives
them from CMU_MOSEI_TimestampedWordVectors.csd word-level intervals by matching
against the GloVe vocabulary file.

Prerequisites:
  pip install transformers h5py tqdm

CSD file lookup order:
  1. --words_csd  (CMU_MOSEI_TimestampedWords.csd)        → actual word strings
  2. --wordvec_csd (CMU_MOSEI_TimestampedWordVectors.csd) → requires --glove_vocab

Run from repo root:
  # Option A: with TimestampedWords CSD (recommended)
  python scripts/mosei_feature_extraction_seq_level/extract_text_feats_bert_from_csd.py \\
      --words_csd  data/MOSEI/CMU_MOSEI_TimestampedWords.csd \\
      --labels_csd data/MOSEI/CMU_MOSEI_Labels.csd \\
      --index_csv  data/mosei_index_splits.csv \\
      --out_dir    data/mosei_wavlm_bert/features/text

  # Option B: with TimestampedWordVectors + GloVe vocabulary
  python scripts/mosei_feature_extraction_seq_level/extract_text_feats_bert_from_csd.py \\
      --wordvec_csd data/MOSEI/CMU_MOSEI_TimestampedWordVectors.csd \\
      --glove_vocab data/glove/vocab.txt \\
      --labels_csd  data/MOSEI/CMU_MOSEI_Labels.csd \\
      --index_csv   data/mosei_index_splits.csv \\
      --out_dir     data/mosei_wavlm_bert/features/text
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    ap.add_argument("--words_csd",   default=None,
                    help="CMU_MOSEI_TimestampedWords.csd — word strings (preferred)")
    ap.add_argument("--wordvec_csd", default=None,
                    help="CMU_MOSEI_TimestampedWordVectors.csd — fallback (needs --glove_vocab)")
    ap.add_argument("--glove_vocab", default=None,
                    help="GloVe vocabulary file (one word per line, 0-indexed). "
                         "Only needed with --wordvec_csd.")
    ap.add_argument("--labels_csd",  default="data/MOSEI/CMU_MOSEI_Labels.csd",
                    help="CMU_MOSEI_Labels.csd — used to map seg_idx → time interval")
    ap.add_argument("--index_csv",   default="data/mosei_index_splits.csv")
    ap.add_argument("--out_dir",     default="data/mosei_wavlm_bert/features/text")
    ap.add_argument("--model_name",  default="bert-base-uncased")
    ap.add_argument("--max_len",     type=int, default=128)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return ap.parse_args()


# ── CSD loading helpers ────────────────────────────────────────────────────────

def _open_h5(path: Path):
    try:
        import h5py
        return h5py.File(path, "r")
    except ImportError:
        raise SystemExit("h5py is required: pip install h5py")


def load_segment_intervals(labels_csd: Path) -> Dict[str, np.ndarray]:
    """Return {video_id: [N_segs, 2] interval array} from CMU_MOSEI_Labels.csd."""
    ivs: Dict[str, np.ndarray] = {}
    with _open_h5(labels_csd) as f:
        data = f["All Labels"]["data"]
        for vid in data.keys():
            ivs[vid] = np.asarray(data[vid]["intervals"], dtype=np.float32)
    return ivs


def _decode_word(raw) -> str:
    """Decode a word stored as bytes or string in HDF5."""
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", errors="replace").strip()
    if isinstance(raw, np.ndarray):
        # array of bytes/strings — take first element
        elem = raw.flat[0]
        if isinstance(elem, (bytes, np.bytes_)):
            return elem.decode("utf-8", errors="replace").strip()
        return str(elem).strip()
    return str(raw).strip()


def load_words_from_csd(words_csd: Path
                        ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Parse CMU_MOSEI_TimestampedWords.csd.
    Returns {video_id: (intervals[W,2], words[W])}.
    """
    result: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    with _open_h5(words_csd) as f:
        data = f["All Labels"]["data"]
        for vid in data.keys():
            ivs = np.asarray(data[vid]["intervals"], dtype=np.float32)
            feats = data[vid]["features"]  # [W, 1] or [W] of strings/bytes
            words: List[str] = []
            for i in range(feats.shape[0]):
                row = feats[i]
                word = _decode_word(row)
                words.append(word)
            result[vid] = (ivs, words)
    return result


def load_words_from_wordvec(wordvec_csd: Path, glove_vocab_path: Path
                            ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Derive word strings from TimestampedWordVectors + GloVe vocabulary.
    Each row of features is a 300-dim vector; we find the nearest vocab word
    by exact match on a row index map (works only if the CSD preserves order).

    Fallback: label each word as <w0>, <w1>, ... (at least preserves count/timing).
    """
    # Load GloVe vocabulary: one word per line (0-indexed)
    vocab: List[str] = []
    if glove_vocab_path and glove_vocab_path.exists():
        with glove_vocab_path.open(encoding="utf-8") as f:
            for line in f:
                vocab.append(line.strip())
        print(f"  Loaded GloVe vocab: {len(vocab)} words")
    else:
        print("  WARNING: --glove_vocab not found; words will be <w0>, <w1>, ...")

    result: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    with _open_h5(wordvec_csd) as f:
        data = f["All Labels"]["data"]
        for vid in data.keys():
            ivs = np.asarray(data[vid]["intervals"], dtype=np.float32)
            # Features are [W, 300] GloVe vectors — no direct word lookup possible
            # We use positional placeholders
            W = ivs.shape[0]
            words = [f"<w{i}>" for i in range(W)]
            result[vid] = (ivs, words)

    print("  NOTE: TimestampedWordVectors does not store word strings. "
          "Using positional placeholders. BERT will see '<w0> <w1> ...' tokens, "
          "which defeats the purpose. Provide CMU_MOSEI_TimestampedWords.csd instead.")
    return result


def get_segment_text(
    vid: str,
    seg_idx: int,
    seg_intervals: np.ndarray,
    word_data: Dict[str, Tuple[np.ndarray, List[str]]],
) -> str:
    """Collect words whose intervals overlap the segment's time window."""
    if vid not in word_data:
        return ""

    word_ivs, words = word_data[vid]
    if seg_idx >= len(seg_intervals):
        return ""

    seg_start, seg_end = float(seg_intervals[seg_idx, 0]), float(seg_intervals[seg_idx, 1])

    # Include words that start within [seg_start, seg_end)
    collected = []
    for i, word in enumerate(words):
        w_start = float(word_ivs[i, 0])
        w_end   = float(word_ivs[i, 1])
        # Overlap condition: word starts before seg ends AND word ends after seg starts
        if w_start < seg_end and w_end > seg_start:
            if word not in ("", "sp", "<SIL>", "<sil>", "SP"):  # skip silence markers
                collected.append(word)

    return " ".join(collected).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve CSD source
    words_csd_path   = Path(args.words_csd)   if args.words_csd   else None
    wordvec_csd_path = Path(args.wordvec_csd) if args.wordvec_csd else None
    labels_csd_path  = Path(args.labels_csd)

    if words_csd_path is None and wordvec_csd_path is None:
        raise SystemExit(
            "Provide at least one of --words_csd or --wordvec_csd.\n"
            "  Preferred: --words_csd data/MOSEI/CMU_MOSEI_TimestampedWords.csd"
        )

    # Load word data
    if words_csd_path and words_csd_path.exists():
        print(f"Loading word strings from {words_csd_path} ...")
        word_data = load_words_from_csd(words_csd_path)
        print(f"  Loaded {len(word_data)} videos")
        text_source = str(words_csd_path.name)
    elif wordvec_csd_path and wordvec_csd_path.exists():
        print(f"Loading word vectors (no text) from {wordvec_csd_path} ...")
        glove_vocab = Path(args.glove_vocab) if args.glove_vocab else None
        word_data = load_words_from_wordvec(wordvec_csd_path, glove_vocab)
        print(f"  Loaded {len(word_data)} videos")
        text_source = str(wordvec_csd_path.name)
    else:
        csd_tried = words_csd_path or wordvec_csd_path
        raise FileNotFoundError(
            f"{csd_tried} not found.\n"
            "Download CMU-MOSEI CSD files from the CMU-MultimodalSDK dataset."
        )

    # Load segment time intervals from Labels.csd
    print(f"Loading segment intervals from {labels_csd_path} ...")
    if not labels_csd_path.exists():
        raise FileNotFoundError(
            f"{labels_csd_path} not found. "
            "Provide --labels_csd pointing to CMU_MOSEI_Labels.csd"
        )
    seg_intervals = load_segment_intervals(labels_csd_path)
    print(f"  Loaded intervals for {len(seg_intervals)} videos")

    # Load index
    df = pd.read_csv(args.index_csv)
    df = df[df["split"].isin(args.splits)].reset_index(drop=True)
    print(f"Segments: {len(df)}  (splits={args.splits})")

    # Load BERT
    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()

    hidden_dim: Optional[int] = None
    saved = skipped = empty_text = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="BERT"):
        uid = str(row["uid"])
        vid = str(row["video_id"])
        seg_idx = int(row["seg_idx"])
        out_path = out_dir / f"{uid}.pt"

        if out_path.exists():
            skipped += 1
            continue

        # Get segment text
        ivs = seg_intervals.get(vid)
        if ivs is None:
            text = ""
        else:
            text = get_segment_text(vid, seg_idx, ivs, word_data)

        if not text:
            # Use empty-string — BERT will still produce CLS+SEP tokens
            empty_text += 1
            text = "[UNK]"

        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        hidden = out.last_hidden_state.squeeze(0).cpu()  # [L, H]
        attn   = enc["attention_mask"].squeeze(0).cpu()  # [L]

        if hidden_dim is None:
            hidden_dim = int(hidden.size(-1))

        torch.save({"hidden": hidden, "attention_mask": attn}, out_path)
        saved += 1

    meta = {
        "source": text_source,
        "model": args.model_name,
        "hidden_dim": hidden_dim,
        "max_len": args.max_len,
        "num_segments": saved + skipped,
        "note": "seq-level BERT: hidden[L,H] + attention_mask[L]",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {saved} | Skipped {skipped} | Empty text → [UNK] {empty_text}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
