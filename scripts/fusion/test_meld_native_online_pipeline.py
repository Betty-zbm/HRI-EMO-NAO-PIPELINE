#!/usr/bin/env python3
"""
Online validation for MELD's own checkpoints (4-class + 2-class sentiment).

Unlike training (which used gold "Utterance" transcripts + pre-extracted
offline WavLM/BERT features), this runs the live production-style pipeline:
  raw WAV -> Whisper ASR (transcribe) -> WavLM + BERT (live, via
  IemocapFeatureExtractor) -> trained MELD checkpoint.

This measures the real train/inference gap from (a) using Whisper transcripts
instead of gold transcripts, and (b) live feature extraction instead of
pre-extracted offline features -- same kind of check already done for
IEMOCAP (test_online_pipeline.py) and MOSEI (test_mosei_online_pipeline.py).

Run from repo root:
  python scripts/fusion/test_meld_native_online_pipeline.py --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from server.feature_extraction.iemocap import IemocapFeatureExtractor
from server.feature_extraction.whisper import WhisperTranscriber, decode_wav_bytes

CKPT_4CLS = "runs/meld_4cls_wavlm_bert/best_fusion_seq_decoder.pt"
CKPT_2CLS = "runs/meld_sentiment_wavlm_bert/best_fusion_seq_decoder.pt"

# Offline (gold transcript, pre-extracted features) test-set results for comparison
OFFLINE_4CLS = {"WA": 0.6612, "UA": 0.6063, "macro_F1": 0.5806, "weighted_F1": 0.6734}
OFFLINE_2CLS = {"WA": 0.7433, "UA": 0.7184, "macro_F1": 0.7122, "weighted_F1": 0.7464}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="data/meld_raw/wav")
    ap.add_argument("--index_4cls", default="data/meld_index_full.csv")
    ap.add_argument("--index_2cls", default="data/meld_index_sentiment.csv")
    ap.add_argument("--cache_dir", default="runs/meld_native_online/features_cache")
    ap.add_argument("--out_dir",   default="runs/meld_native_online")
    ap.add_argument("--whisper_size", default="base")
    ap.add_argument("--limit", type=int, default=500, help="Subsample size (0 = all test rows)")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    label2id = ckpt["label2id"]
    model = MoseiFusionWithEmotionDecoder(
        d_audio=768, d_text=768, d_model=a["d_model"],
        num_emotions=len(label2id), n_heads=a["n_heads"],
        num_layers_fusion=a["num_layers_fusion"],
        num_layers_decoder=a["num_layers_decoder"],
        beta_hidden=a["beta_hidden"], dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    max_a = a.get("max_len_audio", 300)
    max_t = a.get("max_len_text", 128)
    merge_labels = a.get("merge_labels", "")
    exclude_labels = a.get("exclude_labels", "")
    return model, label2id, max_a, max_t, merge_labels, exclude_labels, ckpt.get("epoch")


def apply_label_transforms(df: pd.DataFrame, merge_labels: str, exclude_labels: str) -> pd.DataFrame:
    if merge_labels:
        remap = {}
        for pair in merge_labels.split(","):
            src, dst = pair.strip().split("=")
            remap[src.strip()] = dst.strip()
        df = df.copy()
        df["label"] = df["label"].map(lambda x: remap.get(x, x))
    if exclude_labels:
        drop = {s.strip() for s in exclude_labels.split(",") if s.strip()}
        df = df[~df["label"].isin(drop)].reset_index(drop=True)
    return df


def crop(h: torch.Tensor, m: torch.Tensor, max_len: int):
    if h.size(1) > max_len:
        return h[:, :max_len, :], m[:, :max_len]
    return h, m


def compute_metrics(preds, labels):
    wa  = accuracy_score(labels, preds)
    ua  = recall_score(labels, preds, average="macro", zero_division=0)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"WA": wa, "UA": ua, "macro_F1": mf1, "weighted_F1": wf1}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "log.txt", "a", encoding="utf-8")

    def log(msg: object) -> None:
        safe = str(msg).encode("utf-8", errors="replace").decode("utf-8")
        print(safe, flush=True)
        log_f.write(safe + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("=" * 70)
    log(f"Device: {device}")

    has_4cls = Path(CKPT_4CLS).is_file()
    has_2cls = Path(CKPT_2CLS).is_file()
    log(f"4-class checkpoint found: {has_4cls}")
    log(f"2-class checkpoint found: {has_2cls}  (train it first if missing)")

    model4 = model2 = None
    if has_4cls:
        model4, l2id4, maxa4, maxt4, merge4, excl4, ep4 = load_model(CKPT_4CLS, device)
        log(f"Loaded 4-class model (epoch {ep4})  labels={l2id4}  max_a={maxa4}  max_t={maxt4}")
    if has_2cls:
        model2, l2id2, maxa2, maxt2, merge2, excl2, ep2 = load_model(CKPT_2CLS, device)
        log(f"Loaded 2-class model (epoch {ep2})  labels={l2id2}  max_a={maxa2}  max_t={maxt2}")

    if not has_4cls and not has_2cls:
        log("Neither checkpoint exists yet. Exiting.")
        return

    # Build evaluation row set from the 4-class index (test split), since it's a superset
    df_full = pd.read_csv(args.index_4cls)
    test_rows = df_full[df_full["split"] == "test"].reset_index(drop=True)

    df_sent = pd.read_csv(args.index_2cls)
    sent_map = dict(zip(df_sent["uid"], df_sent["label"]))

    if has_4cls:
        test_rows_4 = apply_label_transforms(test_rows.copy(), merge4, excl4)
    if has_2cls:
        test_rows["label_2cls"] = test_rows["uid"].map(sent_map)

    wav_dir = Path(args.wav_dir)
    rows_df = test_rows.copy()
    if args.limit and args.limit > 0 and len(rows_df) > args.limit:
        rows_df = rows_df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
    log(f"Evaluating {len(rows_df)} MELD test clips (of {len(test_rows)} total)")

    log(f"Loading Whisper ({args.whisper_size})...")
    transcriber = WhisperTranscriber(model_size=args.whisper_size)
    log("Loading WavLM + BERT live feature extractor...")
    feat_extractor = IemocapFeatureExtractor(device=str(device))

    results = []
    n_failed = 0

    for _, row in tqdm(rows_df.iterrows(), total=len(rows_df), desc="MELD native online"):
        uid = row["uid"]
        wav_path = wav_dir / f"{uid}.wav"
        cache_path = cache_dir / f"{uid}.pt"

        try:
            if cache_path.is_file():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                transcript = cached["transcript"]
                h_a, m_a = cached["h_a"], cached["m_a"]
                h_t, m_t = cached["h_t"], cached["m_t"]
            else:
                data = wav_path.read_bytes()
                wav_np, sr = decode_wav_bytes(data, filename_hint=wav_path.name)
                transcript = transcriber.transcribe(wav_np, sample_rate=sr)
                if not isinstance(transcript, str):
                    transcript = transcript.text
                audio_feats, text_feats = feat_extractor.extract(wav_np, transcript)
                h_a, m_a = audio_feats.hidden, audio_feats.key_padding_mask
                h_t, m_t = text_feats.hidden, text_feats.key_padding_mask
                torch.save({"transcript": transcript, "h_a": h_a, "m_a": m_a,
                           "h_t": h_t, "m_t": m_t}, cache_path)
        except Exception as e:
            n_failed += 1
            log(f"  [SKIP] {uid}: {e}")
            continue

        rec = {"uid": uid, "transcript": transcript}

        if model4 is not None:
            label = row["label"]
            label_remapped = label
            if merge4:
                remap = dict(p.split("=") for p in merge4.split(","))
                label_remapped = remap.get(label, label)
            if (not excl4) or (label_remapped not in {s.strip() for s in excl4.split(",") if s.strip()}):
                ha, ma = crop(h_a, m_a, maxa4)
                ht, mt = crop(h_t, m_t, maxt4)
                with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
                    logits, _, _ = model4(ha.to(device), ht.to(device), ma.to(device), mt.to(device))
                rec["pred4"] = int(logits.argmax(-1).item())
                rec["true4"] = l2id4[label_remapped]

        if model2 is not None and pd.notna(row.get("label_2cls")):
            ha, ma = crop(h_a, m_a, maxa2)
            ht, mt = crop(h_t, m_t, maxt2)
            with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
                logits, _, _ = model2(ha.to(device), ht.to(device), ma.to(device), mt.to(device))
            rec["pred2"] = int(logits.argmax(-1).item())
            rec["true2"] = l2id2[row["label_2cls"]]

        results.append(rec)

    log(f"\nProcessed: {len(results)}  Failed: {n_failed}")

    if model4 is not None:
        sub = [r for r in results if "pred4" in r]
        if sub:
            preds = [r["pred4"] for r in sub]
            labels = [r["true4"] for r in sub]
            m = compute_metrics(preds, labels)
            log("\n" + "=" * 60)
            log(f"MELD 4-CLASS -- ONLINE (live Whisper+WavLM+BERT) RESULTS  N={len(sub)}")
            log(f"WA={m['WA']:.4f}  UA={m['UA']:.4f}  macro-F1={m['macro_F1']:.4f}  wt-F1={m['weighted_F1']:.4f}")
            log(f"\nOffline (gold transcript, pre-extracted): WA={OFFLINE_4CLS['WA']:.4f}  "
                f"macro-F1={OFFLINE_4CLS['macro_F1']:.4f}  wt-F1={OFFLINE_4CLS['weighted_F1']:.4f}")
            log(f"Gap (online - offline) macro-F1: {m['macro_F1'] - OFFLINE_4CLS['macro_F1']:+.4f}")

    if model2 is not None:
        sub = [r for r in results if "pred2" in r]
        if sub:
            preds = [r["pred2"] for r in sub]
            labels = [r["true2"] for r in sub]
            m = compute_metrics(preds, labels)
            log("\n" + "=" * 60)
            log(f"MELD 2-CLASS SENTIMENT -- ONLINE (live Whisper+WavLM+BERT) RESULTS  N={len(sub)}")
            log(f"WA={m['WA']:.4f}  UA={m['UA']:.4f}  macro-F1={m['macro_F1']:.4f}  wt-F1={m['weighted_F1']:.4f}")
            if OFFLINE_2CLS:
                log(f"\nOffline (gold transcript, pre-extracted): WA={OFFLINE_2CLS['WA']:.4f}  "
                    f"macro-F1={OFFLINE_2CLS['macro_F1']:.4f}  wt-F1={OFFLINE_2CLS['weighted_F1']:.4f}")
                log(f"Gap (online - offline) macro-F1: {m['macro_F1'] - OFFLINE_2CLS['macro_F1']:+.4f}")
            else:
                log("\n(Offline test-set number not filled in yet -- run eval after training finishes)")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved: {out_dir / 'results.json'}")
    log_f.close()


if __name__ == "__main__":
    main()
