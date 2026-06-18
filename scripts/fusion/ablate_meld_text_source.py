#!/usr/bin/env python3
"""
Ablation: decompose the MELD online-vs-offline gap into
  (1) live audio-extraction-pipeline differences, and
  (2) Whisper-transcript-imprecision differences.

Reuses the cached live-extracted audio features (h_a, m_a) from
test_meld_native_online_pipeline.py's run (same 500 clips), and swaps in
the GOLD transcript instead of the cached Whisper transcript -- re-encoding
text with the exact same live BERT path (IemocapFeatureExtractor.extract_text).

Three conditions compared:
  (A) Offline = gold text + offline-extracted audio        [already measured]
  (B) Hybrid  = gold text + LIVE-extracted audio            [this script]
  (C) Online  = Whisper text + LIVE-extracted audio         [already measured]

Gap(B-A) = audio-pipeline-difference contribution
Gap(C-B) = Whisper-transcript-imprecision contribution

Run from repo root:
  python scripts/fusion/ablate_meld_text_source.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder
from server.feature_extraction.iemocap import IemocapFeatureExtractor

CKPT_4CLS = "runs/meld_4cls_wavlm_bert/best_fusion_seq_decoder.pt"
CKPT_2CLS = "runs/meld_sentiment_wavlm_bert/best_fusion_seq_decoder.pt"
CACHE_DIR = Path("runs/meld_native_online/features_cache")
INDEX_4CLS = "data/meld_index_full.csv"
INDEX_2CLS = "data/meld_index_sentiment.csv"

# Already-measured reference points (offline + online), for the 3-way table
OFFLINE_4CLS = {"WA": 0.6612, "macro_F1": 0.5806, "weighted_F1": 0.6734}
ONLINE_4CLS  = {"WA": 0.5619, "macro_F1": 0.4773, "weighted_F1": 0.5685}
OFFLINE_2CLS = {"WA": 0.7433, "macro_F1": 0.7122, "weighted_F1": 0.7464}
ONLINE_2CLS  = {"WA": 0.7280, "macro_F1": 0.6905, "weighted_F1": 0.7276}


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
    merge_labels = a.get("merge_labels", "")
    exclude_labels = a.get("exclude_labels", "")
    return model, label2id, a.get("max_len_audio", 300), a.get("max_len_text", 128), merge_labels, exclude_labels


def apply_transforms(label: str, merge_labels: str, exclude_labels: str):
    if merge_labels:
        remap = dict(p.split("=") for p in merge_labels.split(","))
        label = remap.get(label, label)
    if exclude_labels:
        drop = {s.strip() for s in exclude_labels.split(",") if s.strip()}
        if label in drop:
            return None
    return label


def crop(h, m, max_len):
    if h.size(1) > max_len:
        return h[:, :max_len, :], m[:, :max_len]
    return h, m


def compute_metrics(preds, labels):
    return {
        "WA": accuracy_score(labels, preds),
        "UA": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_F1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_F1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model4, l2id4, maxa4, maxt4, merge4, excl4 = load_model(CKPT_4CLS, device)
    model2, l2id2, maxa2, maxt2, merge2, excl2 = load_model(CKPT_2CLS, device)

    df4 = pd.read_csv(INDEX_4CLS).set_index("uid")
    df2 = pd.read_csv(INDEX_2CLS).set_index("uid")

    feat_extractor = IemocapFeatureExtractor(device=str(device))

    cache_files = sorted(CACHE_DIR.glob("*.pt"))
    print(f"Cached clips available: {len(cache_files)}")

    preds4, labels4 = [], []
    preds2, labels2 = [], []

    for f in cache_files:
        uid = f.stem
        cached = torch.load(f, map_location="cpu", weights_only=False)
        h_a, m_a = cached["h_a"], cached["m_a"]  # live-extracted audio (unchanged)

        gold_text = str(df4.loc[uid, "text"]) if uid in df4.index else ""
        if not gold_text.strip():
            gold_text = "[UNK]"
        text_feats = feat_extractor.extract_text(gold_text)
        h_t, m_t = text_feats.hidden, text_feats.key_padding_mask

        # 4-class
        if uid in df4.index:
            raw_label = df4.loc[uid, "label"]
            label = apply_transforms(raw_label, merge4, excl4)
            if label is not None and label in l2id4:
                ha, ma = crop(h_a, m_a, maxa4)
                ht, mt = crop(h_t, m_t, maxt4)
                with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
                    logits, _, _ = model4(ha.to(device), ht.to(device), ma.to(device), mt.to(device))
                preds4.append(int(logits.argmax(-1).item()))
                labels4.append(l2id4[label])

        # 2-class
        if uid in df2.index:
            label2 = df2.loc[uid, "label"]
            ha, ma = crop(h_a, m_a, maxa2)
            ht, mt = crop(h_t, m_t, maxt2)
            with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
                logits, _, _ = model2(ha.to(device), ht.to(device), ma.to(device), mt.to(device))
            preds2.append(int(logits.argmax(-1).item()))
            labels2.append(l2id2[label2])

    m4 = compute_metrics(preds4, labels4)
    m2 = compute_metrics(preds2, labels2)

    print("\n" + "=" * 70)
    print(f"MELD 4-CLASS  -- HYBRID (gold text + LIVE audio)  N={len(preds4)}")
    print(f"WA={m4['WA']:.4f}  macro-F1={m4['macro_F1']:.4f}  wt-F1={m4['weighted_F1']:.4f}")

    print(f"\n3-way decomposition (macro-F1):")
    print(f"  (A) Offline (gold text + offline audio): {OFFLINE_4CLS['macro_F1']:.4f}")
    print(f"  (B) Hybrid  (gold text + live audio):     {m4['macro_F1']:.4f}")
    print(f"  (C) Online  (Whisper text + live audio):  {ONLINE_4CLS['macro_F1']:.4f}")
    print(f"  Gap(B-A) audio-pipeline contribution:     {m4['macro_F1'] - OFFLINE_4CLS['macro_F1']:+.4f}")
    print(f"  Gap(C-B) Whisper-text contribution:       {ONLINE_4CLS['macro_F1'] - m4['macro_F1']:+.4f}")

    print("\n" + "=" * 70)
    print(f"MELD 2-CLASS SENTIMENT -- HYBRID (gold text + LIVE audio)  N={len(preds2)}")
    print(f"WA={m2['WA']:.4f}  macro-F1={m2['macro_F1']:.4f}  wt-F1={m2['weighted_F1']:.4f}")

    print(f"\n3-way decomposition (macro-F1):")
    print(f"  (A) Offline (gold text + offline audio): {OFFLINE_2CLS['macro_F1']:.4f}")
    print(f"  (B) Hybrid  (gold text + live audio):     {m2['macro_F1']:.4f}")
    print(f"  (C) Online  (Whisper text + live audio):  {ONLINE_2CLS['macro_F1']:.4f}")
    print(f"  Gap(B-A) audio-pipeline contribution:     {m2['macro_F1'] - OFFLINE_2CLS['macro_F1']:+.4f}")
    print(f"  Gap(C-B) Whisper-text contribution:       {ONLINE_2CLS['macro_F1'] - m2['macro_F1']:+.4f}")

    with open("runs/meld_native_online/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({"4cls_hybrid": m4, "2cls_hybrid": m2}, f, indent=2)


if __name__ == "__main__":
    main()
