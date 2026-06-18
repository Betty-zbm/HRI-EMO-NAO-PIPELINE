#!/usr/bin/env python3
"""Evaluate the MELD 2-class sentiment WavLM+BERT checkpoint on the held-out MELD test split."""
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder

CKPT = "runs/meld_sentiment_wavlm_bert/best_fusion_seq_decoder.pt"

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
args = ckpt["args"]
label2id = ckpt["label2id"]
id2label = {v: k for k, v in label2id.items()}
print("Saved at epoch:", ckpt.get("epoch"), "  val macro-F1:", round(ckpt.get("macro_F1"), 4))
print("Labels:", label2id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoseiFusionWithEmotionDecoder(
    d_audio=768, d_text=768, d_model=args["d_model"],
    num_emotions=len(label2id), n_heads=args["n_heads"],
    num_layers_fusion=args["num_layers_fusion"],
    num_layers_decoder=args["num_layers_decoder"],
    beta_hidden=args["beta_hidden"], dropout=args["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

df = pd.read_csv(args["csv"])
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"\nTest set: {len(test_df)} samples")
print(test_df["label"].value_counts().to_string())

audio_dir = Path(args["audio_dir"])
text_dir  = Path(args["text_dir"])
max_a = args.get("max_len_audio", 300)
max_t = args.get("max_len_text", 128)

all_preds, all_labels = [], []
missing = 0
with torch.no_grad():
    for _, row in test_df.iterrows():
        uid = str(row["uid"])
        af = audio_dir / f"{uid}.pt"
        tf = text_dir  / f"{uid}.pt"
        if not af.is_file() or not tf.is_file():
            missing += 1
            continue
        oa = torch.load(af, map_location="cpu", weights_only=True)
        ot = torch.load(tf, map_location="cpu", weights_only=True)
        h_a = oa["hidden"].float()
        m_a = (oa["attention_mask"].long() == 0)
        h_t = ot["hidden"].float()
        m_t = (ot["attention_mask"].long() == 0)
        if h_a.size(0) > max_a:
            h_a, m_a = h_a[:max_a], m_a[:max_a]
        if h_t.size(0) > max_t:
            h_t, m_t = h_t[:max_t], m_t[:max_t]
        h_a = h_a.unsqueeze(0).to(device)
        m_a = m_a.unsqueeze(0).to(device)
        h_t = h_t.unsqueeze(0).to(device)
        m_t = m_t.unsqueeze(0).to(device)
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, _, _ = model(h_a, h_t, m_a, m_t)
        all_preds.append(logits.argmax(-1).item())
        all_labels.append(label2id[row["label"]])

print(f"\nEvaluated: {len(all_preds)}  |  Missing features: {missing}")

names = [id2label[i] for i in sorted(id2label)]
wa  = accuracy_score(all_labels, all_preds)
ua  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\nWA          = {wa:.4f}")
print(f"UA          = {ua:.4f}")
print(f"macro-F1    = {mf1:.4f}")
print(f"weighted-F1 = {wf1:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))
