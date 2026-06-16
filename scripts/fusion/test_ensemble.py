#!/usr/bin/env python3
"""
Test-set evaluation: 4-seed selective ensemble + best single model (seed=7777).
Usage:
  python scripts/fusion/test_ensemble.py
"""
import sys, torch, torch.nn.functional as F, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from torch.amp import autocast
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder

ENSEMBLE_CKPTS = [
    ("seed=42",   "runs/iemocap_4cls_expC/best_fusion_seq_decoder.pt"),
    ("seed=2024", "runs/iemocap_4cls_seed2024/best_fusion_seq_decoder.pt"),
    ("seed=7777", "runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"),
    ("seed=2025", "runs/iemocap_4cls_seed2025/best_fusion_seq_decoder.pt"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("data/iemocap_index_splits.csv")
df["label"] = df["label"].map(lambda x: "happy" if x == "excited" else x)
test_df = df[(df["split"] == "test") & (~df["label"].isin(["frustration"]))].reset_index(drop=True)
print(f"Test set: {len(test_df)} samples")
print(test_df["label"].value_counts().to_string())
print()


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    label2id = ckpt["label2id"]
    m = MoseiFusionWithEmotionDecoder(
        d_audio=768, d_text=768, d_model=args["d_model"],
        num_emotions=len(label2id), n_heads=args["n_heads"],
        num_layers_fusion=args["num_layers_fusion"],
        num_layers_decoder=args["num_layers_decoder"],
        beta_hidden=args["beta_hidden"], dropout=args["dropout"],
    ).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, args, label2id, ckpt.get("epoch", "?")


# --- Best single model ---
m, args, label2id, ep = load_model(ENSEMBLE_CKPTS[2][1])  # seed=7777
id2label = {v: k for k, v in label2id.items()}
names = [id2label[i] for i in sorted(id2label)]
audio_dir = Path(args["audio_dir"]); text_dir = Path(args["text_dir"])
max_a = args.get("max_len_audio", 300); max_t = args.get("max_len_text", 128)
preds, labels = [], []
with torch.no_grad():
    for _, row in test_df.iterrows():
        uid = str(row["utter_id"])
        af = audio_dir / f"{uid}.pt"; tf = text_dir / f"{uid}.pt"
        if not af.is_file() or not tf.is_file(): continue
        oa = torch.load(af, map_location="cpu", weights_only=True)
        ot = torch.load(tf, map_location="cpu", weights_only=True)
        h_a = oa["hidden"].float(); m_a = (oa["attention_mask"].long() == 0)
        h_t = ot["hidden"].float(); m_t = (ot["attention_mask"].long() == 0)
        if h_a.size(0) > max_a: h_a, m_a = h_a[:max_a], m_a[:max_a]
        if h_t.size(0) > max_t: h_t, m_t = h_t[:max_t], m_t[:max_t]
        h_a = h_a.unsqueeze(0).to(device); m_a = m_a.unsqueeze(0).to(device)
        h_t = h_t.unsqueeze(0).to(device); m_t = m_t.unsqueeze(0).to(device)
        with autocast("cuda"):
            logits, _, _ = m(h_a, h_t, m_a, m_t)
        preds.append(logits.argmax(-1).item())
        labels.append(label2id[row["label"]])

wa = accuracy_score(labels, preds)
ua = recall_score(labels, preds, average="macro", zero_division=0)
mf1 = f1_score(labels, preds, average="macro", zero_division=0)
print(f"=== Best single model: seed=7777 (epoch={ep}) ===")
print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}")
print(classification_report(labels, preds, target_names=names, zero_division=0))


# --- 4-seed selective ensemble ---
models_data = []
label2id_ref = None
for tag, path in ENSEMBLE_CKPTS:
    m2, args2, l2id, ep2 = load_model(path)
    if label2id_ref is None: label2id_ref = l2id
    print(f"  Loaded {tag}  epoch={ep2}")
    models_data.append((m2, args2, l2id))

all_preds, all_labels = [], []
with torch.no_grad():
    for _, row in test_df.iterrows():
        uid = str(row["utter_id"])
        probs_sum = None; skip = False
        for m2, args2, l2id in models_data:
            af = Path(args2["audio_dir"]) / f"{uid}.pt"
            tf = Path(args2["text_dir"]) / f"{uid}.pt"
            if not af.is_file() or not tf.is_file(): skip = True; break
            oa = torch.load(af, map_location="cpu", weights_only=True)
            ot = torch.load(tf, map_location="cpu", weights_only=True)
            h_a = oa["hidden"].float(); m_a = (oa["attention_mask"].long() == 0)
            h_t = ot["hidden"].float(); m_t = (ot["attention_mask"].long() == 0)
            ma2 = args2.get("max_len_audio", 300); mt2 = args2.get("max_len_text", 128)
            if h_a.size(0) > ma2: h_a, m_a = h_a[:ma2], m_a[:ma2]
            if h_t.size(0) > mt2: h_t, m_t = h_t[:mt2], m_t[:mt2]
            h_a = h_a.unsqueeze(0).to(device); m_a = m_a.unsqueeze(0).to(device)
            h_t = h_t.unsqueeze(0).to(device); m_t = m_t.unsqueeze(0).to(device)
            with autocast("cuda"):
                logits, _, _ = m2(h_a, h_t, m_a, m_t)
            probs = F.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        if skip: continue
        all_preds.append(probs_sum.argmax(-1).item())
        all_labels.append(label2id_ref[row["label"]])

id2label_r = {v: k for k, v in label2id_ref.items()}
names_r = [id2label_r[i] for i in sorted(id2label_r)]
wa = accuracy_score(all_labels, all_preds)
ua = recall_score(all_labels, all_preds, average="macro", zero_division=0)
mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
print(f"\n=== 4-seed selective ensemble ===")
print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}")
print(classification_report(all_labels, all_preds, target_names=names_r, zero_division=0))
