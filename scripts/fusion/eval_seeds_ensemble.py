#!/usr/bin/env python3
"""
Evaluate seed=7777 (best single model) + 7-model selective ensemble.
Usage:
  python scripts/fusion/eval_seeds_ensemble.py
"""
import sys, torch, torch.nn.functional as F, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from torch.amp import autocast
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder

BEST_SINGLE = "runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"

# All 6 seeds + Exp B (different architecture for diversity)
ENSEMBLE_CKPTS = [
    ("seed=42",   "runs/iemocap_4cls_expC/best_fusion_seq_decoder.pt"),
    ("seed=100",  "runs/iemocap_4cls_seed100/best_fusion_seq_decoder.pt"),
    ("seed=999",  "runs/iemocap_4cls_seed999/best_fusion_seq_decoder.pt"),
    ("seed=2024", "runs/iemocap_4cls_seed2024/best_fusion_seq_decoder.pt"),
    ("seed=7777", "runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"),
    ("seed=2025", "runs/iemocap_4cls_seed2025/best_fusion_seq_decoder.pt"),
    ("expB d512", "runs/iemocap_4cls_expB/best_fusion_seq_decoder.pt"),
]

# Selective ensemble: only seeds with WA >= 0.675
SELECT_CKPTS = [
    ("seed=42",   "runs/iemocap_4cls_expC/best_fusion_seq_decoder.pt"),    # WA=67.98%
    ("seed=2024", "runs/iemocap_4cls_seed2024/best_fusion_seq_decoder.pt"),# WA=68.16%
    ("seed=7777", "runs/iemocap_4cls_seed7777/best_fusion_seq_decoder.pt"),# WA=68.91%
    ("seed=2025", "runs/iemocap_4cls_seed2025/best_fusion_seq_decoder.pt"),# WA=67.79%
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("data/iemocap_index_splits.csv")
df["label"] = df["label"].map(lambda x: "happy" if x == "excited" else x)
val_df = df[(df["split"] == "val") & (~df["label"].isin(["frustration"]))].reset_index(drop=True)
print(f"Val set: {len(val_df)} samples\n")


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


def eval_single(path, title):
    m, args, label2id, ep = load_model(path)
    id2label = {v: k for k, v in label2id.items()}
    names = [id2label[i] for i in sorted(id2label)]
    audio_dir = Path(args["audio_dir"]); text_dir = Path(args["text_dir"])
    max_a = args.get("max_len_audio", 300); max_t = args.get("max_len_text", 128)
    preds, labels = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
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
    print(f"=== {title} (epoch={ep}, d_model={args['d_model']}) ===")
    print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}")
    print(classification_report(labels, preds, target_names=names, zero_division=0))
    return id2label, label2id


def eval_ensemble(ckpt_list, title):
    models_data = []
    label2id_ref = None
    for tag, path in ckpt_list:
        m, args, label2id, ep = load_model(path)
        if label2id_ref is None: label2id_ref = label2id
        print(f"  Loaded {tag}  epoch={ep}  d_model={args['d_model']}")
        models_data.append((m, args, label2id))

    id2label = {v: k for k, v in label2id_ref.items()}
    names = [id2label[i] for i in sorted(id2label)]
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            uid = str(row["utter_id"])
            probs_sum = None; skip = False
            for m, args, label2id in models_data:
                audio_dir = Path(args["audio_dir"]); text_dir = Path(args["text_dir"])
                af = audio_dir / f"{uid}.pt"; tf = text_dir / f"{uid}.pt"
                if not af.is_file() or not tf.is_file(): skip = True; break
                oa = torch.load(af, map_location="cpu", weights_only=True)
                ot = torch.load(tf, map_location="cpu", weights_only=True)
                h_a = oa["hidden"].float(); m_a = (oa["attention_mask"].long() == 0)
                h_t = ot["hidden"].float(); m_t = (ot["attention_mask"].long() == 0)
                max_a = args.get("max_len_audio", 300); max_t = args.get("max_len_text", 128)
                if h_a.size(0) > max_a: h_a, m_a = h_a[:max_a], m_a[:max_a]
                if h_t.size(0) > max_t: h_t, m_t = h_t[:max_t], m_t[:max_t]
                h_a = h_a.unsqueeze(0).to(device); m_a = m_a.unsqueeze(0).to(device)
                h_t = h_t.unsqueeze(0).to(device); m_t = m_t.unsqueeze(0).to(device)
                with autocast("cuda"):
                    logits, _, _ = m(h_a, h_t, m_a, m_t)
                probs = F.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            if skip: continue
            all_preds.append(probs_sum.argmax(-1).item())
            all_labels.append(label2id_ref[row["label"]])

    wa = accuracy_score(all_labels, all_preds)
    ua = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\n=== {title} ({len(ckpt_list)} models) ===")
    print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))


print("=" * 60)
print("BEST SINGLE MODEL")
print("=" * 60)
eval_single(BEST_SINGLE, "seed=7777")

print("\n" + "=" * 60)
print("SELECTIVE ENSEMBLE (4 strongest seeds, WA >= 67.79%)")
print("=" * 60)
eval_ensemble(SELECT_CKPTS, "4-seed selective ensemble")

print("\n" + "=" * 60)
print("FULL ENSEMBLE (6 seeds + expB)")
print("=" * 60)
eval_ensemble(ENSEMBLE_CKPTS, "7-model full ensemble")
