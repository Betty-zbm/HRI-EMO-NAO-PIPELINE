#!/usr/bin/env python3
import sys, torch, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from torch.amp import autocast
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder

EXPS = [
    ("baseline", "runs/iemocap_fusion_seq_decoder_4cls/best_fusion_seq_decoder.pt"),
    ("A  label_smooth", "runs/iemocap_4cls_expA/best_fusion_seq_decoder.pt"),
    ("B  d_model=512",  "runs/iemocap_4cls_expB/best_fusion_seq_decoder.pt"),
    ("C  seed=42",      "runs/iemocap_4cls_expC/best_fusion_seq_decoder.pt"),
]

df_full = pd.read_csv("data/iemocap_index_splits.csv")
df_full["label"] = df_full["label"].map(lambda x: "happy" if x == "excited" else x)
val_df = df_full[(df_full["split"] == "val") & (~df_full["label"].isin(["frustration"]))].reset_index(drop=True)

for tag, ckpt_path in EXPS:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}

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

    audio_dir = Path(args["audio_dir"])
    text_dir  = Path(args["text_dir"])
    max_a = args.get("max_len_audio", 300)
    max_t = args.get("max_len_text", 128)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            uid = str(row["utter_id"])
            af = audio_dir / f"{uid}.pt"
            tf = text_dir  / f"{uid}.pt"
            if not af.is_file() or not tf.is_file():
                continue
            oa = torch.load(af, map_location="cpu", weights_only=True)
            ot = torch.load(tf, map_location="cpu", weights_only=True)
            h_a = oa["hidden"].float(); m_a = (oa["attention_mask"].long() == 0)
            h_t = ot["hidden"].float(); m_t = (ot["attention_mask"].long() == 0)
            if h_a.size(0) > max_a: h_a, m_a = h_a[:max_a], m_a[:max_a]
            if h_t.size(0) > max_t: h_t, m_t = h_t[:max_t], m_t[:max_t]
            h_a = h_a.unsqueeze(0).to(device); m_a = m_a.unsqueeze(0).to(device)
            h_t = h_t.unsqueeze(0).to(device); m_t = m_t.unsqueeze(0).to(device)
            with autocast("cuda"):
                logits, _, _ = model(h_a, h_t, m_a, m_t)
            all_preds.append(logits.argmax(-1).item())
            all_labels.append(label2id[row["label"]])

    names = [id2label[i] for i in sorted(id2label)]
    wa  = accuracy_score(all_labels, all_preds)
    ua  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    ep  = ckpt.get("epoch", "?")
    print(f"=== {tag}  (epoch {ep}, d_model={args['d_model']}, seed={args['seed']}) ===")
    print(f"WA={wa:.4f}  UA={ua:.4f}  macro-F1={mf1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=names, zero_division=0))
