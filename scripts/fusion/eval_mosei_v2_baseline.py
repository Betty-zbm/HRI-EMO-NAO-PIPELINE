#!/usr/bin/env python3
"""
Evaluate runs/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt
on val and test splits, using the same metrics as train_mosei_6cls.py.

Reconstructs the model exactly from checkpoint args, applies the
saved calibrated thresholds, and prints a full per-class report.

Usage (from repo root):
  python scripts/fusion/eval_mosei_v2_baseline.py
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder

CKPT_PATH = "runs/mosei_fusion_decoder_v2/best_mosei_fusion_decoder.pt"

# Correct data paths on this machine
AUDIO_DIR = Path("data/mosei_features_for_training/features/mosei/seq_level/audio")
TEXT_DIR  = Path("data/mosei_features_for_training/features/mosei/seq_level/text")
INDEX_CSV = "data/mosei_index_splits.csv"


# ── Dataset (same as train_mosei_6cls.py) ─────────────────────────────────────
def crop_center(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if x.size(0) <= max_len:
        return x
    start = (x.size(0) - max_len) // 2
    return x[start:start + max_len]


class MoseiSeqDataset(Dataset):
    def __init__(self, df, audio_dir, text_dir, uid_col, emo_cols,
                 max_len_audio, max_len_text):
        self.audio_dir     = audio_dir
        self.text_dir      = text_dir
        self.uid_col       = uid_col
        self.emo_cols      = emo_cols
        self.max_len_audio = max_len_audio
        self.max_len_text  = max_len_text
        keep, missing = [], 0
        for i, row in df.reset_index(drop=True).iterrows():
            uid = str(row[uid_col])
            if (audio_dir / f"{uid}.pt").is_file() and (text_dir / f"{uid}.pt").is_file():
                keep.append(i)
            else:
                missing += 1
        if missing:
            print(f"  skipped {missing} rows missing features")
        self.df = df.reset_index(drop=True).iloc[keep].reset_index(drop=True)
        print(f"  {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        h = obj["hidden"].float()
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        m = obj.get("attention_mask", torch.ones(h.size(0), dtype=torch.long))
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, dtype=torch.long)
        return h, (m == 0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])
        h_a, m_a = self._load(self.audio_dir / f"{uid}.pt")
        h_t, m_t = self._load(self.text_dir  / f"{uid}.pt")
        if self.max_len_audio > 0:
            h_a = crop_center(h_a, self.max_len_audio)
            m_a = torch.zeros(h_a.size(0), dtype=torch.bool)
        if self.max_len_text > 0:
            h_t = crop_center(h_t, self.max_len_text)
            m_t = torch.zeros(h_t.size(0), dtype=torch.bool)
        y = torch.tensor([float(row[c]) for c in self.emo_cols], dtype=torch.float32)
        return h_a, m_a, h_t, m_t, torch.nan_to_num(y, nan=0.0)


def collate_fn(batch):
    hs_a, ms_a, hs_t, ms_t, ys = zip(*batch)
    B, d_a, d_t = len(batch), hs_a[0].size(-1), hs_t[0].size(-1)
    La = max(x.size(0) for x in hs_a)
    Lt = max(x.size(0) for x in hs_t)
    pad_ha = torch.zeros(B, La, d_a); pad_ma = torch.ones(B, La, dtype=torch.bool)
    pad_ht = torch.zeros(B, Lt, d_t); pad_mt = torch.ones(B, Lt, dtype=torch.bool)
    for i in range(B):
        la, lt = hs_a[i].size(0), hs_t[i].size(0)
        pad_ha[i, :la] = hs_a[i]; pad_ma[i, :la] = ms_a[i]
        pad_ht[i, :lt] = hs_t[i]; pad_mt[i, :lt] = ms_t[i]
    return pad_ha, pad_ma, pad_ht, pad_mt, torch.stack(ys, 0)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        h_a, m_a, h_t, m_t, y = [x.to(device) for x in batch]
        logits, _, _ = model(h_a, h_t, m_a, m_t)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(y.cpu())
    return torch.cat(all_probs, 0).numpy(), torch.cat(all_targets, 0).numpy()


# ── Metrics ───────────────────────────────────────────────────────────────────
def report(split: str, probs: np.ndarray, y_cont: np.ndarray,
           cal_ths: np.ndarray, names: List[str]):
    y_bin     = (y_cont > 0.0).astype(int)
    y_pred_05 = (probs >= 0.5).astype(int)
    y_pred_cal = (probs >= cal_ths[None, :]).astype(int)

    micro_f1 = f1_score(y_bin, y_pred_05,  average="micro", zero_division=0)
    macro_f1 = f1_score(y_bin, y_pred_05,  average="macro", zero_division=0)
    cal_mf1  = f1_score(y_bin, y_pred_cal, average="macro", zero_division=0)

    pc_auc = []
    for c in range(probs.shape[1]):
        col = y_bin[:, c]
        if col.max() > 0 and col.min() < 1:
            pc_auc.append(float(roc_auc_score(col, probs[:, c])))
        else:
            pc_auc.append(float("nan"))
    macro_auc = float(np.nanmean(pc_auc))

    pc_f1_05  = f1_score(y_bin, y_pred_05,  average=None, zero_division=0)
    pc_f1_cal = f1_score(y_bin, y_pred_cal, average=None, zero_division=0)
    pc_prec   = precision_score(y_bin, y_pred_cal, average=None, zero_division=0)
    pc_rec    = recall_score(y_bin,    y_pred_cal, average=None, zero_division=0)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"{split}  ({len(y_bin)} samples)")
    print(sep)
    print(f"  micro-F1 @ 0.5       : {micro_f1:.4f}")
    print(f"  macro-F1 @ 0.5       : {macro_f1:.4f}")
    print(f"  macro-AUC            : {macro_auc:.4f}")
    print(f"  macro-F1 @ calibrated: {cal_mf1:.4f}")
    print()

    print(f"  Per-class (calibrated thresholds):")
    hdr = f"  {'name':<9}  thr   AUC     F1@0.5  F1@cal  P@cal   R@cal"
    print(hdr)
    print("  " + "-" * 58)
    for i, name in enumerate(names):
        auc_s = f"{pc_auc[i]:.4f}" if not np.isnan(pc_auc[i]) else "  N/A "
        print(f"  {name:<9}  {cal_ths[i]:.2f}  {auc_s}  "
              f"{pc_f1_05[i]:.4f}  {pc_f1_cal[i]:.4f}  "
              f"{pc_prec[i]:.4f}  {pc_rec[i]:.4f}")

    print(f"\n  Classification report (@ calibrated thresholds):")
    print(classification_report(y_bin, y_pred_cal,
                                target_names=names, zero_division=0,
                                digits=4))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    a    = ckpt["args"]
    emo_cols = ckpt["emo_cols"]
    names    = [c.replace("emo_", "") for c in emo_cols]
    cal_ths  = np.array(ckpt["val_calibrated_thresholds"], dtype=np.float32)

    print(f"\nCheckpoint: {CKPT_PATH}")
    print(f"  Saved at epoch       : {ckpt['epoch']}")
    print(f"  val macro-AUC        : {ckpt['val_macro_auc']:.4f}")
    print(f"  val macro-F1 @ 0.5   : {ckpt['val_macro_f1']:.4f}")
    print(f"  val calib macro-F1   : {ckpt['val_macro_f1_calibrated']:.4f}")
    cal_str = "  ".join(f"{n}={t:.2f}" for n, t in zip(names, cal_ths))
    print(f"  Calibrated thresholds: {cal_str}")
    print(f"\n  Architecture: d_model={a['d_model']}  n_heads={a['n_heads']}  "
          f"fusion_layers={a['num_layers_fusion']}  decoder_layers={a['num_layers_decoder']}  "
          f"beta_hidden={a['beta_hidden']}  dropout={a['dropout']}")

    # Read feature dims from meta.json
    audio_meta = json.loads((AUDIO_DIR / "meta.json").read_text())
    text_meta  = json.loads((TEXT_DIR  / "meta.json").read_text())
    d_audio, d_text = int(audio_meta["hidden_dim"]), int(text_meta["hidden_dim"])

    model = MoseiFusionWithEmotionDecoder(
        d_audio=d_audio, d_text=d_text,
        d_model=a["d_model"], num_emotions=len(emo_cols),
        n_heads=a["n_heads"],
        num_layers_fusion=a["num_layers_fusion"],
        num_layers_decoder=a["num_layers_decoder"],
        beta_hidden=a["beta_hidden"], dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nModel loaded. d_audio={d_audio}  d_text={d_text}")

    df = pd.read_csv(INDEX_CSV)

    for split in ["val", "test"]:
        print(f"\nBuilding {split} dataset...")
        split_df = df[df["split"] == split].reset_index(drop=True)
        ds = MoseiSeqDataset(split_df, AUDIO_DIR, TEXT_DIR, "uid", emo_cols,
                             max_len_audio=a["max_len_audio"],
                             max_len_text=a["max_len_text"])
        loader = DataLoader(ds, batch_size=16, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
        probs, y_cont = run_inference(model, loader, device)
        report(split.upper(), probs, y_cont, cal_ths, names)


if __name__ == "__main__":
    main()
