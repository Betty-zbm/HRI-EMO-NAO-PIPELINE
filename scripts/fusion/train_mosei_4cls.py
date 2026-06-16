#!/usr/bin/env python3
"""
MOSEI 4-class single-label emotion training.

Label assignment (threshold=0.5 on normalised score):
  emo_anger dominant  → angry
  emo_happy dominant  → happy
  emo_sad   dominant  → sad
  none of above ≥ thr → neutral
  only disgust/fear/surprise ≥ thr → EXCLUDED

Loss   : CrossEntropyLoss (single-label softmax)
Metrics: WA, UA, macro-F1, weighted-F1, binary per-class F1
Select : macro-F1 on validation
Reports: train / val / test at best checkpoint each epoch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder


# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COLS = {"emo_anger": "angry", "emo_happy": "happy", "emo_sad": "sad"}
OTHER_COLS  = ["emo_fear", "emo_disgust", "emo_surprise"]
EMO_THRESH  = 0.5   # intensity threshold for a class to count as "present"

LABEL2ID = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NAMES    = [ID2LABEL[i] for i in sorted(ID2LABEL)]   # alphabetical


# ── Utilities ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assign_4cls(row) -> str | None:
    """
    Convert one MOSEI row to a 4-class label.
    Returns None for rows that should be excluded.
    """
    target_scores = {name: float(row[col]) for col, name in TARGET_COLS.items()}
    other_max     = max(float(row[c]) for c in OTHER_COLS)
    best_target   = max(target_scores.values())

    if best_target < EMO_THRESH and other_max >= EMO_THRESH:
        return None                              # exclude
    if best_target < EMO_THRESH:
        return "neutral"
    return max(target_scores, key=target_scores.get)


def build_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()
    sub["label_4cls"] = sub.apply(assign_4cls, axis=1)
    sub = sub[sub["label_4cls"].notna()].reset_index(drop=True)
    return sub


def compute_metrics(preds: list, labels: list) -> dict:
    wa  = accuracy_score(labels, preds)
    ua  = recall_score(labels, preds, average="macro",     zero_division=0)
    mf1 = f1_score(labels, preds,     average="macro",     zero_division=0)
    wf1 = f1_score(labels, preds,     average="weighted",  zero_division=0)
    bin_f1 = f1_score(labels, preds,  average=None,        zero_division=0)  # per-class
    return {
        "WA": wa, "UA": ua, "macro_F1": mf1, "weighted_F1": wf1,
        "binary_per_class_F1": {NAMES[i]: float(bin_f1[i]) for i in range(len(NAMES))},
    }


def print_metrics(tag: str, m: dict):
    print(f"  {tag}  WA={m['WA']:.4f}  UA={m['UA']:.4f}  "
          f"macro-F1={m['macro_F1']:.4f}  wt-F1={m['weighted_F1']:.4f}  "
          f"bin-F1: " +
          "  ".join(f"{n}={v:.3f}" for n, v in m["binary_per_class_F1"].items()))


# ── Dataset ────────────────────────────────────────────────────────────────────
def crop_center(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if x.size(0) <= max_len:
        return x
    start = (x.size(0) - max_len) // 2
    return x[start: start + max_len]


class MoseiCls4Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, audio_dir: Path, text_dir: Path,
                 max_len_audio: int, max_len_text: int):
        self.audio_dir     = audio_dir
        self.text_dir      = text_dir
        self.max_len_audio = max_len_audio
        self.max_len_text  = max_len_text

        keep, miss = [], 0
        for _, row in df.iterrows():
            uid = str(row["uid"])
            if (audio_dir / f"{uid}.pt").is_file() and (text_dir / f"{uid}.pt").is_file():
                keep.append(row)
            else:
                miss += 1
        if miss:
            print(f"  [Dataset] skipped {miss} rows (missing features)")
        self.rows = keep
        print(f"  [Dataset] {len(self.rows)} samples")

    def __len__(self): return len(self.rows)

    def _load(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        obj  = torch.load(path, map_location="cpu", weights_only=True)
        h    = torch.nan_to_num(obj["hidden"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = (obj["attention_mask"].long() == 0) if "attention_mask" in obj \
               else torch.zeros(h.size(0), dtype=torch.bool)
        return h, mask

    def __getitem__(self, idx):
        row  = self.rows[idx]
        uid  = str(row["uid"])
        h_a, m_a = self._load(self.audio_dir / f"{uid}.pt")
        h_t, m_t = self._load(self.text_dir  / f"{uid}.pt")
        h_a = crop_center(h_a, self.max_len_audio)
        m_a = torch.zeros(h_a.size(0), dtype=torch.bool)
        h_t = crop_center(h_t, self.max_len_text)
        m_t = torch.zeros(h_t.size(0), dtype=torch.bool)
        label = LABEL2ID[row["label_4cls"]]
        return h_a, m_a, h_t, m_t, label


def collate_fn(batch):
    hs_a, ms_a, hs_t, ms_t, labels = zip(*batch)
    B   = len(batch)
    d_a = hs_a[0].size(-1); d_t = hs_t[0].size(-1)
    La  = max(x.size(0) for x in hs_a)
    Lt  = max(x.size(0) for x in hs_t)

    pad_ha = torch.zeros(B, La, d_a)
    pad_ma = torch.ones(B,  La, dtype=torch.bool)
    pad_ht = torch.zeros(B, Lt, d_t)
    pad_mt = torch.ones(B,  Lt, dtype=torch.bool)

    for i in range(B):
        la = hs_a[i].size(0); lt = hs_t[i].size(0)
        pad_ha[i, :la] = hs_a[i]; pad_ma[i, :la] = ms_a[i]
        pad_ht[i, :lt] = hs_t[i]; pad_mt[i, :lt] = ms_t[i]

    return pad_ha, pad_ma, pad_ht, pad_mt, torch.tensor(labels, dtype=torch.long)


# ── Loss helpers ───────────────────────────────────────────────────────────────
def beta_entropy_loss(beta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = torch.clamp(beta, eps, 1 - eps)
    return -(b * b.log() + (1 - b) * (1 - b).log()).mean()


def symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(p, -1); lq = torch.log_softmax(q, -1)
    pp = lp.exp();               pq = lq.exp()
    return 0.5 * ((pp * (lp - lq)).sum(-1).mean() + (pq * (lq - lp)).sum(-1).mean())


def apply_feat_mask(h: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0 or not h.requires_grad and not torch.is_grad_enabled():
        return h
    valid = ~mask                                 # [B, L]
    noise = torch.rand_like(valid.float()) < p
    drop  = valid & noise
    h = h.clone()
    h[drop] = 0.0
    return h


# ── Train / eval loops ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, scaler, criterion,
              device, args, *, training: bool) -> Tuple[float, list, list]:
    model.train() if training else model.eval()
    total_loss = 0.0; n = 0
    all_preds, all_labels = [], []

    if training:
        optimizer.zero_grad(set_to_none=True)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for step, (h_a, m_a, h_t, m_t, labels) in enumerate(
                tqdm(loader, desc="Train" if training else "Eval", leave=False)):
            h_a, m_a = h_a.to(device), m_a.to(device)
            h_t, m_t = h_t.to(device), m_t.to(device)
            labels    = labels.to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                if training and args.feat_mask_prob > 0:
                    h_a = apply_feat_mask(h_a, m_a, args.feat_mask_prob)
                    h_t = apply_feat_mask(h_t, m_t, args.feat_mask_prob)

                logits, beta, _ = model(h_a, h_t, m_a, m_t)
                loss = criterion(logits, labels)

                if training and args.rdrop_alpha > 0:
                    logits2, beta2, _ = model(h_a, h_t, m_a, m_t)
                    loss = 0.5 * (loss + criterion(logits2, labels)) \
                         + args.rdrop_alpha * symmetric_kl(logits, logits2)

                if beta is not None and args.beta_entropy > 0:
                    loss = loss + args.beta_entropy * beta_entropy_loss(beta)

                if training:
                    loss_scaled = loss / args.grad_accum

            if training:
                scaler.scale(loss_scaled).backward()
                if (step + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            bs = labels.size(0)
            total_loss += loss.detach().float().item() * bs
            n += bs
            all_preds.extend(logits.detach().argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / max(1, n), all_preds, all_labels


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv",  default="data/mosei_index_splits.csv")
    ap.add_argument("--audio_dir",  default="data/mosei_features_for_training/features/mosei/seq_level/audio")
    ap.add_argument("--text_dir",   default="data/mosei_features_for_training/features/mosei/seq_level/text")
    ap.add_argument("--out_dir",    default="runs/mosei_4cls")

    ap.add_argument("--d_model",            type=int,   default=256)
    ap.add_argument("--n_heads",            type=int,   default=4)
    ap.add_argument("--num_layers_fusion",  type=int,   default=2)
    ap.add_argument("--num_layers_decoder", type=int,   default=2)
    ap.add_argument("--beta_hidden",        type=int,   default=128)
    ap.add_argument("--dropout",            type=float, default=0.3)

    ap.add_argument("--batch_size",      type=int,   default=16)
    ap.add_argument("--epochs",          type=int,   default=30)
    ap.add_argument("--lr",              type=float, default=5e-5)
    ap.add_argument("--weight_decay",    type=float, default=0.05)
    ap.add_argument("--grad_accum",      type=int,   default=2)
    ap.add_argument("--warmup_ratio",    type=float, default=0.1)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--feat_mask_prob",  type=float, default=0.10)
    ap.add_argument("--rdrop_alpha",     type=float, default=0.05)
    ap.add_argument("--beta_entropy",    type=float, default=1e-3)

    ap.add_argument("--max_len_audio", type=int, default=600)
    ap.add_argument("--max_len_text",  type=int, default=64)

    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build splits ────────────────────────────────────────────────────────
    df = pd.read_csv(args.index_csv)
    train_df = build_split(df, "train")
    val_df   = build_split(df, "val")
    test_df  = build_split(df, "test")

    print(f"\nTrain {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    for tag, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {tag}: {sub['label_4cls'].value_counts().to_dict()}")

    audio_dir = Path(args.audio_dir)
    text_dir  = Path(args.text_dir)

    print("\nBuilding datasets...")
    train_ds = MoseiCls4Dataset(train_df, audio_dir, text_dir, args.max_len_audio, args.max_len_text)
    val_ds   = MoseiCls4Dataset(val_df,   audio_dir, text_dir, args.max_len_audio, args.max_len_text)
    test_ds  = MoseiCls4Dataset(test_df,  audio_dir, text_dir, args.max_len_audio, args.max_len_text)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn)

    # ── Model ───────────────────────────────────────────────────────────────
    model = MoseiFusionWithEmotionDecoder(
        d_audio=74, d_text=300, d_model=args.d_model, num_emotions=4,
        n_heads=args.n_heads, num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden, dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps  = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_f1 = -1.0
    log_rows    = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")

        tr_loss, tr_preds, tr_labels = run_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, args, training=True)
        va_loss, va_preds, va_labels = run_epoch(
            model, val_loader, None, None, None, criterion,
            device, args, training=False)
        te_loss, te_preds, te_labels = run_epoch(
            model, test_loader, None, None, None, criterion,
            device, args, training=False)

        tr_m = compute_metrics(tr_preds, tr_labels)
        va_m = compute_metrics(va_preds, va_labels)
        te_m = compute_metrics(te_preds, te_labels)

        print_metrics(f"TRAIN loss={tr_loss:.4f}", tr_m)
        print_metrics(f"VAL   loss={va_loss:.4f}", va_m)
        print_metrics(f"TEST  loss={te_loss:.4f}", te_m)

        # Per-class breakdown every epoch
        print(f"\n  VAL classification report:")
        print(classification_report(va_labels, va_preds, target_names=NAMES, zero_division=0))

        row = {"epoch": epoch,
               "train_loss": tr_loss, "val_loss": va_loss, "test_loss": te_loss}
        for split_tag, m in [("train", tr_m), ("val", va_m), ("test", te_m)]:
            row[f"{split_tag}_WA"]  = m["WA"]
            row[f"{split_tag}_UA"]  = m["UA"]
            row[f"{split_tag}_macroF1"]    = m["macro_F1"]
            row[f"{split_tag}_weightedF1"] = m["weighted_F1"]
            for cls_name, f1v in m["binary_per_class_F1"].items():
                row[f"{split_tag}_binF1_{cls_name}"] = f1v
        log_rows.append(row)

        if va_m["macro_F1"] > best_val_f1:
            best_val_f1 = va_m["macro_F1"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "label2id": LABEL2ID,
                "val_macro_F1":  va_m["macro_F1"],
                "val_WA":        va_m["WA"],
                "test_macro_F1": te_m["macro_F1"],
                "test_WA":       te_m["WA"],
                "val_metrics":   va_m,
                "test_metrics":  te_m,
            }
            torch.save(ckpt, out_dir / "best_mosei_4cls.pt")
            print(f"\n  *** New best val macro-F1={best_val_f1:.4f}  "
                  f"(test WA={te_m['WA']:.4f}  test macro-F1={te_m['macro_F1']:.4f})  -> saved ***")

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    print(f"\nTraining done. Best val macro-F1={best_val_f1:.4f}")
    print(f"Checkpoint: {out_dir / 'best_mosei_4cls.pt'}")
    print(f"Log:        {out_dir / 'train_log.csv'}")


if __name__ == "__main__":
    main()
