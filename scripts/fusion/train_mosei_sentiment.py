#!/usr/bin/env python3
"""
MOSEI 2-class sentiment training (positive vs negative).

Labels  : sentiment score >= 0 -> positive (1), score < 0 -> negative (0)
Features: COVAREP (74-dim audio) + GloVe (300-dim text) — all 22,860 segments
Loss    : CrossEntropyLoss with inverse-frequency class weights
Metric  : Acc-2 (binary accuracy) — used for model selection
          Also reports: weighted-F1, macro-F1, per-class F1/P/R, confusion matrix

Reuses MoseiFusionWithEmotionDecoder(num_emotions=2) — no architecture change needed.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv",      default="data/mosei_index_splits.csv")
    ap.add_argument("--audio_dir",
                    default="data/mosei_features_for_training/features/mosei/seq_level/audio")
    ap.add_argument("--text_dir",
                    default="data/mosei_features_for_training/features/mosei/seq_level/text")
    ap.add_argument("--uid_col",        default="uid")
    ap.add_argument("--split_col",      default="split")
    ap.add_argument("--sentiment_col",  default="sentiment")

    # Model
    ap.add_argument("--d_model",             type=int,   default=256)
    ap.add_argument("--n_heads",             type=int,   default=4)
    ap.add_argument("--num_layers_fusion",   type=int,   default=2)
    ap.add_argument("--num_layers_decoder",  type=int,   default=2)
    ap.add_argument("--beta_hidden",         type=int,   default=128)
    ap.add_argument("--dropout",             type=float, default=0.2)

    # Training
    ap.add_argument("--batch_size",     type=int,   default=16)
    ap.add_argument("--epochs",         type=int,   default=30)
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--weight_decay",   type=float, default=1e-2)
    ap.add_argument("--grad_accum",     type=int,   default=2)
    ap.add_argument("--warmup_ratio",   type=float, default=0.1)
    ap.add_argument("--beta_entropy",   type=float, default=1e-3)
    ap.add_argument("--feat_mask_prob", type=float, default=0.10)
    ap.add_argument("--max_len_audio",  type=int,   default=300)
    ap.add_argument("--max_len_text",   type=int,   default=128)
    ap.add_argument("--num_workers",    type=int,   default=0)
    ap.add_argument("--seed",           type=int,   default=1234)
    ap.add_argument("--out_dir",        default="runs/mosei_sentiment")
    return ap.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────
def crop_center(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if x.size(0) <= max_len:
        return x
    start = (x.size(0) - max_len) // 2
    return x[start:start + max_len]


class MoseiSentimentDataset(Dataset):
    def __init__(self, df, audio_dir, text_dir, uid_col, sentiment_col,
                 max_len_audio, max_len_text):
        self.audio_dir     = audio_dir
        self.text_dir      = text_dir
        self.uid_col       = uid_col
        self.sentiment_col = sentiment_col
        self.max_len_audio = max_len_audio
        self.max_len_text  = max_len_text

        keep, missing = [], 0
        for i, row in df.reset_index(drop=True).iterrows():
            uid = str(row[uid_col])
            if ((audio_dir / f"{uid}.pt").is_file() and
                    (text_dir / f"{uid}.pt").is_file()):
                keep.append(i)
            else:
                missing += 1
        if missing:
            print(f"  [Dataset] skipped {missing} rows with missing features")
        self.df = df.reset_index(drop=True).iloc[keep].reset_index(drop=True)
        print(f"  [Dataset] {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _load_feat(self, path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        h = obj["hidden"].float()
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        m = obj.get("attention_mask", torch.ones(h.size(0), dtype=torch.long))
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, dtype=torch.long)
        return h, (m == 0)  # True = PAD

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        uid  = str(row[self.uid_col])
        h_a, m_a = self._load_feat(self.audio_dir / f"{uid}.pt")
        h_t, m_t = self._load_feat(self.text_dir  / f"{uid}.pt")
        if self.max_len_audio > 0:
            h_a = crop_center(h_a, self.max_len_audio)
            m_a = torch.zeros(h_a.size(0), dtype=torch.bool)
        if self.max_len_text > 0:
            h_t = crop_center(h_t, self.max_len_text)
            m_t = torch.zeros(h_t.size(0), dtype=torch.bool)
        score = float(row[self.sentiment_col])
        label = 1 if score >= 0.0 else 0
        return h_a, m_a, h_t, m_t, label, score


def collate_fn(batch):
    hs_a, ms_a, hs_t, ms_t, labels, scores = zip(*batch)
    B = len(batch)
    d_a, d_t = hs_a[0].size(-1), hs_t[0].size(-1)
    La_max = max(x.size(0) for x in hs_a)
    Lt_max = max(x.size(0) for x in hs_t)
    pad_h_a = torch.zeros(B, La_max, d_a)
    pad_m_a = torch.ones(B,  La_max, dtype=torch.bool)
    pad_h_t = torch.zeros(B, Lt_max, d_t)
    pad_m_t = torch.ones(B,  Lt_max, dtype=torch.bool)
    for i in range(B):
        La, Lt = hs_a[i].size(0), hs_t[i].size(0)
        pad_h_a[i, :La] = hs_a[i]; pad_m_a[i, :La] = ms_a[i]
        pad_h_t[i, :Lt] = hs_t[i]; pad_m_t[i, :Lt] = ms_t[i]
    return (pad_h_a, pad_m_a, pad_h_t, pad_m_t,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(scores, dtype=torch.float32),
            d_a, d_t)


# ── Helpers ───────────────────────────────────────────────────────────────────
def beta_entropy_loss(beta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = torch.clamp(beta, eps, 1.0 - eps)
    return (-(b * torch.log(b) + (1 - b) * torch.log(1 - b))).mean()


def apply_feat_mask(h_a: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0.0:
        return h_a
    mask = torch.rand(h_a.shape[:2], device=h_a.device) < prob
    return h_a.masked_fill(mask.unsqueeze(-1), 0.0)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = logits.argmax(dim=-1).cpu().numpy()
    y     = labels.cpu().numpy()
    acc   = float(accuracy_score(y, preds))
    wf1   = float(f1_score(y, preds, average="weighted", zero_division=0))
    mf1   = float(f1_score(y, preds, average="macro",    zero_division=0))
    pc_f1 = f1_score(y, preds, average=None, zero_division=0)
    return {
        "acc2":        acc,
        "weighted_f1": wf1,
        "macro_f1":    mf1,
        "neg_f1":      float(pc_f1[0]) if len(pc_f1) > 0 else 0.0,
        "pos_f1":      float(pc_f1[1]) if len(pc_f1) > 1 else 0.0,
    }


def fmt_row(tag: str, loss: float, m: dict) -> str:
    return (f"  {tag:<6} loss={loss:.4f}  Acc-2={m['acc2']:.4f}  "
            f"W-F1={m['weighted_f1']:.4f}  M-F1={m['macro_f1']:.4f}  "
            f"neg-F1={m['neg_f1']:.4f}  pos-F1={m['pos_f1']:.4f}")


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, scaler, args):
    model.train()
    total_loss, total_samples = 0.0, 0
    all_logits, all_labels = [], []
    optimizer.zero_grad(set_to_none=True)

    for step, (h_a, m_a, h_t, m_t, y_cls, _, _, _) in enumerate(
            tqdm(loader, desc="Train", leave=False)):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y_cls    = y_cls.to(device)
        h_a      = apply_feat_mask(h_a, args.feat_mask_prob)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)
            raw_loss = criterion(logits, y_cls)
            if beta is not None and args.beta_entropy > 0:
                raw_loss = raw_loss + args.beta_entropy * beta_entropy_loss(beta)
            loss = raw_loss / args.grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        bs = y_cls.size(0)
        total_loss   += raw_loss.detach().float().cpu().item() * bs
        total_samples += bs
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_cls.detach().cpu())

    avg_loss = total_loss / max(1, total_samples)
    metrics  = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_logits, all_labels, all_scores = [], [], []

    for h_a, m_a, h_t, m_t, y_cls, y_score, _, _ in tqdm(loader, desc="Eval", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y_cls    = y_cls.to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)
            raw_loss = criterion(logits, y_cls)
            if beta is not None and args.beta_entropy > 0:
                raw_loss = raw_loss + args.beta_entropy * beta_entropy_loss(beta)

        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            continue

        bs = y_cls.size(0)
        total_loss    += raw_loss.item() * bs
        total_samples += bs
        all_logits.append(logits.cpu())
        all_labels.append(y_cls.cpu())
        all_scores.append(y_score)

    avg_loss   = total_loss / max(1, total_samples)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    metrics    = compute_metrics(all_logits, all_labels)
    return avg_loss, metrics, all_logits, all_labels, all_scores


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.txt", "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Args: {vars(args)}\n")

    df       = pd.read_csv(args.index_csv)
    train_df = df[df[args.split_col] == "train"].reset_index(drop=True)
    val_df   = df[df[args.split_col] == "val"].reset_index(drop=True)
    test_df  = df[df[args.split_col] == "test"].reset_index(drop=True)
    log(f"Splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Class balance + inverse-frequency weights
    n_train = len(train_df)
    n_neg   = int((train_df[args.sentiment_col] < 0).sum())
    n_pos   = int((train_df[args.sentiment_col] >= 0).sum())
    log(f"Train class balance: neg={n_neg} ({100*n_neg/n_train:.1f}%)  "
        f"pos={n_pos} ({100*n_pos/n_train:.1f}%)")
    w_neg = n_train / (2.0 * n_neg)
    w_pos = n_train / (2.0 * n_pos)
    class_weight = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    log(f"CrossEntropy weights: neg={w_neg:.3f}  pos={w_pos:.3f}\n")

    audio_dir = Path(args.audio_dir)
    text_dir  = Path(args.text_dir)

    log("Building datasets...")
    train_ds = MoseiSentimentDataset(
        train_df, audio_dir, text_dir, args.uid_col, args.sentiment_col,
        args.max_len_audio, args.max_len_text)
    val_ds   = MoseiSentimentDataset(
        val_df,   audio_dir, text_dir, args.uid_col, args.sentiment_col,
        args.max_len_audio, args.max_len_text)
    test_ds  = MoseiSentimentDataset(
        test_df,  audio_dir, text_dir, args.uid_col, args.sentiment_col,
        args.max_len_audio, args.max_len_text)
    log("")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              collate_fn=collate_fn)

    audio_meta = json.loads((audio_dir / "meta.json").read_text())
    text_meta  = json.loads((text_dir  / "meta.json").read_text())
    d_audio    = int(audio_meta["hidden_dim"])
    d_text     = int(text_meta["hidden_dim"])
    log(f"Feature dims: audio={d_audio} (COVAREP)  text={d_text} (GloVe)  classes=2\n")

    model = MoseiFusionWithEmotionDecoder(
        d_audio=d_audio, d_text=d_text, d_model=args.d_model,
        num_emotions=2, n_heads=args.n_heads,
        num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps  = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc   = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        log("=" * 60)
        log(f"Epoch {epoch}/{args.epochs}")

        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, args)

        va_loss, va_m, _, _, _ = evaluate(model, val_loader,  criterion, device, args)
        te_loss, te_m, _, _, _ = evaluate(model, test_loader, criterion, device, args)

        log(fmt_row("TRAIN", tr_loss, tr_m))
        log(fmt_row("VAL",   va_loss, va_m))
        log(fmt_row("TEST",  te_loss, te_m))

        if va_m["acc2"] > best_acc:
            best_acc   = va_m["acc2"]
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "epoch":            epoch,
                "args":             vars(args),
                "val_acc2":         float(best_acc),
                "val_weighted_f1":  va_m["weighted_f1"],
                "val_macro_f1":     va_m["macro_f1"],
            }
            log(f"  *** New best Acc-2={best_acc:.4f} -> saved ***")

    # ── Save + final evaluation ───────────────────────────────────────────────
    if best_state is None:
        log("No checkpoint saved — training produced no valid epochs.")
        log_f.close()
        return

    ckpt_path = out_dir / "best_mosei_sentiment.pt"
    torch.save(best_state, ckpt_path)
    log(f"\nCheckpoint: {ckpt_path}")
    log(f"Best epoch={best_state['epoch']}  val Acc-2={best_acc:.4f}")

    log("\n" + "=" * 60)
    log(f"FINAL EVALUATION  (best checkpoint: epoch {best_state['epoch']})")
    model.load_state_dict(best_state["model_state_dict"])

    for split_name, loader in [("VAL", val_loader), ("TEST", test_loader)]:
        _, m, all_logits, all_labels, _ = evaluate(
            model, loader, criterion, device, args)
        preds = all_logits.argmax(dim=-1).numpy()
        y     = all_labels.numpy()

        log(f"\n{split_name}:")
        log(f"  Acc-2={m['acc2']:.4f}  Weighted-F1={m['weighted_f1']:.4f}  "
            f"Macro-F1={m['macro_f1']:.4f}")
        log(f"  neg-F1={m['neg_f1']:.4f}  pos-F1={m['pos_f1']:.4f}")

        log(f"\n  Classification report:")
        report = classification_report(
            y, preds, target_names=["negative", "positive"], zero_division=0)
        for line in report.splitlines():
            log("    " + line)

        cm = confusion_matrix(y, preds)
        log(f"\n  Confusion matrix (rows=actual, cols=pred):")
        log(f"               pred_neg  pred_pos")
        log(f"    actual_neg  {cm[0,0]:7d}   {cm[0,1]:7d}")
        log(f"    actual_pos  {cm[1,0]:7d}   {cm[1,1]:7d}")

    log_f.close()


if __name__ == "__main__":
    main()
