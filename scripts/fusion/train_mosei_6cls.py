#!/usr/bin/env python3
"""
MOSEI 6-class multi-label emotion training (improved).

Keeps the original multi-label BCE approach but adds:
  - Test split evaluation every epoch
  - Per-class AUC / F1 / Precision / Recall at fixed + calibrated thresholds
  - Feature masking augmentation (feat_mask_prob)
  - Dropout 0.1 -> 0.2
  - Calibrated thresholds always saved into checkpoint
  - Checkpoint selected by calibrated macro-F1 on val (most principled)
  - 30 epochs (from 20)
  - File logging
  - Windows-compatible paths and num_workers=0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder


# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index_csv",  default="data/mosei_index_splits.csv")
    ap.add_argument("--audio_dir",  default="data/mosei_features_for_training/features/mosei/seq_level/audio")
    ap.add_argument("--text_dir",   default="data/mosei_features_for_training/features/mosei/seq_level/text")
    ap.add_argument("--uid_col",    default="uid")
    ap.add_argument("--split_col",  default="split")
    ap.add_argument("--emo_cols", nargs="+", default=[
        "emo_happy", "emo_sad", "emo_anger", "emo_fear", "emo_disgust", "emo_surprise",
    ])

    # Model
    ap.add_argument("--d_model",             type=int,   default=256)
    ap.add_argument("--n_heads",             type=int,   default=4)
    ap.add_argument("--num_layers_fusion",   type=int,   default=2)
    ap.add_argument("--num_layers_decoder",  type=int,   default=2)
    ap.add_argument("--beta_hidden",         type=int,   default=128)
    ap.add_argument("--dropout",             type=float, default=0.2)

    # Training
    ap.add_argument("--batch_size",    type=int,   default=8)
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--lr",            type=float, default=1e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-2)
    ap.add_argument("--grad_accum",    type=int,   default=4)
    ap.add_argument("--warmup_ratio",  type=float, default=0.1)
    ap.add_argument("--beta_entropy",  type=float, default=1e-3)
    ap.add_argument("--feat_mask_prob",type=float, default=0.15,
                    help="fraction of audio frames to zero-mask during training")
    ap.add_argument("--max_len_audio", type=int,   default=300)
    ap.add_argument("--max_len_text",  type=int,   default=128)
    ap.add_argument("--num_workers",   type=int,   default=0)
    ap.add_argument("--seed",          type=int,   default=1234)

    ap.add_argument("--out_dir", default="runs/mosei_6cls")
    return ap.parse_args()


# ── Label helpers ─────────────────────────────────────────────────────────────
def normalize_mosei_emotions(y: torch.Tensor) -> torch.Tensor:
    return torch.clamp(y, min=0.0, max=3.0) / 3.0


def compute_pos_weight(train_df: pd.DataFrame, emo_cols: List[str]) -> torch.Tensor:
    y = train_df[emo_cols].astype(float).clip(lower=0.0)
    pos = (y > 0.0).sum(axis=0).values
    neg = len(y) - pos
    pw = (neg / np.clip(pos, 1, None)).astype(np.float32)
    return torch.tensor(pw, dtype=torch.float32)


def calibrate_thresholds(probs: np.ndarray, y_true_cont: np.ndarray,
                          steps: int = 19) -> np.ndarray:
    y_true = (y_true_cont > 0.0).astype(int)
    C = probs.shape[1]
    ths = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, steps):
            f1 = f1_score(y_true[:, c], (probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        ths[c] = best_t
    return ths


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5,
                    cal_ths: np.ndarray | None = None,
                    emo_cols: List[str] | None = None,
                    do_calibrate: bool = False):
    """
    Returns dict with aggregate + per-class metrics.
    If do_calibrate=True, also computes and returns calibrated thresholds.
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true_cont = targets.detach().cpu().numpy()
    y_true_bin = (y_true_cont > 0.0).astype(int)
    y_pred_bin = (probs >= threshold).astype(int)

    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    pc_f1    = f1_score(y_true_bin, y_pred_bin, average=None,    zero_division=0)
    pc_prec  = precision_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    pc_rec   = recall_score(y_true_bin,   y_pred_bin, average=None, zero_division=0)

    pc_auc = []
    for c in range(probs.shape[1]):
        col = y_true_bin[:, c]
        if col.max() > 0 and col.min() < 1:
            pc_auc.append(float(roc_auc_score(col, probs[:, c])))
        else:
            pc_auc.append(float("nan"))
    macro_auc = float(np.nanmean(pc_auc))

    out = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "per_class_f1":   pc_f1.tolist(),
        "per_class_prec": pc_prec.tolist(),
        "per_class_rec":  pc_rec.tolist(),
        "per_class_auc":  pc_auc,
    }

    # Calibrated thresholds (computed fresh from this set, only done on val)
    if do_calibrate:
        new_ths = calibrate_thresholds(probs, y_true_cont)
        y_pred_cal = (probs >= new_ths[None, :]).astype(int)
        cal_f1_pc = f1_score(y_true_bin, y_pred_cal, average=None, zero_division=0)
        cal_macro_f1 = float(f1_score(y_true_bin, y_pred_cal, average="macro", zero_division=0))
        out["cal_thresholds"] = new_ths.tolist()
        out["cal_macro_f1"]   = cal_macro_f1
        out["cal_per_class_f1"] = cal_f1_pc.tolist()
    elif cal_ths is not None:
        # Apply externally supplied calibrated thresholds
        y_pred_cal = (probs >= cal_ths[None, :]).astype(int)
        cal_f1_pc = f1_score(y_true_bin, y_pred_cal, average=None, zero_division=0)
        cal_macro_f1 = float(f1_score(y_true_bin, y_pred_cal, average="macro", zero_division=0))
        out["cal_macro_f1"]   = cal_macro_f1
        out["cal_per_class_f1"] = cal_f1_pc.tolist()

    return out, probs, y_true_cont


def print_epoch_metrics(tag: str, loss: float, m: dict, emo_cols: List[str],
                        show_per_class: bool = True, log_f=None):
    names = [c.replace("emo_", "") for c in emo_cols]
    agg = (f"loss={loss:.4f}  micro-F1={m['micro_f1']:.3f}  "
           f"macro-F1={m['macro_f1']:.3f}  macro-AUC={m['macro_auc']:.3f}")
    if "cal_macro_f1" in m:
        agg += f"  calib-macro-F1={m['cal_macro_f1']:.3f}"
    line = f"  {tag:<6} {agg}"
    print(line)
    if log_f:
        log_f.write(line + "\n")

    if show_per_class:
        hdr = f"  {'':6} per-class @ thr=0.50:"
        print(hdr)
        if log_f: log_f.write(hdr + "\n")
        for i, name in enumerate(names):
            auc_s = f"{m['per_class_auc'][i]:.3f}" if not np.isnan(m['per_class_auc'][i]) else "  N/A"
            row = (f"  {'':8} {name:<9}  AUC={auc_s}  "
                   f"F1={m['per_class_f1'][i]:.3f}  "
                   f"P={m['per_class_prec'][i]:.3f}  "
                   f"R={m['per_class_rec'][i]:.3f}")
            print(row)
            if log_f: log_f.write(row + "\n")

        if "cal_thresholds" in m:
            cal_hdr = f"  {'':6} calibrated thresholds & F1:"
            print(cal_hdr)
            if log_f: log_f.write(cal_hdr + "\n")
            parts = []
            for i, name in enumerate(names):
                parts.append(f"{name}={m['cal_thresholds'][i]:.2f}(F1={m['cal_per_class_f1'][i]:.3f})")
            cal_row = "  " + "  ".join(parts)
            print(cal_row)
            if log_f: log_f.write(cal_row + "\n")


# ── Dataset ───────────────────────────────────────────────────────────────────
def crop_center(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if x.size(0) <= max_len:
        return x
    start = (x.size(0) - max_len) // 2
    return x[start:start + max_len]


class MoseiSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, audio_dir: Path, text_dir: Path,
                 uid_col: str, emo_cols: List[str],
                 max_len_audio: int, max_len_text: int):
        self.audio_dir    = audio_dir
        self.text_dir     = text_dir
        self.uid_col      = uid_col
        self.emo_cols     = emo_cols
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
            print(f"  [Dataset] skipped {missing} rows missing audio or text features")
        self.df = df.reset_index(drop=True).iloc[keep].reset_index(drop=True)
        print(f"  [Dataset] {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def _load_feat(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        h = obj["hidden"].float()
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        m = obj.get("attention_mask", torch.ones(h.size(0), dtype=torch.long))
        if not isinstance(m, torch.Tensor):
            m = torch.tensor(m, dtype=torch.long)
        return h, (m == 0)  # True = PAD

    def __getitem__(self, idx: int):
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
        y = torch.tensor([float(row[c]) for c in self.emo_cols], dtype=torch.float32)
        y = torch.nan_to_num(y, nan=0.0)
        return h_a, m_a, h_t, m_t, y


def collate_fn(batch):
    hs_a, ms_a, hs_t, ms_t, ys = zip(*batch)
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
    return pad_h_a, pad_m_a, pad_h_t, pad_m_t, torch.stack(ys, 0), d_a, d_t


# ── Beta entropy loss ─────────────────────────────────────────────────────────
def beta_entropy_loss(beta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = torch.clamp(beta, eps, 1.0 - eps)
    return (-(b * torch.log(b) + (1 - b) * torch.log(1 - b))).mean()


# ── Feature masking ───────────────────────────────────────────────────────────
def apply_feat_mask(h_a: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0.0:
        return h_a
    mask = torch.rand(h_a.shape[:2], device=h_a.device) < prob
    return h_a.masked_fill(mask.unsqueeze(-1), 0.0)


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, scaler, args):
    model.train()
    total_loss, total_samples = 0.0, 0
    all_logits, all_targets = [], []

    optimizer.zero_grad(set_to_none=True)
    for step, (h_a, m_a, h_t, m_t, y, _, _) in enumerate(
            tqdm(loader, desc="Train", leave=False)):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y = y.to(device)

        h_a = apply_feat_mask(h_a, args.feat_mask_prob)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)
            y_loss = normalize_mosei_emotions(y)
            raw_loss = criterion(logits, y_loss)
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

        bs = y.size(0)
        total_loss += raw_loss.detach().float().cpu().item() * bs
        total_samples += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = abs(total_loss / max(1, total_samples))
    all_logits  = torch.cat(all_logits,  0)
    all_targets = torch.cat(all_targets, 0)
    metrics, _, _ = compute_metrics(all_logits, all_targets)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, args,
             cal_ths: np.ndarray | None = None,
             do_calibrate: bool = False):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_logits, all_targets = [], []

    for h_a, m_a, h_t, m_t, y, _, _ in tqdm(loader, desc="Eval", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        y = y.to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)
            y_loss = normalize_mosei_emotions(y)
            raw_loss = criterion(logits, y_loss)
            if beta is not None and args.beta_entropy > 0:
                raw_loss = raw_loss + args.beta_entropy * beta_entropy_loss(beta)

        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            continue

        bs = y.size(0)
        total_loss += raw_loss.item() * bs
        total_samples += bs
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    avg_loss = abs(total_loss / max(1, total_samples))
    all_logits  = torch.cat(all_logits,  0)
    all_targets = torch.cat(all_targets, 0)
    metrics, probs, y_true_cont = compute_metrics(
        all_logits, all_targets,
        cal_ths=cal_ths, do_calibrate=do_calibrate,
    )
    return avg_loss, metrics, probs, y_true_cont


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.txt"
    log_f    = open(log_path, "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Args: {vars(args)}\n")

    df = pd.read_csv(args.index_csv)
    train_df = df[df[args.split_col] == "train"].reset_index(drop=True)
    val_df   = df[df[args.split_col] == "val"].reset_index(drop=True)
    test_df  = df[df[args.split_col] == "test"].reset_index(drop=True)

    log(f"Splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Class imbalance stats
    pos_weight = compute_pos_weight(train_df, args.emo_cols)
    names = [c.replace("emo_", "") for c in args.emo_cols]
    log("pos_weight (neg/pos per class on train):")
    for n, pw in zip(names, pos_weight.tolist()):
        log(f"  {n:<9}: {pw:.2f}")
    log("")

    audio_dir = Path(args.audio_dir)
    text_dir  = Path(args.text_dir)

    log("Building datasets...")
    train_ds = MoseiSeqDataset(train_df, audio_dir, text_dir, args.uid_col,
                                args.emo_cols, args.max_len_audio, args.max_len_text)
    val_ds   = MoseiSeqDataset(val_df,   audio_dir, text_dir, args.uid_col,
                                args.emo_cols, args.max_len_audio, args.max_len_text)
    test_ds  = MoseiSeqDataset(test_df,  audio_dir, text_dir, args.uid_col,
                                args.emo_cols, args.max_len_audio, args.max_len_text)
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

    # Read feature dims from meta.json
    audio_meta = json.loads((audio_dir / "meta.json").read_text())
    text_meta  = json.loads((text_dir  / "meta.json").read_text())
    d_audio    = int(audio_meta["hidden_dim"])
    d_text     = int(text_meta["hidden_dim"])
    num_emo    = len(args.emo_cols)
    log(f"Feature dims: audio={d_audio}  text={d_text}  num_emotions={num_emo}\n")

    model = MoseiFusionWithEmotionDecoder(
        d_audio=d_audio, d_text=d_text, d_model=args.d_model,
        num_emotions=num_emo, n_heads=args.n_heads,
        num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden, dropout=args.dropout,
    ).to(device)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scaler     = GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps  = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_metric = -1.0
    best_state  = None
    best_cal_ths = None

    for epoch in range(1, args.epochs + 1):
        sep = "=" * 60
        log(sep)
        log(f"Epoch {epoch}/{args.epochs}")

        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, args)

        # Val: compute calibrated thresholds fresh each epoch
        va_loss, va_m, va_probs, va_y = evaluate(
            model, val_loader, criterion, device, args, do_calibrate=True)

        # Test: apply val calibrated thresholds
        cal_ths_this_epoch = np.array(va_m["cal_thresholds"])
        te_loss, te_m, _, _ = evaluate(
            model, test_loader, criterion, device, args,
            cal_ths=cal_ths_this_epoch)

        # Print
        print_epoch_metrics("TRAIN", tr_loss, tr_m, args.emo_cols,
                            show_per_class=False, log_f=log_f)
        print_epoch_metrics("VAL",   va_loss, va_m, args.emo_cols,
                            show_per_class=True,  log_f=log_f)
        print_epoch_metrics("TEST",  te_loss, te_m, args.emo_cols,
                            show_per_class=False, log_f=log_f)

        metric_for_selection = va_m["cal_macro_f1"]
        if metric_for_selection > best_metric:
            best_metric  = metric_for_selection
            best_cal_ths = cal_ths_this_epoch.copy()
            best_state = {
                "model_state_dict":          model.state_dict(),
                "epoch":                     epoch,
                "args":                      vars(args),
                "emo_cols":                  args.emo_cols,
                "val_cal_macro_f1":          float(best_metric),
                "val_macro_f1":              va_m["macro_f1"],
                "val_macro_auc":             va_m["macro_auc"],
                "val_calibrated_thresholds": best_cal_ths.tolist(),
            }
            msg = f"  *** New best calib-macro-F1={best_metric:.4f} -> saved ***"
            log(msg)

    # ── Save best checkpoint ──────────────────────────────────────────────────
    if best_state is not None:
        ckpt_path = out_dir / "best_mosei_6cls.pt"
        torch.save(best_state, ckpt_path)
        log(f"\nCheckpoint: {ckpt_path}")
        log(f"Best epoch={best_state['epoch']}  calib-macro-F1={best_metric:.4f}  "
            f"macro-AUC={best_state['val_macro_auc']:.4f}")

    # ── Final evaluation with best checkpoint ─────────────────────────────────
    if best_state is not None:
        log("\n" + "=" * 60)
        log(f"FINAL EVALUATION  (best checkpoint: epoch {best_state['epoch']})")
        model.load_state_dict(best_state["model_state_dict"])

        cal_ths = np.array(best_cal_ths)
        cal_str = "  ".join(f"{n}={t:.2f}" for n, t in zip(names, cal_ths))
        log(f"Val calibrated thresholds: {cal_str}\n")

        for split_name, loader in [("VAL", val_loader), ("TEST", test_loader)]:
            _, m, probs, y_true_cont = evaluate(
                model, loader, criterion, device, args, cal_ths=cal_ths)
            y_true_bin = (y_true_cont > 0.0).astype(int)
            y_pred_cal = (probs >= cal_ths[None, :]).astype(int)

            log(f"\n{split_name} — calibrated thresholds:")
            log(f"  micro-F1={m['micro_f1']:.4f}  macro-F1={m['macro_f1']:.4f}  "
                f"macro-AUC={m['macro_auc']:.4f}  calib-macro-F1={m.get('cal_macro_f1', 0):.4f}")

            log(f"\n  Per-class (@ calibrated thresholds):")
            for i, name in enumerate(names):
                auc_s = f"{m['per_class_auc'][i]:.4f}" if not np.isnan(m['per_class_auc'][i]) else "   N/A"
                f1_c  = m["cal_per_class_f1"][i] if "cal_per_class_f1" in m else m["per_class_f1"][i]
                log(f"    {name:<9}  AUC={auc_s}  F1={f1_c:.4f}  "
                    f"P={m['per_class_prec'][i]:.4f}  R={m['per_class_rec'][i]:.4f}")

            log(f"\n  Classification report (calibrated):")
            report = classification_report(
                y_true_bin, y_pred_cal,
                target_names=names, zero_division=0)
            for line in report.splitlines():
                log("    " + line)

    log_f.close()


if __name__ == "__main__":
    main()
