#!/usr/bin/env python3
"""
v3: improves on the v2 training script with:

  1. Full metrics per epoch -- WA, UA, macro-F1, weighted-F1, per-class recall & F1
  2. Checkpoint on macro-F1  (not unweighted accuracy)
  3. Feature masking  -- randomly zero valid frames during training
  4. R-Drop (correct)  -- same masked input forwarded twice with different dropout
                          masks; symmetric KL added to CE.  alpha=0.05 (gentle).

Everything else (AMP, grad_accum, cosine+warmup, beta_entropy, build_model,
num_workers=0) is identical to v2.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, recall_score
)
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.fusion_with_emotion_decoder import FusionWithEmotionDecoder
from models.mosei_fusion_with_emotion_decoder import MoseiFusionWithEmotionDecoder


# ---------------------------------------------------------------
# Seed
# ---------------------------------------------------------------

def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Args
# ---------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv",       type=str, default="data/iemocap_index_splits.csv")
    ap.add_argument("--audio_dir", type=str, default="features/seq_level/audio")
    ap.add_argument("--text_dir",  type=str, default="features/seq_level/text")

    ap.add_argument("--uid_col",          type=str, default="utter_id")
    ap.add_argument("--label_col",        type=str, default="label")
    ap.add_argument("--split_col",        type=str, default="split")
    ap.add_argument("--train_split_name", type=str, default="train")
    ap.add_argument("--val_split_name",   type=str, default="val")

    # 4-class IEMOCAP setup
    ap.add_argument("--exclude_labels", type=str, default="",
                    help="Comma-separated labels to drop. E.g. 'frustration'")
    ap.add_argument("--merge_labels",   type=str, default="",
                    help="Comma-separated src=dst remappings. E.g. 'excited=happy'")

    # Model (v2 defaults)
    ap.add_argument("--d_model",            type=int,   default=256)
    ap.add_argument("--n_heads",            type=int,   default=4)
    ap.add_argument("--num_layers_fusion",  type=int,   default=1)
    ap.add_argument("--num_layers_decoder", type=int,   default=2)
    ap.add_argument("--beta_hidden",        type=int,   default=64)
    ap.add_argument("--dropout",            type=float, default=0.4)

    ap.add_argument("--loss_type",        type=str,   default="single_label",
                    choices=["single_label", "multi_label"])
    ap.add_argument("--label_smoothing",  type=float, default=0.0,
                    help="Label smoothing for CrossEntropyLoss (0 = off).")

    # Training (v2 defaults)
    ap.add_argument("--batch_size",    type=int,   default=16)
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--lr",            type=float, default=5e-5)
    ap.add_argument("--weight_decay",  type=float, default=0.05)
    ap.add_argument("--grad_accum",    type=int,   default=2)
    ap.add_argument("--warmup_ratio",  type=float, default=0.1)
    ap.add_argument("--beta_entropy",  type=float, default=1e-3)
    ap.add_argument("--max_len_audio", type=int,   default=300)
    ap.add_argument("--max_len_text",  type=int,   default=128)

    # v3 additions
    ap.add_argument("--feat_mask_prob", type=float, default=0.10,
                    help="Fraction of valid frames to zero during training.")
    ap.add_argument("--rdrop_alpha",    type=float, default=0.05,
                    help="Weight for R-Drop symmetric-KL loss. 0 = disabled.")

    ap.add_argument("--out_dir", type=str, default="runs/iemocap_fusion_seq_decoder_v3")
    ap.add_argument("--seed",    type=int, default=1234)

    return ap.parse_args()


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class SeqLevelFusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        text_dir: Path,
        uid_col: str,
        label_col: str,
        label2id: Dict[str, int],
        loss_type: str = "single_label",
        max_len_audio=None,
        max_len_text=None,
    ):
        self.df            = df.reset_index(drop=True)
        self.audio_dir     = audio_dir
        self.text_dir      = text_dir
        self.uid_col       = uid_col
        self.label_col     = label_col
        self.label2id      = label2id
        self.loss_type     = loss_type
        self.max_len_audio = max_len_audio
        self.max_len_text  = max_len_text

        keep = [i for i, row in self.df.iterrows()
                if (audio_dir / f"{row[uid_col]}.pt").is_file()
                and (text_dir  / f"{row[uid_col]}.pt").is_file()]
        dropped = len(self.df) - len(keep)
        if dropped:
            print(f"[Dataset] Dropped {dropped} rows with missing features.")
        self.df = self.df.iloc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load(self, path: Path, max_len) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = torch.load(path, map_location="cpu")
        h = obj["hidden"].float()
        m = (obj["attention_mask"].long() == 0)
        if max_len is not None and h.size(0) > max_len:
            h, m = h[:max_len], m[:max_len]
        return h, m

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = str(row[self.uid_col])
        h_a, m_a = self._load(self.audio_dir / f"{uid}.pt", self.max_len_audio)
        h_t, m_t = self._load(self.text_dir  / f"{uid}.pt", self.max_len_text)
        if self.loss_type == "single_label":
            label = self.label2id[row[self.label_col]]
        else:
            y = torch.zeros(len(self.label2id), dtype=torch.float32)
            y[self.label2id[row[self.label_col]]] = 1.0
            label = y
        return h_a, m_a, h_t, m_t, label


def collate_seq_batch(batch, loss_type: str):
    hs_a, ms_a, hs_t, ms_t, labels = zip(*batch)
    B, d = len(batch), hs_a[0].size(-1)
    L_a = max(x.size(0) for x in hs_a)
    L_t = max(x.size(0) for x in hs_t)

    ph_a = torch.zeros(B, L_a, d);  pm_a = torch.ones(B, L_a, dtype=torch.bool)
    ph_t = torch.zeros(B, L_t, d);  pm_t = torch.ones(B, L_t, dtype=torch.bool)
    for i in range(B):
        la, lt = hs_a[i].size(0), hs_t[i].size(0)
        ph_a[i, :la] = hs_a[i];  pm_a[i, :la] = ms_a[i]
        ph_t[i, :lt] = hs_t[i];  pm_t[i, :lt] = ms_t[i]

    labels_out = (torch.tensor(labels, dtype=torch.long)
                  if loss_type == "single_label" else torch.stack(labels))
    return ph_a, pm_a, ph_t, pm_t, labels_out


# ---------------------------------------------------------------
# Feature masking
# ---------------------------------------------------------------

def apply_feat_mask(h: torch.Tensor, m: torch.Tensor, p: float) -> torch.Tensor:
    """Zero valid (non-PAD) frames with probability p."""
    if p <= 0.0:
        return h
    valid = ~m
    noise = torch.rand_like(h[:, :, 0]) < p
    mask  = (valid & noise).unsqueeze(-1)
    return h.masked_fill(mask, 0.0)


# ---------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------

def beta_entropy_loss(beta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b   = torch.clamp(beta, eps, 1.0 - eps)
    ent = -(b * torch.log(b) + (1 - b) * torch.log(1 - b))
    return ent.mean()


def symmetric_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    return 0.5 * (F.kl_div(p1, p2.exp(), reduction="batchmean") +
                  F.kl_div(p2, p1.exp(), reduction="batchmean"))


def compute_ce(logits, labels, beta, criterion, beta_w):
    loss = criterion(logits, labels)
    if beta is not None and beta_w > 0:
        loss = loss + beta_w * beta_entropy_loss(beta)
    return loss


# ---------------------------------------------------------------
# Data / model builders
# ---------------------------------------------------------------

def create_label_mapping(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    labels = sorted(df[label_col].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    print("[Labels]", label2id)
    return label2id


def apply_label_transforms(df: pd.DataFrame, label_col: str,
                           exclude_labels: str, merge_labels: str) -> pd.DataFrame:
    # merge: "excited=happy,other=happy" → remap column values
    if merge_labels:
        remap = {}
        for pair in merge_labels.split(","):
            src, dst = pair.strip().split("=")
            remap[src.strip()] = dst.strip()
        df = df.copy()
        df[label_col] = df[label_col].map(lambda x: remap.get(x, x))
        print(f"[Labels] merged: {remap}")

    # exclude: "frustration,other" → drop those rows
    if exclude_labels:
        drop = {s.strip() for s in exclude_labels.split(",") if s.strip()}
        before = len(df)
        df = df[~df[label_col].isin(drop)].reset_index(drop=True)
        print(f"[Labels] excluded {drop}: {before} -> {len(df)} rows")

    return df


def get_dataloaders(args):
    df       = pd.read_csv(args.csv)
    df       = apply_label_transforms(df, args.label_col,
                                      args.exclude_labels, args.merge_labels)
    label2id = create_label_mapping(df, args.label_col)
    audio_dir, text_dir = Path(args.audio_dir), Path(args.text_dir)

    train_df = df[df[args.split_col] == args.train_split_name]
    val_df   = df[df[args.split_col] == args.val_split_name]

    kw = dict(uid_col=args.uid_col, label_col=args.label_col, label2id=label2id,
              loss_type=args.loss_type,
              max_len_audio=args.max_len_audio, max_len_text=args.max_len_text)
    train_ds = SeqLevelFusionDataset(train_df, audio_dir, text_dir, **kw)
    val_ds   = SeqLevelFusionDataset(val_df,   audio_dir, text_dir, **kw)

    cf = lambda b: collate_seq_batch(b, args.loss_type)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=cf)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, collate_fn=cf)
    return train_loader, val_loader, label2id


def _read_feat_dim(feat_dir: Path) -> int:
    meta = json.loads((feat_dir / "meta.json").read_text(encoding="utf-8"))
    return int(meta.get("hidden_dim", meta.get("dim", 768)))


def build_model(args, num_emotions, d_audio, d_text, device):
    common = dict(num_emotions=num_emotions, n_heads=args.n_heads,
                  num_layers_fusion=args.num_layers_fusion,
                  num_layers_decoder=args.num_layers_decoder,
                  beta_hidden=args.beta_hidden, dropout=args.dropout)
    if args.d_model != d_audio or args.d_model != d_text:
        print(f"[Model] projection: audio={d_audio}, text={d_text} -> d_model={args.d_model}")
        return MoseiFusionWithEmotionDecoder(
            d_audio=d_audio, d_text=d_text, d_model=args.d_model, **common
        ).to(device)
    return FusionWithEmotionDecoder(d_model=args.d_model, **common).to(device)


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def compute_metrics(all_preds: List[int], all_labels: List[int],
                    id2label: Dict[int, str]) -> Dict:
    names = [id2label[i] for i in sorted(id2label)]
    wa    = accuracy_score(all_labels, all_preds)
    ua    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    mf1   = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    wf1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    per_recall = recall_score(all_labels, all_preds, average=None,
                              labels=sorted(id2label), zero_division=0)
    per_f1     = f1_score(all_labels, all_preds, average=None,
                          labels=sorted(id2label), zero_division=0)
    return {
        "WA": wa, "UA": ua, "macro_F1": mf1, "weighted_F1": wf1,
        "per_recall": {n: r for n, r in zip(names, per_recall)},
        "per_f1":     {n: f for n, f in zip(names, per_f1)},
    }


def print_metrics(m: Dict, prefix: str = "Val"):
    print(f"  {prefix}  WA={m['WA']:.4f}  UA={m['UA']:.4f}  "
          f"macro-F1={m['macro_F1']:.4f}  wt-F1={m['weighted_F1']:.4f}")
    rec_str = "  ".join(f"{k}:{v:.2f}" for k, v in m["per_recall"].items())
    f1_str  = "  ".join(f"{k}:{v:.2f}" for k, v in m["per_f1"].items())
    print(f"  recall  : {rec_str}")
    print(f"  F1      : {f1_str}")


# ---------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, scaler, args):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    optimizer.zero_grad(set_to_none=True)

    for step, (h_a, m_a, h_t, m_t, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels    = labels.to(device)

        # Apply masking ONCE; share the masked input across R-Drop passes
        h_a_aug = apply_feat_mask(h_a, m_a, args.feat_mask_prob)
        h_t_aug = apply_feat_mask(h_t, m_t, args.feat_mask_prob)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits1, beta1, _ = model(h_a_aug, h_t_aug, m_a, m_t)
            loss = compute_ce(logits1, labels, beta1, criterion, args.beta_entropy)

            # R-Drop: second forward on the SAME masked input, different dropout mask
            if args.rdrop_alpha > 0:
                logits2, beta2, _ = model(h_a_aug, h_t_aug, m_a, m_t)
                ce2  = compute_ce(logits2, labels, beta2, criterion, args.beta_entropy)
                loss = 0.5 * (loss + ce2) + args.rdrop_alpha * symmetric_kl(logits1, logits2)

            loss_scaled = loss / args.grad_accum

        scaler.scale(loss_scaled).backward()

        if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        preds = logits1.detach().argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.detach().float().item() * labels.size(0)

    n = max(len(all_labels), 1)
    return total_loss / n, accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, id2label):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for h_a, m_a, h_t, m_t, labels in tqdm(loader, desc="Val", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels    = labels.to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, beta, _ = model(h_a, h_t, m_a, m_t)
            loss = compute_ce(logits, labels, beta, criterion, args.beta_entropy)

        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item() * labels.size(0)

    n = max(len(all_labels), 1)
    metrics = compute_metrics(all_preds, all_labels, id2label)
    return total_loss / n, metrics


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, label2id = get_dataloaders(args)
    id2label     = {v: k for k, v in label2id.items()}
    num_emotions = len(label2id)
    d_audio = _read_feat_dim(Path(args.audio_dir))
    d_text  = _read_feat_dim(Path(args.text_dir))

    print(f"[Info] num_emotions={num_emotions}  d_audio={d_audio}  d_text={d_text}  device={device}")
    print(f"[v3]  feat_mask_prob={args.feat_mask_prob}  rdrop_alpha={args.rdrop_alpha}")

    model     = build_model(args, num_emotions, d_audio, d_text, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = (nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
                 if args.loss_type == "single_label" else nn.BCEWithLogitsLoss())

    total_steps  = (len(train_loader) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[Scheduler] total_steps={total_steps}  warmup_steps={warmup_steps}")

    best_macro_f1 = 0.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_wa = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, args)

        val_loss, vm = evaluate(model, val_loader, criterion, device, args, id2label)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Train  Loss={train_loss:.4f}  WA={train_wa:.4f}  |  lr={lr_now:.2e}")
        print_metrics(vm)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "train_wa": train_wa,
                         "val_loss": val_loss, **{k: vm[k] for k in ("WA","UA","macro_F1","weighted_F1")}})

        if vm["macro_F1"] > best_macro_f1:
            best_macro_f1 = vm["macro_F1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "macro_F1": best_macro_f1,
                "val_metrics": vm,
                "epoch": epoch,
                "args": vars(args),
                "label2id": label2id,
            }, out_dir / "best_fusion_seq_decoder.pt")
            print(f"  [Saved] new best macro-F1 -> {best_macro_f1:.4f}")

    print(f"\n[Done] Best val macro-F1 = {best_macro_f1:.4f}")

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_macro_f1": best_macro_f1, "label2id": label2id,
                   "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
