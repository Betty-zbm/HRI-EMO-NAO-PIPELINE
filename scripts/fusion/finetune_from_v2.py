#!/usr/bin/env python3
"""
Fine-tune from the v2 checkpoint with class-weighted loss.

v2 was trained with MoseiFusionWithEmotionDecoder (audio_proj + text_proj + backbone).
This script loads that checkpoint directly and continues training with:
  - Class-weighted CrossEntropyLoss  (addresses heavy class imbalance)
  - Linear warmup + cosine decay     (smooth convergence from a good init)
  - Low peak LR                      (don't destroy what v2 learned)
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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

    ap.add_argument("--checkpoint", type=str,
                    default="runs/iemocap_fusion_seq_decoder_v2/best_fusion_seq_decoder.pt")
    ap.add_argument("--csv",       type=str, default="data/iemocap_index_splits.csv")
    ap.add_argument("--audio_dir", type=str, default="features/seq_level/audio")
    ap.add_argument("--text_dir",  type=str, default="features/seq_level/text")

    ap.add_argument("--uid_col",          type=str, default="utter_id")
    ap.add_argument("--label_col",        type=str, default="label")
    ap.add_argument("--split_col",        type=str, default="split")
    ap.add_argument("--train_split_name", type=str, default="train")
    ap.add_argument("--val_split_name",   type=str, default="val")

    # Model dims — must match the v2 checkpoint exactly
    ap.add_argument("--d_audio",          type=int, default=768)
    ap.add_argument("--d_text",           type=int, default=768)
    ap.add_argument("--d_model",          type=int, default=256)
    ap.add_argument("--n_heads",          type=int, default=4)
    ap.add_argument("--num_layers_fusion",type=int, default=1)
    ap.add_argument("--num_layers_decoder",type=int,default=2)
    ap.add_argument("--beta_hidden",      type=int, default=64)
    ap.add_argument("--dropout",          type=float, default=0.4)

    # Loss
    ap.add_argument("--label_smoothing",  type=float, default=0.05)

    # Sequence truncation
    ap.add_argument("--max_len_audio", type=int, default=300)
    ap.add_argument("--max_len_text",  type=int, default=128)

    # Fine-tuning schedule
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--lr",           type=float, default=2e-5,
                    help="Peak LR — keep low to avoid catastrophic forgetting.")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_accum",   type=int,   default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--beta_entropy", type=float, default=0.001)

    ap.add_argument("--out_dir", type=str, default="runs/iemocap_fusion_seq_decoder_v3_ft")
    ap.add_argument("--seed",    type=int, default=1234)

    return ap.parse_args()


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class SeqLevelDataset(Dataset):
    def __init__(self, rows, audio_dir, text_dir, uid_col, label_col, label2id,
                 max_len_audio=0, max_len_text=0):
        self.audio_dir     = Path(audio_dir)
        self.text_dir      = Path(text_dir)
        self.uid_col       = uid_col
        self.label_col     = label_col
        self.label2id      = label2id
        self.max_len_audio = max_len_audio
        self.max_len_text  = max_len_text

        self.rows = [r for r in rows
                     if (self.audio_dir / f"{r[uid_col]}.pt").is_file()
                     and (self.text_dir / f"{r[uid_col]}.pt").is_file()]
        dropped = len(rows) - len(self.rows)
        if dropped:
            print(f"[Dataset] Dropped {dropped} rows with missing features.")

    def __len__(self):
        return len(self.rows)

    def _load(self, path: Path, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = torch.load(path, map_location="cpu")
        h = obj["hidden"].float()
        m = (obj["attention_mask"].long() == 0)   # True = PAD
        if max_len > 0 and h.size(0) > max_len:
            h, m = h[:max_len], m[:max_len]
        return h, m

    def __getitem__(self, idx):
        row  = self.rows[idx]
        uid  = row[self.uid_col]
        h_a, m_a = self._load(self.audio_dir / f"{uid}.pt", self.max_len_audio)
        h_t, m_t = self._load(self.text_dir  / f"{uid}.pt", self.max_len_text)
        label = self.label2id[row[self.label_col]]
        return h_a, m_a, h_t, m_t, label


def collate_fn(batch):
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

    return ph_a, pm_a, ph_t, pm_t, torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_class_weights(train_rows, label_col, label2id):
    counts = Counter(r[label_col] for r in train_rows)
    n = len(label2id)
    w = torch.zeros(n)
    for lab, idx in label2id.items():
        w[idx] = 1.0 / max(counts.get(lab, 1), 1)
    w = w / w.sum() * n
    print("[Class weights]", {k: f"{w[v]:.3f}" for k, v in label2id.items()})
    return w


def get_lr_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return fn


# ---------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, grad_accum, beta_entropy, global_step):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()

    for step, (h_a, m_a, h_t, m_t, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels    = labels.to(device)

        logits, beta, _ = model(h_a, h_t, m_a, m_t)
        loss  = criterion(logits, labels)
        loss -= beta_entropy * (beta * (1 - beta)).mean()
        loss  = loss / grad_accum
        loss.backward()

        correct    += (logits.argmax(-1) == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item() * grad_accum * labels.size(0)

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    return total_loss / max(total, 1), correct / max(total, 1), global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for h_a, m_a, h_t, m_t, labels in tqdm(loader, desc="Val", leave=False):
        h_a, m_a = h_a.to(device), m_a.to(device)
        h_t, m_t = h_t.to(device), m_t.to(device)
        labels    = labels.to(device)

        logits, _, _ = model(h_a, h_t, m_a, m_t)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct    += (logits.argmax(-1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    all_rows    = read_csv(args.csv)
    labels_all  = sorted({r[args.label_col] for r in all_rows})
    label2id    = {lab: i for i, lab in enumerate(labels_all)}
    print("[Labels]", label2id)

    train_rows = [r for r in all_rows if r[args.split_col] == args.train_split_name]
    val_rows   = [r for r in all_rows if r[args.split_col] == args.val_split_name]

    train_ds = SeqLevelDataset(train_rows, args.audio_dir, args.text_dir,
                               args.uid_col, args.label_col, label2id,
                               args.max_len_audio, args.max_len_text)
    val_ds   = SeqLevelDataset(val_rows,   args.audio_dir, args.text_dir,
                               args.uid_col, args.label_col, label2id,
                               args.max_len_audio, args.max_len_text)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)

    num_emotions = len(label2id)
    print(f"[Info] num_emotions={num_emotions}  device={device}")

    # ---- model — exact same architecture as v2 ----
    model = MoseiFusionWithEmotionDecoder(
        d_audio=args.d_audio,
        d_text=args.d_text,
        d_model=args.d_model,
        num_emotions=num_emotions,
        n_heads=args.n_heads,
        num_layers_fusion=args.num_layers_fusion,
        num_layers_decoder=args.num_layers_decoder,
        beta_hidden=args.beta_hidden,
        dropout=args.dropout,
    ).to(device)

    # ---- load v2 checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[Checkpoint] loaded from epoch {ckpt.get('epoch','?')}  "
          f"(v2 val_acc={ckpt.get('val_acc', 0):.4f})")
    if missing:
        print(f"  missing keys:    {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Trainable params: {total_params:,}")

    # ---- loss with class weights ----
    class_weights = compute_class_weights(train_rows, args.label_col, label2id).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # ---- optimizer + schedule ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(args.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_lambda(warmup_steps, total_steps)
    )
    print(f"[Schedule] total_steps={total_steps}  warmup_steps={warmup_steps}  peak_lr={args.lr}")

    # ---- fine-tune ----
    best_val_acc = ckpt.get("val_acc", 0.0)  # start from v2's best
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, args.grad_accum, args.beta_entropy, global_step,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}  ||  "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}  |  lr: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "epoch": epoch,
                "args": vars(args),
                "label2id": label2id,
            }, out_dir / "best_fusion_seq_decoder.pt")
            print(f"  [Saved] new best -> {best_val_acc:.4f}")

    print(f"\n[Done] Best val acc = {best_val_acc:.4f}")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val_acc, "label2id": label2id,
                   "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
