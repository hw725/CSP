#!/usr/bin/env python3
"""
Semantic Cross-Lingual Boundary 모델 학습 스크립트

사전계산된 BGE-M3 임베딩을 사용하여 경계 head만 학습.
BGE-M3 모델 로드 불필요 → 매우 빠른 학습.
"""

import argparse
import os
import sys
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── 프로젝트 루트 설정 ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.semantic_boundary_model import (
    SemanticCrossLingualBoundary,
    DiceBCELoss,
    FocalBoundaryLoss,
    BGE_M3_DIM,
    POS_DIM,
)


class PrecomputedBoundaryDataset(Dataset):
    """사전계산된 임베딩 데이터셋"""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "src_emb": s["src_emb"].float(),
            "tgt_emb": s["tgt_emb"].float(),
            "pos_feat": s["pos_feat"].float(),
            "labels": s["labels"].float(),
        }


def collate_fn(batch):
    """가변 길이 배치를 패딩"""
    max_src = max(b["src_emb"].shape[0] for b in batch)
    max_tgt = max(b["tgt_emb"].shape[0] for b in batch)

    B = len(batch)
    src_dim = batch[0]["src_emb"].shape[-1]
    tgt_dim = batch[0]["tgt_emb"].shape[-1]
    pos_dim = batch[0]["pos_feat"].shape[-1]

    src_emb = torch.zeros(B, max_src, src_dim)
    tgt_emb = torch.zeros(B, max_tgt, tgt_dim)
    pos_feat = torch.zeros(B, max_src, pos_dim)
    labels = torch.zeros(B, max_src)
    src_mask = torch.ones(B, max_src, dtype=torch.bool)  # True = pad
    tgt_mask = torch.ones(B, max_tgt, dtype=torch.bool)

    for i, b in enumerate(batch):
        sl = b["src_emb"].shape[0]
        tl = b["tgt_emb"].shape[0]
        src_emb[i, :sl] = b["src_emb"]
        tgt_emb[i, :tl] = b["tgt_emb"]
        pos_feat[i, :sl] = b["pos_feat"]
        labels[i, :sl] = b["labels"]
        src_mask[i, :sl] = False
        tgt_mask[i, :tl] = False

    return {
        "src_emb": src_emb,
        "tgt_emb": tgt_emb,
        "pos_feat": pos_feat,
        "labels": labels,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "valid_mask": ~src_mask,  # True = valid
    }


def compute_f1(logits, labels, valid_mask, threshold_logit=0.0):
    """F1 계산 (logit 공간에서 threshold 적용)"""
    preds = (logits > threshold_logit).float() * valid_mask.float()
    targets = labels * valid_mask.float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets) * valid_mask.float()).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def split_by_book(samples, val_ratio=0.2, seed=42):
    """책명 기준 train/val split"""
    books = list(set(s["book"] for s in samples))
    random.Random(seed).shuffle(books)

    val_count = max(1, int(len(books) * val_ratio))
    val_books = set(books[:val_count])

    train_samples = [s for s in samples if s["book"] not in val_books]
    val_samples = [s for s in samples if s["book"] in val_books]

    return train_samples, val_samples, val_books


def evaluate(model, dataloader, criterion, device):
    """검증 루프"""
    model.eval()
    total_loss = 0
    total_tp = total_fp = total_fn = 0
    n_batches = 0

    # threshold grid search
    thresholds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_f1 = 0
    best_thr = 0.0

    with torch.no_grad():
        all_logits = []
        all_labels = []
        all_masks = []

        for batch in dataloader:
            src_emb = batch["src_emb"].to(device)
            tgt_emb = batch["tgt_emb"].to(device)
            pos_feat = batch["pos_feat"].to(device)
            labels = batch["labels"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            logits = model(src_emb, tgt_emb, pos_feat, src_mask, tgt_mask)
            loss = criterion(logits, labels, valid_mask)

            total_loss += loss.item()
            n_batches += 1

            # 배치마다 패딩 길이가 다르므로 유효 위치만 flatten
            vm = valid_mask.cpu()
            all_logits.append(logits.cpu()[vm])
            all_labels.append(labels.cpu()[vm])

        # 전체 배치에서 best threshold 탐색
        all_logits = torch.cat(all_logits, dim=0)  # [N_valid]
        all_labels = torch.cat(all_labels, dim=0)  # [N_valid]

        for thr in thresholds:
            preds = (all_logits > thr).float()
            tp = (preds * all_labels).sum().item()
            fp = (preds * (1 - all_labels)).sum().item()
            fn = ((1 - preds) * all_labels).sum().item()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, best_f1, best_thr


def main():
    parser = argparse.ArgumentParser(description="Semantic Boundary 모델 학습")
    parser.add_argument(
        "--data",
        default="datasets/precomputed/semantic_boundary/precomputed_all.pt",
        help="사전계산 데이터 경로",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--n-attn-layers", type=int, default=2)
    parser.add_argument("--n-attn-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--bce-pos-weight", type=float, default=1.0,
                        help="BCE의 positive class 가중치 (경계=희소 → 높이면 FN 페널티 증가)")
    parser.add_argument(
        "--loss", choices=["dice_bce", "focal"], default="dice_bce",
        help="손실 함수: dice_bce (기존) 또는 focal (Focal+경계가중치)",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-pos-weight", type=float, default=10.0)
    parser.add_argument("--focal-near-weight", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="models/semantic_boundary.pt", help="모델 저장 경로"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    # ── 데이터 로드 ──
    print(f"📂 데이터 로드: {args.data}")
    samples = torch.load(args.data, map_location="cpu", weights_only=False)
    print(f"  총 {len(samples)}개 샘플")

    # ── Train/Val 분할 ──
    train_samples, val_samples, val_books = split_by_book(samples, seed=args.seed)
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"  Val 책: {val_books}")

    train_ds = PrecomputedBoundaryDataset(train_samples)
    val_ds = PrecomputedBoundaryDataset(val_samples)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # ── 모델 초기화 ──
    model = SemanticCrossLingualBoundary(
        src_dim=BGE_M3_DIM,
        tgt_dim=BGE_M3_DIM,
        pos_dim=POS_DIM,
        proj_dim=args.proj_dim,
        n_attn_heads=args.n_attn_heads,
        n_attn_layers=args.n_attn_layers,
        dropout=args.dropout,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️ 모델: {total_params:,} trainable params")

    # ── 학습 설정 ──
    if args.loss == "focal":
        criterion = FocalBoundaryLoss(
            gamma=args.focal_gamma,
            pos_weight=args.focal_pos_weight,
            near_weight=args.focal_near_weight,
            dice_weight=args.dice_weight,
        )
        print(f"📊 Loss: FocalBoundaryLoss (gamma={args.focal_gamma}, "
              f"pos_w={args.focal_pos_weight}, near_w={args.focal_near_weight})")
    else:
        criterion = DiceBCELoss(dice_weight=args.dice_weight, pos_weight=args.bce_pos_weight)
        if args.bce_pos_weight != 1.0:
            print(f"📊 Loss: DiceBCE (dice_w={args.dice_weight}, bce_pos_w={args.bce_pos_weight})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 학습 루프 ──
    best_val_f1 = 0.0
    patience_counter = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 학습 시작 (epochs={args.epochs}, batch={args.batch}, lr={args.lr})")
    print(f"   dice_weight={args.dice_weight}, bce_pos_weight={args.bce_pos_weight}, proj_dim={args.proj_dim}")
    print(f"   attn_layers={args.n_attn_layers}, heads={args.n_attn_heads}")
    print("=" * 70)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_dl:
            src_emb = batch["src_emb"].to(device)
            tgt_emb = batch["tgt_emb"].to(device)
            pos_feat = batch["pos_feat"].to(device)
            labels = batch["labels"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            logits = model(src_emb, tgt_emb, pos_feat, src_mask, tgt_mask)
            loss = criterion(logits, labels, valid_mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # 검증
        val_loss, val_f1, val_thr = evaluate(model, val_dl, criterion, device)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_F1={val_f1:.4f} (thr={val_thr:.1f}) | "
            f"lr={lr_now:.2e}"
        )

        # Best 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "encoder_model": "BAAI/bge-m3",
                    "max_len": 512,
                    "model_type": "semantic_crosslingual",
                    "tasks": ["pa"],
                    "proj_dim": args.proj_dim,
                    "n_attn_layers": args.n_attn_layers,
                    "n_attn_heads": args.n_attn_heads,
                    "dropout": args.dropout,
                    "lstm_hidden": args.lstm_hidden,
                    "best_threshold": val_thr,
                    "best_val_f1": val_f1,
                    "epoch": epoch + 1,
                },
                str(output_path),
            )
            print(f"  ✅ Best 모델 저장 (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ Early stopping (patience={args.patience})")
                break

    print("=" * 70)
    print(f"🏆 최종 Best Val F1: {best_val_f1:.4f}")
    print(f"💾 모델 저장: {output_path}")


if __name__ == "__main__":
    main()
