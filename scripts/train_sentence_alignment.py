#!/usr/bin/env python3
"""P2S Alignment 모델 학습 (BoundaryAwareDualEncoder)

원문 문장 <-> 번역문 문장 정렬 모델을 학습한다.
- Production 모델(boundary_aware_alignment_loader.py)의 BoundaryAwareDualEncoder 아키텍처와 동일
- 학습 데이터: sentence_train.tsv (원문, 번역문 쌍)
- Contrastive loss + Boundary classifier loss
- Validation + Early stopping + CosineAnnealingLR

Usage:
    python scripts/train_sentence_alignment.py --epochs 20 --batch 128
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Production 모델 클래스 재사용
from common.boundary_aware_alignment_loader import (
    BoundaryAwareCharEncoder,
    BoundaryAwareDualEncoder,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
MODELS_ROOT = WORKSPACE_ROOT / "models"


def load_sentence_pairs(xlsx_path: Path) -> List[Dict]:
    """
    sentence_train.tsv에서 (원문, 번역문, 책명) 쌍을 로드한다.

    Returns:
        [{"src": "원문텍스트", "tgt": "번역문텍스트", "book": "책명"}, ...]
    """
    df = pd.read_csv(xlsx_path, sep='\t')

    for col in ("원문", "번역문"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    samples = []
    for _, row in df.iterrows():
        src = str(row["원문"]).strip()
        tgt = str(row["번역문"]).strip()
        book = str(row.get("책명", ""))

        if src and tgt:
            samples.append({"src": src, "tgt": tgt, "book": book})

    print(f"  유효 문장 쌍: {len(samples)}개")
    return samples


def build_vocabs(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """원문/번역문 각각의 vocab 구축"""
    src_chars = set()
    tgt_chars = set()
    for s in samples:
        src_chars.update(list(s["src"]))
        tgt_chars.update(list(s["tgt"]))

    vocab_src = {c: i + 1 for i, c in enumerate(sorted(src_chars))}
    vocab_tgt = {c: i + 1 for i, c in enumerate(sorted(tgt_chars))}
    return vocab_src, vocab_tgt


class AlignmentDataset(Dataset):
    """원문-번역문 정렬 학습 데이터셋"""

    def __init__(self, samples: List[Dict], vocab_src: Dict, vocab_tgt: Dict, max_len: int = 512):
        self.samples = samples
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def _encode(self, text: str, vocab: Dict) -> torch.Tensor:
        ids = [vocab.get(ch, 0) for ch in text][:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src_enc = self._encode(s["src"], self.vocab_src)
        tgt_enc = self._encode(s["tgt"], self.vocab_tgt)
        return src_enc, tgt_enc


def contrastive_loss(v_src: torch.Tensor, v_tgt: torch.Tensor, temperature: float = 0.07):
    """
    Symmetric contrastive loss (in-batch negatives).
    v_src, v_tgt: [B, 256], L2-normalized
    """
    sim = torch.matmul(v_src, v_tgt.T) / temperature  # [B, B]
    labels = torch.arange(v_src.size(0), device=v_src.device)

    loss_s2t = nn.CrossEntropyLoss()(sim, labels)
    loss_t2s = nn.CrossEntropyLoss()(sim.T, labels)
    return (loss_s2t + loss_t2s) / 2


def boundary_classifier_loss(
    model: BoundaryAwareDualEncoder,
    v_src: torch.Tensor,
    v_tgt: torch.Tensor,
):
    """
    Boundary classifier loss: positive=aligned pair, negative=shifted pair.
    v_src, v_tgt: [B, 256]
    """
    B = v_src.size(0)
    if B < 2:
        return torch.tensor(0.0, device=v_src.device)

    # Positive pairs: (src[i], tgt[i]) -> label=1
    pos_combined = torch.cat([v_src, v_tgt], dim=-1)  # [B, 512]
    pos_scores = model.boundary_classifier(pos_combined).squeeze(-1)  # [B]
    pos_labels = torch.ones(B, device=v_src.device)

    # Negative pairs: (src[i], tgt[(i+1)%B]) -> label=0
    tgt_shifted = torch.roll(v_tgt, shifts=1, dims=0)
    neg_combined = torch.cat([v_src, tgt_shifted], dim=-1)  # [B, 512]
    neg_scores = model.boundary_classifier(neg_combined).squeeze(-1)  # [B]
    neg_labels = torch.zeros(B, device=v_src.device)

    # Combined BCE loss
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([pos_labels, neg_labels])
    loss = nn.BCELoss()(all_scores, all_labels)
    return loss


def compute_val_metrics(
    model: BoundaryAwareDualEncoder,
    val_loader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
):
    """Validation 메트릭 계산: contrastive loss + boundary accuracy"""
    model.eval()
    total_closs = 0.0
    total_bloss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src_ids, tgt_ids in val_loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            zs, zt, _ = model(src_ids, tgt_ids, compute_boundary_match=True)

            # Contrastive loss
            closs = contrastive_loss(zs, zt, temperature)
            total_closs += closs.item()

            # Boundary accuracy (positive pairs)
            B = zs.size(0)
            if B >= 2:
                pos_combined = torch.cat([zs, zt], dim=-1)
                pos_scores = model.boundary_classifier(pos_combined).squeeze(-1)
                correct += (pos_scores >= 0.5).sum().item()
                total += B

                tgt_shifted = torch.roll(zt, shifts=1, dims=0)
                neg_combined = torch.cat([zs, tgt_shifted], dim=-1)
                neg_scores = model.boundary_classifier(neg_combined).squeeze(-1)
                correct += (neg_scores < 0.5).sum().item()
                total += B

            n_batches += 1

    avg_closs = total_closs / max(1, n_batches)
    boundary_acc = correct / max(1, total)
    return avg_closs, boundary_acc


def main():
    parser = argparse.ArgumentParser(description="Train P2S Alignment Model (BoundaryAwareDualEncoder)")
    parser.add_argument(
        "--train-xlsx", type=str,
        default="datasets/splits/sentence_train.tsv",
        help="문장 Excel (train)",
    )
    parser.add_argument(
        "--val-xlsx", type=str,
        default="datasets/splits/sentence_val.tsv",
        help="문장 Excel (val)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
    parser.add_argument("--boundary-weight", type=float, default=0.3, help="Boundary loss 가중치")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="models/dual_encoder_alignment_p2s.pt")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train 데이터 로드
    train_xlsx = WORKSPACE_ROOT / args.train_xlsx
    assert train_xlsx.exists(), f"파일 없음: {train_xlsx}"
    print(f"[Train] 문장: {train_xlsx}")
    train_samples = load_sentence_pairs(train_xlsx)

    if not train_samples:
        print("유효한 학습 샘플이 없습니다.")
        return 1

    # Val 데이터 로드
    val_xlsx = WORKSPACE_ROOT / args.val_xlsx
    assert val_xlsx.exists(), f"파일 없음: {val_xlsx}"
    print(f"[Val] 문장: {val_xlsx}")
    val_samples = load_sentence_pairs(val_xlsx)

    # Vocab 구축 (train + val 통합)
    vocab_src, vocab_tgt = build_vocabs(train_samples + val_samples)
    print(f"Vocab: src={len(vocab_src)}자, tgt={len(vocab_tgt)}자")
    print(f"Train: {len(train_samples)}개, Val: {len(val_samples)}개")

    # Dataset & DataLoader
    train_ds = AlignmentDataset(train_samples, vocab_src, vocab_tgt, max_len=args.max_len)
    val_ds = AlignmentDataset(val_samples, vocab_src, vocab_tgt, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 모델 초기화 (production과 동일 아키텍처)
    model = BoundaryAwareDualEncoder(
        vocab_src=len(vocab_src) + 1,
        vocab_tgt=len(vocab_tgt) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"temperature={args.temperature}, boundary_weight={args.boundary_weight}, patience={args.patience}")
    print("-" * 60)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_closs = 0.0
        train_bloss = 0.0
        n_batches = 0

        for src_ids, tgt_ids in train_loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            # Forward: get embeddings only (boundary는 별도 계산)
            zs, zt = model(src_ids, tgt_ids, compute_boundary_match=False)

            # Contrastive loss
            closs = contrastive_loss(zs, zt, args.temperature)

            # Boundary classifier loss
            bloss = boundary_classifier_loss(model, zs, zt)

            # Combined loss
            loss = closs + args.boundary_weight * bloss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_closs += closs.item()
            train_bloss += bloss.item()
            n_batches += 1

        scheduler.step()
        avg_train_closs = train_closs / max(1, n_batches)
        avg_train_bloss = train_bloss / max(1, n_batches)

        # --- Validation ---
        val_closs, val_boundary_acc = compute_val_metrics(
            model, val_loader, device, args.temperature
        )

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_closs={avg_train_closs:.4f}  train_bloss={avg_train_bloss:.4f}  "
            f"val_closs={val_closs:.4f}  val_bacc={val_boundary_acc:.4f}  "
            f"lr={lr_now:.6f}"
        )

        # Early stopping (val contrastive loss 기준)
        if val_closs < best_val_loss:
            best_val_loss = val_closs
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  -> best val_closs 갱신: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    print("-" * 60)
    print(f"Best val contrastive loss: {best_val_loss:.4f}")

    # 모델 저장 (production loader 호환 형식)
    out_path = WORKSPACE_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_state = best_state if best_state is not None else model.state_dict()
    torch.save(
        {
            "state_dict": save_state,
            "vocab_src": vocab_src,
            "vocab_tgt": vocab_tgt,
            "max_len": args.max_len,
        },
        out_path,
    )

    print(f"Model saved: {out_path}")
    print(f"  vocab_src={len(vocab_src)}, vocab_tgt={len(vocab_tgt)}, max_len={args.max_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
