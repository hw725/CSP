#!/usr/bin/env python3
"""P2S Boundary 모델 학습 (MultiHeadBoundary - 시퀀스 라벨링)

문단 원문을 입력으로 받아 문장 경계 위치를 문자 단위로 예측한다.
- Production 모델(boundary_model_loader.py)의 MultiHeadBoundary 아키텍처와 동일
- 학습 데이터: paragraph_train.tsv + sentence_train.tsv로 경계 라벨 자동 생성
- Validation + Early stopping + CosineAnnealingLR

Usage:
    python scripts/train_p2s_crossattn_boundary.py --epochs 30 --batch 64
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Production 모델 클래스 재사용
from common.boundary_model_loader import CharEncoderForBoundary, MultiHeadBoundary

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
MODELS_ROOT = WORKSPACE_ROOT / "models"


def normalize_text(text: str) -> str:
    """공백/개행 + 편집 마커([, -, ]) 제거하여 정규화 (경계 매칭용)"""
    if pd.isna(text):
        return ""
    return (
        str(text)
        .replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
        .replace("[", "").replace("-", "").replace("]", "")
        .strip()
    )


def load_boundary_samples(
    para_xlsx: Path, sent_xlsx: Path
) -> List[Dict]:
    """
    paragraph + sentence xlsx에서 경계 라벨이 있는 학습 샘플을 생성한다.

    Returns:
        [{"text": "문단원문전체", "labels": [0,0,0,1,0,0,1,...], "book": "책명"}, ...]
        labels[i]=1 → i번째 문자가 새 문장의 시작점 (첫 문장 시작은 제외)
    """
    para_df = pd.read_csv(para_xlsx, sep='\t')
    sent_df = pd.read_csv(sent_xlsx, sep='\t')

    # 컬럼 정리
    for df in (para_df, sent_df):
        for col in ("원문", "번역문"):
            if col in df.columns:
                df[col] = df[col].fillna("")

    # sentence를 (책명, 문단식별자) 기준으로 그룹
    sent_groups = sent_df.groupby(["책명", "문단식별자"], sort=False)

    samples = []
    skipped = 0

    for _, para_row in para_df.iterrows():
        book = str(para_row["책명"])
        pid = para_row["문단식별자"]
        para_src = str(para_row["원문"]).strip()

        if not para_src:
            continue

        # 해당 문단의 문장들 가져오기
        key = (book, pid)
        try:
            sent_group = sent_groups.get_group(key)
        except KeyError:
            skipped += 1
            continue

        # 문장식별자 순서로 정렬
        sent_group = sent_group.sort_values("문장식별자")
        sentences = [str(s).strip() for s in sent_group["원문"].tolist() if str(s).strip()]

        if len(sentences) < 2:
            continue

        # 문단 원문 내에서 각 문장의 시작 위치를 찾아 라벨 생성
        para_norm = normalize_text(para_src)
        labels = [0] * len(para_norm)

        cursor = 0
        valid = True
        for i, sent in enumerate(sentences):
            sent_norm = normalize_text(sent)
            if not sent_norm:
                continue

            # 문단 내에서 이 문장이 시작하는 위치 찾기
            pos = para_norm.find(sent_norm, cursor)
            if pos == -1:
                # 정확 매칭 실패 → 이 문단은 건너뜀
                valid = False
                break

            if i > 0 and pos < len(labels):
                labels[pos] = 1  # 첫 문장 이후의 문장 시작점만 경계

            cursor = pos + len(sent_norm)

        if not valid:
            skipped += 1
            continue

        # 정규화된 텍스트와 라벨 저장
        samples.append({
            "text": para_norm,
            "labels": labels,
            "book": book,
            "task": "pa",
        })

    print(f"  유효 샘플: {len(samples)}개, 건너뜀: {skipped}개")
    return samples


def load_sa_boundary_samples(
    sent_xlsx: Path, phrase_xlsx: Path
) -> List[Dict]:
    """
    sentence + phrase xlsx에서 'sa' (문장→구) 경계 라벨이 있는 학습 샘플을 생성한다.
    최적화: phrase groups를 순회하며 sentence를 lookup (역방향).
    """
    sent_df = pd.read_csv(sent_xlsx, sep='\t', low_memory=False)
    phrase_df = pd.read_csv(phrase_xlsx, sep='\t', low_memory=False)

    for df in (sent_df, phrase_df):
        for col in ("원문", "번역문"):
            if col in df.columns:
                df[col] = df[col].fillna("")

    # sentence를 (책명, 문장식별자) → 원문 dict으로 변환 (벡터화)
    sent_lookup = dict(zip(
        zip(sent_df["책명"].astype(str), sent_df["문장식별자"]),
        sent_df["원문"].astype(str).str.strip(),
    ))

    # phrase를 (책명, 문장식별자) 기준으로 그룹
    phrase_groups = phrase_df.groupby(["책명", "문장식별자"], sort=False)

    samples = []
    skipped = 0

    for key, phrase_group in phrase_groups:
        book, sid = key
        book = str(book)

        sent_src = sent_lookup.get((book, sid), "")
        if not sent_src:
            skipped += 1
            continue

        phrase_group = phrase_group.sort_values("구식별자") if "구식별자" in phrase_group.columns else phrase_group
        phrases = [str(s).strip() for s in phrase_group["원문"].tolist() if str(s).strip()]

        if len(phrases) < 2:
            continue

        sent_norm = normalize_text(sent_src)
        labels = [0] * len(sent_norm)

        cursor = 0
        valid = True
        for i, phrase in enumerate(phrases):
            phrase_norm = normalize_text(phrase)
            if not phrase_norm:
                continue
            pos = sent_norm.find(phrase_norm, cursor)
            if pos == -1:
                valid = False
                break
            if i > 0 and pos < len(labels):
                labels[pos] = 1
            cursor = pos + len(phrase_norm)

        if not valid:
            skipped += 1
            continue

        samples.append({
            "text": sent_norm,
            "labels": labels,
            "book": book,
            "task": "sa",
        })

    print(f"  [sa] 유효 샘플: {len(samples)}개, 건너뜀: {skipped}개")
    return samples


def build_vocab(samples: List[Dict]) -> Dict[str, int]:
    """단일 통합 vocab 구축 (production 모델과 동일)"""
    chars = set()
    for s in samples:
        chars.update(list(s["text"]))
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}
    return vocab


TASK_TO_IDX = {"pa": 0, "sa": 1, "pd": 2}


class BoundarySeqDataset(Dataset):
    """시퀀스 라벨링용 경계 데이터셋"""

    def __init__(self, samples: List[Dict], vocab: Dict[str, int], max_len: int = 512):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text"]
        labels = s["labels"]
        task_idx = TASK_TO_IDX.get(s.get("task", "pa"), 0)

        # 인코딩
        ids = [self.vocab.get(ch, 0) for ch in text][:self.max_len]
        lbl = labels[:self.max_len]
        actual_len = len(ids)

        # 패딩
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
            lbl += [0] * (self.max_len - len(lbl))

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(lbl, dtype=torch.float32),
            torch.tensor(actual_len, dtype=torch.long),
            torch.tensor(task_idx, dtype=torch.long),
        )


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for binary classification (class-imbalanced sequence labeling)"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        # BCE loss per element
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
        )
        # Focal modulation
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE combined loss: directly optimizes F1-like metric + stable BCE.

    Dice loss naturally handles class imbalance without explicit pos_weight.
    Combined with BCE for stable gradients.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5,
                 pos_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Dice loss
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
        )

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def compute_f1(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor, threshold: float = 0.5):
    """Boundary F1 계산 (padding 제외)"""
    tp = fp = fn = 0
    batch_size = logits.size(0)

    for i in range(batch_size):
        length = lengths[i].item()
        pred = (torch.sigmoid(logits[i, :length]) >= threshold).long()
        gold = labels[i, :length].long()

        tp += ((pred == 1) & (gold == 1)).sum().item()
        fp += ((pred == 1) & (gold == 0)).sum().item()
        fn += ((pred == 0) & (gold == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_best_f1(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
    """여러 threshold를 탐색하여 최적 F1과 해당 threshold를 반환"""
    best_f1 = 0.0
    best_thr = 0.5
    best_p = 0.0
    best_r = 0.0
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p, r, f1 = compute_f1(logits, labels, lengths, threshold=thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_p = p
            best_r = r
    return best_p, best_r, best_f1, best_thr


def main():
    parser = argparse.ArgumentParser(description="Train P2S Boundary Model (MultiHeadBoundary)")
    parser.add_argument(
        "--para-xlsx", type=str,
        default="datasets/splits/paragraph_train.tsv",
        help="문단 Excel (train)",
    )
    parser.add_argument(
        "--sent-xlsx", type=str,
        default="datasets/splits/sentence_train.tsv",
        help="문장 Excel (train)",
    )
    parser.add_argument(
        "--val-para-xlsx", type=str,
        default="datasets/splits/paragraph_val.tsv",
        help="문단 Excel (val)",
    )
    parser.add_argument(
        "--val-sent-xlsx", type=str,
        default="datasets/splits/sentence_val.tsv",
        help="문장 Excel (val)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128, help="BiLSTM hidden dim")
    parser.add_argument("--emb-dim", type=int, default=0, help="Char embedding dim (0=hidden_dim//2)")
    parser.add_argument("--n-attn-layers", type=int, default=0, help="Self-attention layers (0=BiLSTM only)")
    parser.add_argument("--n-attn-heads", type=int, default=4, help="Self-attention heads")
    parser.add_argument("--use-focal", action="store_true", help="Focal Loss 사용 (기본: BCE)")
    parser.add_argument("--use-dice", action="store_true", help="Dice+BCE Loss 사용 (F1 최적화)")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Dice loss 가중치")
    parser.add_argument("--multitask", action="store_true", help="pa + sa 멀티태스크 학습")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Focal loss alpha")
    parser.add_argument("--pos-weight", type=float, default=3.0, help="경계 클래스 가중치")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="models/boundary_multitask.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smooth", type=float, default=0.0,
                        help="경계 라벨 스무딩 (±1 위치에 epsilon 부여, 0=비활성)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train 데이터 로드 + 경계 라벨 생성
    para_xlsx = WORKSPACE_ROOT / args.para_xlsx
    sent_xlsx = WORKSPACE_ROOT / args.sent_xlsx
    assert para_xlsx.exists(), f"파일 없음: {para_xlsx}"
    assert sent_xlsx.exists(), f"파일 없음: {sent_xlsx}"

    print(f"[Train] 문단: {para_xlsx}")
    print(f"[Train] 문장: {sent_xlsx}")
    train_samples = load_boundary_samples(para_xlsx, sent_xlsx)

    if not train_samples:
        print("유효한 학습 샘플이 없습니다.")
        return 1

    # Multi-task: sa 샘플도 로드
    if args.multitask:
        phrase_train = WORKSPACE_ROOT / "datasets/splits/phrase_train.tsv"
        if phrase_train.exists():
            sa_train = load_sa_boundary_samples(sent_xlsx, phrase_train)
            train_samples.extend(sa_train)
            print(f"  [multitask] 총 학습 샘플: {len(train_samples)}개 (pa + sa)")

    # Val 데이터 로드
    val_para_xlsx = WORKSPACE_ROOT / args.val_para_xlsx
    val_sent_xlsx = WORKSPACE_ROOT / args.val_sent_xlsx
    assert val_para_xlsx.exists(), f"파일 없음: {val_para_xlsx}"
    assert val_sent_xlsx.exists(), f"파일 없음: {val_sent_xlsx}"

    print(f"[Val] 문단: {val_para_xlsx}")
    print(f"[Val] 문장: {val_sent_xlsx}")
    val_samples = load_boundary_samples(val_para_xlsx, val_sent_xlsx)

    # Vocab 구축 (train + val 통합 + sa 포함)
    all_samples_for_vocab = train_samples + val_samples
    if args.multitask:
        phrase_val = WORKSPACE_ROOT / "datasets/splits/phrase_val.tsv"
        if phrase_val.exists():
            sa_val = load_sa_boundary_samples(val_sent_xlsx, phrase_val)
            all_samples_for_vocab.extend(sa_val)
    vocab = build_vocab(all_samples_for_vocab)
    print(f"Vocab: {len(vocab)}자")
    print(f"Train: {len(train_samples)}개, Val: {len(val_samples)}개")

    # 경계 비율 통계
    total_chars = sum(len(s["labels"]) for s in train_samples)
    total_boundaries = sum(sum(s["labels"]) for s in train_samples)
    print(f"경계 비율: {total_boundaries}/{total_chars} ({total_boundaries/max(1,total_chars)*100:.2f}%)")

    # Dataset & DataLoader
    train_ds = BoundarySeqDataset(train_samples, vocab, max_len=args.max_len)
    val_ds = BoundarySeqDataset(val_samples, vocab, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 모델 초기화 (production과 동일 아키텍처)
    tasks = ["pa", "sa", "pd"]
    model = MultiHeadBoundary(
        vocab_size=len(vocab) + 1, tasks=tasks,
        hidden_dim=args.hidden_dim, emb_dim=args.emb_dim,
        dropout=args.dropout,
        n_attn_layers=args.n_attn_layers, n_attn_heads=args.n_attn_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params:,}개 (attn_layers={args.n_attn_layers})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.use_dice:
        criterion = DiceBCELoss(
            dice_weight=args.dice_weight,
            bce_weight=1.0 - args.dice_weight,
            pos_weight=args.pos_weight,
        )
        print(f"Loss: DiceBCE(dice={args.dice_weight}, bce={1-args.dice_weight}, pw={args.pos_weight})")
    elif args.use_focal:
        criterion = FocalBCEWithLogitsLoss(
            alpha=args.focal_alpha, gamma=args.focal_gamma, pos_weight=args.pos_weight,
        )
        print(f"Loss: FocalBCE(gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    else:
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"pos_weight={args.pos_weight}, patience={args.patience}")
    print("-" * 60)

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        idx_to_task = {0: "pa", 1: "sa", 2: "pd"}

        for ids, labels, lengths, task_ids in train_loader:
            ids = ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Multi-task: 배치 내 태스크별로 forward
            unique_tasks = task_ids.unique().tolist()
            total_loss = torch.tensor(0.0, device=device)
            n_tasks = 0

            for tidx in unique_tasks:
                task_name = idx_to_task.get(tidx, "pa")
                task_mask_b = (task_ids == tidx)
                if not task_mask_b.any():
                    continue

                t_ids = ids[task_mask_b]
                t_labels = labels[task_mask_b]
                t_lengths = lengths[task_mask_b]

                logits = model(t_ids, task=task_name)
                pad_mask = torch.arange(logits.size(1), device=device).unsqueeze(0) < t_lengths.unsqueeze(1)
                total_loss = total_loss + criterion(logits[pad_mask], t_labels[pad_mask])
                n_tasks += 1

            loss = total_loss / max(1, n_tasks)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_batches)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_logits, all_labels, all_lengths = [], [], []

        with torch.no_grad():
            for ids, labels, lengths, task_ids in val_loader:
                ids = ids.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                logits = model(ids, task="pa")
                mask = torch.arange(logits.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                loss = criterion(logits[mask], labels[mask])

                val_loss += loss.item()
                val_batches += 1
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_lengths.append(lengths.cpu())

        avg_val_loss = val_loss / max(1, val_batches)

        # F1 계산 (최적 threshold 탐색)
        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)
        cat_lengths = torch.cat(all_lengths)
        val_p, val_r, val_f1, val_thr = compute_best_f1(cat_logits, cat_labels, cat_lengths)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_F1={val_f1:.4f} (P={val_p:.3f} R={val_r:.3f} thr={val_thr:.1f})  "
            f"lr={lr_now:.6f}"
        )

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  -> best F1 갱신: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    print("-" * 60)
    print(f"Best val F1: {best_f1:.4f}")

    # 모델 저장 (production loader 호환 형식)
    out_path = WORKSPACE_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_state = best_state if best_state is not None else model.state_dict()
    torch.save(
        {
            "state_dict": save_state,
            "vocab": vocab,
            "max_len": args.max_len,
            "tasks": tasks,
            "n_attn_layers": args.n_attn_layers,
            "n_attn_heads": args.n_attn_heads,
        },
        out_path,
    )

    print(f"Model saved: {out_path}")
    print(f"  vocab={len(vocab)}, max_len={args.max_len}, tasks={tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
