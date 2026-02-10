#!/usr/bin/env python3
"""S2P Cross-Attention 경계 모델 학습 (P2S 수준으로 개선)

구-번역 쌍을 입력으로 받아 번역문의 구 경계를 예측
- 원문과 번역문 간 Cross-Attention으로 의미 대응 학습
- 원문 구 구조를 참조하여 번역문 경계 결정

개선 사항 (P2S 교훈 적용):
- Train/Val 분리 + 매 에폭 Val F1 측정
- Early stopping (patience 기반, val F1 최적)
- FocalBCE / DiceBCE 선택 가능 (클래스 불균형 대응)
- 최적 threshold 탐색 (0.2~0.8)
- Best model 저장 (val F1 기준)

Usage:
    python scripts/train_s2p_crossattn_boundary.py --epochs 30
    python scripts/train_s2p_crossattn_boundary.py --epochs 30 --use-dice
    python scripts/train_s2p_crossattn_boundary.py --epochs 30 --use-focal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = WORKSPACE_ROOT / "models"


def load_sa_phrase_pairs(excel_path: Path) -> List[Dict]:
    """
    S2P Excel을 로드하여 문장별로:
    - 전체 원문 (구들 연결)
    - 전체 번역문 (구들 연결)
    - 번역문 B/O 레이블

    Returns:
        [{"src": "원문", "tgt": "번역문", "labels": "BOOOOB..."}, ...]
    """
    # TSV 또는 Excel 지원
    if str(excel_path).endswith('.tsv'):
        df = pd.read_csv(excel_path, sep='\t', low_memory=False)
    else:
        df = pd.read_excel(excel_path)

    # 컬럼명 정규화 (책명 컬럼 유무에 따라 대응)
    cols = df.columns.tolist()
    if "문장식별자" in cols:
        col_sent = "문장식별자"
        col_phrase = "구식별자"
        col_src = "원문"
        col_tgt = "번역문"
    elif len(cols) >= 5:
        # 책명, 문장식별자, 구식별자, 원문, 번역문 순서 가정
        col_sent = cols[1]
        col_phrase = cols[2]
        col_src = cols[3]
        col_tgt = cols[4]
    else:
        col_sent = cols[0]
        col_phrase = cols[1]
        col_src = cols[2]
        col_tgt = cols[3]

    # NaN 처리
    for col in (col_src, col_tgt):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 편집 마커 제거 (P2S와 동일 전처리: [, -, ] 문자 제거)
    _marker_tr = str.maketrans("", "", "[-]")
    for col in (col_src, col_tgt):
        if col in df.columns:
            df[col] = df[col].str.translate(_marker_tr)

    # 문장별로 그룹핑
    sent_groups = defaultdict(list)
    for _, row in df.iterrows():
        sent_id = row[col_sent]
        phrase_id = row[col_phrase]
        src = str(row[col_src]).strip()
        tgt = str(row[col_tgt]).strip()
        sent_groups[sent_id].append((phrase_id, src, tgt))

    samples = []
    for sent_id, phrases in sent_groups.items():
        # 구식별자 순으로 정렬
        phrases.sort(key=lambda x: x[0])

        full_src = ""
        full_tgt = ""
        labels = ""

        for i, (_, src_phrase, tgt_phrase) in enumerate(phrases):
            if not src_phrase and not tgt_phrase:
                continue

            # 구 사이 공백 추가
            if i > 0:
                if full_src:
                    full_src += " "
                if full_tgt:
                    full_tgt += " "
                    labels += "O"  # 공백은 O

            full_src += src_phrase

            # 번역문: 첫 문자 B, 나머지 O
            for j, char in enumerate(tgt_phrase):
                full_tgt += char
                labels += "B" if j == 0 else "O"

        if full_src and full_tgt and len(full_tgt) == len(labels):
            samples.append(
                {
                    "src": full_src,
                    "tgt": full_tgt,
                    "labels": labels,
                    "num_phrases": len(phrases),
                }
            )

    return samples


def build_vocab(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """원문/번역문 각각 vocab 구축"""
    src_chars = set()
    tgt_chars = set()
    for s in samples:
        src_chars.update(list(s["src"]))
        tgt_chars.update(list(s["tgt"]))

    src_vocab = {c: i + 1 for i, c in enumerate(sorted(src_chars))}
    tgt_vocab = {c: i + 1 for i, c in enumerate(sorted(tgt_chars))}
    return src_vocab, tgt_vocab


class CrossAttnBoundaryDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        src_vocab: Dict,
        tgt_vocab: Dict,
        src_max_len: int,
        tgt_max_len: int,
    ):
        self.samples = samples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __len__(self):
        return len(self.samples)

    def _encode(self, text: str, vocab: Dict, max_len: int) -> torch.Tensor:
        ids = [vocab.get(ch, 0) for ch in text][:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _encode_labels(self, labels: str, max_len: int) -> torch.Tensor:
        arr = [1.0 if ch == "B" else 0.0 for ch in labels][:max_len]
        arr += [0.0] * (max_len - len(arr))
        return torch.tensor(arr, dtype=torch.float)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = self._encode(s["src"], self.src_vocab, self.src_max_len)
        tgt = self._encode(s["tgt"], self.tgt_vocab, self.tgt_max_len)
        labels = self._encode_labels(s["labels"], self.tgt_max_len)
        tgt_len = min(len(s["tgt"]), self.tgt_max_len)
        return src, tgt, labels, tgt_len


class CrossAttnBoundaryModel(nn.Module):
    """Cross-Attention 기반 경계 태거

    1. Source Encoder: 원문 인코딩
    2. Target Encoder: 번역문 인코딩
    3. Cross-Attention: 원문 참조
    4. Boundary Head: 경계 예측
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_dim: int = 128,
        hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)

        # Source encoder (BiLSTM)
        self.src_encoder = nn.LSTM(
            emb_dim,
            hidden // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Target encoder (BiLSTM)
        self.tgt_encoder = nn.LSTM(
            emb_dim,
            hidden // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Cross-Attention: target attends to source
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True, dropout=0.1
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden)

        # Boundary classifier
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # concat(tgt_hidden, cross_attn_out)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) 원문 토큰 IDs
            tgt: (batch, tgt_len) 번역문 토큰 IDs

        Returns:
            logits: (batch, tgt_len) 각 번역문 문자의 경계 logit
        """
        # Encode source
        src_emb = self.src_emb(src)  # (B, S, E)
        src_hidden, _ = self.src_encoder(src_emb)  # (B, S, H)

        # Encode target
        tgt_emb = self.tgt_emb(tgt)  # (B, T, E)
        tgt_hidden, _ = self.tgt_encoder(tgt_emb)  # (B, T, H)

        # Cross-attention: target queries, source keys/values
        # Create key_padding_mask for source (True = padding)
        src_padding_mask = src == 0  # (B, S)

        cross_out, _ = self.cross_attn(
            query=tgt_hidden,
            key=src_hidden,
            value=src_hidden,
            key_padding_mask=src_padding_mask,
        )  # (B, T, H)

        cross_out = self.norm(cross_out + tgt_hidden)  # residual + norm

        # Concatenate target hidden and cross-attention output
        combined = torch.cat([tgt_hidden, cross_out], dim=-1)  # (B, T, H*2)

        # Predict boundary
        logits = self.boundary_head(combined).squeeze(-1)  # (B, T)

        return logits


# ============================================================
# Loss functions (P2S 수준)
# ============================================================

class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss for class-imbalanced sequence labeling"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE combined loss: directly optimizes F1-like metric + stable BCE."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5,
                 pos_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
        )
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ============================================================
# F1 evaluation (P2S 수준)
# ============================================================

def compute_f1(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor,
               threshold: float = 0.5):
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
    parser = argparse.ArgumentParser(
        description="Train S2P Cross-Attention Boundary Model (P2S 수준 개선)"
    )
    parser.add_argument(
        "--train-excel",
        type=str,
        default="datasets/splits/phrase_train.xlsx",
        help="훈련용 Excel/TSV 파일",
    )
    parser.add_argument(
        "--val-excel",
        type=str,
        default="datasets/splits/phrase_val.xlsx",
        help="검증용 Excel/TSV 파일",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--src-max-len", type=int, default=256)
    parser.add_argument("--tgt-max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Loss 선택
    parser.add_argument("--use-focal", action="store_true", help="Focal Loss 사용")
    parser.add_argument("--use-dice", action="store_true", help="Dice+BCE Loss 사용 (F1 최적화)")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Dice loss 가중치")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Focal loss alpha")
    parser.add_argument("--pos-weight", type=float, default=5.0, help="경계 클래스 가중치")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # 출력
    parser.add_argument("--out", type=str, default="models/s2p_crossattn_boundary.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed 고정
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ======== 데이터 로드 ========
    print("📂 Loading S2P phrase pair data...")
    train_excel = Path(WORKSPACE_ROOT) / args.train_excel
    if not train_excel.exists():
        raise FileNotFoundError(f"파일 없음: {train_excel}")

    train_samples = load_sa_phrase_pairs(train_excel)
    print(f"  [Train] {len(train_samples)}개 문장")

    # Validation 데이터
    val_excel = Path(WORKSPACE_ROOT) / args.val_excel
    if val_excel.exists():
        val_samples = load_sa_phrase_pairs(val_excel)
        print(f"  [Val]   {len(val_samples)}개 문장")
    else:
        # Val 파일 없으면 train에서 10% 분리
        np.random.seed(args.seed)
        indices = np.random.permutation(len(train_samples))
        split_idx = max(1, len(train_samples) // 10)
        val_samples = [train_samples[i] for i in indices[:split_idx]]
        train_samples = [train_samples[i] for i in indices[split_idx:]]
        print(f"  [Val 자동 분리] train={len(train_samples)}, val={len(val_samples)}")

    if not train_samples:
        print("유효한 학습 샘플이 없습니다.")
        return 1

    # 샘플 확인
    s = train_samples[0]
    print(f"  예시: src='{s['src'][:50]}...'")
    print(f"        tgt='{s['tgt'][:50]}...'")
    print(f"        labels='{s['labels'][:50]}...'  phrases={s['num_phrases']}")

    # ======== Vocab (train + val 통합) ========
    all_samples_for_vocab = train_samples + val_samples
    src_vocab, tgt_vocab = build_vocab(all_samples_for_vocab)
    print(f"📚 Vocab: src={len(src_vocab)}, tgt={len(tgt_vocab)}")

    # 경계 비율 통계
    total_chars = sum(len(s["labels"]) for s in train_samples)
    total_boundaries = sum(s["labels"].count("B") for s in train_samples)
    print(f"경계 비율: {total_boundaries}/{total_chars} ({total_boundaries/max(1,total_chars)*100:.2f}%)")

    # ======== Dataset & DataLoader ========
    train_ds = CrossAttnBoundaryDataset(
        train_samples, src_vocab, tgt_vocab, args.src_max_len, args.tgt_max_len
    )
    val_ds = CrossAttnBoundaryDataset(
        val_samples, src_vocab, tgt_vocab, args.src_max_len, args.tgt_max_len
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # ======== 모델 ========
    model = CrossAttnBoundaryModel(
        src_vocab_size=len(src_vocab) + 1,
        tgt_vocab_size=len(tgt_vocab) + 1,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ======== Loss 함수 선택 ========
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
        print(f"Loss: FocalBCE(gamma={args.focal_gamma}, alpha={args.focal_alpha}, pw={args.pos_weight})")
    else:
        pw = torch.tensor([args.pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"Loss: BCEWithLogitsLoss(pw={args.pos_weight})")

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(f"patience={args.patience}, seed={args.seed}")
    print("-" * 60)

    # ======== 학습 루프 ========
    best_f1 = 0.0
    best_state = None
    best_thr = 0.5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for src, tgt, labels, lengths in train_loader:
            src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(src, tgt)

            # 마스킹: 패딩 위치 제외
            mask = torch.arange(logits.shape[1], device=device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)

            loss = criterion(logits[mask], labels[mask])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            for src, tgt, labels, lengths in val_loader:
                src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)
                lengths = lengths.to(device)

                logits = model(src, tgt)
                mask = torch.arange(logits.shape[1], device=device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
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
            best_thr = val_thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  -> best F1 갱신: {best_f1:.4f} (thr={best_thr})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    print("-" * 60)
    print(f"Best val F1: {best_f1:.4f} (thr={best_thr})")

    # ======== 저장 ========
    out_path = WORKSPACE_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_state = best_state if best_state is not None else {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": save_state,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "src_max_len": args.src_max_len,
            "tgt_max_len": args.tgt_max_len,
            "hidden": args.hidden,
            "emb_dim": args.emb_dim,
            "best_threshold": best_thr,
            "best_f1": best_f1,
        },
        out_path,
    )
    print(f"💾 S2P 경계 모델 저장: {out_path}")
    print(f"  vocab: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    print(f"  best_f1={best_f1:.4f}, best_thr={best_thr}")
    print(f"✅ S2P Cross-Attention Boundary 학습 완료!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
