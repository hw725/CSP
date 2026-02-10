#!/usr/bin/env python3
"""S2P Phrase Alignment Model 학습 (v2.1)

v2 대비 v2.1 변경:
├─ Source context BiLSTM: 구 간 순서/문맥 학습 (위치 인코딩 포함)
├─ hidden 256→512: 모델 용량 확장
├─ Guided Attention Loss: attention이 대각선 패턴을 따르도록 유도
└─ epoch 30→100: 충분한 수렴

구조:
  Source phrases → BGE-M3 [N, 1024] → Linear → BiLSTM → [N, hidden]
  Target text → Char Emb → BiLSTM → [T, hidden]
  Cross-Attention: target chars attend to source phrases (의미+위치 주입)
  Bilinear alignment: [T, hidden] @ [hidden, N] → [T, N] (구 소속 확률)
  소속 구가 바뀌는 위치 = 경계

Usage:
    python scripts/train_s2p_phrase_alignment.py --epochs 100 --hidden 512
    python scripts/train_s2p_phrase_alignment.py --epochs 100 --hidden 512 --guided-attn-weight 0.05
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Data loading
# ============================================================


def load_phrase_data(excel_path: Path) -> List[Dict]:
    """구 쌍 데이터를 문장 단위로 로드하여 정렬 라벨 생성

    Returns:
        [{"src_phrases": ["구1", "구2", ...],
          "tgt_text": "번역문 전체",
          "labels": [0, 0, 1, 1, ...],  # 각 문자의 소속 구 인덱스
          "n_phrases": int}, ...]
    """
    if str(excel_path).endswith(".tsv"):
        df = pd.read_csv(excel_path, sep="\t", low_memory=False)
    else:
        df = pd.read_excel(excel_path)

    cols = df.columns.tolist()
    if "문장식별자" in cols:
        col_sent, col_phrase = "문장식별자", "구식별자"
        col_src, col_tgt = "원문", "번역문"
    elif len(cols) >= 5:
        col_sent, col_phrase, col_src, col_tgt = cols[1], cols[2], cols[3], cols[4]
    else:
        col_sent, col_phrase, col_src, col_tgt = cols[0], cols[1], cols[2], cols[3]

    for col in (col_src, col_tgt):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 편집 마커 제거
    _marker_tr = str.maketrans("", "", "[-]")
    for col in (col_src, col_tgt):
        if col in df.columns:
            df[col] = df[col].str.translate(_marker_tr)

    sent_groups = defaultdict(list)
    for _, row in df.iterrows():
        sent_groups[row[col_sent]].append(
            (row[col_phrase], str(row[col_src]).strip(), str(row[col_tgt]).strip())
        )

    samples = []
    for sent_id, phrases in sent_groups.items():
        phrases.sort(key=lambda x: x[0])

        src_phrases = []
        tgt_text = ""
        labels = []

        for i, (_, src_phrase, tgt_phrase) in enumerate(phrases):
            if not src_phrase and not tgt_phrase:
                continue

            phrase_idx = len(src_phrases)
            src_phrases.append(src_phrase)

            # 구 사이 공백
            if i > 0 and tgt_text:
                tgt_text += " "
                labels.append(phrase_idx)  # 공백은 새 구에 소속

            for char in tgt_phrase:
                tgt_text += char
                labels.append(phrase_idx)

        if src_phrases and tgt_text and len(tgt_text) == len(labels):
            samples.append(
                {
                    "src_phrases": src_phrases,
                    "tgt_text": tgt_text,
                    "labels": labels,
                    "n_phrases": len(src_phrases),
                }
            )

    return samples


def precompute_bge_embeddings(
    samples: List[Dict], cache_path: Path = None
) -> Tuple[Dict[str, int], np.ndarray]:
    """BGE-M3로 모든 고유 원문 구의 임베딩을 사전계산"""
    if cache_path and cache_path.exists():
        print(f"  캐시 로드: {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        return cache["phrase_to_idx"], cache["embeddings"]

    unique_phrases = []
    phrase_to_idx = {}
    for sample in samples:
        for phrase in sample["src_phrases"]:
            if phrase not in phrase_to_idx:
                phrase_to_idx[phrase] = len(unique_phrases)
                unique_phrases.append(phrase)

    print(f"  고유 원문 구: {len(unique_phrases)}개")

    try:
        from FlagEmbedding import BGEM3FlagModel

        bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        batch_size = 256
        all_embeddings = []
        for i in range(0, len(unique_phrases), batch_size):
            batch = unique_phrases[i : i + batch_size]
            embs = bge.encode(batch)["dense_vecs"]
            all_embeddings.append(embs)
            if (i // batch_size) % 10 == 0:
                print(f"    BGE 인코딩: {i}/{len(unique_phrases)}")

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        del bge
        torch.cuda.empty_cache()

    except ImportError:
        print("  ⚠️ FlagEmbedding 미설치 — 랜덤 임베딩 (테스트용)")
        embeddings = np.random.randn(len(unique_phrases), 1024).astype(np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"phrase_to_idx": phrase_to_idx, "embeddings": embeddings}, cache_path
        )
        print(f"  캐시 저장: {cache_path}")

    return phrase_to_idx, embeddings


def build_tgt_vocab(samples: List[Dict]) -> Dict[str, int]:
    chars = set()
    for s in samples:
        chars.update(list(s["tgt_text"]))
    return {c: i + 1 for i, c in enumerate(sorted(chars))}


# ============================================================
# Dataset
# ============================================================


class PhraseAlignmentDataset(Dataset):
    def __init__(
        self,
        samples,
        tgt_vocab,
        phrase_to_idx,
        phrase_embeddings,
        max_phrases=64,
        tgt_max_len=1024,
    ):
        # max_phrases 초과 문장 필터링
        self.samples = [s for s in samples if s["n_phrases"] <= max_phrases]
        self.tgt_vocab = tgt_vocab
        self.phrase_to_idx = phrase_to_idx
        self.phrase_embeddings = phrase_embeddings
        self.max_phrases = max_phrases
        self.tgt_max_len = tgt_max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        n_phrases = s["n_phrases"]
        tgt_len = min(len(s["tgt_text"]), self.tgt_max_len)
        bge_dim = self.phrase_embeddings.shape[1]

        # Source phrase BGE embeddings [max_phrases, bge_dim]
        src_embs = np.zeros((self.max_phrases, bge_dim), dtype=np.float32)
        src_mask = np.zeros(self.max_phrases, dtype=np.bool_)
        for i, phrase in enumerate(s["src_phrases"]):
            src_embs[i] = self.phrase_embeddings[self.phrase_to_idx[phrase]]
            src_mask[i] = True

        # Target character IDs [tgt_max_len]
        tgt_ids = np.zeros(self.tgt_max_len, dtype=np.int64)
        for i, char in enumerate(s["tgt_text"][: self.tgt_max_len]):
            tgt_ids[i] = self.tgt_vocab.get(char, 0)

        # Gold labels [tgt_max_len]
        labels = np.full(self.tgt_max_len, -100, dtype=np.int64)
        for i, label in enumerate(s["labels"][: self.tgt_max_len]):
            labels[i] = label

        return {
            "src_embs": torch.from_numpy(src_embs),
            "src_mask": torch.from_numpy(src_mask),
            "tgt_ids": torch.from_numpy(tgt_ids),
            "labels": torch.from_numpy(labels),
            "tgt_length": tgt_len,
            "n_phrases": n_phrases,
        }


# ============================================================
# Model
# ============================================================


class PhraseAlignmentModel(nn.Module):
    """구 단위 정렬 모델 v2.1 — Source BiLSTM + 확장 hidden

    Source: BGE-M3 [N, 1024] → Linear → BiLSTM → [N, hidden] (문맥+순서 학습)
    Target: 문자 임베딩 → BiLSTM → [T, hidden]
    Cross-Attention: target chars attend to source phrases
    Alignment: bilinear dot-product → [T, N] 소속 logits
    """

    def __init__(
        self,
        bge_dim=1024,
        tgt_vocab_size=8000,
        tgt_emb_dim=128,
        hidden=512,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        # Source encoder: BGE projection + BiLSTM (문맥+순서 학습)
        self.src_proj = nn.Sequential(
            nn.Linear(bge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.src_encoder = nn.LSTM(
            hidden,
            hidden // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0,
        )
        self.src_norm = nn.LayerNorm(hidden)

        # Target encoder: Char embedding + BiLSTM
        self.tgt_emb = nn.Embedding(tgt_vocab_size, tgt_emb_dim, padding_idx=0)
        self.tgt_encoder = nn.LSTM(
            tgt_emb_dim,
            hidden // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.tgt_norm = nn.LayerNorm(hidden)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_norm = nn.LayerNorm(hidden)

        # Alignment projection
        self.alignment_proj = nn.Linear(hidden, hidden)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, src_embs, tgt_ids, src_mask=None, return_attn=False):
        """
        src_embs: [B, N, bge_dim]
        tgt_ids: [B, T]
        src_mask: [B, N] bool (True = valid)
        return_attn: True이면 cross-attention weights도 반환

        Returns: alignment_logits [B, T, N], (optional) attn_weights [B, T, N]
        """
        # Source: BGE → projection → BiLSTM (구 간 문맥 학습)
        src_h = self.src_proj(src_embs)  # [B, N, hidden]
        src_h, _ = self.src_encoder(src_h)  # [B, N, hidden]
        src_h = self.src_norm(src_h)

        # Target: char embedding → BiLSTM
        tgt_emb = self.tgt_emb(tgt_ids)  # [B, T, emb_dim]
        tgt_h, _ = self.tgt_encoder(tgt_emb)  # [B, T, hidden]
        tgt_h = self.tgt_norm(tgt_h)

        # Cross-attention
        key_padding_mask = ~src_mask if src_mask is not None else None
        cross_out, attn_weights = self.cross_attn(
            query=tgt_h,
            key=src_h,
            value=src_h,
            key_padding_mask=key_padding_mask,
        )
        tgt_enriched = self.cross_norm(cross_out + tgt_h)

        # Bilinear alignment
        tgt_proj = self.alignment_proj(tgt_enriched)  # [B, T, hidden]
        alignment_logits = torch.bmm(tgt_proj, src_h.transpose(1, 2))
        alignment_logits = alignment_logits / self.temperature.abs().clamp(min=0.01)

        if src_mask is not None:
            alignment_logits = alignment_logits.masked_fill(
                ~src_mask.unsqueeze(1), float("-inf")
            )

        if return_attn:
            return alignment_logits, attn_weights
        return alignment_logits


# ============================================================
# Evaluation
# ============================================================


def compute_metrics(alignment_logits, labels, tgt_lengths, n_phrases_list):
    """정렬 정확도 + 경계 F1 계산"""
    tp = fp = fn = 0
    total_chars = 0
    correct_chars = 0

    batch_size = alignment_logits.shape[0]
    for i in range(batch_size):
        T = tgt_lengths[i]
        N = n_phrases_list[i]

        logits_i = alignment_logits[i, :T, :N]
        pred = logits_i.argmax(dim=-1)
        gold = labels[i, :T]
        valid = gold >= 0

        correct_chars += ((pred == gold) & valid).sum().item()
        total_chars += valid.sum().item()

        # 경계 추출 (소속 구 변경 지점)
        pred_bounds = set()
        for t in range(1, T):
            if valid[t] and pred[t] != pred[t - 1]:
                pred_bounds.add(t)

        gold_bounds = set()
        for t in range(1, T):
            if valid[t] and valid[t - 1] and gold[t] != gold[t - 1]:
                gold_bounds.add(t)

        # Exact matching
        matched_gold = set()
        for pb in pred_bounds:
            if pb in gold_bounds:
                tp += 1
                matched_gold.add(pb)
            else:
                fp += 1
        fn += len(gold_bounds - matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0

    return precision, recall, f1, char_acc


# ============================================================
# Training
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train S2P Phrase Alignment Model (v2)"
    )
    parser.add_argument(
        "--train-excel",
        type=str,
        default="datasets/splits/phrase_train.xlsx",
    )
    parser.add_argument(
        "--val-excel",
        type=str,
        default="datasets/splits/phrase_val.xlsx",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-phrases", type=int, default=64)
    parser.add_argument("--tgt-max-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--tgt-emb-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--monotonic-weight",
        type=float,
        default=0.1,
        help="순서 유지 정규화 가중치",
    )
    parser.add_argument(
        "--guided-attn-weight",
        type=float,
        default=0.05,
        help="Guided Attention Loss 가중치 (대각선 유도)",
    )
    parser.add_argument(
        "--bge-cache",
        type=str,
        default="cache/bge_phrase_embeddings.pt",
    )
    parser.add_argument(
        "--out", type=str, default="models/s2p_phrase_alignment.pt"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ======== Data ========
    print("📂 데이터 로드...")
    train_path = Path(WORKSPACE_ROOT) / args.train_excel
    if not train_path.exists():
        raise FileNotFoundError(f"파일 없음: {train_path}")
    train_samples = load_phrase_data(train_path)
    print(f"  [Train] {len(train_samples)}개 문장")

    val_path = Path(WORKSPACE_ROOT) / args.val_excel
    if val_path.exists():
        val_samples = load_phrase_data(val_path)
        print(f"  [Val]   {len(val_samples)}개 문장")
    else:
        np.random.seed(args.seed)
        indices = np.random.permutation(len(train_samples))
        split_idx = max(1, len(train_samples) // 10)
        val_samples = [train_samples[i] for i in indices[:split_idx]]
        train_samples = [train_samples[i] for i in indices[split_idx:]]
        print(
            f"  [Val 자동 분리] train={len(train_samples)}, val={len(val_samples)}"
        )

    # 통계
    phrase_counts = [s["n_phrases"] for s in train_samples]
    over_max = sum(1 for c in phrase_counts if c > args.max_phrases)
    print(
        f"  구 개수: min={min(phrase_counts)}, max={max(phrase_counts)}, "
        f"mean={np.mean(phrase_counts):.1f}, median={np.median(phrase_counts):.0f}"
    )
    if over_max:
        print(f"  ⚠️ max_phrases={args.max_phrases} 초과: {over_max}개 제외")

    # 예시
    s = train_samples[0]
    print(f"  예시: n_phrases={s['n_phrases']}")
    print(f"    src[:3]={s['src_phrases'][:3]}")
    print(f"    tgt[:50]='{s['tgt_text'][:50]}...'")
    print(f"    labels[:20]={s['labels'][:20]}")

    # ======== BGE ========
    print("🔄 BGE-M3 임베딩 사전계산...")
    all_samples = train_samples + val_samples
    cache_path = Path(WORKSPACE_ROOT) / args.bge_cache
    phrase_to_idx, phrase_embeddings = precompute_bge_embeddings(
        all_samples, cache_path
    )
    bge_dim = phrase_embeddings.shape[1]
    print(f"  BGE dim={bge_dim}, 고유 구={len(phrase_to_idx)}")

    # ======== Vocab ========
    tgt_vocab = build_tgt_vocab(all_samples)
    print(f"📚 tgt_vocab={len(tgt_vocab)}")

    # ======== Dataset ========
    train_ds = PhraseAlignmentDataset(
        train_samples,
        tgt_vocab,
        phrase_to_idx,
        phrase_embeddings,
        max_phrases=args.max_phrases,
        tgt_max_len=args.tgt_max_len,
    )
    val_ds = PhraseAlignmentDataset(
        val_samples,
        tgt_vocab,
        phrase_to_idx,
        phrase_embeddings,
        max_phrases=args.max_phrases,
        tgt_max_len=args.tgt_max_len,
    )
    print(f"  Dataset: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=0
    )

    # ======== Model ========
    model = PhraseAlignmentModel(
        bge_dim=bge_dim,
        tgt_vocab_size=len(tgt_vocab) + 1,
        tgt_emb_dim=args.tgt_emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print(
        f"max_phrases={args.max_phrases}, tgt_max_len={args.tgt_max_len}, "
        f"mono={args.monotonic_weight}, guided_attn={args.guided_attn_weight}"
    )
    print("-" * 70)

    # ======== Train ========
    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            src_embs = batch["src_embs"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            labels_t = batch["labels"].to(device)
            tgt_lengths = batch["tgt_length"]

            optimizer.zero_grad()
            n_phrases_batch = batch["n_phrases"]

            logits, attn_w = model(
                src_embs, tgt_ids, src_mask, return_attn=True
            )  # [B, T, N], [B, T, N]
            B, T, N = logits.shape

            # Cross-entropy (ignores -100)
            ce_loss = F.cross_entropy(
                logits.reshape(B * T, N),
                labels_t.reshape(B * T),
                ignore_index=-100,
            )

            # Monotonic regularization
            mono_loss = torch.tensor(0.0, device=device)
            if args.monotonic_weight > 0:
                probs = F.softmax(logits, dim=-1)
                phrase_idx = torch.arange(N, device=device, dtype=torch.float)
                expected = (probs * phrase_idx).sum(dim=-1)  # [B, T]
                diffs = expected[:, 1:] - expected[:, :-1]
                tgt_mask = torch.arange(T - 1, device=device).unsqueeze(0) < (
                    tgt_lengths.to(device).unsqueeze(1) - 1
                )
                mono_loss = (
                    (F.relu(-diffs) * tgt_mask).sum()
                    / tgt_mask.sum().clamp(min=1)
                )

            # Guided Attention Loss: attention이 대각선 패턴을 따르도록 유도
            ga_loss = torch.tensor(0.0, device=device)
            if args.guided_attn_weight > 0 and attn_w is not None:
                # 각 샘플별 대각선 가이드 행렬 생성
                ga_total = 0.0
                ga_count = 0
                for b in range(B):
                    t_len = tgt_lengths[b].item()
                    n_phr = n_phrases_batch[b].item()
                    if t_len < 2 or n_phr < 2:
                        continue
                    # W[t, n] = (t/T - n/N)^2 → 대각선에서 벗어날수록 패널티
                    t_pos = torch.arange(t_len, device=device, dtype=torch.float) / t_len
                    n_pos = torch.arange(n_phr, device=device, dtype=torch.float) / n_phr
                    guide = (t_pos.unsqueeze(1) - n_pos.unsqueeze(0)) ** 2  # [t_len, n_phr]
                    # attention weights와 가이드 행렬의 가중 합
                    attn_sub = attn_w[b, :t_len, :n_phr]  # [t_len, n_phr]
                    ga_total += (attn_sub * guide).sum()
                    ga_count += t_len * n_phr
                if ga_count > 0:
                    ga_loss = ga_total / ga_count

            loss = (
                ce_loss
                + args.monotonic_weight * mono_loss
                + args.guided_attn_weight * ga_loss
            )
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
        all_logits, all_labels = [], []
        all_tgt_lengths, all_n_phrases = [], []

        with torch.no_grad():
            for batch in val_loader:
                src_embs = batch["src_embs"].to(device)
                src_mask = batch["src_mask"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                labels_t = batch["labels"].to(device)

                logits = model(src_embs, tgt_ids, src_mask)
                B, T, N = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, N),
                    labels_t.reshape(B * T),
                    ignore_index=-100,
                )

                val_loss += loss.item()
                val_batches += 1
                all_logits.append(logits.cpu())
                all_labels.append(labels_t.cpu())
                all_tgt_lengths.extend(batch["tgt_length"].tolist())
                all_n_phrases.extend(batch["n_phrases"].tolist())

        avg_val_loss = val_loss / max(1, val_batches)

        cat_logits = torch.cat(all_logits)
        cat_labels = torch.cat(all_labels)
        precision, recall, f1, char_acc = compute_metrics(
            cat_logits, cat_labels, all_tgt_lengths, all_n_phrases
        )

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs}: "
            f"loss={avg_train_loss:.4f}/{avg_val_loss:.4f}  "
            f"F1={f1:.4f} (P={precision:.3f} R={recall:.3f})  "
            f"acc={char_acc:.4f}  lr={lr_now:.6f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            print(f"  -> best F1 갱신: {best_f1:.4f} (acc={char_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    print("-" * 70)
    print(f"Best boundary F1: {best_f1:.4f}")

    # ======== Save ========
    out_path = (
        WORKSPACE_ROOT / args.out
        if not Path(args.out).is_absolute()
        else Path(args.out)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_state = (
        best_state
        if best_state is not None
        else {k: v.cpu().clone() for k, v in model.state_dict().items()}
    )
    torch.save(
        {
            "state_dict": save_state,
            "tgt_vocab": tgt_vocab,
            "tgt_max_len": args.tgt_max_len,
            "max_phrases": args.max_phrases,
            "hidden": args.hidden,
            "tgt_emb_dim": args.tgt_emb_dim,
            "bge_dim": bge_dim,
            "best_f1": best_f1,
            "model_version": "phrase_alignment_v2.1",
        },
        out_path,
    )

    print(f"💾 모델 저장: {out_path}")
    print(f"  tgt_vocab={len(tgt_vocab)}, best_f1={best_f1:.4f}")
    print("✅ S2P Phrase Alignment Model (v2) 학습 완료!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
