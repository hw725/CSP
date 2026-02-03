#!/usr/bin/env python3
"""P2S Cross-Attention 경계 모델 학습

문단에서 원문과 번역문을 입력으로 받아 번역문의 문장 경계를 예측
- 원문과 번역문 간 Cross-Attention으로 의미 대응 학습
- 원문 문장 구조를 참조하여 번역문 경계 결정

Usage:
    python scripts/train_p2s_crossattn_boundary.py --epochs 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


def load_p2s_sentence_pairs(excel_path: Path) -> List[Dict]:
    """
    Sentence Excel을 로드하여 원문-번역문 쌍과 경계 레이블 생성
    
    Returns:
        [{"src": "원문 문장", "tgt": "번역문 문장", "is_boundary": True}, ...]
    """
    df = pd.read_excel(excel_path)
    
    samples = []
    for _, row in df.iterrows():
        src = str(row.iloc[3]).strip()  # 원문
        tgt = str(row.iloc[4]).strip()  # 번역문
        
        if src and tgt:
            samples.append({
                "src": src,
                "tgt": tgt,
                "is_boundary": 1,  # 정답: 경계임
            })
    
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


class P2SBoundaryDataset(Dataset):
    """P2S 문장 경계 데이터셋"""
    
    def __init__(self, samples: List[Dict], src_vocab: Dict, tgt_vocab: Dict, max_len: int = 512):
        self.samples = samples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def encode_text(self, text: str, vocab: Dict[str, int]) -> torch.Tensor:
        ids = [vocab.get(ch, 0) for ch in text][:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        src_enc = self.encode_text(s["src"], self.src_vocab)
        tgt_enc = self.encode_text(s["tgt"], self.tgt_vocab)
        label = torch.tensor(s["is_boundary"], dtype=torch.float32)
        return src_enc, tgt_enc, label


class CharEncoder(nn.Module):
    """문자 단위 인코더 (BiLSTM)"""
    
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        out, _ = self.lstm(e)
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.proj(pooled)


class BoundaryClassifier(nn.Module):
    """경계 분류 모델 (원문-번역문 Cross-Attention)"""
    
    def __init__(self, vocab_src: int, vocab_tgt: int):
        super().__init__()
        self.enc_src = CharEncoder(vocab_src)
        self.enc_tgt = CharEncoder(vocab_tgt)
        
        # Cross-attention
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        v_src = self.enc_src(src)  # [B, 256]
        v_tgt = self.enc_tgt(tgt)  # [B, 256]
        
        # Cross-attention: tgt가 query, src가 key/value
        v_tgt_attn, _ = self.attention(
            v_tgt.unsqueeze(1),  # [B, 1, 256]
            v_src.unsqueeze(1),  # [B, 1, 256]
            v_src.unsqueeze(1)   # [B, 1, 256]
        )
        v_tgt_attn = v_tgt_attn.squeeze(1)  # [B, 256]
        
        # 원문-번역문 결합
        combined = torch.cat([v_src, v_tgt_attn], dim=1)  # [B, 512]
        
        # 경계 분류
        logits = self.classifier(combined)  # [B, 1]
        return logits.squeeze(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train P2S Boundary Model")
    parser.add_argument("--train-excel", type=str, default="dataset_split/sentence_train.xlsx",
                        help="훈련용 Excel 파일")
    parser.add_argument("--epochs", type=int, default=10, help="에포크 수")
    parser.add_argument("--batch", type=int, default=128, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--max-len", type=int, default=512, help="최대 시퀀스 길이")
    parser.add_argument("--device", type=str, default="cuda", help="디바이스")
    parser.add_argument("--out", type=str, default="models/boundary_multitask.pt",
                        help="출력 모델 경로")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 데이터 로드
    train_excel = Path(DATASETS_ROOT).parent / args.train_excel
    if not train_excel.exists():
        raise FileNotFoundError(f"파일 없음: {train_excel}")
    
    print(f"📂 Loading data from {train_excel}...")
    samples = load_p2s_sentence_pairs(train_excel)
    src_vocab, tgt_vocab = build_vocab(samples)
    
    dataset = P2SBoundaryDataset(samples, src_vocab, tgt_vocab, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    
    print(f"📊 Loaded {len(samples)} samples")
    print(f"📊 Vocab sizes: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    
    # 모델 초기화
    model = BoundaryClassifier(len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    print(f"\n🚀 Training: {args.epochs} epochs")
    print("-" * 50)
    
    # 훈련 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for src, tgt, labels in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            labels = labels.to(device)
            
            logits = model(src, tgt)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}")
    
    print("-" * 50)
    
    # 모델 저장
    out_path = Path(MODELS_ROOT) / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "state_dict": model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "max_len": args.max_len,
    }, out_path)
    
    print(f"💾 Model saved: {out_path}")
    print(f"✅ P2S 경계 모델 훈련 완료!")


if __name__ == "__main__":
    raise SystemExit(main())
