"""SA 의미 대응 경계 모델 로더

BGE 임베딩 기반 의미 대응 모델 추론
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from torch import nn


class SemanticBoundaryModel(nn.Module):
    """의미 대응 기반 경계 모델 (학습 스크립트와 동일)"""
    def __init__(self, tgt_vocab_size: int, emb_dim: int = 128, 
                 hidden: int = 256, src_emb_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        
        self.src_proj = nn.Linear(src_emb_dim, hidden)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)
        
        self.tgt_encoder = nn.LSTM(
            emb_dim, hidden // 2, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.2
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        
        self.norm = nn.LayerNorm(hidden)
        
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, src_embs, tgt_ids, n_phrases):
        batch_size = src_embs.shape[0]
        max_phrases = src_embs.shape[1]
        
        src_hidden = self.src_proj(src_embs)
        tgt_emb = self.tgt_emb(tgt_ids)
        tgt_hidden, _ = self.tgt_encoder(tgt_emb)
        
        phrase_mask = torch.arange(max_phrases, device=src_embs.device).unsqueeze(0) >= n_phrases.unsqueeze(1)
        
        cross_out, attn_weights = self.cross_attn(
            query=tgt_hidden,
            key=src_hidden,
            value=src_hidden,
            key_padding_mask=phrase_mask
        )
        
        cross_out = self.norm(cross_out + tgt_hidden)
        combined = torch.cat([tgt_hidden, cross_out], dim=-1)
        logits = self.boundary_head(combined).squeeze(-1)
        
        return logits, attn_weights


class SemanticBoundaryTaggerLoader:
    """의미 대응 경계 태거 로더"""
    
    def __init__(self, model_path: Path = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "sa_semantic_boundary.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"의미 대응 경계 모델 없음: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.tgt_vocab = checkpoint["tgt_vocab"]
        self.max_phrases = checkpoint["max_phrases"]
        self.tgt_max_len = checkpoint["tgt_max_len"]
        self.src_emb_dim = checkpoint["src_emb_dim"]
        self.test_scores = checkpoint.get("test_scores", {})
        
        self.model = SemanticBoundaryModel(
            tgt_vocab_size=len(self.tgt_vocab) + 1,
            src_emb_dim=self.src_emb_dim,
            hidden=256,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        # BGE 임베더 로드
        from common.embedders.bge import get_embed_func
        self.embed_func = get_embed_func()
        
        print(f"✅ 의미 대응 경계 태거 로드 완료 (tgt_vocab={len(self.tgt_vocab)}, device={self.device})")
    
    def _encode_tgt(self, text: str) -> torch.Tensor:
        ids = [self.tgt_vocab.get(ch, 0) for ch in text][:self.tgt_max_len]
        ids += [0] * (self.tgt_max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def segment_text(self, src_phrases: List[str], tgt_text: str, 
                     n_segments: int = None, threshold: float = 0.5) -> List[str]:
        """원문 구들과 번역문을 기반으로 번역문 분할
        
        Args:
            src_phrases: 원문 구 리스트
            tgt_text: 번역문 전체 텍스트
            n_segments: 목표 세그먼트 개수 (None이면 threshold 기준)
            threshold: 경계 확률 임계값
        """
        if not tgt_text.strip():
            return [tgt_text] if tgt_text else []
        
        # 원문 구 임베딩
        src_embs = self.embed_func(src_phrases, batch_size=64)
        src_embs = np.array(src_embs)
        
        n_phrases = min(len(src_phrases), self.max_phrases)
        src_padded = np.zeros((self.max_phrases, self.src_emb_dim), dtype=np.float32)
        src_padded[:n_phrases] = src_embs[:n_phrases]
        
        src_tensor = torch.tensor([src_padded], dtype=torch.float32, device=self.device)
        n_phrases_tensor = torch.tensor([n_phrases], dtype=torch.long, device=self.device)
        tgt_tensor = self._encode_tgt(tgt_text)
        
        with torch.no_grad():
            logits, _ = self.model(src_tensor, tgt_tensor, n_phrases_tensor)
            probs = torch.sigmoid(logits[0, :len(tgt_text)]).cpu().numpy()
        
        # n_segments가 주어지면 상위 n-1개 경계 선택
        if n_segments is not None and n_segments > 1:
            prob_positions = [(probs[i], i) for i in range(1, len(probs))]
            prob_positions.sort(reverse=True)
            top_positions = sorted([pos for _, pos in prob_positions[:n_segments - 1]])
            
            segments = []
            start = 0
            for pos in top_positions:
                segments.append(tgt_text[start:pos])
                start = pos
            segments.append(tgt_text[start:])
            
            segments = [s for s in segments if s.strip()]
            return segments if segments else [tgt_text]
        
        # threshold 기준 분할
        segments = []
        start = 0
        for i, prob in enumerate(probs):
            if prob >= threshold and i > start:
                segments.append(tgt_text[start:i])
                start = i
        if start < len(tgt_text):
            segments.append(tgt_text[start:])
        
        segments = [s for s in segments if s.strip()]
        return segments if segments else [tgt_text]


_semantic_tagger_instance = None


def get_semantic_boundary_tagger(model_path: Path = None, device: str = "cuda"):
    global _semantic_tagger_instance
    if _semantic_tagger_instance is None:
        _semantic_tagger_instance = SemanticBoundaryTaggerLoader(model_path=model_path, device=device)
    return _semantic_tagger_instance
