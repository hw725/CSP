"""S2P Phrase Alignment Model 로더 (v2)

학습된 s2p_phrase_alignment.pt를 로드하여 구 단위 정렬 기반 경계 추론.
기존 CrossAttnBoundaryTaggerLoader와 동일한 인터페이스 제공.

v1과의 차이:
  v1: 문자 수준 B/O 예측 (128d 문자 임베딩)
  v2: 구 수준 정렬 예측 (BGE-M3 1024d + Viterbi 디코딩)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn


class PhraseAlignmentModel(nn.Module):
    """구 단위 정렬 모델 v2.1 (train_s2p_phrase_alignment.py와 동일 구조)

    v2→v2.1 변경: Source BiLSTM 추가 (구 간 문맥+순서 학습)
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

        # Source encoder: BGE projection + BiLSTM
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

        # Target encoder
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

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.cross_norm = nn.LayerNorm(hidden)

        self.alignment_proj = nn.Linear(hidden, hidden)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, src_embs, tgt_ids, src_mask=None):
        # Source: BGE → projection → BiLSTM
        src_h = self.src_proj(src_embs)
        src_h, _ = self.src_encoder(src_h)
        src_h = self.src_norm(src_h)

        tgt_emb = self.tgt_emb(tgt_ids)
        tgt_h, _ = self.tgt_encoder(tgt_emb)
        tgt_h = self.tgt_norm(tgt_h)

        key_padding_mask = ~src_mask if src_mask is not None else None
        cross_out, _ = self.cross_attn(
            query=tgt_h,
            key=src_h,
            value=src_h,
            key_padding_mask=key_padding_mask,
        )
        tgt_enriched = self.cross_norm(cross_out + tgt_h)

        tgt_proj = self.alignment_proj(tgt_enriched)
        alignment_logits = torch.bmm(tgt_proj, src_h.transpose(1, 2))
        alignment_logits = alignment_logits / self.temperature.abs().clamp(min=0.01)

        if src_mask is not None:
            alignment_logits = alignment_logits.masked_fill(
                ~src_mask.unsqueeze(1), float("-inf")
            )

        return alignment_logits


def _monotonic_viterbi(logits: torch.Tensor, n_phrases: int) -> List[int]:
    """Viterbi 디코딩: 단조 증가 제약 하에 최적 구 할당

    Args:
        logits: [T, N_max] — 각 문자 위치별 구 소속 logits
        n_phrases: 실제 구 개수 (N)

    Returns:
        assignments: [T] — 각 문자의 소속 구 인덱스 (0..N-1)
    """
    T = logits.shape[0]
    N = n_phrases

    if N <= 0 or T <= 0:
        return [0] * T
    if N == 1:
        return [0] * T

    # dp[t][n] = 위치 0..t까지 할당했을 때, 위치 t가 구 n에 속하는 최대 누적 점수
    NEG_INF = float("-inf")
    dp = [[NEG_INF] * N for _ in range(T)]
    bp = [[0] * N for _ in range(T)]  # backpointer

    # Base: 첫 문자는 반드시 구 0
    dp[0][0] = logits[0, 0].item()

    for t in range(1, T):
        for n in range(min(t + 1, N)):
            # 같은 구에 유지
            stay = dp[t - 1][n] + logits[t, n].item()
            best_score = stay
            best_prev = n

            # 이전 구에서 진행
            if n > 0 and dp[t - 1][n - 1] > NEG_INF:
                advance = dp[t - 1][n - 1] + logits[t, n].item()
                if advance > best_score:
                    best_score = advance
                    best_prev = n - 1

            dp[t][n] = best_score
            bp[t][n] = best_prev

    # 역추적: 마지막 문자는 반드시 마지막 구 (N-1)
    assignments = [0] * T
    # 마지막 구가 도달 불가능하면 가장 높은 점수의 구에서 시작
    if dp[T - 1][N - 1] > NEG_INF:
        current = N - 1
    else:
        current = max(range(N), key=lambda n: dp[T - 1][n])

    assignments[T - 1] = current
    for t in range(T - 2, -1, -1):
        current = bp[t + 1][current]
        assignments[t] = current

    return assignments


class PhraseAlignmentTagger:
    """Phrase Alignment 기반 경계 태거 — CrossAttnBoundaryTaggerLoader 대체

    동일한 segment_text / segment_text_batch 인터페이스 제공.
    """

    def __init__(self, model_path: Path = None, device: str = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        if model_path is None:
            model_path = (
                Path(__file__).parent.parent
                / "models"
                / "s2p_phrase_alignment.pt"
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Phrase Alignment 모델 없음: {model_path}"
            )

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        self.tgt_vocab = checkpoint["tgt_vocab"]
        self.tgt_max_len = checkpoint.get("tgt_max_len", 1024)
        self.max_phrases = checkpoint.get("max_phrases", 64)
        self.hidden = checkpoint.get("hidden", 512)
        self.tgt_emb_dim = checkpoint.get("tgt_emb_dim", 128)
        self.bge_dim = checkpoint.get("bge_dim", 1024)

        self.model = PhraseAlignmentModel(
            bge_dim=self.bge_dim,
            tgt_vocab_size=len(self.tgt_vocab) + 1,
            tgt_emb_dim=self.tgt_emb_dim,
            hidden=self.hidden,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self._bge_model = None  # lazy load

        print(
            f"✅ Phrase Alignment 태거 로드 완료 "
            f"(tgt_vocab={len(self.tgt_vocab)}, "
            f"bge_dim={self.bge_dim}, device={self.device})"
        )

    def _get_bge_model(self):
        """BGE-M3 모델 lazy loading"""
        if self._bge_model is None:
            try:
                from common.embedders.bge import get_embedding_manager

                self._bge_model = get_embedding_manager()
            except ImportError:
                from FlagEmbedding import BGEM3FlagModel

                self._bge_model = BGEM3FlagModel(
                    "BAAI/bge-m3", use_fp16=True
                )
        return self._bge_model

    def _encode_phrases_bge(
        self, phrases: List[str], precomputed: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """원문 구들의 BGE 임베딩 계산 또는 사전계산값 사용

        Returns: [1, max_phrases, bge_dim] tensor
        """
        n = min(len(phrases), self.max_phrases)
        embs = np.zeros((self.max_phrases, self.bge_dim), dtype=np.float32)

        if precomputed is not None:
            for i in range(n):
                embs[i] = precomputed[i]
        else:
            bge = self._get_bge_model()
            # FlagEmbedding 직접 사용 (dense_vecs = 1024d)
            if hasattr(bge, "model") and hasattr(bge.model, "encode"):
                # EmbeddingManager → 내부 BGEM3FlagModel 직접 사용
                raw = bge.model.encode(phrases[:n])["dense_vecs"]
                embs[:n] = raw
            elif hasattr(bge, "encode"):
                # BGEM3FlagModel 직접
                raw = bge.encode(phrases[:n])["dense_vecs"]
                embs[:n] = raw
            else:
                # compute_embeddings_with_cache 폴백 (1636d → 1024d 잘라내기)
                raw = bge.compute_embeddings_with_cache(phrases[:n])
                for i, e in enumerate(raw):
                    vec = e if isinstance(e, np.ndarray) else e.numpy()
                    embs[i] = vec[: self.bge_dim]

        return torch.from_numpy(embs).unsqueeze(0).to(self.device)

    def _encode_tgt(self, tgt_text: str) -> torch.Tensor:
        """번역문 문자 인코딩"""
        ids = [self.tgt_vocab.get(c, 0) for c in tgt_text][: self.tgt_max_len]
        ids += [0] * (self.tgt_max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def predict_boundaries(
        self,
        src_phrases: List[str],
        tgt_text: str,
        src_phrase_embeddings: Optional[np.ndarray] = None,
    ) -> List[int]:
        """구 정렬 기반 경계 위치 예측

        Args:
            src_phrases: 원문 구 리스트
            tgt_text: 번역문 텍스트
            src_phrase_embeddings: 사전계산된 BGE 임베딩 [N, 1024]. None이면 내부 계산.

        Returns:
            boundary positions: 번역문에서 경계 문자 인덱스 리스트
        """
        n_phrases = min(len(src_phrases), self.max_phrases)
        tgt_len = min(len(tgt_text), self.tgt_max_len)

        if n_phrases <= 1 or tgt_len == 0:
            return []

        src_embs = self._encode_phrases_bge(src_phrases, src_phrase_embeddings)
        src_mask = torch.zeros(
            1, self.max_phrases, dtype=torch.bool, device=self.device
        )
        src_mask[0, :n_phrases] = True

        tgt_ids = self._encode_tgt(tgt_text)

        with torch.no_grad():
            logits = self.model(src_embs, tgt_ids, src_mask)  # [1, T, N]

        logits_np = logits[0, :tgt_len, :n_phrases]

        # Viterbi 디코딩 (단조 증가 제약)
        assignments = _monotonic_viterbi(logits_np, n_phrases)

        # 소속 구 변경 지점 = 경계
        boundaries = []
        for t in range(1, tgt_len):
            if assignments[t] != assignments[t - 1]:
                boundaries.append(t)

        return boundaries

    def segment_text(
        self,
        src_text: str,
        tgt_text: str,
        n_segments: int = None,
        src_phrase_embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[str]:
        """기존 CrossAttnBoundaryTaggerLoader와 동일한 인터페이스

        Args:
            src_text: 원문 (공백으로 구 분할)
            tgt_text: 번역문
            n_segments: 무시됨 (구 개수는 src_text에서 자동 추출)
            src_phrase_embeddings: 사전계산된 BGE 임베딩
        """
        if not tgt_text or not tgt_text.strip():
            return [tgt_text] if tgt_text else []

        src_phrases = src_text.split()
        if len(src_phrases) <= 1:
            return [tgt_text]

        boundaries = self.predict_boundaries(
            src_phrases, tgt_text, src_phrase_embeddings
        )

        if not boundaries:
            return [tgt_text]

        segments = []
        start = 0
        for pos in boundaries:
            seg = tgt_text[start:pos]
            if seg.strip():
                segments.append(seg)
            elif segments:
                segments[-1] += seg  # 빈 세그먼트는 이전에 병합
            start = pos
        last = tgt_text[start:]
        if last.strip():
            segments.append(last)
        elif segments:
            segments[-1] += last

        return segments if segments else [tgt_text]

    def segment_text_batch(
        self,
        src_texts: List[str],
        tgt_texts: List[str],
        n_segments_list: List[int] = None,
        src_phrase_embeddings_list: List[Optional[np.ndarray]] = None,
        **kwargs,
    ) -> List[List[str]]:
        """배치 경계 분할"""
        results = []
        for i in range(len(src_texts)):
            embs = (
                src_phrase_embeddings_list[i]
                if src_phrase_embeddings_list
                else None
            )
            segments = self.segment_text(
                src_texts[i],
                tgt_texts[i],
                src_phrase_embeddings=embs,
            )
            results.append(segments)
        return results


# 전역 인스턴스 캐싱
_phrase_alignment_tagger_instance = None


def get_phrase_alignment_tagger(
    model_path: Path = None, device: str = "cuda"
) -> PhraseAlignmentTagger:
    """Phrase Alignment 태거 싱글톤 반환"""
    global _phrase_alignment_tagger_instance

    if _phrase_alignment_tagger_instance is None:
        _phrase_alignment_tagger_instance = PhraseAlignmentTagger(
            model_path=model_path, device=device
        )

    return _phrase_alignment_tagger_instance
