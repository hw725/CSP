"""
Semantic Cross-Lingual Boundary 모델 Production 로더

BGE-M3 + kiwipiepy + 학습된 cross-attention head로
원문의 문장 경계를 예측.

기존 BoundaryModelLoader와 동일한 인터페이스 제공:
  - segment_text(text, task, threshold)
  - predict_boundary_logits(text, task)
  + 새 메서드: segment_with_target(src_text, tgt_text, threshold)
"""

import math
import warnings
from pathlib import Path
from typing import List, Optional

import torch

from common.semantic_boundary_model import (
    SemanticCrossLingualBoundary,
    BGE_M3_DIM,
    POS_DIM,
)


def _build_eojeol_end_set(raw_text: str) -> set:
    """원본 텍스트에서 어절 끝 위치를 정규화 좌표(norm_pos)로 반환.

    어절 = 공백으로 구분된 단위.  어절의 마지막 문자 위치만 유효한 경계.
    """
    valid = set()
    norm_idx = -1
    for i, ch in enumerate(raw_text):
        if ch in (" ", "\n", "\t", "\r"):
            continue
        norm_idx += 1
        next_i = i + 1
        if next_i >= len(raw_text) or raw_text[next_i] in (" ", "\n", "\t", "\r"):
            valid.add(norm_idx)
    return valid


def _snap_peaks_to_eojeol(
    raw_text: str,
    peaks: List[tuple],
) -> List[tuple]:
    """경계 위치를 가장 가까운 어절 끝으로 스냅 (어절 내부 분리 방지)."""
    if not peaks or not raw_text:
        return peaks

    valid_ends = _build_eojeol_end_set(raw_text)
    if not valid_ends:
        return peaks

    norm_len = sum(1 for ch in raw_text if ch not in (" ", "\n", "\t", "\r"))

    adjusted = []
    for pos, logit in peaks:
        if pos in valid_ends:
            adjusted.append((pos, logit))
            continue
        # forward snap
        snapped = None
        for candidate in range(pos, norm_len):
            if candidate in valid_ends:
                snapped = candidate
                break
        if snapped is None:
            for candidate in range(pos - 1, -1, -1):
                if candidate in valid_ends:
                    snapped = candidate
                    break
        adjusted.append((snapped if snapped is not None else pos, logit))

    return adjusted


class SemanticBoundaryLoader:
    """Semantic Cross-Lingual Boundary 모델 로더"""

    def __init__(self, model_path: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")

        # 체크포인트 로드
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # 모델 하이퍼파라미터
        proj_dim = checkpoint.get("proj_dim", 256)
        n_attn_layers = checkpoint.get("n_attn_layers", 2)
        n_attn_heads = checkpoint.get("n_attn_heads", 4)
        dropout = checkpoint.get("dropout", 0.1)
        lstm_hidden = checkpoint.get("lstm_hidden", 128)
        self.best_threshold = checkpoint.get("best_threshold", 0.0)
        self.max_len = checkpoint.get("max_len", 512)

        # 모델 초기화
        self.model = SemanticCrossLingualBoundary(
            src_dim=BGE_M3_DIM,
            tgt_dim=BGE_M3_DIM,
            pos_dim=POS_DIM,
            proj_dim=proj_dim,
            n_attn_heads=n_attn_heads,
            n_attn_layers=n_attn_layers,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # BGE-M3 & kiwipiepy는 lazy loading
        self._bgem3 = None
        self._tokenizer = None
        self._kiwi = None

        print(f"✅ Semantic Boundary 모델 로드: {self.model_path}")
        print(f"   best_threshold(logit)={self.best_threshold:.2f}")

    def _load_bgem3(self):
        """BGE-M3 lazy loading"""
        if self._bgem3 is not None:
            return

        warnings.filterwarnings("ignore")
        from FlagEmbedding import BGEM3FlagModel

        self._bgem3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self._tokenizer = self._bgem3.tokenizer
        print("  BGE-M3 로드 완료")

    def _load_kiwi(self):
        """kiwipiepy lazy loading"""
        if self._kiwi is not None:
            return

        from common.tokenizers.kiwi_tokenizer import get_kiwi_tokenizer

        self._kiwi = get_kiwi_tokenizer()
        self._kiwi._initialize()

    def _get_char_embeddings(self, text: str) -> torch.Tensor:
        """텍스트의 문자별 BGE-M3 임베딩 [L, 1024]"""
        self._load_bgem3()

        encoded = self._tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoded.pop("offset_mapping")[0]  # [L, 2]
        # bgem3.model = 래퍼, bgem3.model.model = XLMRobertaModel
        xlm_model = self._bgem3.model.model
        bgem3_device = next(xlm_model.parameters()).device
        inputs = {k: v.to(bgem3_device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = xlm_model(**inputs)
            hidden = outputs.last_hidden_state[0].cpu()  # [L, 1024]

        # 토큰 → 문자 매핑
        char_emb = torch.zeros(len(text), hidden.shape[-1])
        for t_idx, (start, end) in enumerate(offset_mapping.tolist()):
            if start == 0 and end == 0:
                continue
            for c_idx in range(int(start), min(int(end), len(text))):
                char_emb[c_idx] = hidden[t_idx]

        return char_emb

    def _get_pos_features(self, text: str) -> torch.Tensor:
        """원문의 POS features [L, 14]"""
        self._load_kiwi()

        from common.semantic_boundary_model import POS_TAGS

        features = torch.zeros(len(text), POS_DIM)
        try:
            pos_results = self._kiwi.pos(text)
            cursor = 0
            for token_text, pos_tag in pos_results:
                pos = text.find(token_text, cursor)
                if pos == -1:
                    pos = cursor
                if pos_tag in POS_TAGS:
                    tag_idx = POS_TAGS.index(pos_tag)
                    for j in range(len(token_text)):
                        char_idx = pos + j
                        if 0 <= char_idx < len(text):
                            features[char_idx, tag_idx] = 1.0
                cursor = pos + len(token_text)
        except Exception:
            pass

        return features

    def predict_boundary_logits_with_target(
        self, src_text: str, tgt_text: str
    ) -> List[float]:
        """
        원문+번역문으로 경계 logits 예측.

        Note: 입력 텍스트가 이미 정규화되어 있다고 가정.
              segment_with_target은 내부에서 정규화 후 이 메서드를 호출.
              외부에서 직접 호출 시에도 정규화된 텍스트를 전달해야 정확.

        Returns:
            [len(src_text)] 크기의 logit 리스트
        """
        if not src_text or not tgt_text:
            return [0.0] * len(src_text) if src_text else []

        # 임베딩 추출
        src_emb = self._get_char_embeddings(src_text)  # [L_src, 1024]
        tgt_emb = self._get_char_embeddings(tgt_text)  # [L_tgt, 1024]
        pos_feat = self._get_pos_features(src_text)  # [L_src, 14]

        # 배치 차원 추가
        src_emb = src_emb.unsqueeze(0).to(self.device)
        tgt_emb = tgt_emb.unsqueeze(0).to(self.device)
        pos_feat = pos_feat.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(src_emb, tgt_emb, pos_feat)
            return logits[0].cpu().tolist()

    @staticmethod
    def _normalize(text: str) -> str:
        """학습 시 사용한 정규화와 동일: 공백/개행/탭 제거"""
        return text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "").strip()

    @staticmethod
    def _norm_to_raw_map(raw_text: str) -> List[int]:
        """정규화 텍스트의 각 위치가 원본 텍스트에서 어느 위치인지 매핑.
        Returns: norm_pos -> raw_pos 매핑 리스트"""
        mapping = []
        for i, ch in enumerate(raw_text):
            if ch not in (" ", "\n", "\t", "\r"):
                mapping.append(i)
        return mapping

    def segment_with_target(
        self,
        src_text: str,
        tgt_text: str,
        threshold: float = None,
        min_len_override: int = None,
        precomputed_logits: Optional[List[float]] = None,
    ) -> List[str]:
        """
        원문+번역문으로 원문을 분할.

        내부에서 정규화(공백 제거)하여 모델 추론 후, 원본 텍스트에 매핑하여 분할.
        (학습 시 정규화 텍스트로 학습했으므로, 추론도 정규화 텍스트에서 수행)

        Args:
            src_text: 원문 (raw, 공백 포함 가능)
            tgt_text: 번역문 (raw, 공백 포함 가능)
            threshold: None이면 학습된 best_threshold(logit) 사용
            min_len_override: 최소 세그먼트 길이
            precomputed_logits: 사전계산된 logit 리스트 (정규화 텍스트 기준).
                                전달 시 BGE-M3 forward pass를 건너뛰어 속도 향상.

        Returns:
            분할된 세그먼트 리스트 (원본 텍스트 기준)
        """
        if not src_text:
            return []

        # 정규화: 학습 시와 동일한 좌표계에서 추론
        src_norm = self._normalize(src_text)
        tgt_norm = self._normalize(tgt_text) if tgt_text else ""

        if not src_norm:
            return [src_text]

        if precomputed_logits is not None and len(precomputed_logits) == len(src_norm):
            logits = precomputed_logits
        else:
            logits = self.predict_boundary_logits_with_target(src_norm, tgt_norm)
        if not logits:
            return [src_text]

        # threshold가 명시되면 사용, 아니면 학습 시 결정된 best_threshold (logit 공간)
        if threshold is not None:
            # processor는 확률 공간(0-1)으로 threshold를 전달하므로 logit으로 변환
            if 0.0 < threshold < 1.0:
                logit_thr = math.log(threshold / (1.0 - threshold))
            elif threshold <= 0.0:
                logit_thr = -float("inf")
            elif threshold >= 1.0:
                logit_thr = float("inf")
            else:
                logit_thr = threshold
        else:
            logit_thr = self.best_threshold

        # 디코딩: logit → 후보 → peak 선택 (정규화 좌표계)
        probs = [1.0 / (1.0 + math.exp(-x)) if x < 20 else 1.0 for x in logits]
        candidates = [
            (i, p) for i, (l, p) in enumerate(zip(logits, probs))
            if l >= logit_thr and i > 0
        ]

        if not candidates:
            return [src_text]

        candidates.sort(key=lambda x: x[0])

        # 인접 후보 그루핑 → peak 선택
        grouped_peaks = []
        cur_group = [candidates[0]]
        for i, p in candidates[1:]:
            prev_i, _ = cur_group[-1]
            if i <= prev_i + 1:
                cur_group.append((i, p))
            else:
                grouped_peaks.append(max(cur_group, key=lambda x: x[1]))
                cur_group = [(i, p)]
        if cur_group:
            grouped_peaks.append(max(cur_group, key=lambda x: x[1]))

        # 최소 세그먼트 길이
        min_len = 6 if min_len_override is None else max(1, min_len_override)

        filtered_peaks = []
        for i, p in grouped_peaks:
            if not filtered_peaks:
                filtered_peaks.append((i, p))
                continue
            prev_i, prev_p = filtered_peaks[-1]
            if (i - prev_i) < min_len:
                if p > prev_p:
                    filtered_peaks[-1] = (i, p)
            else:
                filtered_peaks.append((i, p))

        # 어절 내부 분리 방지: 경계를 어절 끝으로 스냅
        filtered_peaks = _snap_peaks_to_eojeol(src_text, filtered_peaks)
        # 스냅 후 중복 제거
        seen = set()
        deduped = []
        for i, p in filtered_peaks:
            if i not in seen:
                seen.add(i)
                deduped.append((i, p))
        filtered_peaks = sorted(deduped, key=lambda x: x[0])

        norm_boundaries = [i for i, _ in filtered_peaks]
        if not norm_boundaries:
            return [src_text]

        # 정규화 경계 → 원본 텍스트 경계로 매핑
        # 라벨 정의: 문장 마지막 문자 위치에 1 → 해당 문자 포함하여 분할
        norm_map = self._norm_to_raw_map(src_text)

        segments = []
        start = 0
        for norm_pos in norm_boundaries:
            if norm_pos < len(norm_map):
                raw_pos = norm_map[norm_pos]
                # raw_pos+1: 해당 문자 포함
                segments.append(src_text[start:raw_pos + 1])
                start = raw_pos + 1
        if start < len(src_text):
            segments.append(src_text[start:])

        # 고립 구두점 병합
        punctuation = set('.,!?;:\'"。、，！？；：""…—·)]}）〉》」』】〕〗〙〛〉')
        merged = []
        for seg in segments:
            if seg and len(seg.strip()) <= 1 and seg.strip() in punctuation:
                if merged:
                    merged[-1] += seg
                else:
                    merged.append(seg)
            else:
                merged.append(seg)

        return merged if merged else segments

    # ── 기존 BoundaryModelLoader 호환 인터페이스 ──

    def segment_text(
        self,
        text: str,
        task: str = "pa",
        threshold: float = None,
        min_len_override: int = None,
        tgt_text: str = None,
        precomputed_logits: Optional[List[float]] = None,
    ) -> List[str]:
        """
        기존 BoundaryModelLoader 호환 인터페이스.

        tgt_text가 주어지면 cross-lingual 모드,
        없으면 src만으로 추론 (tgt=src로 fallback).
        threshold=None이면 학습된 best_threshold 자동 사용.
        precomputed_logits가 주어지면 BGE-M3 forward pass를 건너뜀.
        """
        if tgt_text:
            return self.segment_with_target(
                text, tgt_text, threshold, min_len_override,
                precomputed_logits=precomputed_logits,
            )
        else:
            # tgt 없으면 src를 tgt로 사용 (self-attention 모드)
            return self.segment_with_target(
                text, text, threshold, min_len_override,
                precomputed_logits=precomputed_logits,
            )

    def predict_boundary_logits(
        self, text: str, task: str = "pa", tgt_text: str = None
    ) -> List[float]:
        """기존 BoundaryModelLoader 호환 인터페이스."""
        if tgt_text:
            return self.predict_boundary_logits_with_target(text, tgt_text)
        else:
            return self.predict_boundary_logits_with_target(text, text)

    def predict_boundary_probs(
        self, text: str, task: str = "pa", tgt_text: str = None
    ) -> List[float]:
        """경계 확률 반환"""
        logits = self.predict_boundary_logits(text, task, tgt_text)
        out = []
        for x in logits:
            if x >= 0:
                z = math.exp(-x)
                out.append(1.0 / (1.0 + z))
            else:
                z = math.exp(x)
                out.append(z / (1.0 + z))
        return out
