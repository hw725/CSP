"""
Boundary-aware Alignment Model Loader 및 추론 엔진

기존 AlignmentMatcher 대체용:
- 경계 정보를 입력으로 받음
- Boundary match score 추가 출력
"""

from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch import nn

class BoundaryAwareCharEncoder(nn.Module):
    """Boundary 정보를 포함한 Character Encoder"""

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # Checkpoint는 boundary embedding 없이 저장됨 (embedding만 128)
        self.lstm = nn.LSTM(
            emb_dim,  # embedding only, no boundary embedding
            hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden * 2, 256)

    def forward(self, x, b=None):
        # Checkpoint는 boundary 없이 학습됨, inference에서도 무시
        # x: [batch_size, seq_len]
        if x is None or x.numel() == 0:
            # Empty input handling
            return torch.zeros(
                1,
                256,
                device=x.device if x is not None else self.char_emb.weight.device,
            )

        char_emb = self.char_emb(x)
        lstm_out, _ = self.lstm(char_emb)
        pooled = lstm_out.mean(dim=1) if lstm_out.shape[0] > 0 else lstm_out[0]
        z = self.proj(pooled)
        z = nn.functional.normalize(z, dim=-1)
        return z

class BoundaryAwareDualEncoder(nn.Module):
    """Context-aware Dual Encoder"""

    def __init__(self, vocab_src: int, vocab_tgt: int):
        super().__init__()
        self.enc_src = BoundaryAwareCharEncoder(vocab_src)
        self.enc_tgt = BoundaryAwareCharEncoder(vocab_tgt)

        self.boundary_classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, xs, xt, bs=None, bt=None, compute_boundary_match=True):
        # Checkpoint는 boundary 없이 학습됨 (bs, bt 무시)
        zs = self.enc_src(xs)
        zt = self.enc_tgt(xt)

        if compute_boundary_match:
            combined = torch.cat([zs, zt], dim=-1)
            boundary_score = self.boundary_classifier(combined).squeeze(-1)
            return zs, zt, boundary_score
        else:
            return zs, zt

class BoundaryAwareAlignmentMatcher:
    """
    Boundary-aware Alignment Matcher

    기존 AlignmentMatcher와 호환되는 인터페이스 + boundary match score 추가
    """

    def __init__(
        self, model_path: Path, device: str = "cuda", boundary_weight: float = 0.3
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.boundary_weight = float(boundary_weight)

        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 모델 파일 없음: {self.model_path}")

        # 체크포인트 로드
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # 모델 파일 형식에 따라 처리
        if "model_src" in checkpoint and "model_tgt" in checkpoint:
            # 새 형식: model_src, model_tgt는 OrderedDict (state_dict)
            self.vocab_src = checkpoint.get("vocab_src", {})
            self.vocab_tgt = checkpoint.get("vocab_tgt", {})
            # vocab 사이즈는 vocab 딕셔너리의 크기 + 1 (UNK token)
            actual_vocab_src = len(self.vocab_src) + 1 if self.vocab_src else 5000
            actual_vocab_tgt = len(self.vocab_tgt) + 1 if self.vocab_tgt else 5000
        else:
            # 기존 형식: state_dict 키 하에 전체 state_dict
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.vocab_src = checkpoint.get("vocab_src", {})
            self.vocab_tgt = checkpoint.get("vocab_tgt", {})
            actual_vocab_src = (
                state_dict["enc_src.char_emb.weight"].shape[0]
                if "enc_src.char_emb.weight" in state_dict
                else len(self.vocab_src) + 1
            )
            actual_vocab_tgt = (
                state_dict["enc_tgt.char_emb.weight"].shape[0]
                if "enc_tgt.char_emb.weight" in state_dict
                else len(self.vocab_tgt) + 1
            )

        self.model = BoundaryAwareDualEncoder(
            vocab_src=actual_vocab_src, vocab_tgt=actual_vocab_tgt
        ).to(self.device)

        # 상태 딕셔너리 로드
        if "state_dict" in checkpoint:
            # 기존 형식
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif "model_src" in checkpoint and "model_tgt" in checkpoint:
            # 새 형식: model_src와 model_tgt가 각각 encoder state_dict
            try:
                # 모델의 full state_dict를 가져옴
                full_state = self.model.state_dict()

                # model_src (src encoder) 가중치를 enc_src로 매핑
                src_state = checkpoint["model_src"]
                for key, value in src_state.items():
                    full_key = f"enc_src.{key}"
                    if full_key in full_state:
                        full_state[full_key] = value

                # model_tgt (tgt encoder) 가중치를 enc_tgt로 매핑
                tgt_state = checkpoint["model_tgt"]
                for key, value in tgt_state.items():
                    full_key = f"enc_tgt.{key}"
                    if full_key in full_state:
                        full_state[full_key] = value

                # 부분 로드 (strict=False는 매핑되지 않은 가중치는 초기화된 상태로 유지)
                self.model.load_state_dict(full_state, strict=False)
            except Exception as e:
                print(f"⚠️ 가중치 로드 실패, 모델 구조만 사용: {e}")
        else:
            try:
                self.model.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                print(f"⚠️ 상태 로드 실패: {e}")

        self.model.eval()

        print(f"✅ Boundary-aware Alignment 모델 로드: {self.model_path}")
        print(f"   vocab_src={len(self.vocab_src)}, vocab_tgt={len(self.vocab_tgt)}")

    def _extract_boundaries(
        self, text: str, is_src: bool = True, extra_boundaries: List[int] | None = None
    ) -> List[int]:
        """텍스트에서 어절/구절 경계 추출"""
        import re

        if not text:
            return []

        boundaries = [0]

        if is_src:
            # 원문: 공백 기준
            for match in re.finditer(r"\s+", text):
                boundary_pos = match.end()
                if boundary_pos < len(text):
                    boundaries.append(boundary_pos)
        else:
            # 번역문: 공백 + 구두점
            for match in re.finditer(r"[\s,\.!?\)\]\}]+", text):
                boundary_pos = match.end()
                if boundary_pos < len(text):
                    boundaries.append(boundary_pos)

        if extra_boundaries:
            boundaries.extend([int(x) for x in extra_boundaries if x is not None])

        # 모델의 boundary flag는 text index 좌표계(0<=idx<len(text))여야 한다.
        cleaned: List[int] = []
        for b in boundaries:
            if 0 <= int(b) < len(text):
                cleaned.append(int(b))
        return sorted(set(cleaned))

    def _encode_text(
        self, text: str, is_src: bool = True, max_len: int = 512
    ) -> torch.Tensor:
        """텍스트를 character ID로 변환"""
        vocab = self.vocab_src if is_src else self.vocab_tgt
        ids = [vocab.get(ch, 0) for ch in text]
        ids = ids[:max_len]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return torch.tensor([ids], dtype=torch.long).to(self.device)

    def _encode_boundaries(
        self, text_len: int, boundaries: List[int], max_len: int = 512
    ) -> torch.Tensor:
        """Boundary 위치를 binary flag로 변환"""
        flags = [0] * min(text_len, max_len)
        for b in boundaries:
            if 0 <= b < len(flags):
                flags[b] = 1

        pad_len = max_len - len(flags)
        if pad_len > 0:
            flags += [0] * pad_len

        return torch.tensor([flags], dtype=torch.float32).to(self.device)

    def _segment_boundaries_from_segments(self, segments: List[str]) -> List[int]:
        """세그먼트 리스트를 concat 했을 때의 '세그먼트 시작 위치'를 boundary로 반환.

        - 첫 세그먼트 시작(0)은 항상 포함되므로 여기서는 0을 제외한 offset만 반환한다.
        - 공백 없이 join("".join)하는 좌표계 기준.
        """
        offsets: List[int] = []
        pos = 0
        for i, seg in enumerate(segments):
            seg = "" if seg is None else str(seg)
            if i > 0:
                offsets.append(pos)
            pos += len(seg)
        return offsets

    def _compute_similarity_with_boundaries(
        self,
        src_text: str,
        tgt_text: str,
        *,
        src_boundaries: List[int] | None = None,
        tgt_boundaries: List[int] | None = None,
    ) -> Tuple[float, float]:
        """외부에서 주어진 boundary list를 그대로 사용해 (sim, boundary) 계산.

        Checkpoint는 boundary flags 없이 저장되어 있으므로 flags를 보내지 않음.
        """
        if not src_text or not tgt_text:
            return 0.0, 0.0

        src_ids = self._encode_text(src_text, is_src=True)
        tgt_ids = self._encode_text(tgt_text, is_src=False)

        # Checkpoint는 boundary flags 없이 학습됨 - None이 아니라 보내지 않음
        with torch.no_grad():
            v_src, v_tgt, boundary_score = self.model(
                src_ids,
                tgt_ids,
                compute_boundary_match=True,
            )
            cos_sim = (v_src * v_tgt).sum(dim=-1).item()
            boundary_match = boundary_score.item()

        return cos_sim, boundary_match

    def compute_similarity_with_boundary(
        self, src_text: str, tgt_text: str
    ) -> Tuple[float, float]:
        """
        원문과 번역문 사이의 유사도 및 경계 일치 점수 계산

        Returns:
            (similarity, boundary_match_score)
        """
        if not src_text or not tgt_text:
            return 0.0, 0.0

        # Text encoding
        src_ids = self._encode_text(src_text, is_src=True)
        tgt_ids = self._encode_text(tgt_text, is_src=False)

        src_boundaries = self._extract_boundaries(src_text, is_src=True)
        tgt_boundaries = self._extract_boundaries(tgt_text, is_src=False)

        # Forward
        cos_sim, boundary_match = self._compute_similarity_with_boundaries(
            src_text,
            tgt_text,
            src_boundaries=src_boundaries,
            tgt_boundaries=tgt_boundaries,
        )

        return cos_sim, boundary_match

    def compute_similarity_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """(src, tgt) 쌍 리스트를 한 번의 GPU forward pass로 처리.

        개별 compute_similarity() 반복 호출 대비 GPU utilization 향상.
        """
        if not pairs:
            return []
        n = len(pairs)
        max_len = 512

        # 배치 인코딩 (CPU)
        src_batch = torch.zeros(n, max_len, dtype=torch.long)
        tgt_batch = torch.zeros(n, max_len, dtype=torch.long)
        for i, (src_text, tgt_text) in enumerate(pairs):
            for j, ch in enumerate(src_text[:max_len]):
                src_batch[i, j] = self.vocab_src.get(ch, 0)
            for j, ch in enumerate(tgt_text[:max_len]):
                tgt_batch[i, j] = self.vocab_tgt.get(ch, 0)

        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)

        with torch.no_grad():
            v_src, v_tgt, boundary_scores = self.model(
                src_batch, tgt_batch, compute_boundary_match=True
            )
            cos_sims = (v_src * v_tgt).sum(dim=-1)  # [n]
            w = self.boundary_weight
            combined = (1 - w) * cos_sims + w * boundary_scores  # [n]
            return combined.cpu().tolist()

    def compute_similarity(self, src_text: str, tgt_text: str) -> float:
        """
        기존 AlignmentMatcher와 호환되는 인터페이스
        - 기본값으로 '의미 유사도 + 경계 일치' 결합 점수를 반환한다.
        - strict 후보 선택/매칭에서 이 값이 직접 사용되므로, PA 파이프라인 교체 시 별도 수정이 필요 없다.
        """
        return self.compute_combined_score(
            src_text, tgt_text, boundary_weight=self.boundary_weight
        )

    def compute_combined_score(
        self, src_text: str, tgt_text: str, boundary_weight: float = 0.3
    ) -> float:
        """
        의미 유사도 + 경계 일치를 결합한 최종 점수

        Args:
            boundary_weight: 경계 점수 가중치 (0~1)

        Returns:
            combined_score = (1-w)*similarity + w*boundary_match
        """
        sim, boundary = self.compute_similarity_with_boundary(src_text, tgt_text)
        combined = (1 - boundary_weight) * sim + boundary_weight * boundary
        return combined

    def match_segments(
        self, src_segments: List[str], tgt_segments: List[str]
    ) -> List[str]:
        """원문 세그먼트를 번역문 세그먼트와 greedy matching.

        반환 개수/순서 무결성 규칙은 기존 AlignmentMatcher.match_segments와 동일하게 유지한다.
        차이점:
        - 스코어링은 compute_similarity(=combined score)를 사용한다.
        - candidate concat 시, 세그먼트 경계 offset을 boundary flag에 추가하여 경계-aware 성질을 강화한다.
        """
        if not tgt_segments:
            return []

        joined_src = "".join([s for s in src_segments if s is not None])
        if joined_src and len(src_segments) < len(tgt_segments):
            target_count = min(len(joined_src), len(tgt_segments))
            if target_count >= 1:
                base_segments: List[str] = [
                    s for s in src_segments if s is not None and s != ""
                ]
                if not base_segments:
                    base_segments = [joined_src]

                pieces_per_seg = [1 for _ in base_segments]
                extra = target_count - len(base_segments)
                seg_lens = [len(s) for s in base_segments]

                def _pick_idx_to_expand() -> int:
                    best_i = 0
                    best_score = -1.0
                    for i, (ln, k) in enumerate(zip(seg_lens, pieces_per_seg)):
                        score = ln / max(1, k)
                        if score > best_score:
                            best_score = score
                            best_i = i
                    return best_i

                while extra > 0:
                    i = _pick_idx_to_expand()
                    pieces_per_seg[i] += 1
                    extra -= 1

                expanded: List[str] = []
                for seg, k in zip(base_segments, pieces_per_seg):
                    if k <= 1 or len(seg) <= 1:
                        expanded.append(seg)
                        continue
                    base = len(seg) // k
                    rem = len(seg) % k
                    pos = 0
                    for j in range(k):
                        step = base + (1 if j < rem else 0)
                        nxt = pos + step
                        expanded.append(seg[pos:nxt])
                        pos = nxt
                    if pos < len(seg):
                        expanded[-1] = expanded[-1] + seg[pos:]

                expanded = [p for p in expanded if p is not None and p != ""]
                if not expanded:
                    expanded = [joined_src]

                src_segments = expanded

        matched_src: List[str] = []
        src_idx = 0
        total_tgt_len = sum(len(t) for t in tgt_segments) or 1

        def _score_pair(
            src_text: str,
            tgt_text: str,
            *,
            src_extra_boundaries: List[int] | None = None,
        ) -> float:
            # segment concat 경계를 boundary flag로 추가한다.
            src_boundaries = self._extract_boundaries(
                src_text, is_src=True, extra_boundaries=src_extra_boundaries
            )
            tgt_boundaries = self._extract_boundaries(tgt_text, is_src=False)
            sim, boundary = self._compute_similarity_with_boundaries(
                src_text,
                tgt_text,
                src_boundaries=src_boundaries,
                tgt_boundaries=tgt_boundaries,
            )
            return (1 - self.boundary_weight) * sim + self.boundary_weight * boundary

        for t_i, tgt_seg in enumerate(tgt_segments):
            remaining_tgt = len(tgt_segments) - t_i

            best_score = -1.0
            best_end = min(src_idx + 1, len(src_segments))

            latest_end_allowed = max(
                src_idx + 1, len(src_segments) - (remaining_tgt - 1)
            )
            max_end = min(src_idx + 5, latest_end_allowed + 1)

            # 후보 수집 후 배치 스코어링
            candidates = []
            for end_idx in range(src_idx + 1, max_end):
                window = src_segments[src_idx:end_idx]
                src_text = "".join(window)
                candidates.append((end_idx, src_text))

            if len(candidates) == 1:
                best_end = candidates[0][0]
            elif len(candidates) > 1:
                pairs = [(src_text, tgt_seg) for _, src_text in candidates]
                scores = self.compute_similarity_batch(pairs)
                for (end_idx, _), score in zip(candidates, scores):
                    if score > best_score:
                        best_score = score
                        best_end = end_idx

            if best_end <= src_idx:
                best_end = min(src_idx + 1, len(src_segments))

            if src_idx >= len(src_segments):
                matched_src.append("")
                continue

            chosen = "".join(src_segments[src_idx:best_end])
            if not chosen:
                remaining_src_len = sum(len(s) for s in src_segments[src_idx:])
                ratio = len(tgt_seg) / total_tgt_len
                target_len = max(1, int(remaining_src_len * ratio))

                accum = 0
                end = src_idx
                while end < len(src_segments) and accum < target_len:
                    accum += len(src_segments[end])
                    end += 1
                chosen = "".join(src_segments[src_idx:end])
                best_end = end

            matched_src.append(chosen)
            src_idx = best_end

        if src_idx < len(src_segments) and matched_src:
            matched_src[-1] = matched_src[-1] + "".join(src_segments[src_idx:])

        return matched_src
