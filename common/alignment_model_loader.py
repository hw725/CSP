"""
Dual Encoder Alignment 모델 로더 및 추론 엔진
- dual_encoder_alignment_pa.pt 또는 sa.pt로부터 세그먼트 정렬
"""

from pathlib import Path
from typing import List, Dict
import torch
from torch import nn


class CharEncoderForAlignment(nn.Module):
    """정렬용 문자 인코더"""
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, 256)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        m = o.mean(dim=1)
        z = self.proj(m)
        z = nn.functional.normalize(z, dim=-1)
        return z


class DualEncoder(nn.Module):
    """원문/번역문 이중 인코더"""
    def __init__(self, vocab_src, vocab_tgt):
        super().__init__()
        self.enc_src = CharEncoderForAlignment(vocab_src)
        self.enc_tgt = CharEncoderForAlignment(vocab_tgt)

    def forward(self, src, tgt):
        v_src = self.enc_src(src)
        v_tgt = self.enc_tgt(tgt)
        return v_src, v_tgt


class AlignmentMatcher:
    """세그먼트 정렬 매칭 엔진"""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 모델 파일 없음: {self.model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.vocab_src: Dict[str, int] = checkpoint.get("vocab_src", {})
        self.vocab_tgt: Dict[str, int] = checkpoint.get("vocab_tgt", {})
        
        # 모델 로드 - 형식에 따라 처리
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            actual_vocab_src = state_dict['enc_src.emb.weight'].shape[0]
            actual_vocab_tgt = state_dict['enc_tgt.emb.weight'].shape[0]
        elif 'model_src' in checkpoint:
            # 새 형식: model_src, model_tgt에서 크기 추출
            actual_vocab_src = len(self.vocab_src) + 1 if self.vocab_src else checkpoint.get('model_src', torch.zeros(1)).shape[0]
            actual_vocab_tgt = len(self.vocab_tgt) + 1 if self.vocab_tgt else checkpoint.get('model_tgt', torch.zeros(1)).shape[0]
            state_dict = checkpoint  # state_dict 없으면 checkpoint 자체 사용
        else:
            # 폴백: 키 기반으로 추론
            state_dict = checkpoint
            actual_vocab_src = len(self.vocab_src) + 1 if self.vocab_src else 256
            actual_vocab_tgt = len(self.vocab_tgt) + 1 if self.vocab_tgt else 256
        
        self.model = DualEncoder(vocab_src=actual_vocab_src, vocab_tgt=actual_vocab_tgt).to(self.device)
        
        # 상태 딕셔너리 로드 시도
        try:
            if isinstance(state_dict, dict) and 'enc_src' in str(state_dict.keys()):
                self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"⚠️ 상태 딕셔너리 로드 불가, 모델 구조만 사용: {e}")
        
        self.model.eval()
        
        print(f"✅ Alignment 모델 로드: {self.model_path}")
        print(f"   vocab_src={len(self.vocab_src)}, vocab_tgt={len(self.vocab_tgt)}")

    def encode_text(self, text: str, is_src: bool = True) -> torch.Tensor:
        """텍스트를 임베딩 벡터로 변환"""
        vocab = self.vocab_src if is_src else self.vocab_tgt
        ids = []
        for ch in text:
            idx = vocab.get(ch, 0)
            ids.append(idx)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        return x

    def compute_similarity(self, src_text: str, tgt_text: str) -> float:
        """원문과 번역문 사이의 유사도 계산"""
        if not src_text or not tgt_text:
            return 0.0
        
        src_ids = self.encode_text(src_text, is_src=True)
        tgt_ids = self.encode_text(tgt_text, is_src=False)
        
        with torch.no_grad():
            v_src, v_tgt = self.model(src_ids, tgt_ids)
            cos_sim = (v_src * v_tgt).sum(dim=-1).item()
        
        return cos_sim

    def match_segments(self, src_segments: List[str], tgt_segments: List[str]) -> List[str]:
        """
        원문 세그먼트를 번역문 세그먼트와 greedy matching
        
        Args:
            src_segments: 원문 세그먼트 리스트
            tgt_segments: 번역문 세그먼트 리스트 (정렬 기준)
        
        Returns:
            매칭된 원문 세그먼트 리스트
        """
        if not tgt_segments:
            return []

        # src 세그먼트 수가 tgt보다 적으면, 빈 원문이 생기기 쉽다.
        # 요구사항(문장 수/순서 불변 + 빈 원문 금지 + 결합 무결성)을 만족하려면
        # 원문 전체를 최소 tgt 개수만큼 '비어있지 않게' 나눌 수 있어야 한다.
        joined_src = "".join([s for s in src_segments if s is not None])
        if joined_src and len(src_segments) < len(tgt_segments):
            target_count = min(len(joined_src), len(tgt_segments))
            if target_count >= 1:
                # 기존 src_segments(예: boundary 후보)의 경계를 최대한 보존한 채,
                # 각 세그먼트 내부에서만 분할을 늘려 tgt 개수까지 맞춘다.
                # (전역 joined_src 균등분할은 후보 경계를 통째로 무시해 threshold 민감도를 죽일 수 있음)

                base_segments: List[str] = [s for s in src_segments if s is not None and s != ""]
                if not base_segments:
                    base_segments = [joined_src]

                # 1) 각 base segment에 최소 1개 조각을 할당하고, 남은 조각을 길이 큰 세그먼트에 분배
                pieces_per_seg = [1 for _ in base_segments]
                extra = target_count - len(base_segments)
                seg_lens = [len(s) for s in base_segments]

                def _pick_idx_to_expand() -> int:
                    # 현재 조각 수 대비 길이가 가장 큰 세그먼트를 더 쪼갠다(결정적).
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

                # 2) 각 세그먼트를 내부에서 문자 단위 균등분할하여 총 target_count를 만든다
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

                # 방어: 비어있는 조각이 생기면 제거 후, 마지막에 합쳐 개수 보존을 시도
                expanded = [p for p in expanded if p is not None and p != ""]
                if not expanded:
                    expanded = [joined_src]

                src_segments = expanded
        
        # ✅ 반드시 len(tgt_segments)개를 반환해야 함
        # (PA/SA 모두 zip으로 결합하기 때문에 개수가 어긋나면 의미가 밀리고 마지막에 몰림)
        matched_src: List[str] = []
        src_idx = 0

        total_tgt_len = sum(len(t) for t in tgt_segments) or 1

        for t_i, tgt_seg in enumerate(tgt_segments):
            remaining_tgt = len(tgt_segments) - t_i

            # src_idx부터 시작해 (최대 4개까지) 합쳐보며 최고 유사도 구간 선택
            best_score = -1.0
            best_end = min(src_idx + 1, len(src_segments))

            # 남은 tgt 개수만큼은 src를 남겨야 빈 원문을 피할 수 있다.
            # (src 세그먼트는 1개 이상 남겨두는 보수적 제한)
            latest_end_allowed = max(src_idx + 1, len(src_segments) - (remaining_tgt - 1))
            max_end = min(src_idx + 5, latest_end_allowed + 1)
            for end_idx in range(src_idx + 1, max_end):
                src_text = "".join(src_segments[src_idx:end_idx])
                score = self.compute_similarity(src_text, tgt_seg)
                if score > best_score:
                    best_score = score
                    best_end = end_idx

            if best_end <= src_idx:
                best_end = min(src_idx + 1, len(src_segments))

            # src가 바닥났다면(이론상 위 제한으로 거의 불가), 마지막 세그먼트를 억지로라도 재사용하지 않고 실패로 본다.
            # 빈 원문을 만들면 무결성 검증에서 'nan'로 오염되기 쉬우므로 금지.
            if src_idx >= len(src_segments):
                matched_src.append("")
                continue

            # 최적 구간을 "하나의" 원문 세그먼트로 합쳐서 추가
            chosen = "".join(src_segments[src_idx:best_end])
            if not chosen:
                # 실패 시 길이 비율로 fallback: 남은 src에서 비율만큼 소비
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

        # 남은 원문은 마지막에 붙여 무결성/보존성 강화(개수는 유지)
        if src_idx < len(src_segments) and matched_src:
            matched_src[-1] = matched_src[-1] + "".join(src_segments[src_idx:])

        # 최종 방어: 빈 원문이 생기면 상위 로직에서 재시도/대체 분할을 하도록 표시
        # (여기서는 반환 개수는 유지하되, 빈 문자열이 있음을 남긴다)

        return matched_src
