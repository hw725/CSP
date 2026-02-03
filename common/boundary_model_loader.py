"""
Boundary Multitask 모델 로더 및 추론 엔진
- boundary_multitask.pt로부터 문단/문장/구 경계 태깅
"""

from pathlib import Path
from typing import List, Dict
import torch
from torch import nn
import json
import math


class CharEncoderForBoundary(nn.Module):
    """Boundary 태깅용 문자 인코더"""
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return h


class MultiHeadBoundary(nn.Module):
    """멀티태스크 경계 태깅 모델"""
    def __init__(self, vocab_size: int, tasks: List[str]):
        super().__init__()
        self.encoder = CharEncoderForBoundary(vocab_size)
        hidden_dim = 128
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_dim * 2, 1) for t in tasks})

    def forward(self, x: torch.Tensor, task: str):
        h = self.encoder(x)
        return self.heads[task](h).squeeze(-1)


class BoundaryModelLoader:
    """Boundary Multitask 모델 로더"""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 모델 파일 없음: {self.model_path}")
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 체크포인트 형식에 따라 vocab 로드
        if "vocab" in checkpoint:
            # 기존 형식
            self.vocab: Dict[str, int] = checkpoint["vocab"]
        elif "src_vocab" in checkpoint:
            # 새 형식 (src_vocab, tgt_vocab)
            # 둘을 병합하여 vocab으로 사용
            self.vocab: Dict[str, int] = {}
            if isinstance(checkpoint["src_vocab"], dict):
                self.vocab.update(checkpoint["src_vocab"])
            if isinstance(checkpoint.get("tgt_vocab"), dict):
                self.vocab.update(checkpoint["tgt_vocab"])
            if not self.vocab:
                # 만약 둘 다 dict가 아니면 임시 vocab 생성
                self.vocab = {chr(i): i+1 for i in range(256)}
        else:
            # 폴백: 임시 vocab
            self.vocab = {chr(i): i+1 for i in range(256)}
        
        self.max_len: int = checkpoint.get("max_len", 1024)
        tasks: List[str] = checkpoint.get("tasks", ["pa", "sa", "pd"])
        
        # 모델 초기화
        self.model = MultiHeadBoundary(vocab_size=len(self.vocab) + 1, tasks=tasks).to(self.device)

        # state_dict 키 호환성 처리 (훈련 시점에 encoder 없이 저장된 체크포인트 지원)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # 훈련 스크립트(train_boundary_multitask.py)는 emb.*, lstm.*를 최상위로 저장
        # 현재 추론 모델은 encoder.emb.*, encoder.lstm.* 구조를 사용하므로 키를 재매핑
        needs_remap = any(k.startswith("emb.") or k.startswith("lstm.") for k in state_dict.keys())
        if needs_remap:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("emb.") or k.startswith("lstm."):
                    remapped[f"encoder.{k}"] = v
                else:
                    remapped[k] = v
            state_dict = remapped
            print("🔧 Boundary 체크포인트 키 재매핑 수행 (emb./lstm. → encoder.*)")

        # 엄격 매칭을 완화하여 호환성 확보
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        print(f"✅ Boundary 모델 로드: {self.model_path}")
        print(f"   vocab={len(self.vocab)}, max_len={self.max_len}, tasks={tasks}")

    def segment_text(self, text: str, task: str = "pa", threshold: float = 0.5, min_len_override: int = None) -> List[str]:
        """
        텍스트를 경계 위치에서 분할
        
        Args:
            text: 입력 텍스트
            task: "pa" (문단→문장) 또는 "sa" (문장→구)
            threshold: 경계 확률 임계값 (0.5)
        
        Returns:
            분할된 세그먼트 리스트
        """
        if not text:
            return []
        
        # 텍스트를 tensor로 변환
        original_length = len(text)
        ids = [self.vocab.get(ch, 0) for ch in text][:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        
        # 추론
        with torch.no_grad():
            logits = self.model(x, task)[0].detach().cpu()

        # ⚠️ 중요: sigmoid 확률은 큰 logit에서 1.0으로 포화되어
        # threshold(예: 0.9~1.0)가 사실상 동일하게 동작할 수 있다.
        # 따라서 threshold를 logit 공간으로 변환하여 raw logits에서 비교한다.
        # - threshold=0.5 → logit 0
        # - threshold→1.0 → +inf
        # - threshold→0.0 → -inf
        if threshold <= 0.0:
            logit_thr = -float('inf')
        elif threshold >= 1.0:
            logit_thr = float('inf')
        else:
            logit_thr = math.log(threshold / (1.0 - threshold))
        
        # 경계 위치 찾기 (실제 텍스트 길이만큼만 확인!)
        # boundary는 "이 위치 직후에 분할" (즉, 다음 segment의 시작점)
        #
        # ⚠️ 주의: 태깅 모델이 공백/연속 구간에서 높은 점수를 주면
        # 경계가 과도하게 촘촘해져 단어 단위로 쪼개질 수 있다.
        # 이를 방지하기 위해 다음의 일반적인 디코딩(후처리)을 적용한다.
        # - threshold 이상인 연속(인접) 후보는 하나의 그룹으로 묶고 최고점(peak)만 남김
        # - task별 최소 세그먼트 길이(min_len)를 강제하여 경계 폭증 방지

        # logit_thr 이상인 위치를 후보로 수집. 점수는 후속 peak 선택을 위해 sigmoid(prob)로 보관.
        # (여기서만 sigmoid를 쓰되, 비교는 logits로 수행)
        probs = torch.sigmoid(logits[:original_length]).tolist()
        candidates = [(i, p) for i, p in enumerate(probs) if logits[i].item() >= logit_thr]

        # 후보가 없으면 전체 텍스트 반환
        if not candidates:
            return [text] if text else []

        # 0 위치는 분할점이 될 수 없음
        candidates = [(i, p) for i, p in candidates if i > 0]
        if not candidates:
            return [text] if text else []

        candidates.sort(key=lambda x: x[0])

        # 인접 후보는 하나로 묶어 peak만 유지
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

        # task별 최소 세그먼트 길이
        if task == "pa":
            min_len = 20
        elif task == "sa":
            min_len = 6
        else:
            min_len = 12

        # 필요 시 호출부에서 강제 오버라이드 (실험/튜닝용)
        if min_len_override is not None:
            try:
                v = int(min_len_override)
                if v >= 1:
                    min_len = v
            except Exception:
                pass

        filtered_peaks = []
        for i, p in grouped_peaks:
            if not filtered_peaks:
                filtered_peaks.append((i, p))
                continue
            prev_i, prev_p = filtered_peaks[-1]
            if (i - prev_i) < min_len:
                # 너무 가까우면 더 높은 점수의 경계를 유지
                if p > prev_p:
                    filtered_peaks[-1] = (i, p)
            else:
                filtered_peaks.append((i, p))

        boundaries = [i for i, _ in filtered_peaks]

        # 디코딩 결과 경계가 없으면 전체 텍스트 반환
        if not boundaries:
            return [text] if text else []
        
        # 세그먼트 생성
        # boundaries[i]는 i번째 문자 **다음**에서 분할
        # 예: text="ABCDEF", boundaries=[3] → ["ABC", "DEF"]
        segments = []
        start = 0
        for boundary_pos in boundaries:
            segments.append(text[start:boundary_pos])
            start = boundary_pos
        # 마지막 세그먼트 (항상 추가)
        if start < original_length:
            segments.append(text[start:])
        
        # 🔧 고립된 구두점 병합: 1자 segment가 구두점이면 이전 segment에 병합
        punctuation = set('.,!?;:\'"。、，！？；：""''…—·)]}）〉》」』】〕〗〙〛〉')
        merged = []
        for seg in segments:
            if seg and len(seg) == 1 and seg in punctuation:
                if merged:
                    merged[-1] += seg
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
        
        return merged if merged else segments

    def predict_boundary_logits(self, text: str, task: str = "pa") -> List[float]:
        """텍스트의 각 문자 위치에 대한 boundary logit을 반환한다.

        - 반환 길이는 원본 텍스트 길이와 동일
        - logit은 '해당 위치 직후에 경계가 존재'할 점수(학습 정의와 동일)

        NOTE: threshold 비교/디코딩 없이 raw score가 필요할 때 사용.
        """

        if not text:
            return []

        original_length = len(text)
        ids = [self.vocab.get(ch, 0) for ch in text][: self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))

        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(x, task)[0].detach().cpu()
        return logits[:original_length].tolist()

    def predict_boundary_probs(self, text: str, task: str = "pa") -> List[float]:
        """텍스트의 각 문자 위치에 대한 boundary 확률(sigmoid(logit))을 반환한다."""

        logits = self.predict_boundary_logits(text, task=task)
        if not logits:
            return []
        # torch 없이도 안정적으로
        out: List[float] = []
        for x in logits:
            # sigmoid
            if x >= 0:
                z = math.exp(-x)
                out.append(1.0 / (1.0 + z))
            else:
                z = math.exp(x)
                out.append(z / (1.0 + z))
        return out

    def segment_paragraphs_to_sentences(self, paragraphs: List[str], threshold: float = 0.5) -> List[str]:
        """
        문단 리스트를 문장으로 분할
        
        Args:
            paragraphs: 문단 텍스트 리스트
            threshold: 경계 확률 임계값
        
        Returns:
            분할된 문장 리스트
        """
        sentences = []
        for para in paragraphs:
            segs = self.segment_text(para, task="pa", threshold=threshold)
            sentences.extend(segs)
        return sentences

    def segment_sentences_to_phrases(self, sentences: List[str], threshold: float = 0.5) -> List[str]:
        """
        문장 리스트를 구로 분할
        
        Args:
            sentences: 문장 텍스트 리스트
            threshold: 경계 확률 임계값
        
        Returns:
            분할된 구 리스트
        """
        phrases = []
        for sent in sentences:
            segs = self.segment_text(sent, task="sa", threshold=threshold)
            phrases.extend(segs)
        return phrases
