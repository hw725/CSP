"""S2P 경계 태거 로더

학습된 s2p_boundary_tagger.pt를 로드하여 번역문 경계 추론
"""

from pathlib import Path
from typing import List, Dict
import torch
from torch import nn

class SaBoundaryTagger(nn.Module):
    """BiLSTM 기반 경계 태거 (학습 스크립트와 동일 구조)"""

    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.emb(x))
        return self.fc(h).squeeze(-1)

class SaBoundaryTaggerLoader:
    """S2P 경계 태거 로더"""

    def __init__(self, model_path: Path = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        candidates = []
        if model_path is None:
            candidates = [
                Path(__file__).parent.parent / "models" / "s2p_boundary_tagger.pt",
                Path(__file__).parent.parent / "models" / "sa_boundary_tagger.pt",
            ]
            model_path = next((p for p in candidates if p.exists()), candidates[0])

        if not model_path.exists():
            hint = ", ".join(str(p) for p in candidates) if candidates else str(model_path)
            raise FileNotFoundError(f"S2P 경계 모델 없음: {model_path} (expected one of: {hint})")

        # 모델 로드
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        self.vocab = checkpoint["vocab"]
        self.max_len = checkpoint.get("max_len", 512)
        self.test_scores = checkpoint.get("test_scores", {})

        # 모델 초기화 및 로드
        self.model = SaBoundaryTagger(len(self.vocab) + 1).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        print(
            f"✅ S2P 경계 태거 로드 완료 (vocab={len(self.vocab)}, device={self.device})"
        )

    def encode_text(self, text: str) -> torch.Tensor:
        """텍스트를 토큰 ID로 인코딩"""
        ids = [self.vocab.get(ch, 0) for ch in text][: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def segment_text(
        self, text: str, threshold: float = 0.5, task: str = None, **kwargs
    ) -> List[str]:
        """텍스트를 경계 위치에서 분할

        Args:
            text: 입력 텍스트 (번역문)
            threshold: 경계 확률 임계값
            task: 무시됨 (호환성용)

        Returns:
            분할된 세그먼트 리스트
        """
        if not text.strip():
            return [text] if text else []

        with torch.no_grad():
            x = self.encode_text(text)
            logits = self.model(x)[0][: len(text)]  # 실제 텍스트 길이만큼만
            probs = torch.sigmoid(logits).cpu().numpy()

        # 경계 위치 찾기 (B 태그)
        segments = []
        start = 0

        for i, prob in enumerate(probs):
            if prob >= threshold and i > start:
                segments.append(text[start:i])
                start = i

        # 마지막 세그먼트
        if start < len(text):
            segments.append(text[start:])

        # 빈 세그먼트 제거
        segments = [s for s in segments if s.strip()]

        return segments if segments else [text]

    def predict_boundary_probs(self, text: str) -> List[float]:
        """각 문자 위치의 경계 확률 반환"""
        if not text:
            return []

        with torch.no_grad():
            x = self.encode_text(text)
            logits = self.model(x)[0][: len(text)]
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs.tolist()

# 전역 인스턴스 캐싱
_sa_tagger_instance = None

def get_sa_boundary_tagger(
    model_path: Path = None, device: str = "cuda"
) -> SaBoundaryTaggerLoader:
    """S2P 경계 태거 싱글톤 반환"""
    global _sa_tagger_instance

    if _sa_tagger_instance is None:
        _sa_tagger_instance = SaBoundaryTaggerLoader(
            model_path=model_path, device=device
        )

    return _sa_tagger_instance
