"""
RoBERTa Korean-Hanja 토크나이저 모듈
한국어 텍스트의 한자 부분 처리 전용
"""

import logging
from typing import List, Optional, Dict, Any, Union
import re
import torch

logger = logging.getLogger(__name__)

class RobertaKoreanHanjaTokenizer:
    """RoBERTa Korean-Hanja 토크나이저 래퍼 클래스"""

    def __init__(self, model_name: str = "hwp0725/roberta-korean-hanja-stdict-mlm", device: str = "auto"):
        """
        Args:
            model_name: 사용할 모델명 (기본값은 roberta-korean-hanja-stdict-mlm)
            device: 사용할 디바이스 ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self._initialized = False

        # 한자 패턴 정의
        self.hanja_pattern = re.compile(r"[\u4e00-\u9fff]+")

    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize(self):
        """지연 초기화"""
        if self._initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast

            # 토크나이저와 모델 로드
            # hwp0725/roberta-korean-hanja-stdict-mlm의 tokenizer_config.json에
            # tokenizer_class 오류가 있어 AutoTokenizer가 실패할 수 있으므로
            # PreTrainedTokenizerFast → AutoTokenizer 순서로 시도
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except (ValueError, OSError):
                logger.info(f"AutoTokenizer 실패, PreTrainedTokenizerFast로 재시도: {self.model_name}")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # GPU로 이동
            if self.device == "cuda":
                self.model = self.model.to(self.device)

            logger.info(
                f"RoBERTa Korean-Hanja 모델 로드 완료: {self.model_name} on {self.device}"
            )
            self._initialized = True

        except ImportError as e:
            logger.error(f"Transformers 라이브러리 설치되지 않음: {e}")
            raise
        except Exception as e:
            logger.error(f"RoBERTa Korean-Hanja 모델 초기화 실패: {e}")
            raise

    def extract_hanja_parts(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 한자 부분 추출"""
        hanja_parts = []

        for match in self.hanja_pattern.finditer(text):
            hanja_parts.append(
                {"text": match.group(), "start": match.start(), "end": match.end()}
            )

        return hanja_parts

    def tokenize_hanja(self, hanja_text: str) -> List[str]:
        """한자 텍스트 토큰화"""
        self._initialize()

        try:
            # RoBERTa 토크나이저로 한자 처리
            tokens = self.tokenizer.tokenize(hanja_text)

            # 특수 토큰 제거
            cleaned_tokens = []
            for token in tokens:
                if not token.startswith("[") and not token.startswith("<"):
                    # 서브워드 토큰 처리 (## 제거)
                    if token.startswith("##"):
                        token = token[2:]
                    cleaned_tokens.append(token)

            return cleaned_tokens

        except Exception as e:
            logger.error(f"RoBERTa 한자 토큰화 실패: {e}")
            return list(hanja_text)  # 폴백: 문자 단위 분할

    def tokenize_text_with_hanja_priority(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트를 한자 우선으로 토큰화
        한자 부분은 RoBERTa로, 나머지는 별도 처리를 위해 표시
        """
        self._initialize()

        result = []
        hanja_parts = self.extract_hanja_parts(text)

        last_end = 0
        for hanja_part in hanja_parts:
            # 한자 이전 부분 (한글 등)
            if hanja_part["start"] > last_end:
                hangul_text = text[last_end : hanja_part["start"]]
                if hangul_text.strip():
                    result.append(
                        {
                            "text": hangul_text,
                            "type": "hangul",
                            "tokens": None,  # 별도 토크나이저에서 처리
                            "start": last_end,
                            "end": hanja_part["start"],
                        }
                    )

            # 한자 부분
            hanja_tokens = self.tokenize_hanja(hanja_part["text"])
            result.append(
                {
                    "text": hanja_part["text"],
                    "type": "hanja",
                    "tokens": hanja_tokens,
                    "start": hanja_part["start"],
                    "end": hanja_part["end"],
                }
            )

            last_end = hanja_part["end"]

        # 마지막 한글 부분
        if last_end < len(text):
            hangul_text = text[last_end:]
            if hangul_text.strip():
                result.append(
                    {
                        "text": hangul_text,
                        "type": "hangul",
                        "tokens": None,
                        "start": last_end,
                        "end": len(text),
                    }
                )

        return result

    def get_embeddings(self, text: str) -> torch.Tensor:
        """텍스트 임베딩 생성"""
        self._initialize()

        try:
            # 토큰화
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**inputs)
                # [CLS] 토큰의 임베딩 사용
                embeddings = outputs.last_hidden_state[:, 0, :]

            return embeddings.cpu()

        except Exception as e:
            logger.error(f"RoBERTa 임베딩 생성 실패: {e}")
            return torch.zeros(1, 768)  # 기본 차원

# 전역 인스턴스
_roberta_hanja_tokenizers = {}

def get_roberta_hanja_tokenizer(
    model_name: str = "hwp0725/roberta-korean-hanja-stdict-mlm", device: str = "auto"
) -> RobertaKoreanHanjaTokenizer:
    """RoBERTa Korean-Hanja 토크나이저 싱글톤 인스턴스 반환"""
    global _roberta_hanja_tokenizers

    key = f"{model_name}_{device}"
    if key not in _roberta_hanja_tokenizers:
        _roberta_hanja_tokenizers[key] = RobertaKoreanHanjaTokenizer(
            model_name=model_name, device=device
        )

    return _roberta_hanja_tokenizers[key]

def roberta_extract_hanja(
    text: str, model_name: str = "hwp0725/roberta-korean-hanja-stdict-mlm"
) -> List[Dict[str, Any]]:
    """RoBERTa로 한자 부분 추출 및 토큰화 편의 함수"""
    tokenizer = get_roberta_hanja_tokenizer(model_name=model_name)
    return tokenizer.extract_hanja_parts(text)

def roberta_tokenize_hanja(
    hanja_text: str, model_name: str = "hwp0725/roberta-korean-hanja-stdict-mlm"
) -> List[str]:
    """RoBERTa 한자 토큰화 편의 함수"""
    tokenizer = get_roberta_hanja_tokenizer(model_name=model_name)
    return tokenizer.tokenize_hanja(hanja_text)
