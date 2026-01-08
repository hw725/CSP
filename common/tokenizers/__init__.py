"""
공통 토크나이저 모듈
하이브리드 토크나이저 통합 인터페이스
- 중국어: SikuBERT (GPU 가속)
- 한국어: RoBERTa-Korean-Hanja (한자) + Kiwipiepy (한글)
"""

# 중국 고전 전용 모델들
from .siku_tokenizer import SikuBertTokenizer, get_siku_tokenizer, siku_get_embeddings, siku_similarity

# 한국어 전용 모델들
from .roberta_hanja_tokenizer import RobertaKoreanHanjaTokenizer, get_roberta_hanja_tokenizer
from .kiwi_tokenizer import KiwipieTokenizer, get_kiwi_tokenizer
from .hybrid_korean_tokenizer import HybridKoreanTokenizer, get_hybrid_korean_tokenizer, hybrid_tokenize_korean

# 유틸리티
try:
    from .period_detector import PeriodDetector, detect_chinese_period
except ImportError:
    # PeriodDetector 대안 구현
    class PeriodDetector:
        def detect_period(self, text):
            return '.' in text or '。' in text or '!' in text or '？' in text
    
    def detect_chinese_period(text):
        return '。' in text or '！' in text or '？' in text

__all__ = [
    # 중국 고전 전용
    'SikuBertTokenizer',
    'get_siku_tokenizer', 
    'siku_get_embeddings',
    'siku_similarity',
    
    # 한국어 하이브리드
    'HybridKoreanTokenizer',
    'get_hybrid_korean_tokenizer',
    'hybrid_tokenize_korean',
    'RobertaKoreanHanjaTokenizer',
    'get_roberta_hanja_tokenizer', 
    'KiwipieTokenizer',
    'get_kiwi_tokenizer',
    
    # 유틸리티
    'PeriodDetector',
    'detect_chinese_period',
]
