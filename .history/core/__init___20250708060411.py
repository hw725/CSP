"""Core 모듈 - 공통 기능 및 유틸리티"""

# 로깅 설정
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 주요 모듈들 import
try:
    from .io_utils import IOManager
except ImportError as e:
    logging.warning(f"io_utils 모듈 로드 실패: {e}")

try:
    from .tokenizers import get_tokenizer, BaseTokenizer, SimpleTokenizer, ChineseTokenizer, KoreanTokenizer
except ImportError as e:
    logging.warning(f"tokenizers 모듈 로드 실패: {e}")

try:
    from .embedders import get_embedder, BaseEmbedder, BGEEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder
except ImportError as e:
    logging.warning(f"embedders 모듈 로드 실패: {e}")

try:
    from .sentence_splitter import (
        split_target_sentences_advanced,
        split_with_spacy,
        split_source_by_whitespace_and_align,
        split_source_by_meaning_units
    )
except ImportError as e:
    logging.warning(f"sentence_splitter 모듈 로드 실패: {e}")

__version__ = "0.1.0"
__all__ = [
    "IOManager",
    "get_tokenizer", "BaseTokenizer", "SimpleTokenizer", "ChineseTokenizer", "KoreanTokenizer",
    "get_embedder", "BaseEmbedder", "BGEEmbedder", "SentenceTransformerEmbedder", "OpenAIEmbedder",
    "split_target_sentences_advanced", "split_with_spacy", 
    "split_source_by_whitespace_and_align", "split_source_by_meaning_units"
]