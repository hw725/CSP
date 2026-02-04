"""서종 메타데이터 추출기

메타데이터는 book_metadata.json에서 로드됩니다.
코드에 하드코딩하지 않고 JSON 파일을 수정하여 메타데이터를 관리합니다.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# 메타데이터 JSON 파일 경로
METADATA_FILE = Path(__file__).parent / "book_metadata.json"

class BookMetadataExtractor:
    """서종 메타데이터 추출기

    사용법:
        extractor = BookMetadataExtractor()
        metadata = extractor.get_metadata("당송팔대가문초한유1")
        # {'author': '한유', 'period': '당대(768-824)', 'sibu': '集', 'translator': '한국고전번역원'}
    """

    def __init__(self, metadata_file: Path = METADATA_FILE):
        self.metadata_file = metadata_file
        self._metadata: Dict = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """JSON 파일에서 메타데이터 로드"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                logger.info(
                    f"메타데이터 로드 완료: {len(self._metadata.get('books', {}))}개 서종"
                )
            else:
                logger.warning(f"메타데이터 파일 없음: {self.metadata_file}")
                self._metadata = {"books": {}, "default_translator": "한국고전번역원"}
        except Exception as e:
            logger.error(f"메타데이터 로드 실패: {e}")
            self._metadata = {"books": {}, "default_translator": "한국고전번역원"}

    def reload(self) -> None:
        """메타데이터 다시 로드 (파일 변경 시)"""
        self._load_metadata()

    @property
    def books(self) -> Dict[str, Dict]:
        """등록된 모든 서종 메타데이터"""
        return self._metadata.get("books", {})

    @property
    def default_translator(self) -> str:
        return self._metadata.get("default_translator", "한국고전번역원")

    def get_metadata(self, book_name: str) -> Dict[str, str]:
        """서종명으로 메타데이터 조회

        Args:
            book_name: 서종명 (예: "당송팔대가문초한유1")

        Returns:
            메타데이터 딕셔너리 {author, period, sibu, translator}
        """
        # 정확히 일치하는 경우
        if book_name in self.books:
            meta = self.books[book_name].copy()
            meta.setdefault("translator", self.default_translator)
            return meta

        # 숫자 제거 후 매칭 시도 (예: "춘추좌씨전1" -> "춘추좌씨전")
        base_name = re.sub(r"\d+$", "", book_name)
        for key, meta in self.books.items():
            if re.sub(r"\d+$", "", key) == base_name:
                result = meta.copy()
                result.setdefault("translator", self.default_translator)
                return result

        # 부분 매칭 시도
        for key, meta in self.books.items():
            if key in book_name or book_name in key:
                result = meta.copy()
                result.setdefault("translator", self.default_translator)
                return result

        # 매칭 실패 시 기본값
        logger.warning(f"메타데이터 없음: {book_name}")
        return {
            "author": "미상",
            "period": "미상",
            "sibu": "未詳",
            "translator": self.default_translator,
        }

    def get_author(self, book_name: str) -> str:
        """저자 조회"""
        return self.get_metadata(book_name).get("author", "미상")

    def get_period(self, book_name: str) -> str:
        """시대 조회"""
        return self.get_metadata(book_name).get("period", "미상")

    def get_sibu(self, book_name: str) -> str:
        """사부 분류 조회"""
        return self.get_metadata(book_name).get("sibu", "未詳")

    def get_translator(self, book_name: str) -> str:
        """역자 조회"""
        return self.get_metadata(book_name).get("translator", self.default_translator)

    # === 하위 호환성을 위한 레거시 메서드 ===

    def extract_metadata(self, book_name: str) -> Dict[str, str]:
        """레거시 호환: get_metadata와 동일"""
        meta = self.get_metadata(book_name)
        return {
            "author": meta.get("author", "미상"),
            "translator": meta.get("translator", self.default_translator),
            "period": meta.get("period", "미상"),
            "sibu_classification": meta.get("sibu", "未詳"),
        }

    def extract_author_translator(self, book_name: str) -> tuple:
        """레거시 호환: (author, translator) 튜플 반환"""
        meta = self.get_metadata(book_name)
        return meta.get("author", "미상"), meta.get(
            "translator", self.default_translator
        )

    def get_detailed_author_info(self, book_name: str) -> Dict[str, str]:
        """레거시 호환: extract_metadata와 동일"""
        return self.extract_metadata(book_name)

# 싱글톤 인스턴스 (편의용)
_extractor_instance: Optional[BookMetadataExtractor] = None

def get_extractor() -> BookMetadataExtractor:
    """싱글톤 메타데이터 추출기 반환"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = BookMetadataExtractor()
    return _extractor_instance

if __name__ == "__main__":
    # 테스트
    extractor = BookMetadataExtractor()

    test_books = ["당송팔대가문초한유1", "춘추좌씨전3", "예기집설대전1", "unknown_book"]

    for book in test_books:
        meta = extractor.get_metadata(book)
        print(f"{book}: {meta}")
