#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XLSX 파이프라인 프로세서
Excel 파일(구병렬, 문장병렬, 문단병렬)을 입력으로 사용하는 처리 시스템

Excel 파일 구조:
- 구병렬: 문장식별자, 구식별자, 원문, 번역문
- 문장병렬: 문단식별자, 문장식별자, 원문, 번역문
- 문단병렬: 문단식별자, 원문, 번역문
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging
import json
import sys

# 통합 분석기 import
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from xlsx_pipeline.xlsx_integrated_analyzer import XLSXIntegratedAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from xlsx_integrated_analyzer import XLSXIntegratedAnalyzer
        ANALYZER_AVAILABLE = True
    except ImportError:
        ANALYZER_AVAILABLE = False
        XLSXIntegratedAnalyzer = None

logger = logging.getLogger(__name__)

def normalize_text(text: Any) -> Any:
    """
    텍스트 정규화: 내부 개행 제거 및 중복 공백 정리
    
    Args:
        text: 입력 텍스트 (문자열 또는 기타 타입)
    
    Returns:
        정규화된 텍스트 (문자열이 아닌 경우 원본 반환)
    """
    if not isinstance(text, str):
        return text
    
    # 내부 개행 제거 (\n, \r, \r\n)
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # 탭을 공백으로 변환
    text = text.replace('\t', ' ')
    
    # 중복 공백을 단일 공백으로
    text = ' '.join(text.split())
    
    return text

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 모든 텍스트 열 정규화
    
    Args:
        df: 입력 DataFrame
    
    Returns:
        정규화된 DataFrame
    """
    df_normalized = df.copy()
    
    for col in df_normalized.columns:
        if df_normalized[col].dtype == 'object':  # 문자열 열만 처리
            df_normalized[col] = df_normalized[col].apply(normalize_text)
    
    return df_normalized

class XLSXBook:
    """Excel 기반 책 데이터 클래스"""
    
    def __init__(self, book_id: str, book_name: str, book_dir: Path):
        """
        Args:
            book_id: 책 식별자 (예: "당송팔대가문초구양수1")
            book_name: 책 이름 (선택적, 기본값은 book_id)
            book_dir: 책 디렉토리 경로 (xlsx/{책이름}/)
        """
        self.book_id = book_id
        self.book_name = book_name or book_id
        self.book_dir = Path(book_dir)
        
        # Excel 파일 경로 설정
        self.word_parallel_path = self.book_dir / f"{book_id}_구병렬.xlsx"
        self.sentence_parallel_path = self.book_dir / f"{book_id}_문장병렬.xlsx"
        self.paragraph_parallel_path = self.book_dir / f"{book_id}_문단병렬.xlsx"
        
        # 데이터프레임 캐시
        self._word_df = None
        self._sentence_df = None
        self._paragraph_df = None
    
    def exists(self) -> Dict[str, bool]:
        """각 Excel 파일 존재 여부 확인"""
        return {
            "word": self.word_parallel_path.exists(),
            "sentence": self.sentence_parallel_path.exists(),
            "paragraph": self.paragraph_parallel_path.exists()
        }
    
    def load_word_parallel(self, force_reload: bool = False) -> pd.DataFrame:
        """구병렬 데이터 로드 (정규화 포함)"""
        if self._word_df is None or force_reload:
            if not self.word_parallel_path.exists():
                raise FileNotFoundError(f"구병렬 파일을 찾을 수 없습니다: {self.word_parallel_path}")
            
            df = pd.read_excel(self.word_parallel_path)
            self._word_df = normalize_dataframe(df)
            logger.info(f"✅ 구병렬 로드 완료: {len(self._word_df)}행 (정규화 적용)")
        
        return self._word_df
    
    def load_sentence_parallel(self, force_reload: bool = False) -> pd.DataFrame:
        """문장병렬 데이터 로드 (정규화 포함)"""
        if self._sentence_df is None or force_reload:
            if not self.sentence_parallel_path.exists():
                raise FileNotFoundError(f"문장병렬 파일을 찾을 수 없습니다: {self.sentence_parallel_path}")
            
            df = pd.read_excel(self.sentence_parallel_path)
            self._sentence_df = normalize_dataframe(df)
            logger.info(f"✅ 문장병렬 로드 완료: {len(self._sentence_df)}행 (정규화 적용)")
        
        return self._sentence_df
    
    def load_paragraph_parallel(self, force_reload: bool = False) -> pd.DataFrame:
        """문단병렬 데이터 로드 (정규화 포함)"""
        if self._paragraph_df is None or force_reload:
            if not self.paragraph_parallel_path.exists():
                raise FileNotFoundError(f"문단병렬 파일을 찾을 수 없습니다: {self.paragraph_parallel_path}")
            
            df = pd.read_excel(self.paragraph_parallel_path)
            self._paragraph_df = normalize_dataframe(df)
            logger.info(f"✅ 문단병렬 로드 완료: {len(self._paragraph_df)}행 (정규화 적용)")
        
        return self._paragraph_df
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = {
            "book_id": self.book_id,
            "book_name": self.book_name,
            "files_exist": self.exists()
        }
        
        try:
            word_df = self.load_word_parallel()
            stats["word_count"] = len(word_df)
            stats["word_nan_count"] = word_df.isna().sum().to_dict()
        except Exception as e:
            stats["word_error"] = str(e)
        
        try:
            sent_df = self.load_sentence_parallel()
            stats["sentence_count"] = len(sent_df)
            stats["sentence_nan_count"] = sent_df.isna().sum().to_dict()
        except Exception as e:
            stats["sentence_error"] = str(e)
        
        try:
            para_df = self.load_paragraph_parallel()
            stats["paragraph_count"] = len(para_df)
            stats["paragraph_nan_count"] = para_df.isna().sum().to_dict()
        except Exception as e:
            stats["paragraph_error"] = str(e)
        
        return stats


class XLSXPipelineProcessor:
    """XLSX 파이프라인 프로세서"""
    
    def __init__(self, xlsx_root_dir: str = "xlsx", output_dir: str = "xlsx_pipeline_results"):
        """
        Args:
            xlsx_root_dir: Excel 파일들이 있는 루트 디렉토리
            output_dir: 결과 파일 저장 디렉토리
        """
        self.xlsx_root_dir = Path(xlsx_root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.books: Dict[str, XLSXBook] = {}
        
        # 통합 분석기 초기화
        if ANALYZER_AVAILABLE:
            self.analyzer = XLSXIntegratedAnalyzer(str(self.output_dir))
            logger.info(f"✅ 통합 분석기 로드됨 (PA/SA/Accuracy)")
        else:
            self.analyzer = None
            logger.warning(f"⚠️ 통합 분석기를 사용할 수 없습니다")
        
        logger.info(f"📂 XLSX 루트 디렉토리: {self.xlsx_root_dir}")
        logger.info(f"📂 결과 출력 디렉토리: {self.output_dir}")
    
    def discover_books(self) -> List[str]:
        """xlsx_root_dir에서 모든 책 디렉토리 자동 발견"""
        if not self.xlsx_root_dir.exists():
            logger.error(f"❌ XLSX 루트 디렉토리를 찾을 수 없습니다: {self.xlsx_root_dir}")
            return []
        
        discovered = []
        
        for item in self.xlsx_root_dir.iterdir():
            if item.is_dir():
                # 구병렬 파일이 있는지 확인
                word_file = item / f"{item.name}_구병렬.xlsx"
                if word_file.exists():
                    book_id = item.name
                    self.books[book_id] = XLSXBook(book_id, book_id, item)
                    discovered.append(book_id)
                    logger.info(f"✅ 책 발견: {book_id}")
        
        logger.info(f"📚 총 {len(discovered)}개 책 발견됨")
        return discovered
    
    def add_book(self, book_id: str, book_dir: Optional[Path] = None):
        """책을 수동으로 추가"""
        if book_dir is None:
            book_dir = self.xlsx_root_dir / book_id
        
        self.books[book_id] = XLSXBook(book_id, book_id, book_dir)
        logger.info(f"✅ 책 추가: {book_id}")
    
    def get_book(self, book_id: str) -> Optional[XLSXBook]:
        """책 객체 가져오기"""
        return self.books.get(book_id)
    
    def list_books(self) -> List[Dict[str, Any]]:
        """모든 책 목록 및 정보 반환"""
        book_list = []
        
        for book_id, book in self.books.items():
            info = {
                "book_id": book_id,
                "book_name": book.book_name,
                "exists": book.exists()
            }
            book_list.append(info)
        
        return book_list
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """모든 책의 통계 정보 수집"""
        all_stats = {
            "total_books": len(self.books),
            "timestamp": datetime.now().isoformat(),
            "books": {}
        }
        
        for book_id, book in self.books.items():
            try:
                all_stats["books"][book_id] = book.get_statistics()
            except Exception as e:
                all_stats["books"][book_id] = {
                    "book_id": book_id,
                    "error": str(e)
                }
        
        return all_stats
    
    def save_statistics(self, output_path: Optional[Path] = None):
        """통계 정보를 JSON 파일로 저장"""
        if output_path is None:
            output_path = self.output_dir / f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats = self.get_all_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 통계 정보 저장: {output_path}")
        return output_path
    
    def process_book_pipeline(self, book_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        특정 책에 대한 파이프라인 실행
        
        Args:
            book_id: 책 식별자
            config: 파이프라인 설정
                - levels: 처리할 레벨 리스트 ['word', 'sentence', 'paragraph']
                - analysis: 수행할 분석 리스트 ['statistics', 'pa', 'sa', 'accuracy']
                - run_pa: PA 분석 실행 여부
                - run_sa: SA 분석 실행 여부
                - run_accuracy: 정확도 평가 실행 여부
                - project: 프로젝트 이름 (정확도 평가용)
        
        Returns:
            파이프라인 실행 결과
        """
        config = config or {}
        levels = config.get('levels', ['word', 'sentence', 'paragraph'])
        analysis = config.get('analysis', ['statistics'])
        
        book = self.get_book(book_id)
        if book is None:
            raise ValueError(f"책을 찾을 수 없습니다: {book_id}")
        
        results = {
            "book_id": book_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "data": {},
            "analysis": {}
        }
        
        # 데이터 로드
        logger.info(f"🚀 파이프라인 시작: {book_id}")
        
        word_df = None
        sent_df = None
        para_df = None
        
        if 'word' in levels:
            try:
                word_df = book.load_word_parallel()
                results["data"]["word"] = {
                    "rows": len(word_df),
                    "columns": list(word_df.columns)
                }
            except Exception as e:
                results["data"]["word"] = {"error": str(e)}
        
        if 'sentence' in levels:
            try:
                sent_df = book.load_sentence_parallel()
                results["data"]["sentence"] = {
                    "rows": len(sent_df),
                    "columns": list(sent_df.columns)
                }
            except Exception as e:
                results["data"]["sentence"] = {"error": str(e)}
        
        if 'paragraph' in levels:
            try:
                para_df = book.load_paragraph_parallel()
                results["data"]["paragraph"] = {
                    "rows": len(para_df),
                    "columns": list(para_df.columns)
                }
            except Exception as e:
                results["data"]["paragraph"] = {"error": str(e)}
        
        # 기본 통계 분석
        if 'statistics' in analysis:
            results["analysis"]["statistics"] = book.get_statistics()
        
        # PA/SA/Accuracy 통합 분석
        if self.analyzer and any(a in analysis for a in ['pa', 'sa', 'accuracy', 'full']):
            try:
                # full 분석 실행
                if 'full' in analysis:
                    if word_df is not None and sent_df is not None and para_df is not None:
                        full_results = self.analyzer.run_full_pipeline(
                            word_df, sent_df, para_df, book_id, config
                        )
                        results["analysis"]["full_pipeline"] = full_results
                else:
                    # 개별 분석 실행
                    if 'pa' in analysis and para_df is not None:
                        pa_result = self.analyzer.run_pa_analysis(para_df, book_id)
                        results["analysis"]["pa"] = pa_result
                    
                    if 'sa' in analysis and sent_df is not None:
                        sa_result = self.analyzer.run_sa_analysis(sent_df, book_id)
                        results["analysis"]["sa"] = sa_result
                    
                    if 'accuracy' in analysis:
                        logger.warning("⚠️ accuracy 분석은 'full' 모드에서만 실행됩니다")
            
            except Exception as e:
                logger.error(f"❌ 통합 분석 실패: {e}")
                results["analysis"]["error"] = str(e)
        
        logger.info(f"✅ 파이프라인 완료: {book_id}")
        
        return results
    
    def batch_process(self, book_ids: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        여러 책에 대한 배치 처리
        
        Args:
            book_ids: 처리할 책 ID 리스트 (None이면 전체)
            config: 파이프라인 설정
        
        Returns:
            배치 처리 결과
        """
        if book_ids is None:
            book_ids = list(self.books.keys())
        
        batch_results = {
            "total_books": len(book_ids),
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": {}
        }
        
        for book_id in book_ids:
            logger.info(f"📖 처리 중: {book_id}")
            try:
                result = self.process_book_pipeline(book_id, config)
                batch_results["results"][book_id] = result
            except Exception as e:
                logger.error(f"❌ 처리 실패: {book_id} - {e}")
                batch_results["results"][book_id] = {
                    "book_id": book_id,
                    "error": str(e)
                }
        
        return batch_results


def create_books_from_directory(xlsx_root_dir: str = "xlsx") -> List[XLSXBook]:
    """디렉토리에서 모든 책 자동 생성"""
    processor = XLSXPipelineProcessor(xlsx_root_dir)
    discovered = processor.discover_books()
    
    return [processor.get_book(book_id) for book_id in discovered]


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 프로세서 생성 및 테스트
    processor = XLSXPipelineProcessor()
    
    # 책 자동 발견
    discovered = processor.discover_books()
    
    if discovered:
        print(f"\n📚 발견된 책: {len(discovered)}개")
        
        # 통계 정보 저장
        stats_file = processor.save_statistics()
        print(f"💾 통계 파일 저장: {stats_file}")
        
        # 첫 번째 책 파이프라인 테스트
        first_book_id = discovered[0]
        print(f"\n🧪 테스트 실행: {first_book_id}")
        result = processor.process_book_pipeline(first_book_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("❌ 책을 찾을 수 없습니다.")
