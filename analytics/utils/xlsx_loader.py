#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytics용 XLSX 로더 유틸리티
- xlsx_pipeline_results 폴더의 PA/SA 출력물을 읽어들여 표준화된 DataFrame을 제공합니다.
- 책 ID 자동 탐색 및 메타데이터 결합을 지원합니다.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from common.config import get_results_dir

DEFAULT_RESULTS_DIR = get_results_dir()

def list_books(results_dir: Optional[Path] = None) -> List[str]:
    """결과 폴더에서 이용 가능한 책 ID 목록을 반환"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        return []
    # *_pa_output.xlsx 또는 *_sa_output.xlsx 패턴에서 책 ID 추출
    book_ids = set()
    for f in results_dir.glob("*_pa_output.xlsx"):
        book_ids.add(f.name.replace("_pa_output.xlsx", ""))
    for f in results_dir.glob("*_sa_output.xlsx"):
        book_ids.add(f.name.replace("_sa_output.xlsx", ""))
    return sorted(book_ids)

def load_pa(book_id: str, results_dir: Optional[Path] = None) -> pd.DataFrame:
    """PA 출력 XLSX 로드"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    file = results_dir / f"{book_id}_pa_output.xlsx"
    if not file.exists():
        raise FileNotFoundError(f"PA 결과 파일이 없습니다: {file}")
    df = pd.read_excel(file)
    # 표준 컬럼 보정
    # 기대 컬럼: 문단식별자(optional), 원문, 번역문, similarity(optional), 기타 메타
    # 결측 컬럼 채우기
    for col in ["원문", "번역문"]:
        if col not in df.columns:
            df[col] = ""
    if "similarity" not in df.columns:
        df["similarity"] = 0.0
    return df

def load_sa(book_id: str, results_dir: Optional[Path] = None) -> pd.DataFrame:
    """SA 출력 XLSX 로드"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    file = results_dir / f"{book_id}_sa_output.xlsx"
    if not file.exists():
        raise FileNotFoundError(f"SA 결과 파일이 없습니다: {file}")
    df = pd.read_excel(file)
    for col in ["원문", "번역문"]:
        if col not in df.columns:
            df[col] = ""
    return df

def load_pa_sa(book_id: str, results_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """동시에 PA/SA 결과를 로드"""
    return load_pa(book_id, results_dir), load_sa(book_id, results_dir)

def concat_books_pa_sa(book_ids: Optional[List[str]] = None, results_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    여러 책의 PA/SA 결과를 합쳐서 반환
    Returns: {"pa": DataFrame, "sa": DataFrame}
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    if book_ids is None:
        book_ids = list_books(results_dir)
    pa_frames = []
    sa_frames = []
    for bid in book_ids:
        try:
            pa_df = load_pa(bid, results_dir)
            pa_df["book_id"] = bid
            pa_frames.append(pa_df)
        except FileNotFoundError:
            pass
        try:
            sa_df = load_sa(bid, results_dir)
            sa_df["book_id"] = bid
            sa_frames.append(sa_df)
        except FileNotFoundError:
            pass
    return {
        "pa": pd.concat(pa_frames, ignore_index=True) if pa_frames else pd.DataFrame(),
        "sa": pd.concat(sa_frames, ignore_index=True) if sa_frames else pd.DataFrame(),
    }
