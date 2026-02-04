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
    # *_p2s_output.xlsx 또는 *_s2p_output.xlsx 패턴에서 책 ID 추출
    book_ids = set()
    for f in results_dir.glob("*_p2s_output.xlsx"):
        book_ids.add(f.name.replace("_p2s_output.xlsx", ""))
    for f in results_dir.glob("*_s2p_output.xlsx"):
        book_ids.add(f.name.replace("_s2p_output.xlsx", ""))
    return sorted(book_ids)

def load_p2s(book_id: str, results_dir: Optional[Path] = None) -> pd.DataFrame:
    """P2S 출력 XLSX 로드"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    file = results_dir / f"{book_id}_p2s_output.xlsx"
    if not file.exists():
        raise FileNotFoundError(f"P2S 결과 파일이 없습니다: {file}")
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

def load_s2p(book_id: str, results_dir: Optional[Path] = None) -> pd.DataFrame:
    """S2P 출력 XLSX 로드"""
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    file = results_dir / f"{book_id}_s2p_output.xlsx"
    if not file.exists():
        raise FileNotFoundError(f"S2P 결과 파일이 없습니다: {file}")
    df = pd.read_excel(file)
    for col in ["원문", "번역문"]:
        if col not in df.columns:
            df[col] = ""
    return df

def load_p2s_s2p(
    book_id: str, results_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """동시에 P2S/S2P 결과를 로드"""
    return load_p2s(book_id, results_dir), load_s2p(book_id, results_dir)

def concat_books_p2s_s2p(
    book_ids: Optional[List[str]] = None, results_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    여러 책의 P2S/S2P 결과를 합쳐서 반환
    Returns: {"p2s": DataFrame, "s2p": DataFrame}
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    if book_ids is None:
        book_ids = list_books(results_dir)
    p2s_frames = []
    s2p_frames = []
    for bid in book_ids:
        try:
            p2s_df = load_p2s(bid, results_dir)
            p2s_df["book_id"] = bid
            p2s_frames.append(p2s_df)
        except FileNotFoundError:
            pass
        try:
            s2p_df = load_s2p(bid, results_dir)
            s2p_df["book_id"] = bid
            s2p_frames.append(s2p_df)
        except FileNotFoundError:
            pass
    return {
        "p2s": pd.concat(p2s_frames, ignore_index=True) if p2s_frames else pd.DataFrame(),
        "s2p": pd.concat(s2p_frames, ignore_index=True) if s2p_frames else pd.DataFrame(),
    }
