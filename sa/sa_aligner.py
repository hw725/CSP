"""
SA (Semantic Alignment) 모듈
원문을 공백 단위로 분할하고 번역문을 정렬하는 핵심 기능

완전한 원본 텍스트 무결성 보장:
- 원문: 공백으로만 분할, 토크나이저는 분석용만
- 번역문: 원문 단위에 맞춰 정렬
- 메타데이터: 시대 정보 등 참고용 데이터
"""

import logging
from typing import List, Dict, Optional, Any
import sys
import os
import pandas as pd
import numpy as np

# 공통 토크나이저 모듈 import - 전근대 고전 전용 모델 우선
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.tokenizers import (
    detect_chinese_period,
    # 전근대 고전 전용 모델들
    get_siku_tokenizer,
    siku_get_embeddings,
    siku_similarity,
    # 전근대 고전 전용 토크나이저 사용 (교체 완료)
    get_siku_tokenizer,
    siku_get_embeddings,
    siku_similarity
)

logger = logging.getLogger(__name__)

def split_src_meaning_units(text: str, **kwargs) -> List[str]:
    """
    SA 핵심 기능: 원문을 공백 단위로 분할 (무결성 보장)
    
    Args:
        text: 분할할 원문 텍스트
        **kwargs: 추가 옵션 (호환성용, 분할에는 영향 없음)
    
    Returns:
        공백으로 분할된 의미 단위 리스트 (원본 구조 완전 보존)
    """
    if not text or not text.strip():
        return []
    
    # 🎯 SA 핵심 원칙: 무조건 공백 단위로만 분할
    # 사용자가 입력한 공백 구조를 그대로 보존
    words = text.split()
    
    # 📊 내부 분석 (분할에는 절대 영향 없음, 로깅/메타데이터용만)
    try:
        if logger.isEnabledFor(logging.DEBUG):
            # 전근대 고전 텍스트로 가정 (시대 감지 불필요)
            logger.debug(f"전근대 고전 텍스트 처리 중 (분할에 영향 없음)")
            logger.debug(f"실제 분할: 공백 기준 {len(words)}개 - {words}")
    except Exception as e:
        logger.warning(f"내부 분석 실패 (분할에는 영향 없음): {e}")
    
    return words

def split_tgt_meaning_units(text: str, src_units_count: int, src_units: List[str] = None, use_semantic: bool = True, **kwargs) -> List[str]:
    """
    번역문을 원문 단위에 맞춰 분할 (의미 기반 매칭)
    
    Args:
        text: 번역문 텍스트  
        src_units_count: 원문 단위 수
        src_units: 원문 단위 리스트 (의미 매칭용)
        use_semantic: 의미 기반 매칭 사용 여부
        **kwargs: 분할 옵션
    
    Returns:
        원문 단위 수에 맞춰 분할된 번역문 리스트
    """
    if not text or not text.strip() or src_units_count <= 0:
        return [''] * max(1, src_units_count)
    
    # 의미 기반 분할 시도
    if use_semantic and src_units and len(src_units) == src_units_count:
        try:
            result = _split_tgt_by_src_units_semantic(src_units, text, **kwargs)
            logger.debug(f"의미 기반 분할 성공: {result}")
            return result
        except Exception as e:
            logger.warning(f"의미 기반 분할 실패, 단순 분할로 대체: {e}")
    
    # 폴백: 단순 분할
    return _split_tgt_by_src_units_simple(text, src_units_count)

def _split_tgt_by_src_units_semantic(src_units: List[str], tgt_text: str, min_tokens: int = 1, **kwargs) -> List[str]:
    """원문 단위에 따른 번역문 분할 (BGE-M3 Multi-Vector 의미 매칭)"""
    
    tgt_tokens = tgt_text.split()
    N, T = len(src_units), len(tgt_tokens)
    
    if N == 0 or T == 0:
        return [''] * N if N > 0 else []
    
    # 🎯 빠른 처리: 단일 단위면 전체 반환
    if N == 1:
        return [tgt_text]
    
    # 🎯 빠른 처리: 토큰이 단위보다 적으면 1:1 매칭
    if T <= N:
        result = []
        for i in range(N):
            if i < T:
                result.append(tgt_tokens[i])
            else:
                result.append("")
        return result
    
    # 🧠 BGE-M3 Multi-Vector 의미 기반 매칭 시도
    try:
        from common.embedders.bge import get_embedding_manager
        embedder = get_embedding_manager()
        
        logger.debug("✅ BGE-M3 Multi-Vector 의미 기반 매칭 시작")
        
        # 원문 단위별 Multi-Vector 임베딩 (Dense + Sparse + ColBERT)
        src_embeddings = embedder.compute_embeddings_with_cache(
            src_units, 
            batch_size=4,  # SA는 작은 배치 사용
            use_multi_vector=True  # Multi-vector 활성화
        )
        
        # 번역문 토큰들의 Multi-Vector 임베딩
        tgt_embeddings = embedder.compute_embeddings_with_cache(
            tgt_tokens, 
            batch_size=8,  # 토큰은 더 작은 단위
            use_multi_vector=True  # Multi-vector 활성화
        )
        
        # Dynamic Programming으로 최적 분할 찾기
        optimal_split = _find_optimal_split_dp(
            src_embeddings, tgt_embeddings, tgt_tokens, N, T
        )
        
        if optimal_split:
            logger.debug(f"✅ BGE-M3 Multi-Vector 최적 분할 성공: {len(optimal_split)}개 단위")
            return optimal_split
    
    except Exception as e:
        logger.warning(f"⚠️ BGE-M3 Multi-Vector 매칭 실패, 순차 분할로 대체: {e}")
    
    # ⚡ 폴백: 순차적 분할 (토큰 순서 절대 변경 금지)
    avg_len = T // N
    remainder = T % N
    
    result = []
    start_idx = 0
    
    for i in range(N):
        # 각 원문 단위당 할당할 토큰 수
        tokens_for_this_unit = avg_len
        if i < remainder:  # 나머지를 앞쪽 단위들에 분배
            tokens_for_this_unit += 1
        
        end_idx = min(start_idx + tokens_for_this_unit, T)
        
        if start_idx < T:
            unit_text = " ".join(tgt_tokens[start_idx:end_idx])
            result.append(unit_text)
        else:
            result.append("")
        
        start_idx = end_idx
    
    return result

def _find_optimal_split_dp(src_embeddings, tgt_embeddings, tgt_tokens, N, T) -> List[str]:
    """BGE-M3 Multi-Vector 기반 Dynamic Programming 최적 분할"""
    
    # 각 토큰과 각 원문 단위 간의 Multi-Vector 유사도 행렬 계산
    similarity_matrix = np.zeros((T, N))
    
    for t in range(T):
        for s in range(N):
            # Multi-Vector 코사인 유사도 (1636차원)
            sim = np.dot(tgt_embeddings[t], src_embeddings[s]) / (
                np.linalg.norm(tgt_embeddings[t]) * np.linalg.norm(src_embeddings[s]) + 1e-8
            )
            similarity_matrix[t, s] = float(sim)
    
    # DP 테이블: dp[i][j] = i번째 토큰까지 j개 단위로 분할하는 최대 점수
    dp = np.full((T + 1, N + 1), -np.inf)
    parent = np.full((T + 1, N + 1), -1, dtype=int)
    
    dp[0][0] = 0  # 기저 사례
    
    # DP 수행
    for i in range(1, T + 1):
        for j in range(1, min(i, N) + 1):
            # k는 j번째 단위의 시작 위치 (0-indexed)
            for k in range(j - 1, i):
                if dp[k][j - 1] == -np.inf:
                    continue
                
                # k부터 i-1까지 토큰들을 j번째 단위에 할당
                unit_score = 0
                token_count = i - k
                
                for t in range(k, i):
                    unit_score += similarity_matrix[t, j - 1]
                
                # 평균 점수로 정규화
                if token_count > 0:
                    unit_score /= token_count
                
                new_score = dp[k][j - 1] + unit_score
                
                if new_score > dp[i][j]:
                    dp[i][j] = new_score
                    parent[i][j] = k
    
    # 백트래킹으로 최적 분할 복원
    if dp[T][N] == -np.inf:
        return None
    
    splits = []
    i, j = T, N
    
    while j > 0:
        start = parent[i][j]
        end = i
        
        if start >= 0:
            unit_tokens = tgt_tokens[start:end]
            splits.append(" ".join(unit_tokens))
            i = start
            j -= 1
        else:
            break
    
    splits.reverse()
    
    # 결과 검증
    if len(splits) == N:
        return splits
    else:
        return None

def _split_tgt_by_src_units_simple(text: str, target_count: int) -> List[str]:
    """번역문 단순 분할 (폴백용)"""
    trans_words = text.split()
    
    if len(trans_words) == target_count:
        return trans_words
    elif len(trans_words) < target_count:
        return trans_words + [''] * (target_count - len(trans_words))
    else:
        return _distribute_words_evenly(trans_words, target_count)

def process_single_row(row_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
    단일 행 처리: 원문 분할 + 번역문 정렬 → 여러 행으로 분할
    
    Args:
        row_data: {'원문': '원문', '번역문': '번역문', '문장식별자': 'id', ...}
        **kwargs: 처리 옵션
    
    Returns:
        분할된 행 데이터 리스트 (원문 단위별로 개별 행)
    """
    source_text = row_data.get('원문', '')
    translation_text = row_data.get('번역문', '')
    base_id = row_data.get('문장식별자', '')
    
    # 원문 분할 (공백 기준)
    src_units = split_src_meaning_units(source_text, **kwargs)
    
    # 번역문 정렬 (의미 기반)
    trans_units = split_tgt_meaning_units(
        translation_text, 
        len(src_units), 
        src_units=src_units,  # 원문 단위 전달
        use_semantic=True,    # 의미 기반 매칭 사용
        **kwargs
    )
    
    # 시대 정보 분석
    try:
        period = detect_chinese_period(source_text)  # 전근대 고전 가정
    except Exception as e:
        logger.warning(f"시대 감지 실패: {e}")
        period = 'unknown'
    
    # 각 단위별로 개별 행 생성
    result_rows = []
    for i, (src_unit, trans_unit) in enumerate(zip(src_units, trans_units)):
        row = {
            '문장식별자': base_id if base_id else f"row_{i+1}",
            '원문': src_unit,
            '번역문': trans_unit,
            # 원본 데이터의 다른 컬럼들 유지
            **{k: v for k, v in row_data.items() 
               if k not in ['문장식별자', '원문', '번역문']}
        }
        result_rows.append(row)
    
    # 🆕 한글 토씨 매칭으로 SA 결과 보완
    try:
        from common.korean_particle_matcher import enhance_sa_results_with_particles
        result_rows = enhance_sa_results_with_particles(result_rows)
        logger.debug(f"SA 토씨 매칭 보완 완료: {len(result_rows)}개 행")
    except Exception as e:
        logger.warning(f"SA 토씨 매칭 보완 실패 (기존 결과 유지): {e}")
        # 실패해도 기존 result_rows 그대로 사용

    logger.info(f"SA 처리 완료: {len(src_units)}개 단위, 시대: {period}")
    
    return result_rows

def align_translation_to_source(src_units: List[str], translation: str, use_semantic: bool = True, **kwargs) -> List[str]:
    """
    번역문을 원문 단위에 맞춰 정렬 (의미 기반)
    
    Args:
        src_units: 원문 의미 단위들 (공백 분할된)
        translation: 번역문 전체
        use_semantic: 의미 기반 매칭 사용 여부
        **kwargs: 정렬 옵션
    
    Returns:
        원문 단위 수에 맞춰 분할된 번역문 리스트
    """
    if not src_units or not translation:
        return []
    
    # 의미 기반 분할 사용
    return split_tgt_meaning_units(
        translation, 
        len(src_units), 
        src_units=src_units,
        use_semantic=use_semantic,
        **kwargs
    )
    
    logger.debug(f"번역문 정렬: {len(src_units)}개 원문 → {len(aligned)}개 번역문")
    
    return aligned

def _distribute_words_evenly(words: List[str], target_count: int) -> List[str]:
    """단어들을 목표 개수에 맞춰 균등 분배"""
    if not words or target_count <= 0:
        return []
    
    if target_count == 1:
        return [' '.join(words)]
    
    # 단어를 그룹으로 나누기
    words_per_group = len(words) / target_count
    result = []
    
    for i in range(target_count):
        start_idx = int(i * words_per_group)
        end_idx = int((i + 1) * words_per_group) if i < target_count - 1 else len(words)
        
        group_words = words[start_idx:end_idx]
        result.append(' '.join(group_words) if group_words else '')
    
    return result

def process_sa_alignment(src_text: str, translation: str, **kwargs) -> Dict[str, List[str]]:
    """
    SA 통합 처리: 원문 분할 + 번역문 정렬
    
    Args:
        src_text: 원문 텍스트
        translation: 번역문 텍스트
        **kwargs: 처리 옵션
    
    Returns:
        {
            'source_units': [...],    # 분할된 원문
            'translation_units': [...], # 정렬된 번역문
            'metadata': {...}         # 메타데이터
        }
    """
    # 원문 분할 (공백 단위)
    src_units = split_src_meaning_units(src_text, **kwargs)
    
    # 번역문 정렬
    trans_units = align_translation_to_source(src_units, translation, **kwargs)
    
    # 메타데이터
    metadata = {
        'source_count': len(src_units),
        'translation_count': len(trans_units),
        'alignment_method': 'space_based'
    }
    
    # 시대 정보 추가 (분석용)
    try:
        period = detect_chinese_period(src_text)  # 전근대 고전 가정
        metadata['detected_period'] = period
    except Exception as e:
        logger.warning(f"시대 감지 실패: {e}")
        metadata['detected_period'] = 'unknown'
    
    result = {
        'source_units': src_units,
        'translation_units': trans_units,
        'metadata': metadata
    }
    
    logger.info(f"SA 처리 완료: {metadata['source_count']}개 단위, 시대: {metadata['detected_period']}")
    
    return result
