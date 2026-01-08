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
import json
import hashlib

# OpenAI wrapper for direct access with parallel processing
class OpenAIWrapper:
    """OpenAI embedder wrapper - uses common module with parallel processing"""
    def __init__(self, max_workers=4, batch_size=100):
        try:
            # 패키지 경로로 안전하게 임포트
            from common.embedders.openai_embedder import compute_embeddings_batch
            self.compute_embeddings_batch = compute_embeddings_batch
            self.max_workers = max_workers
            self.batch_size = batch_size
            logger.debug(f"✅ OpenAI 임베더 초기화 완료 (max_workers={max_workers}, batch_size={batch_size})")
        except Exception as e:
            raise ImportError(f"OpenAI 설정 실패: {e}")
    
    def compute_embeddings_with_cache(self, texts, batch_size=None):
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        actual_batch_size = batch_size if batch_size is not None else self.batch_size
        
        try:
            embeddings = self.compute_embeddings_batch(
                texts, 
                model="text-embedding-3-large",
                batch_size=actual_batch_size,  # 🚀 매개변수명 수정
                max_workers=self.max_workers
            )
            
            if return_single:
                return np.array(embeddings[0])
            else:
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"OpenAI 임베딩 생성 실패: {e}")
            raise

logger = logging.getLogger(__name__)

# 공통 토크나이저 모듈 import - 전근대 고전 전용 모델 우선
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from common.tokenizers import (
        detect_chinese_period,
        # 전근대 고전 전용 모델들
        get_siku_tokenizer,
        siku_get_embeddings,
        siku_similarity
    )
except ImportError as e:
    logger.warning(f"⚠️ SA: 하이브리드 토크나이저 초기화 실패: {e}")
    # 폴백: 기본 split() 사용

logger = logging.getLogger(__name__)

# ===== SuPar-Kanbun 안전 로딩 준비 (Torch 2.6 weights_only 대비) =====
def _prepare_supar_safe_loading():
    """Torch 2.6의 안전 로딩 정책으로 인한 SuPar 체크포인트 로딩 실패를 완화.
    supar.utils.config.Config 클래스를 안전 목록에 추가한다.
    실패 시 조용히 무시(폴백 유지).
    """
    try:
        # 지연 임포트: 환경에 없으면 조용히 패스
        try:
            import importlib  # 표준 라이브러리
            torch_serialization = importlib.import_module('torch.serialization')
            add_safe_globals = getattr(torch_serialization, 'add_safe_globals', None)
            if callable(add_safe_globals):
                # SuPar 관련 모듈의 클래스들을 폭넓게 허용 (특히 *Field, Transform/CoNLL 등)
                try:
                    import inspect
                    module_names = [
                        'supar.utils.config',
                        'supar.utils.transform',
                        'supar.utils.field',
                        'supar.utils.vocab',
                        'dill._dill',
                    ]
                    to_allow = []
                    for mod_name in module_names:
                        try:
                            mod = importlib.import_module(mod_name)
                        except Exception:
                            continue
                        for name, obj in vars(mod).items():
                            try:
                                # 클래스 및 함수 모두 허용 (dill._dill._load_type 등을 위해)
                                if inspect.isclass(obj) or inspect.isfunction(obj):
                                    # SuPar 관련: Field 계열, Transform/CoNLL 계열, Vocab/Vectors 등
                                    # dill 관련: _load_type, _create_type 등
                                    if (
                                        'Field' in name or
                                        name in {'Transform', 'CoNLL', 'CoNLLU', 'Vocab', 'Vectors', 'Config'} or
                                        name.startswith('_load') or name.startswith('_create') or name.startswith('_import')
                                    ):
                                        to_allow.append(obj)
                            except Exception:
                                continue
                    if to_allow:
                        try:
                            add_safe_globals(to_allow)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # torch가 없거나 구버전인 경우 무시
            pass
    except Exception:
        pass

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
    """원문 단위에 따른 번역문 분할 (고어 패턴 감지 개선)"""
    
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
    
    # 🧠 동적 임베더 기반 분할 (순서 보장 모드)
    try:
        # 설정된 임베더 가져오기 (CLI --embedder 옵션 반영)
        embedder_name = kwargs.get('embedder_name', kwargs.get('embedder', 'bge'))
        max_workers = kwargs.get('max_workers', 4)
        # CLI에서는 chunk_size 이름을 사용하므로 호환 처리
        batch_size = kwargs.get('batch_size', kwargs.get('chunk_size', 100))
        
        # 임베더 없이 순차 분할만 사용하는 경우
        if embedder_name.lower() == 'none':
            logger.debug("⚡ 순차 분할 모드: 임베더 미사용으로 빠른 처리")
            # 바로 순차 분할로 넘어감
            raise Exception("순차 분할 모드 선택됨")
        
        if embedder_name.lower() == 'openai':
            # OpenAI 직접 사용 (병렬 처리 적용)
            embedder = OpenAIWrapper(max_workers=max_workers, batch_size=batch_size)
            logger.debug(f"✅ OpenAI 임베더로 순서 보장 의미 매칭 시작 (max_workers={max_workers})")
            compute_embeddings_func = embedder.compute_embeddings_with_cache
        else:
            # BGE 등 다른 임베더 사용 - 함수 직접 가져오기
            from common.embedders import get_embedder
            compute_embeddings_func = get_embedder(embedder_name)
            logger.debug(f"✅ {embedder_name.upper()} 임베더로 순서 보장 의미 매칭 시작")
        
        # 원문 단위별 임베딩
        src_embeddings = compute_embeddings_func(
            src_units, 
            batch_size=batch_size
        )
        
        # 번역문 토큰들의 임베딩
        tgt_embeddings = compute_embeddings_func(
            tgt_tokens, 
            batch_size=batch_size
        )
        
        # 🎯 순서 보장 Dynamic Programming (가중치 파라미터 전달)
        # DP에 원문 단위/텍스트 컨텍스트도 전달(구문 힌트용)
        dp_kwargs = dict(kwargs)
        dp_kwargs.setdefault('src_units', src_units)
        dp_kwargs.setdefault('source_text', ' '.join(src_units))
        optimal_split = _find_optimal_split_dp_sequential(
            src_embeddings, tgt_embeddings, tgt_tokens, N, T, **dp_kwargs
        )
        
        if optimal_split and len(optimal_split) == N:
            logger.debug(f"✅ {embedder_name.upper()} 순서 보장 분할 성공: {len(optimal_split)}개 단위")
            return optimal_split
    
    except Exception as e:
        if embedder_name and embedder_name.lower() != 'none':
            logger.warning(f"⚠️ 임베더 순서 보장 매칭 실패, 순차 분할로 대체: {e}")
        else:
            logger.debug("⚡ 순차 분할 모드로 진행")
    
    # ⚡ 폴백: 순차적 분할만 사용 (순서 무결성 보장)
    # 기본 모드: 평균 길이 기반 분할 (고어 패턴 보정 적용)
    try:
        from common.korean_particle_matcher import get_archaic_bonus
        
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
                
                # 🆕 고어 패턴 보정 적용
                try:
                    archaic_bonus = get_archaic_bonus(unit_text, mode='SA')
                    if archaic_bonus > 0.05:
                        logger.debug(f"기본 분할에서 고어 패턴 감지: {unit_text} (보너스: {archaic_bonus})")
                except:
                    pass  # 고어 패턴 실패해도 계속 진행
                
                result.append(unit_text)
            else:
                result.append("")
            
            start_idx = end_idx
        
        return result
        
    except Exception as e:
        logger.warning(f"고어 패턴 보정 실패, 기본 분할 사용: {e}")
        # 최종 폴백: 기본 균등 분할
        avg_len = T // N
        remainder = T % N
        
        result = []
        start_idx = 0
        
        for i in range(N):
            tokens_for_this_unit = avg_len
            if i < remainder:
                tokens_for_this_unit += 1
            
            end_idx = min(start_idx + tokens_for_this_unit, T)
            
            if start_idx < T:
                unit_text = " ".join(tgt_tokens[start_idx:end_idx])
                result.append(unit_text)
            else:
                result.append("")
            
            start_idx = end_idx
        
        return result

def _find_optimal_split_dp_sequential(
    src_embeddings,
    tgt_embeddings,
    tgt_tokens,
    N,
    T,
    **kwargs,
) -> List[str]:
    """
    순서 보장 Dynamic Programming 분할 (의미 기반 경계 조정만)
    - 가중치 파라미터(옵션, 기본은 기존 동작 유지):
      dp_window: 예상 위치 대비 허용 창(정수, 기본 1)
      distance_decay: 거리 감쇠 알파(실수, 기본 0.0 → 비활성)
      boundary_bonus: 경계 보너스(실수, 기본 0.0)
      particle_bonus: 토씨 경계 보너스(실수, 기본 0.0)
      length_penalty: 기대 길이 대비 차이에 대한 패널티 알파(실수, 기본 0.0)
      sim_gamma: 유사도 샤프닝 지수(실수, 기본 1.0)
    """

    # ===== 파라미터 로드 (기본값은 기존 동작 유지) =====
    dp_window: int = int(kwargs.get('dp_window', 1))
    distance_decay_alpha: float = float(kwargs.get('distance_decay', 0.0))
    boundary_bonus: float = float(kwargs.get('boundary_bonus', 0.0))
    particle_bonus: float = float(kwargs.get('particle_bonus', 0.0))
    length_penalty_alpha: float = float(kwargs.get('length_penalty', 0.0))
    sim_gamma: float = float(kwargs.get('sim_gamma', 1.0))

    # ===== 경계 힌트 준비 =====
    def _is_boundary_token(tok: str) -> bool:
        if not tok:
            return False
        punct_ends = set(list(".!?;:。！？；、…·”’'\"」』》)]>))"))
        return tok[-1] in punct_ends

    def _is_particle_ending(tok: str) -> bool:
        if not tok:
            return False
        particles = (
            '은','는','이','가','을','를','에','에게','에서','으로','로','와','과','랑','하고','도','만','까지','부터',
            '마다','처럼','보다','조차','마저','께','께서','이라','라','이라도','라도','이나','나','이라서','라서'
        )
        stripped = tok.rstrip(".,!?;:。！？；、…·”’'\"」』》)]>))")
        for p in particles:
            if stripped.endswith(p):
                return True
        return False

    boundary_flags = [_is_boundary_token(t) for t in tgt_tokens]
    particle_flags = [_is_particle_ending(t) for t in tgt_tokens]

    # ===== 한국어/중국어 구문 힌트 기반 경계 (옵션) - 파싱 게이팅 준비 =====
    comma_bonus = float(kwargs.get('comma_bonus', 0.0) or 0.0)
    comma_mode = kwargs.get('comma_mode', 'soft')
    syntax_hints = kwargs.get('syntax_hints', 'none')
    syntax_when = kwargs.get('syntax_when', 'ambiguous')
    # 강도 가중치를 위한 배열 (토큰 끝 경계 강도)
    ko_boundary_token_strengths = [0.0] * T
    src_boundary_indices = set()

    # ===== 유사도 매트릭스 계산 =====
    similarity_matrix = np.zeros((T, N))
    for t in range(T):
        for s in range(N):
            sim = np.dot(tgt_embeddings[t], src_embeddings[s]) / (
                np.linalg.norm(tgt_embeddings[t]) * np.linalg.norm(src_embeddings[s]) + 1e-8
            )
            sim_val = float(sim)
            if sim_gamma != 1.0:
                sign = 1.0 if sim_val >= 0 else -1.0
                sim_val = sign * (abs(sim_val) ** sim_gamma)
            similarity_matrix[t, s] = sim_val

    # 🎯 순서 보장 + 거리 감쇠
    enhanced_similarity = np.full_like(similarity_matrix, -1000.0)
    for t in range(T):
        expected_unit = min(int(t * N / T), N - 1)
        left = max(0, expected_unit - dp_window)
        right = min(N, expected_unit + dp_window + 1)
        for s in range(left, right):
            val = similarity_matrix[t, s]
            if distance_decay_alpha > 0:
                dist = abs(s - expected_unit)
                decay = np.exp(-distance_decay_alpha * dist)
                val = val * decay
            enhanced_similarity[t, s] = val

    # ===== 파싱 모호성 판단 이후에 구문 힌트 계산 =====
    def is_ambiguous(sim_mat: np.ndarray) -> bool:
        try:
            gaps = []
            for tt in range(sim_mat.shape[0]):
                row = sim_mat[tt]
                vals = [v for v in row if v > -999]
                if len(vals) >= 2:
                    top2 = sorted(vals, reverse=True)[:2]
                    gaps.append(top2[0] - top2[1])
            if not gaps:
                return False
            avg_gap = sum(gaps) / len(gaps)
            return avg_gap < 0.05
        except Exception:
            return False

    allow_parse = (syntax_when == 'always') or (syntax_when == 'ambiguous' and is_ambiguous(enhanced_similarity))

    # 한국어 경계 강도 계산 (Stanza 우선, 실패 시 콤마 휴리스틱) — syntax_hints가 ko/both일 때만
    if allow_parse and syntax_hints in ('ko', 'both'):
        try:
            from common.new_parsers import (
                get_korean_clause_offsets_with_strength,
                get_korean_clause_boundary_commas,
            )
            # 토큰을 공백 없이 이어 붙여 문자 오프셋 공간으로 매핑
            token_spans = []
            offset = 0
            joined = "".join(tgt_tokens)
            for tok in tgt_tokens:
                token_spans.append((offset, offset + len(tok)))
                offset += len(tok)
            # 강도 사전 조회 (실패/미가용 시 콤마 1.0 강도)
            try:
                strength_map = get_korean_clause_offsets_with_strength(joined)
            except Exception:
                offs = get_korean_clause_boundary_commas(joined, mode=comma_mode)
                strength_map = {o: 1.0 for o in offs}
            if strength_map:
                for idx, (s_off, e_off) in enumerate(token_spans):
                    # 경계는 토큰 끝 직전 문자 위치(e_off-1)
                    if (e_off - 1) in strength_map:
                        ko_boundary_token_strengths[idx] = max(
                            ko_boundary_token_strengths[idx], float(strength_map[e_off - 1])
                        )
        except Exception:
            pass

    # 중국어 원문 단위 경계 (syntax_hints가 zh/both일 때만)
    if allow_parse and syntax_hints in ('zh', 'both'):
        try:
            from common.new_parsers import get_chinese_unit_boundary_indices
            src_boundary_indices = get_chinese_unit_boundary_indices(kwargs.get('src_units', []))
        except Exception:
            src_boundary_indices = set()

    # DP로 최적 분할 찾기
    dp = np.full((T + 1, N + 1), -np.inf)
    parent = np.full((T + 1, N + 1), -1, dtype=int)
    dp[0][0] = 0

    for i in range(1, T + 1):
        for j in range(1, min(i, N) + 1):
            for k in range(j - 1, i):
                if dp[k][j - 1] == -np.inf:
                    continue
                unit_score = 0.0
                token_count = i - k
                for t in range(k, i):
                    unit_score += enhanced_similarity[t, j - 1]
                if token_count > 0:
                    unit_score /= token_count
                if length_penalty_alpha > 0 and N > 0:
                    expected_len = max(1, int(round(T / N)))
                    diff = abs(token_count - expected_len)
                    unit_score -= length_penalty_alpha * diff
                end_tok_idx = i - 1
                if 0 <= end_tok_idx < T:
                    if boundary_bonus != 0.0 and boundary_flags[end_tok_idx]:
                        unit_score += boundary_bonus
                    if particle_bonus != 0.0 and particle_flags[end_tok_idx]:
                        unit_score += particle_bonus
                    # 한국어 경계 보너스: 강도 가중치 적용
                    if comma_bonus != 0.0:
                        strength = ko_boundary_token_strengths[end_tok_idx]
                        if strength > 0.0:
                            bonus = comma_bonus * strength
                            # 양측 동시 지지(원문 경계와 현재 단위 경계 j 정렬) 시 강화
                            if src_boundary_indices and (j in src_boundary_indices):
                                bonus *= 1.5
                            unit_score += bonus
                new_score = dp[k][j - 1] + unit_score
                if new_score > dp[i][j]:
                    dp[i][j] = new_score
                    parent[i][j] = k

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
    return splits if len(splits) == N else None

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

    # 🔒 SA 무결성 검증: 번역문 텍스트 보존 확인
    try:
        # 원본 번역문 (공백 제거)
        original_trans = translation_text.replace(' ', '')
        
        # 처리된 번역문 재결합 (공백 제거)
        processed_trans = ''.join([row['번역문'].replace(' ', '') for row in result_rows])
        
        # 무결성 검증
        if original_trans != processed_trans:
            logger.error(f"SA 무결성 실패: {base_id}")
            logger.error(f"  원본 길이: {len(original_trans)}자")
            logger.error(f"  처리 후 길이: {len(processed_trans)}자")
            logger.error(f"  길이 차이: {len(processed_trans) - len(original_trans)}자")
            
            # 문자 정확도 계산
            correct_chars = sum(1 for a, b in zip(original_trans, processed_trans) if a == b)
            accuracy = correct_chars / max(len(original_trans), len(processed_trans))
            logger.error(f"  문자 정확도: {accuracy:.3f}")
            
            # 차이점 상세 분석 (처음 100자만)
            if len(original_trans) > 0 and len(processed_trans) > 0:
                import difflib
                diff = list(difflib.unified_diff(
                    original_trans[:100], processed_trans[:100], 
                    fromfile='원본', tofile='처리후', lineterm=''
                ))
                if diff:
                    logger.error(f"SA 텍스트 차이점 분석: {base_id}")
                    logger.error(f"  원본 샘플: '{original_trans[:50]}{'...' if len(original_trans) > 50 else ''}'")
                    logger.error(f"  처리 샘플: '{processed_trans[:50]}{'...' if len(processed_trans) > 50 else ''}'")
            
            # 무결성 실패시에도 결과는 반환 (분석용)
        else:
            logger.debug(f"SA 무결성 확인: {base_id} ✅")
            
    except Exception as e:
        logger.warning(f"SA 무결성 검증 실패: {e}")

    # 🔧 기본 모드에서는 상세 로깅 제거, verbose에서만 출력
    logger.debug(f"SA 처리 완료: {len(src_units)}개 단위, 시대: {period}")
    
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
    
    # 로깅 제거 - 너무 상세함
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
    
    logger.debug(f"SA 처리 완료: {metadata['source_count']}개 단위, 시대: {metadata['detected_period']}")
    
    return result