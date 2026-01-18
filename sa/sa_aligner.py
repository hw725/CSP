"""
SA (Semantic Alignment) 모듈
원문을 공백 단위로 분할하고 번역문을 정렬하는 핵심 기능

완전한 원본 텍스트 무결성 보장:
- 원문: 공백으로만 분할, 토크나이저는 분석용만
- 번역문: 원문 단위에 맞춰 정렬
- 메타데이터: 시대 정보 등 참고용 데이터
"""

import logging
# Use the third-party 'regex' module to support Unicode properties like \p{Han}
import regex as re
from typing import List, Dict, Optional, Any, Tuple
import sys
import os
import pandas as pd
import numpy as np
import json
import hashlib

try:
    from common.llm_boundary_refiner import refine_boundaries_with_llm
except Exception:
    refine_boundaries_with_llm = None

# 전역 구식별자 (파일 처리마다 리셋)
_global_segment_id = 0


def reset_segment_counter(start: int = 0):
    """구식별자를 리셋하고 시작값을 설정한다."""
    global _global_segment_id
    _global_segment_id = start

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

# 공용 한자 토큰 패턴 (SikuBERT 기준: \p{Han}+)
_han_token_pattern = re.compile(r"\p{Han}+")

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

    # 🧭 원문 토크나이징(분석용): 한자(SikuBERT), 한글 토씨(Kiwipiepy)
    # - 결과는 반환하지 않고 필요 시 kwargs['token_capture']에 담아 전달
    siku_tokens: List[str] = []
    kiwi_tokens: List[str] = []
    kiwi_particles: List[Any] = []

    try:
        siku_tokens = _han_token_pattern.findall(text)
    except Exception as e:
        logger.debug(f"SA: SikuBERT 한자 토큰 추출 실패: {e}")

    try:
        from common.tokenizers import get_kiwi_tokenizer

        kiwi = get_kiwi_tokenizer()
        kiwi_tokens = kiwi.morphs(text)
        try:
            kiwi_particles = kiwi.extract_particles(text)
        except Exception as e:
            logger.debug(f"SA: Kiwipiepy 토씨 추출 실패: {e}")
    except Exception as e:
        logger.debug(f"SA: Kiwipiepy 초기화 실패: {e}")

    token_capture = kwargs.get('token_capture')
    if isinstance(token_capture, dict):
        token_capture['siku_tokens'] = siku_tokens
        token_capture['kiwi_tokens'] = kiwi_tokens
        token_capture['kiwi_particles'] = kiwi_particles
    
    return words


def _mask_unaligned_segments(text: str):
    """비대응 표시 구간([- (... )])에서 [, -, ] 부호만 토큰으로 마스킹"""
    pattern = re.compile(r"\[-\(([^)]*)\)\]")
    mapping = []  # (token, symbol)

    def repl(match):
        seq = len(mapping) // 3
        token_l = f"__UNALIGNED_L_{seq}__"
        token_h = f"__UNALIGNED_H_{seq}__"
        token_r = f"__UNALIGNED_R_{seq}__"
        mapping.extend([
            (token_l, "["),
            (token_h, "-"),
            (token_r, "]"),
        ])
        inner = match.group(1)
        return f"{token_l}{token_h}({inner}){token_r}"

    masked = pattern.sub(repl, text)
    return masked, mapping


def _unmask_text(text: str, mapping):
    for token, original in mapping:
        text = text.replace(token, original)
    return text


def _unmask_list(chunks: List[str], mapping):
    return [_unmask_text(chunk, mapping) for chunk in chunks]


def _merge_leading_quotation_markers(units: List[str]) -> List[str]:
    """인용 표지가 문장 앞에 오면 바로 앞 단위로 이동시키되 길이는 유지한다."""
    if len(units) <= 1:
        return units

    quotation_particles = r'(고|[이]?라?고|하고|며|면서)'
    speech_verbs = r'(하|말하|말씀하|명하|이르|대답하|답하|묻|문|여쭙|아뢰|전하|칭하|부르|외치)'
    honorific_tense = r'(?:셨|ㅆ|시었|시어|시는|시ㄴ|시ㄹ|시|었|았|였|는|ㄴ|ㄹ|을)?'
    endings = r'(다|ㄴ다|는다|습니다|ㅂ니다|까|ㄹ까|을까|느냐|ㄴ가|는가|라|거라|소|오|어라|아라|니|으니)'
    closing_quote = r'["”’]?'
    punctuation = r'[\.。?!,，]?'
    marker_chunk = (
        closing_quote +
        r'\s*' +
        quotation_particles +
        r'\s+' +
        speech_verbs +
        honorific_tense +
        endings +
        r'\s*' +
        punctuation +
        r'\s*' +
        closing_quote +
        r'\s*'
    )
    marker_prefix_regex = re.compile(r'^\s*' + marker_chunk, re.IGNORECASE)

    merged = list(units)
    for idx in range(1, len(merged)):
        seg = merged[idx]
        if seg and marker_prefix_regex.match(seg):
            merged[idx - 1] = (merged[idx - 1].rstrip() + ' ' + seg.lstrip()).strip()
            merged[idx] = ''
    return merged

def _adjust_segments_to_count(
    segments: List[str], 
    target_count: int, 
    mode: str,
    text: str = None,
    src_units: List[str] = None
) -> List[str]:
    """
    LLM이 제안한 세그먼트를 원문 의미에 맞게 목표 개수로 조정
    
    Args:
        segments: LLM이 제안한 세그먼트 리스트
        target_count: 목표 세그먼트 개수 (원문 단위 수)
        mode: 'split' (추가 분할) 또는 'merge' (병합)
        text: 원본 번역문 텍스트 (LLM 재요청용)
        src_units: 원문 세그먼트 리스트 (의미 대응 정보)
    
    Returns:
        조정된 세그먼트 리스트 (정확히 target_count 개)
    """
    if len(segments) == target_count:
        return segments
    
    # LLM에게 의미 기반 재조정 요청 시도
    if text and src_units and len(src_units) == target_count and refine_boundaries_with_llm:
        try:
            adjusted = _adjust_with_llm_semantic(text, segments, src_units, target_count, mode)
            if adjusted and len(adjusted) == target_count:
                # 텍스트 무결성 검증
                original_flat = ''.join(text.split())
                adjusted_flat = ''.join(''.join(adjusted).split())
                if original_flat == adjusted_flat:
                    logger.info(f"✅ LLM 의미 기반 조정 성공: {len(segments)}개 → {target_count}개")
                    return adjusted
        except Exception as e:
            logger.debug(f"LLM 의미 기반 조정 실패, 폴백 사용: {e}")
    
    # 폴백: 길이 기반 조정
    result = list(segments)
    
    if mode == 'split':
        # 가장 긴 세그먼트를 반복적으로 분할
        while len(result) < target_count:
            # 가장 긴 세그먼트 찾기 (공백으로 분할 가능한 것 우선)
            longest_idx = -1
            longest_len = 0
            for i, seg in enumerate(result):
                tokens = seg.split()
                if len(tokens) > 1 and len(tokens) > longest_len:
                    longest_len = len(tokens)
                    longest_idx = i
            
            if longest_idx == -1:
                # 분할 가능한 세그먼트가 없음 - 마지막에 빈 문자열 추가
                result.append('')
                continue
            
            # 세그먼트를 중간에서 분할
            seg = result[longest_idx]
            tokens = seg.split()
            mid = len(tokens) // 2
            left = ' '.join(tokens[:mid])
            right = ' '.join(tokens[mid:])
            result[longest_idx] = left
            result.insert(longest_idx + 1, right)
    
    elif mode == 'merge':
        # 가장 짧은 인접 세그먼트 쌍을 반복적으로 병합
        while len(result) > target_count:
            # 가장 짧은 인접 쌍 찾기
            shortest_pair_idx = -1
            shortest_pair_len = float('inf')
            for i in range(len(result) - 1):
                pair_len = len(result[i]) + len(result[i + 1])
                if pair_len < shortest_pair_len:
                    shortest_pair_len = pair_len
                    shortest_pair_idx = i
            
            if shortest_pair_idx == -1:
                # 병합할 수 없음 - 마지막 제거
                result.pop()
                continue
            
            # 두 세그먼트 병합
            merged = result[shortest_pair_idx] + ' ' + result[shortest_pair_idx + 1]
            result[shortest_pair_idx] = merged.strip()
            result.pop(shortest_pair_idx + 1)
    
    return result


def _adjust_with_llm_semantic(
    text: str,
    current_segments: List[str],
    src_units: List[str],
    target_count: int,
    mode: str
) -> List[str]:
    """
    LLM을 사용해 원문 의미에 맞게 번역문 세그먼트 재조정
    
    Args:
        text: 원본 번역문 전체 텍스트
        current_segments: 현재 LLM이 제안한 세그먼트
        src_units: 원문 세그먼트 리스트
        target_count: 목표 개수
        mode: 'split' 또는 'merge'
    
    Returns:
        재조정된 세그먼트 리스트
    """
    import os
    import json
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    model_name = os.getenv("LLM_BOUNDARY_MODEL", "gpt-4o-mini")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None
    
    # 원문 세그먼트 번호와 함께 제시
    src_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(src_units)])
    
    # 현재 LLM이 제안한 세그먼트
    current_list = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(current_segments)])
    
    if mode == 'split':
        instruction = f"""당신이 제안한 {len(current_segments)}개 세그먼트를 {target_count}개로 세분화해야 합니다.
        
원문은 {target_count}개의 의미 단위로 구성되어 있습니다:
{src_list}

현재 당신이 제안한 번역문 세그먼트 {len(current_segments)}개:
{current_list}

번역문을 정확히 {target_count}개 세그먼트로 나누되, 각 세그먼트가 원문의 대응하는 의미 단위와 맞도록 분할하세요.
번역문의 일부 세그먼트가 원문의 여러 의미 단위를 포함하고 있다면, 그것을 원문 단위에 맞게 나누세요."""
    else:  # merge
        instruction = f"""당신이 제안한 {len(current_segments)}개 세그먼트를 {target_count}개로 병합해야 합니다.

원문은 {target_count}개의 의미 단위로 구성되어 있습니다:
{src_list}

현재 당신이 제안한 번역문 세그먼트 {len(current_segments)}개:
{current_list}

번역문을 정확히 {target_count}개 세그먼트로 병합하되, 각 세그먼트가 원문의 대응하는 의미 단위와 맞도록 조정하세요.
번역문의 여러 세그먼트가 원문의 하나의 의미 단위에 대응한다면, 그것들을 병합하세요."""
    
    prompt = f"""{instruction}

원본 번역문 전체:
{text}

중요 규칙:
1. 텍스트의 문자나 순서를 절대 변경하지 마세요. 오직 경계만 조정하세요.
2. 반드시 정확히 {target_count}개 세그먼트를 반환하세요.
3. 각 세그먼트가 원문의 대응하는 의미 단위를 번역한 부분이 되도록 하세요.

JSON 형식으로만 응답하세요:
{{"segments": ["세그먼트1", "세그먼트2", ...]}}"""
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise boundary adjuster. Adjust boundaries based on source-target semantic correspondence. Never alter text content."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        
        content = resp.choices[0].message.content.strip()
        
        # JSON 파싱
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        adjusted = data.get("segments", [])
        
        if not isinstance(adjusted, list) or len(adjusted) != target_count:
            logger.warning(f"LLM 의미 조정: 잘못된 개수 반환 ({len(adjusted)} != {target_count})")
            return None
        
        return adjusted
        
    except Exception as e:
        logger.debug(f"LLM 의미 조정 요청 실패: {e}")
        return None

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
    
    tgt_tokens = text.split()
    N, T = src_units_count, len(tgt_tokens)
    
    # 원본 텍스트 정규화 (무결성 검증용)
    original_normalized = text.replace(' ', '').replace('\n', '').replace('\t', '')

    # 의미 기반 분할 시도
    result: List[str]
    if use_semantic and src_units and len(src_units) == src_units_count:
        try:
            result = _split_tgt_by_src_units_semantic(src_units, text, **kwargs)
            logger.debug(f"의미 기반 분할 성공: {result}")
            
            # 🔒 무결성 검증: 분할 결과가 원본과 동일한지 확인 (경고만)
            result_normalized = ''.join(result).replace(' ', '').replace('\n', '').replace('\t', '')
            if result_normalized != original_normalized:
                diff = len(original_normalized) - len(result_normalized)
                logger.warning(f"⚠️ 의미 기반 분할 무결성 경고: 원본 {len(original_normalized)}자 != 분할 {len(result_normalized)}자 (차이: {diff:+d}자)")
                # 차이가 크면 (10자 이상) 폴백으로 재시도
                if abs(diff) > 10:
                    logger.warning(f"   차이가 커서 단순 분할로 대체")
                    del result  # 무결성 문제가 크면 폴백
                # 차이가 작으면 (공백 차이 등) 그대로 사용
        except Exception as e:
            logger.warning(f"의미 기반 분할 실패, 단순 분할로 대체: {e}")
    
    # 폴백: 단순 분할
    if 'result' not in locals():
        result = _split_tgt_by_src_units_simple(text, src_units_count, src_units=src_units, **kwargs)
        
        # 🔒 단순 분할 무결성 검증 (경고만)
        result_normalized = ''.join(result).replace(' ', '').replace('\n', '').replace('\t', '')
        if result_normalized != original_normalized:
            diff = len(original_normalized) - len(result_normalized)
            logger.warning(f"⚠️ 단순 분할 무결성 경고: 원본 {len(original_normalized)}자 != 분할 {len(result_normalized)}자 (차이: {diff:+d}자)")
            # 차이가 크면 (10자 이상) 균등 분할로 재시도
            if abs(diff) > 10:
                logger.warning(f"   차이가 커서 균등 분할로 재시도")
                words = text.split()
                result = _distribute_words_evenly(words, src_units_count)
                result_normalized_2 = ''.join(result).replace(' ', '').replace('\n', '').replace('\t', '')
                if result_normalized_2 == original_normalized:
                    logger.info(f"✅ 균등 분할로 무결성 복원")
                else:
                    logger.error(f"❌ 균등 분할도 무결성 문제: 차이 {len(original_normalized) - len(result_normalized_2):+d}자")

    # 인용 표지가 문장 앞에 오면 앞 단위로 이동 (빈 칸 유지)
    result = _merge_leading_quotation_markers(result)

    # 선택적 LLM 경계 재검증 (길이/문자 무결성 유지)
    original_result = list(result)
    original_flat = ''.join(original_result).replace("\n", "")
    if refine_boundaries_with_llm:
        try:
            # 의미 대응 강화를 위해 원문 단위들을 참조 텍스트로 제공
            ref_text = None
            try:
                if src_units and isinstance(src_units, list) and len(src_units) == src_units_count:
                    ref_text = " ".join([s for s in src_units if isinstance(s, str)])
            except Exception:
                ref_text = None
            llm_checked = refine_boundaries_with_llm(
                text,
                result,
                task="sa",
                max_segments=20,
                reference_text=ref_text,
            )
            if llm_checked:
                llm_flat = ''.join(llm_checked).replace("\n", "")
                same_text = ''.join(llm_flat.split()) == ''.join(original_flat.split())
                
                if not same_text:
                    logger.warning(f"SA LLM boundary rejected: text mismatch")
                elif len(llm_checked) == src_units_count:
                    # 개수 정확히 일치 - 바로 적용
                    result = llm_checked
                    logger.info(f"✅ LLM 경계 적용: {len(llm_checked)}개 (정확 일치)")
                elif len(llm_checked) < src_units_count:
                    # LLM이 적게 나눔 - 긴 세그먼트를 추가 분할
                    logger.info(f"🔧 LLM 경계 조정: {len(llm_checked)}개 → {src_units_count}개 (추가 분할)")
                    adjusted = _adjust_segments_to_count(
                        llm_checked, src_units_count, mode='split',
                        text=text, src_units=src_units
                    )
                    if adjusted and len(adjusted) == src_units_count:
                        result = adjusted
                        logger.info(f"✅ LLM 경계 적용: 조정 후 {len(adjusted)}개")
                    else:
                        logger.warning(f"⚠️ LLM 경계 조정 실패, 원본 유지")
                elif len(llm_checked) > src_units_count:
                    # LLM이 많이 나눔 - 짧은 세그먼트를 병합
                    logger.info(f"🔧 LLM 경계 조정: {len(llm_checked)}개 → {src_units_count}개 (병합)")
                    adjusted = _adjust_segments_to_count(
                        llm_checked, src_units_count, mode='merge',
                        text=text, src_units=src_units
                    )
                    if adjusted and len(adjusted) == src_units_count:
                        result = adjusted
                        logger.info(f"✅ LLM 경계 적용: 조정 후 {len(adjusted)}개")
                    else:
                        logger.warning(f"⚠️ LLM 경계 조정 실패, 원본 유지")
        except Exception:
            pass

    return result

def _split_tgt_by_src_units_semantic(src_units: List[str], tgt_text: str, min_tokens: int = 1, **kwargs) -> List[str]:
    """원문 단위에 따른 번역문 분할 (고어 패턴 감지 개선)"""
    # 🎯 원본 토큰 보존 (최종 출력용)
    tgt_tokens_original = tgt_text.split()
    
    # 🚨 augmentation 비활성화: 토큰 개수 불일치 문제 해결
    # (한자 괄호 처리가 토큰 개수를 바꾸는 문제 발생)
    tgt_tokens_aug = tgt_tokens_original[:]
    
    N, T = len(src_units), len(tgt_tokens_aug)
    
    if N == 0 or T == 0:
        return [''] * N if N > 0 else []
    
    # 🎯 빠른 처리: 단일 단위면 전체 반환
    if N == 1:
        return [tgt_text]
    
    # 🎯 빠른 처리: 토큰이 단위보다 적으면 1:1 매칭 (원본 사용)
    if T <= N:
        result = []
        for i in range(N):
            if i < len(tgt_tokens_original):
                result.append(tgt_tokens_original[i])
            else:
                result.append("")
        return result
    
    # 🧠 동적 임베더 기반 분할 (순서 보장 모드)
    try:
        # 설정된 임베더/디바이스 가져오기 (환경변수/CLI 옵션 반영)
        embedder_name = kwargs.get('embedder_name', kwargs.get('embedder', 'bge'))
        embedder_device = kwargs.get('embedder_device', kwargs.get('device', os.getenv('CSP_DEVICE', 'cuda')))
        embedder_device_id = kwargs.get('embedder_device_id', None)
        if embedder_device_id is None and embedder_device and embedder_device.lower() == 'cuda':
            embedder_device_id = 0  # 기본 GPU:0

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
            # BGE 등 다른 임베더 사용 - 함수 직접 가져오기 (GPU 디바이스 반영)
            from common.embedders import get_embedder
            compute_embeddings_func = get_embedder(embedder_name, device_id=embedder_device_id)
            logger.debug(f"✅ {embedder_name.upper()} 임베더로 순서 보장 의미 매칭 시작 (device_id={embedder_device_id})")
        
        # 원문 단위별 임베딩
        src_embeddings = compute_embeddings_func(
            src_units, 
            batch_size=batch_size
        )
        
        # 번역문 토큰들의 임베딩 (augmented 사용)
        tgt_embeddings = compute_embeddings_func(
            tgt_tokens_aug, 
            batch_size=batch_size
        )
        
        # 🎯 순서 보장 Dynamic Programming (가중치 파라미터 전달)
        # DP에 원문 단위/텍스트 컨텍스트도 전달(구문 힌트용)
        dp_kwargs = dict(kwargs)
        dp_kwargs.setdefault('src_units', src_units)
        dp_kwargs.setdefault('source_text', ' '.join(src_units))
        dp_kwargs['tgt_tokens_original'] = tgt_tokens_original  # 원본 토큰 전달
        optimal_split = _find_optimal_split_dp_sequential(
            src_embeddings, tgt_embeddings, tgt_tokens_aug, N, T, **dp_kwargs
        )
        
        if optimal_split and len(optimal_split) == N:
            logger.debug(f"✅ {embedder_name.upper()} 순서 보장 분할 성공: {len(optimal_split)}개 단위")
            return optimal_split
        else:
            # DP 실패 - 길이 불일치 경고
            if optimal_split:
                logger.error(f"❌ DP 분할 실패: 기대 {N}개, 실제 {len(optimal_split)}개 → 폴백")
            else:
                logger.warning(f"⚠️ DP 분할 실패 (None 반환) → 폴백")
    
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
                # 🎯 반드시 원본 토큰 사용 (인덱스 범위는 tgt_tokens_aug 기준, 값은 tgt_tokens_original)
                # 주의: T = len(tgt_tokens_aug)이므로, 범위가 tgt_tokens_original을 초과할 수 있음
                # → tgt_tokens_aug와 tgt_tokens_original 개수가 같다고 가정해야 함
                if start_idx >= len(tgt_tokens_original):
                    logger.error(f"🚨 인덱스 오류: start_idx={start_idx}, len(tgt_tokens_original)={len(tgt_tokens_original)}")
                    unit_text = ""
                elif end_idx > len(tgt_tokens_original):
                    # 경계를 초과한 경우 (이론적으로 발생 불가, 하지만 안전성 위해)
                    logger.warning(f"⚠️ 경계 초과: end_idx={end_idx}, len(tgt_tokens_original)={len(tgt_tokens_original)}")
                    unit_text = " ".join(tgt_tokens_original[start_idx:])
                else:
                    unit_text = " ".join(tgt_tokens_original[start_idx:end_idx])
                
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
        
        # 🔥 필수 검증: 반드시 N개 반환 보장
        if len(result) != N:
            logger.error(f"❌ 폴백 분할 실패: 기대 {N}개, 실제 {len(result)}개")
            # 강제 패딩
            if len(result) < N:
                result.extend([''] * (N - len(result)))
            else:
                result = result[:N]
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 고어 패턴 보정 실패: {e}, 긴급 균등 분할 사용")
        # 최후의 폴백: 균등 분할
        return _split_tgt_by_src_units_simple(tgt_text, N)

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
    - 가중치 파라미터(옵션, 기본은 config에서 로드):
      dp_window: 예상 위치 대비 허용 창(정수, 기본 2)
      distance_decay: 거리 감쇠 알파(실수, 기본 0.05)
      boundary_bonus: 경계 보너스(실수, 기본 0.15)
      particle_bonus: 토씨 경계 보너스(실수, 기본 0.2)
      length_penalty: 기대 길이 대비 차이에 대한 패널티 알파(실수, 기본 0.1)
      sim_gamma: 유사도 샤프닝 지수(실수, 기본 1.2)
    """

    # ===== 파라미터 로드 (config 우선, kwargs override, 최종 기본값) =====
    from common.config import get_alignment_params
    cfg_params = get_alignment_params()
    
    dp_window: int = int(kwargs.get('dp_window', cfg_params.get('dp_window', 2)))
    distance_decay_alpha: float = float(kwargs.get('distance_decay', cfg_params.get('distance_decay', 0.05)))
    boundary_bonus: float = float(kwargs.get('boundary_bonus', cfg_params.get('boundary_bonus', 0.15)))
    particle_bonus: float = float(kwargs.get('particle_bonus', cfg_params.get('particle_bonus', 0.2)))
    length_penalty_alpha: float = float(kwargs.get('length_penalty', cfg_params.get('length_penalty', 0.1)))
    sim_gamma: float = float(kwargs.get('sim_gamma', cfg_params.get('sim_gamma', 1.2)))
    similarity_threshold: float = float(kwargs.get('similarity_threshold', cfg_params.get('similarity_threshold', 0.5)))

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
    
    # 🎯 원본 토큰으로도 플래그 생성 (경계 감지는 원본 패턴 사용)
    tgt_tokens_orig = kwargs.get('tgt_tokens_original', tgt_tokens)
    boundary_flags_orig = [_is_boundary_token(t) for t in tgt_tokens_orig]
    particle_flags_orig = [_is_particle_ending(t) for t in tgt_tokens_orig]
    
    # tgt_tokens와 tgt_tokens_orig 크기 같다고 가정 (같아야 함)
    # 원본 기반 플래그 우선 사용, 없으면 augmented 사용
    boundary_flags = boundary_flags_orig if len(boundary_flags_orig) == len(boundary_flags) else boundary_flags
    particle_flags = particle_flags_orig if len(particle_flags_orig) == len(particle_flags) else particle_flags

    # ===== 한글 토씨 기반 경계 강도(가중) 계산: common.korean_particle_matcher 사용 시 더 정교하게 =====
    particle_strengths = [1.0 if f else 0.0 for f in particle_flags]
    try:
        # 토큰별로 토씨를 추출하여 카테고리 가중치를 반영한 강도 계산
        from common.korean_particle_matcher import get_korean_particle_matcher
        _matcher = get_korean_particle_matcher()
        weights = getattr(_matcher, 'particle_weights', {})
        strengths = []
        # 원본 토큰 기준으로 분석(무결성 보존: 읽기만 함)
        tokens_for_analysis = tgt_tokens_orig if len(tgt_tokens_orig) == len(tgt_tokens) else tgt_tokens
        for tok in tokens_for_analysis:
            parts = []
            try:
                parts = _matcher.extract_particles_from_text(tok)
            except Exception:
                parts = []
            if not parts:
                strengths.append(0.0)
                continue
            # 동일 토큰 내 다수 조사 → 가중치 합을 [0,1]로 정규화(최대 1.0)
            s = 0.0
            for (_form, cat, _pos) in parts:
                s += float(weights.get(cat, 0.1))
            # 간단 정규화: 1.0로 클램프
            s = max(0.0, min(1.0, s))
            strengths.append(s)
        if len(strengths) == len(particle_strengths):
            particle_strengths = strengths
    except Exception:
        # 매처 불가 시 기존 휴리스틱 강도 유지(0/1)
        pass

    # ===== 한국어/중국어 구문 힌트 기반 경계 (옵션) - 파싱 게이팅 준비 =====
    comma_bonus = float(kwargs.get('comma_bonus', cfg_params.get('comma_bonus', 0.1)))
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
            # 🎯 원본 토큰으로 문자 오프셋 계산 (augmented 아님!)
            tgt_tokens_for_parsing = kwargs.get('tgt_tokens_original', tgt_tokens)
            token_spans = []
            offset = 0
            joined = "".join(tgt_tokens_for_parsing)
            for tok in tgt_tokens_for_parsing:
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
                valid_similarities = []
                for t in range(k, i):
                    sim = enhanced_similarity[t, j - 1]
                    # similarity threshold 적용: 낮은 매칭 제외
                    if sim >= similarity_threshold:
                        valid_similarities.append(sim)
                
                if valid_similarities:
                    unit_score = sum(valid_similarities) / len(valid_similarities)
                elif token_count > 0:
                    # threshold 미달이지만 일부라도 반영 (페널티)
                    for t in range(k, i):
                        unit_score += enhanced_similarity[t, j - 1]
                    unit_score = unit_score / token_count * 0.5  # 페널티 적용
                if length_penalty_alpha > 0 and N > 0:
                    expected_len = max(1, int(round(T / N)))
                    diff = abs(token_count - expected_len)
                    unit_score -= length_penalty_alpha * diff
                end_tok_idx = i - 1
                if 0 <= end_tok_idx < T:
                    if boundary_bonus != 0.0 and boundary_flags[end_tok_idx]:
                        unit_score += boundary_bonus
                    # 토씨 보너스: 강도 가중치 적용(매처가 없으면 0/1로 동작)
                    if particle_bonus != 0.0:
                        strength = particle_strengths[end_tok_idx]
                        if strength > 0.0:
                            unit_score += particle_bonus * strength
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

    # 🎯 원본 토큰으로 재구성 (augmented는 임베딩 계산용만)
    tgt_tokens_orig = kwargs.get('tgt_tokens_original', None)
    if tgt_tokens_orig is None:
        logger.error("🚨 CRITICAL: tgt_tokens_original이 DP에 전달되지 않았습니다! 원본 텍스트가 손상될 수 있습니다.")
        # 무조건 None 반환 (원본 손상 방지)
        return None
    
    splits = []
    i, j = T, N
    while j > 0:
        start = parent[i][j]
        end = i
        if start >= 0:
            # 반드시 원본 토큰만 사용 (인덱스 범위 확인)
            if start < 0 or end > len(tgt_tokens_orig) or start > end:
                logger.error(f"🚨 인덱스 오류: start={start}, end={end}, len(original)={len(tgt_tokens_orig)}")
                return None
            
            unit_tokens_orig = tgt_tokens_orig[start:end]
            splits.append(" ".join(unit_tokens_orig))
            i = start
            j -= 1
        else:
            break
    splits.reverse()
    return splits if len(splits) == N else None

def _split_tgt_by_src_units_simple(text: str, target_count: int, src_units: List[str] = None, **kwargs) -> List[str]:
    """번역문 폴백 분할: 형태(EC/EF)+의미 임베딩 결합 점수 (어절 경계 보존)"""
    if target_count <= 1:
        return [text.strip()]

    # 어절 단위 spans 추출 (공백 포함 원본 위치 유지)
    import regex as _re
    word_matches = list(_re.finditer(r"\S+", text))
    words = [m.group(0) for m in word_matches]
    if not words:
        return [''] * target_count
    word_starts = [m.start() for m in word_matches]
    word_ends = [m.end() for m in word_matches]
    word_count = len(words)

    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        denom = (np.linalg.norm(u) * np.linalg.norm(v))
        if denom == 0:
            return 0.0
        return float(np.dot(u, v) / denom)

    def _get_embed_func():
        embedder_name = kwargs.get('embedder_name', kwargs.get('embedder', 'bge'))
        embedder_device_id = kwargs.get('embedder_device_id', 0)
        if embedder_name.lower() == 'none':
            return None
        try:
            if embedder_name.lower() == 'openai':
                embedder = OpenAIWrapper(max_workers=kwargs.get('max_workers', 4))
                return embedder.compute_embeddings_with_cache
            else:
                from common.embedders import get_embedder
                return get_embedder(embedder_name, device_id=embedder_device_id)
        except Exception as e:
            logger.debug(f"⚠️ SA 폴백 임베더 초기화 실패: {e}")
            return None

    embed_func = _get_embed_func()

    def _semantic_delta(left: str, right: str) -> float:
        if not embed_func:
            return 0.0
        try:
            embs = embed_func([left, right], batch_size=kwargs.get('batch_size', 100))
            if not embs or len(embs) < 2:
                return 0.0
            a = np.array(embs[0])
            b = np.array(embs[1])
            sim = _cosine(a, b)
            return max(0.0, 1.0 - sim)
        except Exception as e:
            logger.debug(f"⚠️ SA 폴백 의미 스코어 실패: {e}")
            return 0.0

    candidates: List[Tuple[int, str]] = []
    semantic_window_words = kwargs.get('semantic_window_words', 30)
    text_len = len(text)

    # 구두점 기반 후보
    try:
        from sentence_splitter import split_target_sentences_advanced
        cand = split_target_sentences_advanced(text, max_length=400, splitter="punctuation") or []
        offset = 0
        for seg in cand:
            seg = seg or ""
            offset += len(seg)
            if 0 < offset < text_len:
                candidates.append((offset, "PUNC"))
    except Exception as e:
        logger.debug(f"⚠️ SA 구두점 후보 추출 실패: {e}")

    # Kiwi EC/EF 기반 후보
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        analysis = kiwi.analyze(text, top_n=1)
        tokens = analysis[0][0] if analysis and analysis[0] else []
        for tok in tokens:
            tag = getattr(tok, "tag", "") or ""
            if tag.startswith(("EF", "EC")):
                candidates.append((tok.start + tok.len, tag[:2]))
    except Exception as e:
        logger.debug(f"⚠️ SA Kiwi 후보 추출 실패: {e}")

    # 후보 없으면 기존 단순 분배
    if not candidates:
        trans_words = text.split()
        if len(trans_words) == target_count:
            return trans_words
        elif len(trans_words) < target_count:
            return trans_words + [''] * (target_count - len(trans_words))
        else:
            return _distribute_words_evenly(trans_words, target_count)

    # 후보를 어절 경계로 스냅
    def _charpos_to_word_boundary(pos: int) -> Optional[int]:
        # boundary index in [0, word_count]
        for idx, (s, e) in enumerate(zip(word_starts, word_ends)):
            if pos <= s:
                return idx
            if s < pos < e:
                return idx + 1
        return word_count

    unique_candidates = {}
    for pos, tag in candidates:
        if not (0 < pos < text_len):
            continue
        w_idx = _charpos_to_word_boundary(pos)
        if w_idx is None or w_idx <= 0 or w_idx >= word_count:
            continue
        unique_candidates[w_idx] = tag

    if not unique_candidates:
        trans_words = words
        if len(trans_words) == target_count:
            return trans_words
        elif len(trans_words) < target_count:
            return trans_words + [''] * (target_count - len(trans_words))
        else:
            return _distribute_words_evenly(trans_words, target_count)

    scored: List[Tuple[float, int]] = []
    for w_idx, tag in sorted(unique_candidates.items()):
        lidx = max(0, w_idx - semantic_window_words)
        ridx = min(word_count, w_idx + semantic_window_words)
        left = " ".join(words[lidx:w_idx]).strip()
        right = " ".join(words[w_idx:ridx]).strip()
        sem = _semantic_delta(left, right) if left and right else 0.0
        morph = 1.0 if tag.startswith("EF") else 0.6 if tag.startswith("EC") else 0.7
        score = sem + 0.2 * morph
        scored.append((score, w_idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    choose = sorted([w for _, w in scored[:max(0, target_count - 1)]])

    boundaries = [0] + choose + [word_count]
    segments = []
    for i in range(len(boundaries) - 1):
        seg_words = words[boundaries[i]:boundaries[i+1]]
        seg = " ".join(seg_words).strip()
        segments.append(seg)

    if len(segments) < target_count:
        segments += [''] * (target_count - len(segments))
    elif len(segments) > target_count:
        segments = segments[:target_count]

    return segments

def process_single_row(row_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
    단일 행 처리: 원문 분할 + 번역문 정렬 → 여러 행으로 분할
    
    🆕 통합 모드: use_boundary_model=True이면
    - 기존 방식으로 초기 분할
    - 경계 모델로 refinement
    - Alignment 모델로 정렬 개선
    
    Args:
        row_data: {'원문': '원문', '번역문': '번역문', '문장식별자': 'id', ...}
        **kwargs: 처리 옵션
    
    Returns:
        분할된 행 데이터 리스트 (원문 단위별로 개별 행)
    """
    source_text = row_data.get('원문', '')
    translation_text = row_data.get('번역문', '')
    base_id = row_data.get('문장식별자', '')
    
    # 1️⃣ 기존 방식으로 초기 분할
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
    method_label = 'bge_initial'

    # 2️⃣ (옵션) boundary + alignment 모델로 refinement
    use_boundary_model = kwargs.get('use_boundary_model', False)
    
    if use_boundary_model:
        try:
            from sa.io_manager import safe_process_sa_row

            boundary_model = getattr(safe_process_sa_row, '_boundary_model', None)
            alignment_model = getattr(safe_process_sa_row, '_alignment_model', None)
            threshold = float(kwargs.get('boundary_threshold', 0.5))

            if boundary_model is not None:
                # 🆕 Cross-Attention 모델은 (src, tgt) 인자 필요
                try:
                    import inspect
                    sig = inspect.signature(boundary_model.segment_text)
                    if 'src_text' in sig.parameters or len(sig.parameters) >= 2:
                        tgt_units_by_model = boundary_model.segment_text(
                            str(source_text),
                            str(translation_text),
                            n_segments=len(src_units) if len(src_units) > 1 else None, # 🆕 원문이 2개 이상일 때만 힌트 사용
                            threshold=threshold,
                        )
                    else:
                        tgt_units_by_model = boundary_model.segment_text(
                            str(translation_text),
                            task='sa',
                            threshold=threshold,
                        )
                    logger.debug(f"DEBUG: 모델 예측 세그먼트 수: {len(tgt_units_by_model)}")
                except TypeError:
                     tgt_units_by_model = boundary_model.segment_text(
                        str(translation_text),
                        threshold=threshold,
                    )
                     logger.debug(f"DEBUG: 폴백 모델 예측 세그먼트 수: {len(tgt_units_by_model)}")

                # 너무 극단적인 분할(0개/1개)은 이득이 없으므로 스킵
                if tgt_units_by_model and len(tgt_units_by_model) >= 2:
                    # 무결성 검증: 번역문
                    tgt_ok = ''.join(str(translation_text).split()) == ''.join(''.join(tgt_units_by_model).split())
                    
                    if tgt_ok:
                        # alignment 모델이 있으면 사용, 없으면 DP로 원문 재분할
                        if alignment_model is not None:
                            src_units_by_model = alignment_model.match_segments(src_units, tgt_units_by_model)
                        else:
                            # 🆕 Alignment 모델 없이: DP로 원문을 번역문 개수에 맞춰 재분할
                            logger.debug(f"DEBUG: DP 재분할 시도 (Target Count: {len(tgt_units_by_model)})")
                            src_units_by_model = split_tgt_meaning_units(
                                source_text, len(tgt_units_by_model),
                                src_units=[source_text], use_semantic=False, **kwargs
                            )
                            logger.debug(f"DEBUG: DP 재분할 결과 수: {len(src_units_by_model)}")
                        
                        # 원문 무결성 검증
                        src_ok = ''.join(str(source_text).split()) == ''.join(''.join(src_units_by_model).split())
                        len_ok = len(src_units_by_model) == len(tgt_units_by_model)

                        if src_ok and len_ok:
                            # 🆕 LLM 보정 단계 (USE_LLM_BOUNDARY_VERIFY 환경변수가 설정된 경우만)
                            import os
                            if os.getenv("USE_LLM_BOUNDARY_VERIFY"):
                                try:
                                    ref_text = ' '.join(src_units_by_model)
                                    llm_refined = refine_boundaries_with_llm(
                                        str(translation_text),
                                        tgt_units_by_model,
                                        task="sa",
                                        max_segments=30,
                                        reference_text=ref_text,
                                    )
                                    if llm_refined and len(llm_refined) == len(tgt_units_by_model):
                                        # LLM 결과 무결성 검증
                                        llm_text = ''.join(''.join(llm_refined).split())
                                        orig_text = ''.join(str(translation_text).split())
                                        if llm_text == orig_text:
                                            tgt_units_by_model = llm_refined
                                            method_label = method_label + '+llm_refine'
                                            logger.info(f"✅ LLM 경계 보정 적용 (id={base_id})")
                                except Exception as llm_err:
                                    logger.debug(f"LLM 보정 스킵: {llm_err}")
                            
                            src_units = src_units_by_model
                            trans_units = tgt_units_by_model
                            method_label = method_label + '+boundary_tgt'
                            if alignment_model is not None:
                                method_label += '+align_src'
                            else:
                                method_label += '+dp_src'
                        else:
                            logger.warning(
                                f"SA refinement 무결성 실패로 폴백 (id={base_id}, src_ok={src_ok}, len_ok={len_ok}, src_len={len(src_units_by_model)}, tgt_len={len(tgt_units_by_model)})"
                            )
                    else:
                        logger.warning(f"SA boundary tgt 무결성 실패로 폴백 (id={base_id})")
                else:
                    logger.debug(f"DEBUG: 모델 분할 개수 부족 ({len(tgt_units_by_model)})")
        except Exception as e:
            logger.warning(f"SA refinement 실패로 폴백 (id={base_id}): {e}")
            import traceback
            logger.warning(traceback.format_exc())
    
    # 시대 정보 분석
    try:
        period = detect_chinese_period(source_text)  # 전근대 고전 가정
    except Exception as e:
        logger.warning(f"시대 감지 실패: {e}")
        period = 'unknown'
    
    # 무결성 가드레일: 번역문 결합(공백/개행 제거 기준)이 원본과 다르면 안전 폴백으로 재분할
    try:
        original_flat_tgt = ''.join(str(translation_text).split())
        processed_flat_tgt = ''.join(''.join(trans_units).split())

        if original_flat_tgt != processed_flat_tgt:
            diff_chars = len(original_flat_tgt) - len(processed_flat_tgt)
            logger.warning(
                f"⚠️ SA 번역문 무결성 불일치: {diff_chars:+d}자 (id={base_id}) → 단순 분할로 폴백"
            )

            # split_tgt_meaning_units는 내부에서 semantic→simple→evenly 폴백을 하므로,
            # 여기서는 semantic을 끄고 재시도해서 텍스트 보존성을 최우선으로 한다.
            trans_units = split_tgt_meaning_units(
                translation_text,
                len(src_units),
                src_units=src_units,
                use_semantic=False,
                **kwargs,
            )
            method_label = method_label + '+integrity_fallback'
    except Exception as e:
        logger.warning(f"SA 무결성 가드레일 실패(기존 결과 유지): {e}")

    # 각 단위별로 개별 행 생성
    result_rows = []
    
    # 문장식별자: PA 결과의 문장식별자 사용
    sentence_id = row_data.get('문장식별자', 1)
    
    # 원문-번역문 쌍별 유사도 계산
    segment_similarities = []
    try:
        # BGE/임베더 기반 유사도 (기존 로직 유지)
        from common.embedders import get_embedder
        embedder_name = kwargs.get('embedder_name', kwargs.get('embedder', 'bge'))
        embedder_device_id = kwargs.get('embedder_device_id', 0)
        if embedder_name.lower() != 'none':
            if embedder_name.lower() == 'openai':
                embedder = OpenAIWrapper(max_workers=kwargs.get('max_workers', 4))
                compute_embeddings = embedder.compute_embeddings_with_cache
            else:
                compute_embeddings = get_embedder(embedder_name, device_id=embedder_device_id)

            all_texts = src_units + trans_units
            embeddings = compute_embeddings(all_texts, batch_size=kwargs.get('batch_size', 100))
            src_embeddings = embeddings[:len(src_units)]
            tgt_embeddings = embeddings[len(src_units):]

            from sklearn.metrics.pairwise import cosine_similarity
            for src_emb, tgt_emb in zip(src_embeddings, tgt_embeddings):
                sim = float(cosine_similarity([src_emb], [tgt_emb])[0][0])
                segment_similarities.append(sim)
    except Exception as e:
        logger.warning(f"세그먼트 유사도 계산 실패: {e}")
        segment_similarities = [row_data.get('similarity', 1.0)] * len(src_units)
    
    # 각 분할된 구(segment)에 대해
    global _global_segment_id

    for _, (src_unit, trans_unit, sim) in enumerate(zip(src_units, trans_units, segment_similarities), start=1):
        _global_segment_id += 1
        row = {
            '문장식별자': sentence_id,
            '구식별자': _global_segment_id,
            '원문': src_unit,
            '번역문': trans_unit,
            '유사도': sim,
            '분할방법': method_label
        }
        result_rows.append(row)
    
    # 🔥 한글 토씨 매칭 비활성화: 행 수 초과 문제 해결
    # try:
    #     from common.korean_particle_matcher import enhance_sa_results_with_particles
    #     result_rows = enhance_sa_results_with_particles(result_rows)
    #     logger.debug(f"SA 토씨 매칭 보완 완료: {len(result_rows)}개 행")
    # except Exception as e:
    #     logger.warning(f"SA 토씨 매칭 보완 실패 (기존 결과 유지): {e}")
    #     # 실패해도 기존 result_rows 그대로 사용

    # 🔒 SA 무결성 검증: 번역문 텍스트 보존 확인
    # 주의: 마스킹된 상태에서 검증하면 마스크 토큰 길이 차이로 오류가 발생할 수 있으므로
    # 원본 입력과 비교하는 것이 정확함
    try:
        # 입력/출력 모두 공백·개행을 제거해 순수 텍스트만 비교 (개행 손실로 인한 오탐 방지)
        input_trans_for_check = ''.join(translation_text.split())

        processed_joined = ''.join(row['번역문'] for row in result_rows)
        processed_trans = ''.join(processed_joined.split())
        
        # 무결성 검증
        if input_trans_for_check != processed_trans:
            logger.error(f"SA 무결성 실패: {base_id}")
            logger.error(f"  입력 길이: {len(input_trans_for_check)}자")
            logger.error(f"  처리 후 길이: {len(processed_trans)}자")
            logger.error(f"  길이 차이: {len(processed_trans) - len(input_trans_for_check)}자")
            
            # 문자 정확도 계산
            correct_chars = sum(1 for a, b in zip(input_trans_for_check, processed_trans) if a == b)
            accuracy = correct_chars / max(len(input_trans_for_check), len(processed_trans))
            logger.error(f"  문자 정확도: {accuracy:.3f}")
            
            # 차이점 상세 분석 (처음 100자만)
            if len(input_trans_for_check) > 0 and len(processed_trans) > 0:
                import difflib
                diff = list(difflib.unified_diff(
                    input_trans_for_check[:100], processed_trans[:100], 
                    fromfile='입력', tofile='처리후', lineterm=''
                ))
                if diff:
                    logger.error(f"SA 텍스트 차이점 분석: {base_id}")
                    logger.error(f"  입력 샘플: '{input_trans_for_check[:50]}{'...' if len(input_trans_for_check) > 50 else ''}'")
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
    # 비대응 구간 마스킹 ([-(...)]) 후 처리
    masked_src, src_map = _mask_unaligned_segments(src_text)
    masked_trans, trans_map = _mask_unaligned_segments(translation)

    # 원문 분할 (공백 단위)
    src_units = split_src_meaning_units(masked_src, **kwargs)
    
    # 번역문 정렬
    trans_units = align_translation_to_source(src_units, masked_trans, **kwargs)

    # 마스크 복원
    src_units = _unmask_list(src_units, src_map)
    trans_units = _unmask_list(trans_units, trans_map)
    
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
    
    # 🎯 유사도 계산 추가 (PA와 동일하게)
    similarities = []
    try:
        embedder_name = kwargs.get('embedder_name', kwargs.get('embedder', 'bge'))
        if embedder_name and embedder_name.lower() != 'none':
            from common.embedders import get_embedder
            embedder_device_id = kwargs.get('embedder_device_id', 0)
            compute_embeddings_func = get_embedder(embedder_name, device_id=embedder_device_id)
            batch_size = kwargs.get('batch_size', 100)
            
            for src_unit, tgt_unit in zip(src_units, trans_units):
                if src_unit.strip() and tgt_unit.strip():
                    try:
                        src_emb = compute_embeddings_func([src_unit], batch_size=batch_size)[0]
                        tgt_emb = compute_embeddings_func([tgt_unit], batch_size=batch_size)[0]
                        sim = float(np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-8))
                        similarities.append(sim)
                    except:
                        similarities.append(0.0)
                else:
                    similarities.append(0.0)
        else:
            similarities = [0.0] * len(src_units)
    except Exception as e:
        logger.warning(f"유사도 계산 실패: {e}, 0.0으로 채움")
        similarities = [0.0] * len(src_units)
    
    result = {
        'source_units': src_units,
        'translation_units': trans_units,
        'similarities': similarities,  # 유사도 추가
        'metadata': metadata
    }
    
    logger.debug(f"SA 처리 완료: {metadata['source_count']}개 단위, 시대: {metadata['detected_period']}")
    
    return result


# === 번역문 괄호 한자 적극 반영 ===
# Python's re doesn't support Unicode property escapes like \p{Han}.
# Use explicit CJK ranges (Unified Ideographs + Extension A + Compatibility). 
# This covers common Han characters used in texts.
_han_regex = re.compile(r"\p{Han}")

def _augment_translation_with_hanja_parentheses(text: str) -> str:
    """번역문에서 괄호 속 한자를 의미 매칭에 반영 (토큰 개수 유지).
    예: '태사공(太史公)은' → '太史公태사공은' (하나의 토큰 유지)
    """
    # 🎯 토큰 단위로 처리하여 원본과 augmented의 토큰 개수 일치 보장
    tokens = text.split()
    augmented_tokens = []
    
    for token in tokens:
        # 소괄호와 대괄호에서 한자 추출
        hanja_parts = []
        # 소괄호
        for m in re.finditer(r"\(([^)]*)\)", token):
            inner = m.group(1)
            hanja = ''.join(ch for ch in inner if _han_regex.match(ch))
            if hanja:
                hanja_parts.append(hanja)
        # 대괄호
        for m in re.finditer(r"\[([^\]]*)\]", token):
            inner = m.group(1)
            hanja = ''.join(ch for ch in inner if _han_regex.match(ch))
            if hanja:
                hanja_parts.append(hanja)
        
        # 괄호 제거하고 한자를 토큰 앞에 붙임
        token_clean = re.sub(r"\([^)]*\)", "", token)
        token_clean = re.sub(r"\[[^\]]*\]", "", token_clean)
        
        if hanja_parts:
            # 한자를 앞에 배치 (임베딩 시 더 강조됨)
            augmented_token = ''.join(hanja_parts) + token_clean
        else:
            augmented_token = token_clean
        
        augmented_tokens.append(augmented_token)
    
    return ' '.join(augmented_tokens)


def _try_split_by_korean_particles(tgt_text: str, N: int) -> Optional[List[str]]:
    """
    한글 번역문을 조사/어미 경계에서 분할 (현재 비활성화 - 실행 단계에서 더 개선 필요)
    """
    # 현재는 비활성화 - 추후 개선 예정
    return None

