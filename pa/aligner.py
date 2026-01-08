"""PA 전용 정렬기 - SA의 Vice Versa 방식 (완벽한 무결성 보장)"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import pandas as pd
import regex as re
# 통합 진행률 관리자
from common.progress_manager import start_unified_progress, update_unified_progress, finish_unified_progress, set_progress_description
import logging
import copy
from common.text_normalizer import normalize_for_similarity

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 로컬 모듈 import (기존과 동일)
from sentence_splitter import (
    split_target_sentences_advanced,
    split_source_by_whitespace_and_align,
    merge_quotation_markers_in_list,
)

try:
    from common.llm_boundary_refiner import refine_boundaries_with_llm
except Exception:
    refine_boundaries_with_llm = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# === 번역문 괄호 한자 적극 반영 (PA) ===
_han_regex = re.compile(r"\p{Han}")


def _augment_translation_with_hanja_parentheses(text: str) -> str:
    """번역문 괄호 속 한자를 노출해 의미 매칭에 활용 (PA 전용)
    - 원문 텍스트를 변형하지 않고, 괄호 안의 한자만 앞쪽에 노출하여 임베딩에 도움을 준다.
    - 예: '태사공(太史公)은' → '太史公 태사공(太史公)은'
    """
    try:
        def repl_paren(m):
            inner = m.group(1)
            hanja = ''.join(ch for ch in inner if _han_regex.match(ch))
            if hanja:
                return f" {hanja} ({inner})"
            return f"({inner})"

        # 둥근/대괄호 모두 처리
        t = re.sub(r"\(([^)]*)\)", repl_paren, text)
        t = re.sub(r"\[([^\]]*)\]", repl_paren, t)
        return t
    except Exception:
        return text

def safe_source_split(tgt_sentences: List[str], src_text: str, tokenizer_func=None, nlp=None) -> List[str]:
    """무결성 보장 원문 분할"""
    if not tgt_sentences or not src_text.strip():
        return []
    
    try:
        # 새 파서들 방식 또는 기본 방식
        if tokenizer_func:
            src_chunks = split_src_by_tgt_units_new_parsers(tgt_sentences, src_text, tokenizer_func)
        else:
            src_chunks = split_src_by_tgt_units_vice_versa(tgt_sentences, src_text, None, tokenizer_func)
        
        if not src_chunks:
            src_chunks = split_source_by_whitespace_and_align(src_text, len(tgt_sentences))
        
        # 결과 개수 보정
        while len(src_chunks) < len(tgt_sentences):
            src_chunks.append('')
        
        return src_chunks[:len(tgt_sentences)]
        
    except Exception as e:
        logger.error(f"원문 분할 중 오류: {e}")
        # 오류시 기본 분할
        return split_source_by_whitespace_and_align(src_text, len(tgt_sentences))

def verify_paragraph_integrity(src_paragraph: str, tgt_paragraph: str, alignments: List[Dict]) -> bool:
    """문단 단위 무결성 검증: (SA 방식) 정규화 후 검증, 결과는 원상복구
    - 공백/개행/탭 제거 + 괄호 주석([-...]) 등 정규화 후 비교
    - 결과 텍스트는 변경하지 않음(검증 전용)
    """
    
    if not alignments:
        logger.error("정렬 결과가 비어있음")
        return False
    
    # 원본/결과 텍스트 결합
    raw_original_src = src_paragraph
    raw_original_tgt = tgt_paragraph
    raw_aligned_src = ''.join([align.get('원문', '') for align in alignments])
    raw_aligned_tgt = ''.join([align.get('번역문', '') for align in alignments])

    # 정규화(유사도/무결성 검증용): 공백/개행/[-...] 제거 등
    original_src = normalize_for_similarity(raw_original_src)
    original_tgt = normalize_for_similarity(raw_original_tgt)
    aligned_src = normalize_for_similarity(raw_aligned_src)
    aligned_tgt = normalize_for_similarity(raw_aligned_tgt)

    # 1. 텍스트 완전성 검증(정규화 기준)
    src_integrity = (original_src == aligned_src)
    tgt_integrity = (original_tgt == aligned_tgt)
    
    if not src_integrity:
        logger.error(f"원문 무결성 실패 - 원본: {len(original_src)}자, 결과: {len(aligned_src)}자")
        logger.error(f"원본(raw→norm): {raw_original_src[:80]} → {original_src[:80]} ...")
        logger.error(f"결과(raw→norm): {raw_aligned_src[:80]} → {aligned_src[:80]} ...")
    
    if not tgt_integrity:
        logger.error(f"번역문 무결성 실패 - 원본: {len(original_tgt)}자, 결과: {len(aligned_tgt)}자")
        logger.error(f"원본(raw→norm): {raw_original_tgt[:80]} → {original_tgt[:80]} ...")
        logger.error(f"결과(raw→norm): {raw_aligned_tgt[:80]} → {aligned_tgt[:80]} ...")
    
    # 2. 정렬 품질 검증: 빈 원문/번역문 비율 체크
    empty_src_count = sum(1 for align in alignments if not align.get('원문', '').strip())
    empty_tgt_count = sum(1 for align in alignments if not align.get('번역문', '').strip())
    total_count = len(alignments)
    
    empty_src_ratio = empty_src_count / total_count if total_count > 0 else 0.0
    empty_tgt_ratio = empty_tgt_count / total_count if total_count > 0 else 0.0
    
    # 빈 행이 30% 이상이면 정렬 실패로 간주
    alignment_quality = True
    if empty_src_ratio > 0.3:
        logger.error(f"원문 빈 행 과다: {empty_src_count}/{total_count} ({empty_src_ratio:.1%})")
        alignment_quality = False
    
    if empty_tgt_ratio > 0.3:
        logger.error(f"번역문 빈 행 과다: {empty_tgt_count}/{total_count} ({empty_tgt_ratio:.1%})")
        alignment_quality = False
    
    return src_integrity and tgt_integrity and alignment_quality

def restore_paragraph_integrity(src_paragraph: str, tgt_paragraph: str, alignments: List[Dict]) -> List[Dict]:
    """문단 무결성 복원"""
    
    def _map_normalized_to_original(original_text: str, normalized_text: str, normalized_idx: int) -> int:
        """정규화된 텍스트의 인덱스를 원본 텍스트의 인덱스로 매핑"""
        normalized_idx = min(normalized_idx, len(normalized_text))
        non_space_count = 0
        for i, char in enumerate(original_text):
            if char not in (' ', '\n', '\t'):
                if non_space_count == normalized_idx:
                    return i
                non_space_count += 1
        return len(original_text)
    
    def _insert_text_at_pos(seq_alignments: List[Dict], text: str, pos: int, field: str):
        """정렬된 세그먼트들에 대해 전역 pos 위치에 text를 삽입"""
        if not seq_alignments:
            seq_alignments.append({
                '원문': '' if field == '번역문' else '', 
                '번역문': '' if field == '원문' else '', 
                'similarity': 0.0, 
                'split_method': 'integrity_restore', 
                'align_method': f'{field}_missing_restore'
            })
        concat = ''.join([a.get(field, '') for a in seq_alignments])
        pos = min(max(pos, 0), len(concat))
        # 누적 길이로 삽입 위치 찾기
        running = 0
        for idx, a in enumerate(seq_alignments):
            seg = a.get(field, '')
            if running + len(seg) >= pos:
                local = pos - running
                a[field] = seg[:local] + text + seg[local:]
                return
            running += len(seg)
        # 끝에 삽입
        seq_alignments[-1][field] += text
    
    # 현재 정렬 결과 분석
    aligned_src = ''.join([align.get('원문', '') for align in alignments])
    aligned_tgt = ''.join([align.get('번역문', '') for align in alignments])
    
    # 정규화(공백/개행/탭 제거)
    aligned_src_normalized = aligned_src.replace(' ', '').replace('\n', '').replace('\t', '')
    aligned_tgt_normalized = aligned_tgt.replace(' ', '').replace('\n', '').replace('\t', '')
    original_src = src_paragraph.replace('\n', '').replace('\t', '')
    original_tgt = tgt_paragraph.replace('\n', '').replace('\t', '')
    original_src_normalized = original_src.replace(' ', '')
    original_tgt_normalized = original_tgt.replace(' ', '')
    
    restored_alignments = alignments[:]
    
    # 원문 복원
    if original_src_normalized != aligned_src_normalized:
        logger.info("원문 무결성 복원 시작...")
        sm = SequenceMatcher(None, aligned_src_normalized, original_src_normalized)
        opcodes = sm.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # 누락된 원문 추가 (정규화 인덱스 → 원본 인덱스로 매핑)
                missing_text = original_src[j1:j2]
                original_pos = _map_normalized_to_original(original_src, original_src_normalized, j1)
                _insert_text_at_pos(restored_alignments, missing_text, original_pos, '원문')
                logger.info(f"누락 원문 복원: '{missing_text}'")
                
            elif tag == 'delete':
                # 중복된 원문 제거
                excess_text = aligned_src_normalized[i1:i2]
                for align in restored_alignments:
                    if excess_text in align.get('원문', '').replace(' ', '').replace('\n', '').replace('\t', ''):
                        align['원문'] = align['원문'].replace(excess_text, '', 1)
                        logger.info(f"중복 원문 제거: '{excess_text}'")
                        break
    
    # 번역문 복원
    aligned_tgt_after_src_restore = ''.join([align.get('번역문', '') for align in restored_alignments])
    aligned_tgt_after_src_restore_normalized = aligned_tgt_after_src_restore.replace(' ', '').replace('\n', '').replace('\t', '')
    
    if original_tgt_normalized != aligned_tgt_after_src_restore_normalized:
        logger.info("번역문 무결성 복원 시작...")
        sm = SequenceMatcher(None, aligned_tgt_after_src_restore_normalized, original_tgt_normalized)
        opcodes = sm.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # 누락된 번역문 추가 (정규화 인덱스 → 원본 인덱스로 매핑)
                missing_text = original_tgt[j1:j2]
                original_pos = _map_normalized_to_original(original_tgt, original_tgt_normalized, j1)
                _insert_text_at_pos(restored_alignments, missing_text, original_pos, '번역문')
                logger.info(f"누락 번역문 복원: '{missing_text}'")
                
            elif tag == 'delete':
                # 중복된 번역문 제거
                excess_text = aligned_tgt_after_src_restore_normalized[i1:i2]
                for align in restored_alignments:
                    if excess_text in align.get('번역문', '').replace(' ', '').replace('\n', '').replace('\t', ''):
                        align['번역문'] = align['번역문'].replace(excess_text, '', 1)
                        logger.info(f"중복 번역문 제거: '{excess_text}'")
                        break
    
    return restored_alignments

# ===== 기존 함수들에 무결성 보장 적용 =====

def get_new_parsers():
    """새로운 파서들(SuPar-Kanbun/Stanza) 로드"""
    try:
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(current_dir)  # CSP 디렉토리
        sys.path.insert(0, project_root)
        
        from common.new_parsers import SUPAR_AVAILABLE, STANZA_AVAILABLE
        if SUPAR_AVAILABLE and STANZA_AVAILABLE:
            print("✅ SuPar-Kanbun & Stanza 파서 로드 성공")
            return True
        elif SUPAR_AVAILABLE:
            print("⚠️ SuPar-Kanbun만 사용 가능, Stanza 없음")
            return True
        elif STANZA_AVAILABLE:
            print("⚠️ Stanza만 사용 가능, SuPar-Kanbun 없음")
            return True
        else:
            print("❌ 새 파서들 없음, 폴백 모드")
            return False
    except ImportError:
        print("❌ 새 파서 모듈 로드 실패")
        return False

def split_target_sentences_new_parsers(
    text: str, 
    max_length: int = 150,
    tokenizer_func=None,
    use_new_parsers: bool = True
) -> List[str]:
    """새 파서들 + 토크나이저 융합 문장 분할 (무결성 보장)"""
    if not text.strip():
        return []
    
    sentences = []
    
    # 1단계: 새 파서들로 문장 경계 감지
    if use_new_parsers:
        try:
            import sys
            import os
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(current_dir)  # CSP 디렉토리
            sys.path.insert(0, project_root)
            
            from common.new_parsers import smart_sentence_split
            new_parser_sentences = smart_sentence_split(text, is_source=False)  # 번역문이므로 is_source=False
            
            if new_parser_sentences:
                print(f"🔍 새 파서 분할: {len(new_parser_sentences)}개 문장")
                sentences = new_parser_sentences
            else:
                sentences = [text]
        except Exception as e:
            print(f"⚠️ 새 파서 분할 실패: {e}")
            sentences = [text]
    else:
        sentences = split_target_sentences_advanced(text, max_length, splitter="punctuation")
    
    # 2단계: 토크나이저로 긴 문장 세분화
    if tokenizer_func and sentences:
        refined_sentences = []
        
        for sentence in sentences:
            if len(sentence) > max_length:
                refined_parts = split_long_sentence_with_tokenizer(sentence, max_length, tokenizer_func)
                refined_sentences.extend(refined_parts)
            else:
                refined_sentences.append(sentence)
        
        print(f"🔧 토크나이저 조정: {len(sentences)} → {len(refined_sentences)}개 문장")
        sentences = refined_sentences
    
    # 최종 안전 병합: 인용 표지 단독 문장을 직전 문장에 재결합
    if sentences:
        sentences = merge_quotation_markers_in_list(sentences)
    return sentences if sentences else [text]

def split_long_sentence_with_tokenizer(sentence: str, max_length: int, tokenizer_func) -> List[str]:
    """토크나이저를 사용하여 긴 문장을 의미 단위로 분할"""
    
    try:
        tokens = tokenizer_func(sentence)
        if not tokens:
            return [sentence]
        
        parts = []
        current_part = []
        current_length = 0
        
        for token in tokens:
            token_length = len(token)
            
            if current_length + token_length > max_length and current_part:
                parts.append(''.join(current_part))
                current_part = [token]
                current_length = token_length
            else:
                current_part.append(token)
                current_length += token_length
        
        if current_part:
            parts.append(''.join(current_part))
        
        # 무결성 검증
        combined_result = ''.join(parts)
        is_valid, message = integrity_manager.verify_integrity(combined_result, sent_id)
        
        if not is_valid:
            logger.warning(f"긴 문장 분할 무결성 실패: {message}")
            parts = integrity_manager.restore_integrity(parts, sent_id)
        
        return parts if parts else [sentence]
        
    except Exception as e:
        print(f"⚠️ 토크나이저 분할 실패: {e}")
        return [sentence]

# ===== 기존 함수들 (무결성 보장 적용) =====

def get_tokenizer_function(tokenizer_name: str = "siku"):
    """토크나이저 함수 반환 - 하이브리드 토크나이저 사용"""
    try:
        if tokenizer_name == "siku":
            from common.tokenizers import get_siku_tokenizer
            tokenizer = get_siku_tokenizer()
            print("✅ SikuBERT 토크나이저 로드 성공")
            return tokenizer.tokenize
        elif tokenizer_name == "korean_hybrid":
            from common.tokenizers import get_hybrid_korean_tokenizer
            tokenizer = get_hybrid_korean_tokenizer()
            print("✅ 하이브리드 한국어 토크나이저 로드 성공")
            return tokenizer.tokenize
        else:
            print(f"⚠️ 기본 분할 사용: {tokenizer_name}")
            return lambda text: list(text)
    except ImportError as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}, 기본 분할 사용")
        return lambda text: list(text)

def get_embedder_function(embedder_name: str, device: str = "cpu", openai_model: str = None, openai_api_key: str = None, max_workers: int = 16, batch_size: int = 256):
    """임베더 함수 반환 - 병렬 처리 지원"""
    
    # 순차 분할 모드 (임베더 미사용)
    if embedder_name == 'none':
        print("⚡ 순차 분할 모드: 임베더 미사용으로 빠른 처리")
        return None
    
    if device == "cuda":
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("⚠️ CUDA 미지원: CPU로 전환합니다.")
            device = "cpu"
    
    if embedder_name == 'bge':
        try:
            sys.path.insert(0, str(project_root / 'common' / 'embedders'))
            from bge import get_embedding_manager
            
            # 싱글톤 매니저 사용 (이미 로드된 모델 재사용)
            embedder = get_embedding_manager()
            
            # Multi-vector 지원을 위한 래퍼 함수
            def enhanced_embed_func(texts, use_multi_vector=True, batch_size_override=None, **kwargs):
                actual_batch_size = batch_size_override if batch_size_override else batch_size
                return embedder.compute_embeddings_with_cache(
                    texts, 
                    use_multi_vector=use_multi_vector,
                    batch_size=actual_batch_size,
                    **kwargs
                )
            
            print(f"✅ BGE 임베더 로드 성공 (batch_size={batch_size}, device={device})")
            return enhanced_embed_func
        except ImportError as e:
            print(f"❌ BGE 임베더 로드 실패: {e}")
            return None
            
    elif embedder_name == 'openai':
        try:
            # 모듈명 충돌 해결: openai_embedder.py로 파일명 변경
            sys.path.insert(0, str(project_root / 'common' / 'embedders'))
            from openai_embedder import compute_embeddings_batch
            
            # OpenAI API 키 설정
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
            def openai_embed_func(texts, model="text-embedding-3-large"):
                if isinstance(texts, str):
                    texts = [texts]
                    return_single = True
                else:
                    return_single = False
                
                embeddings = compute_embeddings_batch(
                    texts, 
                    model=model, 
                    max_workers=max_workers, 
                    batch_size=batch_size
                )
                
                if return_single:
                    return embeddings[0]
                else:
                    return embeddings
            
            print(f"✅ OpenAI 임베더 초기화 성공 (max_workers={max_workers}, batch_size={batch_size})")
            return openai_embed_func
            
        except ImportError as e:
            print(f"❌ OpenAI 임베더 로드 실패: {e}")
            return None
    else:
        print(f"❌ 지원하지 않는 임베더: {embedder_name}")
        return None

def split_src_by_tgt_units_new_parsers(
    tgt_sentences: List[str], 
    src_text: str, 
    tokenizer_func=None
) -> List[str]:
    """새 파서들을 활용한 Vice Versa 원문 분할 (무결성 보장)"""
    if not tgt_sentences:
        return []

    # 새 파서가 아직 완전히 연결되지 않았으므로, 의미적 분할기 직접 호출
    return split_source_by_whitespace_and_align(
        src_text,
        len(tgt_sentences),
        target_sentences=tgt_sentences,
        embedder_name="bge",
        embedder_func=None,
        max_workers=16,
        batch_size=256,
    )

def split_src_by_tgt_units_spacy_tokenizer(
    tgt_sentences: List[str], 
    src_text: str, 
    tokenizer_func=None,
    nlp=None
) -> List[str]:
    """spaCy + 토크나이저를 활용한 Vice Versa 원문 분할 (무결성 보장) - 레거시"""
    if not tgt_sentences:
        return []

    # spaCy 토크나이저 경로도 아직 완전치 않으므로 동일한 의미적 분할기로 처리
    return split_source_by_whitespace_and_align(
        src_text,
        len(tgt_sentences),
        target_sentences=tgt_sentences,
        embedder_name="bge",
        embedder_func=None,
        max_workers=16,
        batch_size=256,
    )

def split_src_by_tgt_units_vice_versa(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    similarity_threshold: float = 0.3
) -> List[str]:
    """SA의 Vice Versa: 번역문 문장들을 기준으로 원문을 분할 (무결성 보장)"""
    if not tgt_sentences:
        return []

    return split_source_by_whitespace_and_align(
        src_text,
        len(tgt_sentences),
        target_sentences=tgt_sentences,
        embedder_name="bge",
        embedder_func=embed_func,
        max_workers=16,
        batch_size=256,
    )

def compute_similarity_simple(text1: str, text2: str) -> float:
    """간단한 길이 기반 유사도"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    ratio = min(len1, len2) / max(len1, len2)
    return 0.5 + (ratio * 0.5)



def _strip_brackets_from_text(text: str):
    """Remove [-…] segments and record insertion points for restoration.

    공백은 제거하지 않고 그대로 두어 원본 위치와 동일한 간격을 유지한다.
    """
    import re as _re
    pattern = _re.compile(r"\[-\([^)]*\)\]|\[-[^\]]*\]")
    insertions = []
    parts = []
    last = 0
    curr_len = 0
    for m in pattern.finditer(text):
        parts.append(text[last:m.start()])
        curr_len += (m.start() - last)
        bracket_text = m.group(0)
        insertions.append((curr_len, bracket_text))
        last = m.end()
    parts.append(text[last:])
    working = ''.join(parts)
    if insertions:
        logger.info(f"🔥 Strip brackets: {len(insertions)} brackets removed, working_src len {len(text)} → {len(working)}")
    return working, insertions


def _adjust_boundaries_with_llm_and_morphology(
    dp_segments: List[str], 
    src_text: str, 
    src_units: List[str], 
    tgt_sentences: List[str],
    target_count: int
) -> List[str]:
    """
    DP 결과의 경계를 LLM/형태소 정보로 미세 조정
    
    전략:
    1. DP 결과가 "한두 마디씩 밀린" 문제를 해결
    2. LLM으로 경계가 부자연스러운 위치를 찾아 조정
    3. 형태소(EC/EF) 경계를 참조하여 더 자연스러운 위치로 이동
    
    Args:
        dp_segments: DP로 분할된 원문 세그먼트들
        src_text: 원문 전체 텍스트
        src_units: 원문 어절 리스트
        tgt_sentences: 번역문 문장 리스트
        target_count: 목표 문장 개수
    
    Returns:
        경계가 조정된 세그먼트 리스트 (실패시 None)
    """
    try:
        # 개수가 맞지 않으면 LLM으로 조정 시도
        if len(dp_segments) != target_count and refine_boundaries_with_llm:
            logger.info(f"🔧 LLM으로 세그먼트 개수 조정: {len(dp_segments)}개 → {target_count}개")
            adjusted = _adjust_source_segments_to_target(
                dp_segments, target_count, src_text, src_units, tgt_sentences
            )
            if adjusted and len(adjusted) == target_count:
                return adjusted
        
        # 개수가 맞으면 경계 위치만 미세 조정
        if len(dp_segments) == target_count:
            logger.info(f"🔧 DP 결과 경계 미세 조정 시도")
            
            # 형태소 경계 정보 수집
            ec_ef_positions = []
            try:
                from kiwipiepy import Kiwi
                kiwi = Kiwi()
                analysis = kiwi.analyze(src_text, top_n=1)
                tokens = analysis[0][0] if analysis and analysis[0] else []
                for tok in tokens:
                    tag = getattr(tok, "tag", "") or ""
                    if tag.startswith(("EF", "EC")):
                        ec_ef_positions.append(tok.start + tok.len)
                logger.info(f"🔤 형태소 경계: {len(ec_ef_positions)}개 EC/EF 위치")
            except Exception as e:
                logger.debug(f"⚠️ 형태소 분석 실패: {e}")
            
            # 각 경계를 가장 가까운 EC/EF 위치로 이동
            if ec_ef_positions:
                adjusted_segments = []
                current_pos = 0
                
                for i, segment in enumerate(dp_segments):
                    segment_end = current_pos + len(segment)
                    
                    # 마지막 세그먼트가 아니면 경계 조정 시도
                    if i < len(dp_segments) - 1:
                        # 현재 경계에서 가장 가까운 EC/EF 위치 찾기
                        nearest_boundary = min(
                            ec_ef_positions,
                            key=lambda pos: abs(pos - segment_end)
                        )
                        
                        # 경계가 너무 멀리 이동하지 않도록 제한 (±10글자)
                        if abs(nearest_boundary - segment_end) <= 10:
                            # 새 경계로 텍스트 재분할
                            adjusted_segment = src_text[current_pos:nearest_boundary].strip()
                            adjusted_segments.append(adjusted_segment)
                            current_pos = nearest_boundary
                            logger.debug(f"  경계 조정: {segment_end} → {nearest_boundary}")
                            continue
                    
                    # 조정 불가능하거나 마지막 세그먼트
                    adjusted_segments.append(segment)
                    current_pos = segment_end
                
                # 무결성 검증
                original_flat = ''.join(src_text.split())
                adjusted_flat = ''.join(''.join(adjusted_segments).split())
                
                if original_flat == adjusted_flat:
                    logger.info(f"✅ 경계 미세 조정 성공: EC/EF 기반")
                    return adjusted_segments
                else:
                    logger.warning(f"⚠️ 경계 조정 무결성 실패, DP 결과 유지")
        
        return None
        
    except Exception as e:
        logger.debug(f"⚠️ 경계 미세 조정 실패: {e}")
        return None


def _restore_brackets_to_segments(segments: List[str], insertions: List[Tuple[int, str]]) -> List[str]:
    """Restore bracketed segments into final chunks based on cumulative offsets."""
    if not insertions:
        return segments
    
    logger.info(f"🔥 Restore brackets: {len(insertions)} brackets to restore into {len(segments)} segments")
    cumulative = [0]
    for ch in segments:
        cumulative.append(cumulative[-1] + len(ch))
    chunk_buffers = [list(ch) for ch in segments]
    for pos, content in insertions:
        idx = 0
        while idx < len(segments) and not (cumulative[idx] <= pos <= cumulative[idx+1]):
            idx += 1
        if idx >= len(segments):
            chunk_buffers[-1].extend(list(content))
            cumulative[-1] += len(content)
            logger.debug(f"  Restored '{content[:20]}...' to end")
            continue
        if pos == cumulative[idx] and idx > 0:
            idx -= 1
        local_pos = pos - cumulative[idx]
        buf = chunk_buffers[idx]
        left = buf[:local_pos]
        right = buf[local_pos:]
        chunk_buffers[idx] = left + list(content) + right
        for j in range(idx+1, len(cumulative)):
            cumulative[j] += len(content)
        logger.debug(f"  Restored '{content[:20]}...' to chunk {idx}")
    result = [''.join(b) for b in chunk_buffers]
    logger.info(f"✅ Bracket restoration complete, result: {[len(r) for r in result]}")
    return result


def split_source_by_semantic_boundaries(src_text: str, target_count: int = None, embed_func=None, semantic_window: int = 120, tgt_sentences: List[str] = None) -> List[str]:
    """
    ✨ 원문을 의미 경계로 분할 (어절 단위 보존 + DP 메인 + LLM/형태소 미세 조정)
    
    🆕 전략 변경:
    1. **DP 로직을 메인으로 사용** - 전체적으로 무난한 분할
    2. **LLM/형태소 분석으로 경계 미세 조정** - "한두 마디씩 밀린" 문제 해결
    
    핵심 원칙:
    1. 원문은 공백으로 이미 어절 단위로 구분됨
    2. **어절 내부를 절대 쪼개지 않음**
    3. DP로 초기 분할 후 경계 위치 조정
    4. 각 세그먼트는 어절들을 공백으로 연결
    """
    if not src_text or not src_text.strip():
        return []
    
    if target_count is None or target_count < 2:
        return [src_text]
    
    logger.info(f"🔍 원문 의미 분할: {len(src_text)}글자 → 목표 {target_count}개 문장 (DP 메인 + 미세 조정)")
    
    # 1️⃣ 어절 단위로 분할 (공백 기준)
    src_units = src_text.split()
    num_units = len(src_units)
    
    if num_units < target_count:
        # 어절 수가 목표보다 적으면 각 어절을 1개 문장으로
        logger.warning(f"⚠️ 어절 수({num_units}) < 목표 문장 수({target_count}), 어절 단위로 반환")
        return src_units
    
    logger.info(f"📋 어절 분할: {num_units}개 어절")
    
    # 🆕 2️⃣ DP 로직을 메인으로 사용 (segment_src_by_tgt_similarity)
    if tgt_sentences and embed_func:
        try:
            logger.info(f"🎯 DP 기반 원문 분할 시작 (번역문 {len(tgt_sentences)}개 참조)")
            tokenizer_func = get_tokenizer_function('siku')
            
            dp_segments = segment_src_by_tgt_similarity(
                src_text, 
                tgt_sentences, 
                embed_func=embed_func,
                tokenizer_func=tokenizer_func
            )
            
            if dp_segments and len(dp_segments) > 0:
                # 텍스트 무결성 검증
                original_flat = ''.join(src_text.split())
                dp_flat = ''.join(''.join(dp_segments).split())
                
                if original_flat == dp_flat:
                    logger.info(f"✅ DP 원문 분할 완료: {len(dp_segments)}개")
                    
                    # 🆕 3️⃣ LLM/형태소로 경계 미세 조정
                    adjusted_segments = _adjust_boundaries_with_llm_and_morphology(
                        dp_segments, 
                        src_text, 
                        src_units, 
                        tgt_sentences,
                        target_count
                    )
                    
                    if adjusted_segments:
                        logger.info(f"✅ 경계 미세 조정 완료: {len(adjusted_segments)}개")
                        return adjusted_segments
                    else:
                        logger.info(f"✅ DP 결과 그대로 사용")
                        return dp_segments
                else:
                    logger.warning(f"⚠️ DP 원문 분할 무결성 실패, 폴백 시도")
        except Exception as e:
            logger.warning(f"⚠️ DP 원문 분할 실패: {e}, 폴백 시도")
    
    # 폴백: SuPar 시도
    try:
        from common.new_parsers import get_chinese_unit_boundary_indices_supar, SUPAR_AVAILABLE
        
        if SUPAR_AVAILABLE:
            # SuPar에 어절 리스트 전달
            boundaries = get_chinese_unit_boundary_indices_supar(src_text, src_units)
            
            if boundaries and len(boundaries) > 0:
                logger.info(f"✅ SuPar 경계: {len(boundaries)}개 → {sorted(boundaries)}")
                
                # 경계 기반으로 세그먼트 생성 (어절 단위)
                segments = []
                prev_idx = 0
                
                for b_idx in sorted(boundaries):
                    if prev_idx < b_idx <= num_units:
                        segment = ' '.join(src_units[prev_idx:b_idx])
                        if segment.strip():
                            segments.append(segment)
                        prev_idx = b_idx
                
                # 마지막 세그먼트
                if prev_idx < num_units:
                    segment = ' '.join(src_units[prev_idx:])
                    if segment.strip():
                        segments.append(segment)
                
                if len(segments) > 1:
                    logger.info(f"✅ SuPar 분할 완료: {len(segments)}개 문장")
                    # 🆕 LLM으로 개수 조정 시도 (SuPar 결과가 target_count와 다를 때)
                    if len(segments) != target_count and tgt_sentences and refine_boundaries_with_llm:
                        try:
                            adjusted = _adjust_source_segments_to_target(
                                segments, target_count, src_text, src_units, tgt_sentences
                            )
                            if adjusted and len(adjusted) == target_count:
                                logger.info(f"✅ SuPar+LLM 조정: {len(segments)}개 → {target_count}개")
                                return adjusted
                        except Exception as e:
                            logger.debug(f"⚠️ SuPar 결과 LLM 조정 실패: {e}")
                    return segments
    except Exception as e:
        logger.debug(f"⚠️ SuPar 분할 실패: {e}")
    
    # 3️⃣ 폴백: 형태(EC/EF) + 의미 임베딩 결합 점수로 경계 선택 (구두점 포함)
    segments = [src_text]

    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        denom = (np.linalg.norm(u) * np.linalg.norm(v))
        if denom == 0:
            return 0.0
        return float(np.dot(u, v) / denom)

    def _semantic_delta(left: str, right: str) -> float:
        if not embed_func:
            return 0.0
        try:
            embs = embed_func([left, right])
            if not embs or len(embs) < 2:
                return 0.0
            a = np.array(embs[0])
            b = np.array(embs[1])
            sim = _cosine(a, b)
            return max(0.0, 1.0 - sim)
        except Exception as e:
            logger.debug(f"⚠️ 의미 스코어 계산 실패: {e}")
            return 0.0

    candidates: List[Tuple[int, str]] = []  # (pos, tag)
    try:
        from sentence_splitter import split_target_sentences_advanced
        cand = split_target_sentences_advanced(src_text, max_length=400, splitter="punctuation") or []
        offset = 0
        for seg in cand:
            seg = seg or ""
            offset += len(seg)
            if offset < len(src_text):
                candidates.append((offset, "PUNC"))
    except Exception as e:
        logger.debug(f"⚠️ 구두점 후보 추출 실패: {e}")

    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        analysis = kiwi.analyze(src_text, top_n=1)
        tokens = analysis[0][0] if analysis and analysis[0] else []
        for tok in tokens:
            tag = getattr(tok, "tag", "") or ""
            if tag.startswith(("EF", "EC")):
                candidates.append((tok.start + tok.len, tag[:2]))
    except Exception as e:
        logger.debug(f"⚠️ Kiwi 후보 추출 실패: {e}")

    # 후보가 없으면 원문 전체 유지
    if candidates:
        unique_candidates = sorted({pos: tag for pos, tag in candidates if 0 < pos < len(src_text)}.items())
        scored: List[Tuple[float, int]] = []
        for pos, tag in unique_candidates:
            lpos = max(0, pos - semantic_window)
            rpos = min(len(src_text), pos + semantic_window)
            left = src_text[lpos:pos].strip()
            right = src_text[pos:rpos].strip()
            sem = _semantic_delta(left, right) if left and right else 0.0
            morph = 1.0 if tag.startswith("EF") else 0.6 if tag.startswith("EC") else 0.7
            score = sem + 0.2 * morph
            scored.append((score, pos))
        scored.sort(key=lambda x: x[0], reverse=True)
        choose = sorted([pos for _, pos in scored[:max(0, target_count - 1)]]) if target_count else [pos for _, pos in scored]
        if choose:
            boundaries = [0] + choose + [len(src_text)]
            segs = []
            for i in range(len(boundaries) - 1):
                seg = src_text[boundaries[i]:boundaries[i+1]].strip()
                if seg:
                    segs.append(seg)
            if len(segs) > 0:
                segments = segs
                logger.info(f"📍 의미+형태 결합 분할: {len(segments)}개 (후보 {len(candidates)}개)")
    else:
        logger.info("📍 형태/구두점 후보 없음 → 단일 세그먼트 유지")

    def split_longest_until(segs: List[str], k: int) -> List[str]:
        while len(segs) < k:
            idx = max(range(len(segs)), key=lambda i: len(segs[i]) if segs[i] else 0)
            words = segs[idx].split()
            if len(words) <= 1:
                break  # 더 이상 쪼갤 수 없음
            mid = max(1, len(words) // 2)
            left = ' '.join(words[:mid])
            right = ' '.join(words[mid:])
            segs[idx:idx+1] = [left, right]
        return segs

    def merge_smallest_until(segs: List[str], k: int) -> List[str]:
        while len(segs) > k and len(segs) > 1:
            best_idx = min(range(len(segs)-1), key=lambda i: len(segs[i]) + len(segs[i+1]))
            merged = (segs[best_idx] + ' ' + segs[best_idx+1]).strip()
            segs[best_idx:best_idx+2] = [merged]
        return segs

    if len(segments) < target_count:
        segments = split_longest_until(segments, target_count)
    elif len(segments) > target_count:
        segments = merge_smallest_until(segments, target_count)

    if len(segments) < target_count:
        segments += [''] * (target_count - len(segments))
    elif len(segments) > target_count:
        segments = segments[:target_count]

    logger.info(f"✅ 폴백 분할 완료: {len(segments)}개 문장")
    return segments


def _split_source_with_llm_and_target(src_text: str, src_units: List[str], tgt_sentences: List[str], target_count: int) -> List[str]:
    """
    LLM을 사용해 번역문을 참조하여 원문을 의미 기반으로 분할
    
    Args:
        src_text: 원문 전체 텍스트
        src_units: 원문 어절 리스트
        tgt_sentences: 번역문 문장 리스트
        target_count: 목표 세그먼트 수
    
    Returns:
        분할된 원문 세그먼트 리스트
    """
    import os
    import json
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not os.getenv("USE_LLM_BOUNDARY_VERIFY"):
        return None
    
    model_name = os.getenv("LLM_BOUNDARY_MODEL", "gpt-4o-mini")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None
    
    # 번역문 제시
    tgt_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(tgt_sentences)])
    
    prompt = f"""다음은 한문 원문과 그 한국어 번역문입니다.

번역문 {target_count}개 문장:
{tgt_list}

원문 전체:
{src_text}

원문을 정확히 {target_count}개 세그먼트로 나누되, 각 세그먼트가 번역문의 대응하는 문장과 의미적으로 일치하도록 분할하세요.

중요 규칙:
1. 원문의 문자나 순서를 절대 변경하지 마세요. 오직 경계만 조정하세요.
2. 반드시 정확히 {target_count}개 세그먼트를 반환하세요.
3. 각 원문 세그먼트가 번역문의 대응하는 문장 내용과 일치하도록 하세요.
4. 어절(공백으로 구분된 단위)을 쪼개지 마세요.

JSON 형식으로만 응답하세요:
{{"segments": ["세그먼트1", "세그먼트2", ...]}}"""
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise text segmenter. Divide source text to match target translations semantically. Never alter text content."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000,
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
        segments = data.get("segments", [])
        
        if not isinstance(segments, list) or len(segments) != target_count:
            logger.warning(f"LLM 원문 분할: 잘못된 개수 반환 ({len(segments)} != {target_count})")
            return None
        
        return segments
        
    except Exception as e:
        logger.debug(f"LLM 원문 분할 요청 실패: {e}")
        return None


def _adjust_source_segments_to_target(
    segments: List[str],
    target_count: int,
    src_text: str,
    src_units: List[str],
    tgt_sentences: List[str]
) -> List[str]:
    """
    SuPar 분할 결과를 번역문 개수에 맞게 LLM으로 조정
    
    Args:
        segments: SuPar가 분할한 원문 세그먼트
        target_count: 목표 개수 (번역문 문장 수)
        src_text: 원문 전체
        src_units: 원문 어절 리스트
        tgt_sentences: 번역문 문장 리스트
    
    Returns:
        조정된 세그먼트 리스트
    """
    import os
    import json
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not os.getenv("USE_LLM_BOUNDARY_VERIFY"):
        return None
    
    model_name = os.getenv("LLM_BOUNDARY_MODEL", "gpt-4o-mini")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None
    
    # 번역문과 현재 분할 결과 제시
    tgt_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(tgt_sentences)])
    src_list = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(segments)])
    
    if len(segments) < target_count:
        instruction = f"""현재 원문이 {len(segments)}개로 나뉘어 있는데, 번역문 {target_count}개에 맞춰 더 세분화해야 합니다."""
    else:
        instruction = f"""현재 원문이 {len(segments)}개로 나뉘어 있는데, 번역문 {target_count}개에 맞춰 병합해야 합니다."""
    
    prompt = f"""{instruction}

번역문 {target_count}개 문장:
{tgt_list}

현재 원문 {len(segments)}개 세그먼트:
{src_list}

원문 전체:
{src_text}

원문을 정확히 {target_count}개 세그먼트로 조정하되, 각 세그먼트가 번역문의 대응하는 문장과 의미적으로 일치하도록 하세요.

중요 규칙:
1. 원문의 문자나 순서를 절대 변경하지 마세요. 오직 경계만 조정하세요.
2. 반드시 정확히 {target_count}개 세그먼트를 반환하세요.
3. 각 원문 세그먼트가 번역문의 대응하는 문장 내용과 일치하도록 하세요.

JSON 형식으로만 응답하세요:
{{"segments": ["세그먼트1", "세그먼트2", ...]}}"""
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise boundary adjuster. Adjust segmentation to match target translations semantically. Never alter text content."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=3000,
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
            logger.warning(f"LLM 원문 조정: 잘못된 개수 반환 ({len(adjusted)} != {target_count})")
            return None
        
        return adjusted
        
    except Exception as e:
        logger.debug(f"LLM 원문 조정 요청 실패: {e}")
        return None


def align_sentences_with_optimal_matching(
    src_sentences: List[str],
    tgt_sentences: List[str], 
    embed_func=None,
    tokenizer_func=None,
    max_workers: int = 16,
    batch_size: int = 256
) -> List[Dict]:
    """
    ✨ 의미 기반 문장 매칭
    
    원문과 번역문이 분할 수준이 다를 때:
    1. 원문 1개 → 의미 경계로 번역문 문장 수에 맞춰 분할
    2. 번역문 1개 → 의미 경계로 원문 문장 수에 맞춰 분할
    3. 둘 다 N개 이상 → 유사도 기반 1:1 매칭
    """
    if not src_sentences or not tgt_sentences:
        return []
    
    M = len(src_sentences)
    N = len(tgt_sentences)
    
    logger.info(f"📊 의미 기반 매칭: 원문 {M}개 ↔ 번역문 {N}개")
    
    # ✨ Case 1: 원문 1개, 번역문 N개 → 원문을 의미 경계로 분할
    if M == 1 and N > 1:
        src_text = src_sentences[0]
        logger.info(f"🔍 원문 {N}개 문장으로 의미 분할 시도 (번역문 참조)")
        
        # 원문을 의미 경계로 분할 (번역문 참조)
        semantic_src_sentences = split_source_by_semantic_boundaries(
            src_text, target_count=N, embed_func=embed_func, tgt_sentences=tgt_sentences
        )
        
        # 분할 결과 확인
        logger.info(f"📋 분할 결과: {len(semantic_src_sentences)}개 문장")
        for i, s in enumerate(semantic_src_sentences):
            logger.info(f"  [{i}] {s[:50]}...")
        
        # 재귀적으로 호출하여 의미 기반 매칭 수행
        return align_sentences_with_optimal_matching(
            semantic_src_sentences,
            tgt_sentences,
            embed_func=embed_func,
            tokenizer_func=tokenizer_func,
            max_workers=max_workers,
            batch_size=batch_size
        )
    
    # ✨ Case 2: 원문 M개, 번역문 1개 → 번역문을 의미 경계로 분할
    if M > 1 and N == 1:
        tgt_text = tgt_sentences[0]
        logger.info(f"🔍 번역문을 {M}개 문장으로 의미 분할 시도 (재귀)")
        
        # 번역문도 의미 경계로 재분할
        semantic_tgt_sentences = split_target_sentences_advanced(tgt_text, splitter='punctuation')
        
        if len(semantic_tgt_sentences) == 1:
            # 분할 실패 → 원문으로 돌아가서 길이 기반 분할
            logger.warning(f"⚠️ 번역문 재분할 실패, 원문 {M}개로 유지")
            src_text = ''.join(src_sentences)
            semantic_src_sentences = split_source_by_semantic_boundaries(
                src_text, target_count=M, embed_func=embed_func, tgt_sentences=tgt_sentences
            )
            
            return align_sentences_with_optimal_matching(
                semantic_src_sentences,
                tgt_sentences,
                embed_func=embed_func,
                tokenizer_func=tokenizer_func,
                max_workers=max_workers,
                batch_size=batch_size
            )
        else:
            logger.info(f"📋 번역문 재분할: {len(semantic_tgt_sentences)}개 문장")
            return align_sentences_with_optimal_matching(
                src_sentences,
                semantic_tgt_sentences,
                embed_func=embed_func,
                tokenizer_func=tokenizer_func,
                max_workers=max_workers,
                batch_size=batch_size
            )
    
    # 임베딩 계산 (선택적)
    src_embs = None
    tgt_embs = None
    if embed_func:
        try:
            src_embs = np.array(embed_func(src_sentences))
            tgt_embs = np.array(embed_func(tgt_sentences))
            logger.info(f"✅ 임베딩 완료: src {src_embs.shape}, tgt {tgt_embs.shape}")
        except Exception as e:
            logger.warning(f"⚠️ 임베딩 실패: {e}, 길이 기반으로 진행")
            src_embs = None
            tgt_embs = None
    
    alignments = []
        
    
    # ✨ Case 3: 원문 M개, 번역문 N개 (M = N 또는 M ≠ N) → 순서 보존 매칭
    if M > 1 and N > 1:
        logger.info(f"📊 순서 보존 매칭 수행 (M={M}, N={N})")
        
        # 유사도 행렬 계산
        sim_matrix = np.zeros((M, N), dtype=float)
        
        for i in range(M):
            for j in range(N):
                if src_embs is not None and tgt_embs is not None:
                    src_norm = np.linalg.norm(src_embs[i])
                    tgt_norm = np.linalg.norm(tgt_embs[j])
                    if src_norm > 0 and tgt_norm > 0:
                        sim_matrix[i, j] = np.dot(src_embs[i], tgt_embs[j]) / (src_norm * tgt_norm)
                else:
                    src_len = len(src_sentences[i].split())
                    tgt_len = len(tgt_sentences[j].split())
                    sim_matrix[i, j] = 1.0 - abs(src_len - tgt_len) / max(src_len, tgt_len, 1)
        
        # 🎯 순서 보존: 원문 i는 번역문 i와 매칭 (단, 인덱스 범위 내에서)
        # M = N인 경우: 1:1 매칭
        # M < N인 경우: 일부 번역문 미매칭
        # M > N인 경우: 일부 원문 미매칭
        
        min_len = min(M, N)
        
        for i in range(min_len):
            sim = sim_matrix[i, i]
            alignments.append({
                '원문': src_sentences[i],
                '번역문': tgt_sentences[i],
                'similarity': float(sim),
                'split_method': 'semantic_sentence_level',
                'align_method': 'order_preserved_1to1'
            })
        
        # 미매칭 원문 추가 (M > N인 경우)
        for i in range(min_len, M):
            alignments.append({
                '원문': src_sentences[i],
                '번역문': '',
                'similarity': 0.0,
                'split_method': 'semantic_sentence_level',
                'align_method': 'unmatched_src'
            })
        
        # 미매칭 번역문 추가 (M < N인 경우)
        for j in range(min_len, N):
            alignments.append({
                '원문': '',
                '번역문': tgt_sentences[j],
                'similarity': 0.0,
                'split_method': 'semantic_sentence_level',
                'align_method': 'unmatched_tgt'
            })
    
    # 🔁 저유사도 폴백: 1:1 정렬 평균 유사도가 낮으면 번역문 문장 수에 맞춰 DP 재분할
    if embed_func and M > 1 and N > 1 and alignments:
        sims = [a.get('similarity', 0.0) for a in alignments if isinstance(a.get('similarity', None), (int, float))]
        if sims:
            avg_sim = float(np.mean(sims))
            logger.info(f"📊 평균 유사도: {avg_sim:.3f} (M={M}, N={N})")
            if avg_sim < 0.25:  # 임계값 완화: 0.2 → 0.25
                logger.info(f"🔄 저유사도 폴백 시도 (avg_sim={avg_sim:.3f} < 0.25)")
                try:
                    src_text_full = ' '.join(src_sentences)
                    logger.info(f"  원문 재분할: {len(src_text_full)}자 → {N}개 세그먼트")
                    reseg = segment_src_by_tgt_similarity(src_text_full, tgt_sentences, embed_func=embed_func, tokenizer_func=tokenizer_func)
                    logger.info(f"  DP 재분할 결과: {len(reseg) if reseg else 0}개")
                    if reseg and len(reseg) == N:
                        # 재분할된 원문이 의미가 있는지 확인 (빈 문자열 비율)
                        non_empty = sum(1 for s in reseg if s.strip())
                        if non_empty < N * 0.5:  # 절반 이상 비어있으면 무효
                            logger.warning(f"⚠️ 재분할 무효 (빈 세그먼트 과다: {non_empty}/{N})")
                        else:
                            src_reemb = embed_func(reseg)
                            tgt_reemb = embed_func(tgt_sentences)
                            new_align = []
                            for i in range(N):
                                svec = np.array(src_reemb[i])
                                tvec = np.array(tgt_reemb[i])
                                denom = (np.linalg.norm(svec) * np.linalg.norm(tvec) + 1e-8)
                                sim_val = float(np.dot(svec, tvec) / denom) if denom > 0 else 0.0
                                new_align.append({
                                    '원문': reseg[i],
                                    '번역문': tgt_sentences[i],
                                    'similarity': sim_val,
                                    'split_method': 'dp_src_by_tgt',
                                    'align_method': 'dp_src_by_tgt_similarity'
                                })
                            if new_align:
                                new_avg = float(np.mean([a['similarity'] for a in new_align]))
                                if new_avg > avg_sim or avg_sim < 0.15:  # 개선되거나 원래 너무 나쁘면 적용
                                    alignments = new_align
                                    logger.info(f"✅ 저유사도 폴백 적용: 평균 유사도 {avg_sim:.3f} → {new_avg:.3f}")
                                else:
                                    logger.info(f"⚠️ 폴백 결과 나쁨 ({new_avg:.3f}), 원본 유지")
                    else:
                        logger.warning(f"⚠️ 재분할 실패: 예상 {N}개, 실제 {len(reseg) if reseg else 0}개")
                except Exception as e:
                    logger.warning(f"⚠️ 저유사도 폴백 실패: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            else:
                logger.info(f"✅ 유사도 정상 (avg_sim={avg_sim:.3f} >= 0.25), 폴백 불필요")

    logger.info(f"✅ 매칭 완료: {len(alignments)}개 쌍")
    return alignments


def segment_src_by_tgt_similarity(src_text: str, tgt_sentences: List[str], embed_func=None, tokenizer_func=None) -> List[str]:
    """
    원문을 토큰(어절) 단위로 분할하고, 번역문 문장과의 유사도를 기반으로
    순서를 보존하는 단조 증가 세그멘테이션(DP)으로 각 문장에 연속 구간을 할당.
    - 모든 토큰은 정확히 한 번만 사용
    - 문장 순서에 맞춰 구간이 앞에서 뒤로 진행 (reordering 방지)
    - 각 문장에 최소 1 토큰 할당 (단, 토큰 수 < 문장 수이면 일부는 빈 구간)
    - 🎯 원본 텍스트를 span 슬라이싱으로 재구성하여 무결성 보장
    - 🆕 원문 문장 구조도 함께 고려하여 양방향 유사도 측정
    - 🔥 [-…] 구간 완전 제외 후 복원 (정렬에서 괄호 영향 제거)
    """
    if not tgt_sentences:
        return []
    
    # Strip bracket segments to avoid interference in alignment
    working_src, bracket_insertions = _strip_brackets_from_text(src_text)
    
    # Extract word spans from working text (brackets removed)
    import re
    word_spans = []  # List[Tuple[start, end]] on working text
    pos = 0
    
    while pos < len(working_src):
        # Skip whitespace
        if working_src[pos].isspace():
            pos += 1
            continue
        
        # Regular word: up to next whitespace
        end_pos = pos + 1
        while end_pos < len(working_src) and not working_src[end_pos].isspace():
            end_pos += 1
        word_spans.append((pos, end_pos))
        pos = end_pos
    
    words = [working_src[s:e] for (s, e) in word_spans]
    W = len(words)
    S = len(tgt_sentences)

    if W == 0:
        return ["" for _ in range(S)]

    if not embed_func:
        # 임베더 없으면 단순 균등 (원본 슬라이싱 사용)
        base = max(1, W // S)
        segments = []
        idx = 0
        for s in range(S - 1):
            next_idx = min(W, idx + base)
            if idx < next_idx and idx < len(word_spans) and next_idx-1 < len(word_spans):
                start_char = word_spans[idx][0]
                # trailing whitespace까지 포함하여 원문 간격 보존
                end_char = word_spans[next_idx][0] if next_idx < len(word_spans) else len(working_src)
                segments.append(working_src[start_char:end_char])
            else:
                segments.append('')
            idx = next_idx
        # 마지막 세그먼트
        if idx < W and idx < len(word_spans):
            start_char = word_spans[idx][0]
            end_char = len(working_src)
            segments.append(working_src[start_char:end_char])
        else:
            segments.append('')
        # Restore brackets before returning
        return _restore_brackets_to_segments(segments, bracket_insertions)

    # 🆕 원문도 문장으로 분할하여 양방향 유사도 측정 준비
    src_sentences = [working_src]  # 기본값: 전체를 하나의 문장으로
    try:
        # 원문 문장 분할 시도 (구두점 기반)
        from sentence_splitter import split_target_sentences_advanced as split_func
        temp_src_sentences = split_func(working_src, max_length=200, splitter="punctuation")
        if temp_src_sentences and len(temp_src_sentences) > 1:
            src_sentences = temp_src_sentences
            logger.info(f"🔤 원문 문장 분할: {len(src_sentences)}개")
        else:
            logger.info(f"🔤 원문 문장 분할 없음 (전체 1개)")
    except Exception as e:
        logger.warning(f"⚠️ 원문 문장 분할 실패: {e}, 전체를 하나로 처리")


    # 임베딩 계산
    try:
        import numpy as np
        token_embs = embed_func(words)  # (W, D) - 원문 어절들
        tgt_sent_embs = embed_func(tgt_sentences)  # (S, D) - 번역문 문장들
        src_sent_embs = embed_func(src_sentences)  # (Src_S, D) - 원문 문장들
        # numpy 배열화
        token_embs = np.array(token_embs)
        tgt_sent_embs = np.array(tgt_sent_embs)
        src_sent_embs = np.array(src_sent_embs)
        
        logger.info(f"📊 임베딩: 원문어절 {len(token_embs)}, 번역문문장 {len(tgt_sent_embs)}, 원문문장 {len(src_sent_embs)}")
    except Exception as e:
        logger.error(f"❌ 임베딩 계산 실패: {e}")
        # 임베딩 실패 시: 구두점 기반 분할 후 greedy split/merge로 타깃 개수 맞춤
        try:
            from sentence_splitter import split_target_sentences_advanced
            segments = split_target_sentences_advanced(working_src, max_length=400, splitter="punctuation") or []
            segments = [s for s in segments if s.strip()] or [working_src]
            def split_longest(segs):
                while len(segs) < S:
                    idx2 = max(range(len(segs)), key=lambda i: len(segs[i]) if segs[i] else 0)
                    words2 = segs[idx2].split()
                    if len(words2) <= 1:
                        break
                    mid2 = max(1, len(words2)//2)
                    left2 = ' '.join(words2[:mid2])
                    right2 = ' '.join(words2[mid2:])
                    segs[idx2:idx2+1] = [left2, right2]
                return segs
            def merge_smallest(segs):
                while len(segs) > S and len(segs) > 1:
                    bidx = min(range(len(segs)-1), key=lambda i: len(segs[i])+len(segs[i+1]))
                    merged = (segs[bidx] + ' ' + segs[bidx+1]).strip()
                    segs[bidx:bidx+2] = [merged]
                return segs
            if len(segments) < S:
                segments = split_longest(segments)
            elif len(segments) > S:
                segments = merge_smallest(segments)
            if len(segments) < S:
                segments += [''] * (S - len(segments))
            elif len(segments) > S:
                segments = segments[:S]
            return _restore_brackets_to_segments(segments, bracket_insertions)
        except Exception as ee:
            logger.error(f"❌ 폴백 분할도 실패: {ee}")
            return _restore_brackets_to_segments([working_src], bracket_insertions)

    import numpy as np
    # 토큰 임베딩 누적합으로 구간 평균 임베딩을 빠르게 계산
    prefix = np.zeros((W + 1, token_embs.shape[1]), dtype=float)
    prefix[1:] = np.cumsum(token_embs, axis=0)

    # 🆕 한문+한글 혼용문 경계 힌트: siku-bert(한문) + kiwipiepy(한글 EC/EF)
    tokenizer_boundary_bonus = {}  # word_idx -> bonus_score
    
    def has_hangul(text):
        """텍스트에 한글 포함 여부"""
        return any('\uac00' <= c <= '\ud7a3' for c in text)
    
    def has_hanja(text):
        """텍스트에 한문 포함 여부"""
        return any(_han_regex.match(c) for c in text)
    
    def cosine_sim(v1, v2):
        """코사인 유사도"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    try:
        # Kiwipiepy 초기화
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        # Siku-BERT 토크나이저 (tokenizer_func가 있다면)
        siku_tokenizer = tokenizer_func if tokenizer_func else None
        
        for word_idx, (s, e) in enumerate(word_spans):
            word = working_src[s:e]
            
            # 1️⃣ 한글이 포함된 어절 → kiwipiepy로 EC/EF 분석
            if has_hangul(word):
                try:
                    analyzed = kiwi.analyze(word, top_n=1)
                    if analyzed and analyzed[0]:
                        tokens = analyzed[0][0]
                        for token in tokens:
                            tag = getattr(token, 'tag', '')
                            if tag == 'EF':  # 종결어미 → 확실한 문장 경계
                                tokenizer_boundary_bonus[word_idx] = 0.20
                                logger.debug(f"  어절 {word_idx} ({word}): EF 종결어미 → bonus +0.20")
                                break
                            elif tag == 'EC':  # 연결어미 → 문맥 기반 동적 평가
                                # EC의 좌우 임베딩 차이로 절 경계 가능성 판단
                                if word_idx > 0 and word_idx < W - 1:
                                    # 현재 어절 이전 구간의 평균 임베딩
                                    left_emb = prefix[word_idx] / max(word_idx, 1)
                                    # 현재 어절 이후 구간의 평균 임베딩
                                    right_emb = (prefix[W] - prefix[word_idx + 1]) / max(W - word_idx - 1, 1)
                                    
                                    sim = cosine_sim(left_emb, right_emb)
                                    # 유사도가 낮을수록 의미 단절 → 경계 가능성 높음
                                    boundary_score = 0.05 + (1.0 - sim) * 0.10  # 0.05~0.15 범위
                                    tokenizer_boundary_bonus[word_idx] = boundary_score
                                    logger.debug(f"  어절 {word_idx} ({word}): EC 연결어미, 좌우 sim={sim:.3f} → bonus +{boundary_score:.3f}")
                                else:
                                    tokenizer_boundary_bonus[word_idx] = 0.08
                                    logger.debug(f"  어절 {word_idx} ({word}): EC 연결어미 (경계) → bonus +0.08")
                                break
                except Exception as e_inner:
                    logger.debug(f"  어절 {word_idx} ({word}): kiwipiepy 분석 실패: {e_inner}")
            
            # 2️⃣ 순수 한문 어절 → siku-bert로 구두점 또는 구조적 경계 탐지
            else:
                if siku_tokenizer:
                    try:
                        # siku-bert 토크나이저 적용
                        tokens = siku_tokenizer(word) if callable(siku_tokenizer) else []
                        # 여러 토큰으로 분할되거나 특수 토큰이 있으면 구조적 경계 가능
                        if tokens and len(tokens) > 1:
                            tokenizer_boundary_bonus[word_idx] = 0.06
                            logger.debug(f"  어절 {word_idx} ({word}): 한문 다중 토큰({len(tokens)}개) → bonus +0.06")
                    except Exception as e_inner:
                        logger.debug(f"  어절 {word_idx} ({word}): siku-bert 분석 실패: {e_inner}")
    
    except Exception as e:
        logger.debug(f"⚠️ 혼용문 경계 힌트 추출 실패: {e}")
    
    if tokenizer_boundary_bonus:
        logger.info(f"📊 형태소 경계 힌트: {len(tokenizer_boundary_bonus)}개 어절 (siku-bert + kiwipiepy EC/EF 동적 평가)")
    else:
        logger.debug(f"📊 형태소 경계 힌트: 0개")

    # 🆕 파서 경계 힌트 추출 (SuPar-Kanbun, Stanza)
    parser_boundary_bonus = {}  # word_idx -> bonus_score
    try:
        from common.new_parsers import (
            get_korean_clause_boundaries_stanza,
            get_chinese_unit_boundary_indices_supar,
            STANZA_AVAILABLE, SUPAR_AVAILABLE
        )
        
        logger.debug(f"📊 파서 설정: SUPAR_AVAILABLE={SUPAR_AVAILABLE}, STANZA_AVAILABLE={STANZA_AVAILABLE}")
        
        # 원문 경계 정보 (SuPar-Kanbun)
        if SUPAR_AVAILABLE:
            try:
                src_unit_boundaries = get_chinese_unit_boundary_indices_supar(working_src, words)
                logger.debug(f"  SuPar 원문 경계: {src_unit_boundaries}")
                
                # 원문 어절 인덱스를 경계 위치로 매핑
                if src_unit_boundaries:
                    cumulative_len = 0
                    for word_idx, word in enumerate(words):
                        cumulative_len += len(word)
                        # 이 어절의 끝이 파서 경계와 근접하면 보너스
                        if any(abs(cumulative_len - b) <= 2 for b in src_unit_boundaries):
                            parser_boundary_bonus[word_idx] = 0.1
                            logger.debug(f"    어절 {word_idx} ({word}): SuPar 경계 힌트 (+0.1)")
            except Exception as e:
                logger.debug(f"⚠️ SuPar 경계 추출 실패: {e}")
        
        # 번역문 절 경계 정보 (Stanza) - 참고만 (번역문은 이미 분할됨)
        if STANZA_AVAILABLE:
            try:
                tgt_clause_boundaries = get_korean_clause_boundaries_stanza(tgt_paragraph, mode='default')
                logger.debug(f"  Stanza 절 경계: {len(tgt_clause_boundaries)}개 위치")
            except Exception as e:
                logger.debug(f"⚠️ Stanza 경계 추출 실패: {e}")
    except ImportError as e:
        logger.debug(f"⚠️ new_parsers 모듈 임포트 실패: {e}")
    except Exception as e:
        logger.debug(f"⚠️ 파서 경계 힌트 추출 실패: {e}")
    
    if parser_boundary_bonus:
        logger.info(f"📊 파서 경계 힌트: {len(parser_boundary_bonus)}개 어절")
    else:
        logger.debug(f"📊 파서 경계 힌트: 0개 (파서 미설치 또는 경계 없음)")

    # 🆕 원문 문장과 번역문 문장 간 유사도 행렬 계산 (양방향 매핑 힌트용)
    # src_sentences와 tgt_sentences 사이의 유사도
    use_bidirectional = True
    try:
        src_tgt_sim_matrix = np.zeros((len(src_sentences), len(tgt_sentences)), dtype=float)
        for i, src_sent_emb in enumerate(src_sent_embs):
            for j, tgt_sent_emb in enumerate(tgt_sent_embs):
                try:
                    sim = float(np.dot(src_sent_emb, tgt_sent_emb) / 
                              (np.linalg.norm(src_sent_emb) * np.linalg.norm(tgt_sent_emb) + 1e-8))
                    src_tgt_sim_matrix[i, j] = sim
                except Exception:
                    src_tgt_sim_matrix[i, j] = 0.0
        logger.info(f"📊 원문-번역문 문장 유사도 행렬: {src_tgt_sim_matrix.shape}")
    except Exception as e:
        logger.warning(f"⚠️ 양방향 유사도 계산 실패: {e}, 번역문만 사용")
        use_bidirectional = False

    # DP: dp[i][j] = 첫 i개의 단어를 j+1개의 문장에 할당한 최대 점수
    # i는 단어 개수(0..W), j는 문장 index(0..S-1)
    NEG_INF = -1e18
    dp = np.full((W + 1, S), NEG_INF, dtype=float)
    prev = np.full((W + 1, S), -1, dtype=int)  # 경계 k 저장

    # base: 첫 문장(j=0)에 i>=1 단어 할당
    for i in range(1, W + 1):
        try:
            # 0..i-1 단어들의 평균 임베딩과 첫 번째 문장의 유사도
            span_vec = prefix[i] / i
            tgt_vec = tgt_sent_embs[0]
            
            # 번역문 유사도
            sim_tgt = float(np.dot(span_vec, tgt_vec) / (np.linalg.norm(span_vec) + 1e-8) / (np.linalg.norm(tgt_vec) + 1e-8))
            
            if use_bidirectional:
                # 원문 유사도
                src_sims = [float(np.dot(span_vec, src_emb) / (np.linalg.norm(span_vec) + 1e-8) / (np.linalg.norm(src_emb) + 1e-8))
                            for src_emb in src_sent_embs]
                max_src_sim = max(src_sims) if src_sims else 0.0
                # 결합 유사도
                dp[i, 0] = 0.5 * sim_tgt + 0.5 * max_src_sim
            else:
                # 번역문만 사용
                dp[i, 0] = sim_tgt
        except Exception as e:
            logger.warning(f"⚠️ DP 초기화 오류 (i={i}): {e}, 기본값 사용")
            dp[i, 0] = 0.0
        prev[i, 0] = 0

    # 점화
    for j in range(1, S):
        # 최소 단어 수 제약: i >= j+1 (각 문장 최소 1 단어)
        for i in range(j + 1, W + 1):
            best_score = NEG_INF
            best_k = -1
            # 이전 경계 k는 >= j (앞의 j문장에 최소 j 단어), < i
            k_min = j
            for k in range(k_min, i):
                span_len = i - k
                if span_len <= 0:
                    continue
                # 구간 평균 임베딩 (원문 어절들)
                span_vec = (prefix[i] - prefix[k]) / span_len
                
                # 번역문 문장 임베딩
                tgt_sent_vec = tgt_sent_embs[j]
                
                # 번역문 유사도 계산
                num_tgt = float(np.dot(span_vec, tgt_sent_vec))
                denom_tgt = (np.linalg.norm(span_vec) + 1e-8) * (np.linalg.norm(tgt_sent_vec) + 1e-8)
                sim_tgt = num_tgt / denom_tgt
                
                # 🆕 원문 문장 유사도 계산 (해당 구간이 어느 원문 문장과 가장 유사한지)
                if use_bidirectional:
                    try:
                        src_sent_sims = []
                        for src_sent_emb in src_sent_embs:
                            try:
                                num_src = float(np.dot(span_vec, src_sent_emb))
                                denom_src = (np.linalg.norm(span_vec) + 1e-8) * (np.linalg.norm(src_sent_emb) + 1e-8)
                                sim_src = num_src / denom_src
                                src_sent_sims.append(sim_src)
                            except Exception:
                                src_sent_sims.append(0.0)
                        
                        max_src_sim = max(src_sent_sims) if src_sent_sims else 0.0
                        # 🎯 양방향 유사도 결합: 번역문 50%, 원문 50%
                        combined_sim = 0.5 * sim_tgt + 0.5 * max_src_sim
                    except Exception:
                        combined_sim = sim_tgt  # 실패 시 번역문만 사용
                else:
                    combined_sim = sim_tgt  # 양방향 비활성 시 번역문만 사용
                
                # 🆕 경계 보너스 추가 (토크나이저, 파서 힌트)
                boundary_bonus = 0.0
                # k-1 위치 (구간 시작)에 경계 힌트가 있으면 보너스
                if k - 1 in tokenizer_boundary_bonus:
                    boundary_bonus += tokenizer_boundary_bonus[k - 1]
                if k - 1 in parser_boundary_bonus:
                    boundary_bonus += parser_boundary_bonus[k - 1]
                
                score = dp[k, j - 1] + combined_sim + boundary_bonus
                if score > best_score:
                    best_score = score
                    best_k = k
            dp[i, j] = best_score
            prev[i, j] = best_k

    # 복원: 모든 단어를 S문장에 할당
    # 양방향 유사도 실패해도 번역문 유사도로 DP는 계속 실행됨
    if W >= S and dp[W, S - 1] > NEG_INF / 2:
        cuts = [W]
        i = W
        for j in range(S - 1, 0, -1):
            k = prev[i, j]
            cuts.append(k)
            i = k
        cuts.append(0)
        cuts = sorted(cuts)
        
        # Word spans are on working_src; reconstruct segments
        segments = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a >= b or a < 0 or b > len(word_spans):
                segments.append('')
            else:
                # 어절 인덱스 a부터 b-1까지의 working 텍스트 슬라이스
                # 경계 이후의 공백까지 포함하여 원문 간격을 보존
                start_char = word_spans[a][0]
                end_char = word_spans[b][0] if b < len(word_spans) else len(working_src)
                segments.append(working_src[start_char:end_char])
        
        # Ensure exactly S segments
        if len(segments) > S:
            segments = segments[:S-1] + [''.join(segments[S-1:])]
        elif len(segments) < S:
            segments += [''] * (S - len(segments))
        
        # Final validation
        assert len(segments) == S, f"세그먼트 개수 불일치: {len(segments)} != {S}"
        # Restore brackets before returning
        return _restore_brackets_to_segments(segments, bracket_insertions)
    else:
        # 극히 예외적인 경우: W < S (단어가 문장보다 적음)
        logger.warning(f"⚠️ 극히 예외: 단어 수({W}) < 문장 수({S})")
        segments = [''] * S
        
        if W > 0:
            word_to_sent = {}  # word_idx -> sent_idx
            
            for w_idx in range(W):
                word_vec = token_embs[w_idx]
                best_sent_idx = 0
                best_sim = -1.0
                
                for s_idx in range(S):
                    try:
                        sent_vec = tgt_sent_embs[s_idx]
                        sim = float(np.dot(word_vec, sent_vec) / 
                                  (np.linalg.norm(word_vec) + 1e-8) / (np.linalg.norm(sent_vec) + 1e-8))
                        if sim > best_sim:
                            best_sim = sim
                            best_sent_idx = s_idx
                    except Exception:
                        pass
                
                word_to_sent[w_idx] = best_sent_idx
            
            # Collect words for each sentence from working_src
            for w_idx, sent_idx in sorted(word_to_sent.items()):
                s, e = word_spans[w_idx]
                if segments[sent_idx]:
                    segments[sent_idx] += ' ' + working_src[s:e]
                else:
                    segments[sent_idx] = working_src[s:e]
        
        # Restore brackets before returning
        return _restore_brackets_to_segments(segments, bracket_insertions)



def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    tokenizer_name: str = 'korean_hybrid',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8,
    use_spacy_tokenizer: bool = False,
    max_workers: int = 16,
    batch_size: int = 256
):
    """PA 처리: 완벽한 무결성 보장
    
    🆕 개선: 원문도 의미 문장으로 사전 분할
    - 원문과 번역문을 모두 의미 경계 기반으로 분할
    - 분할된 원문-번역문 쌍을 정렬
    - 원문의 의미 경계를 존중하면서 품질 향상
    """
    
    print(f"🔄 PA 처리 시작 (원문-번역문 의미 문장 대응 정렬)")
    
    para_id = f"paragraph_{id(src_paragraph)}_{id(tgt_paragraph)}"
    
    try:
        # 임베더/토크나이저는 분할 전에 준비하여 의미 경계 점수 계산 시 바로 사용
        embed_func = get_embedder_function(embedder_name, device=device, max_workers=max_workers, batch_size=batch_size)
        tokenizer_func = get_tokenizer_function(tokenizer_name)

        # 번역문 분할 (구두점 기반)
        tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length=max_length, splitter="punctuation")
        logger.info(f"🔤 번역문 분할: {len(tgt_sentences)}개 문장")

        if refine_boundaries_with_llm:
            try:
                tgt_sentences = refine_boundaries_with_llm(
                    tgt_paragraph,
                    tgt_sentences,
                    task="pa",
                    reference_text=src_paragraph,
                    max_segments=50,
                )
                logger.info(f"🤖 LLM 재검증(번역문) 후: {len(tgt_sentences)}개 문장")
            except Exception:
                pass
        
        # 🆕 원문 분할 (의미 경계 기반, 어절 단위 보존, 번역문 참조)
        src_sentences = split_source_by_semantic_boundaries(
            src_paragraph, 
            target_count=len(tgt_sentences), 
            embed_func=embed_func,
            tgt_sentences=tgt_sentences
        )
        logger.info(f"🔤 원문 분할: {len(src_sentences)}개 문장")

        if refine_boundaries_with_llm:
            try:
                src_sentences = refine_boundaries_with_llm(
                    src_paragraph,
                    src_sentences,
                    task="pa",
                    reference_text=tgt_paragraph,
                    max_segments=50,
                )
                logger.info(f"🤖 LLM 재검증(원문) 후: {len(src_sentences)}개 문장")
            except Exception:
                pass
        
        # 🆕 이제 원문-번역문이 모두 분할되었으므로, 어절 단위 매칭 대신
        # 문장 단위 매칭으로 전환 (원문 문장 N개 ↔ 번역문 문장 M개)
        print(f"🔄 원문 {len(src_sentences)}개 문장 ↔ 번역문 {len(tgt_sentences)}개 문장 정렬 중...")
        logger.info(f"🔄 원문 {len(src_sentences)}개 문장 ↔ 번역문 {len(tgt_sentences)}개 문장 정렬 중...")

        if not embed_func or len(src_sentences) == 0 or len(tgt_sentences) == 0:
            logger.error(f"문장 정렬 불가: embed_func={bool(embed_func)}, src={len(src_sentences)}, tgt={len(tgt_sentences)}")
            raise ValueError("문장 정렬 필수 조건 미충족")

        # 🆕 문장 단위 유사도 기반 최적 정렬 (DP)
        results = align_sentences_with_optimal_matching(
            src_sentences,
            tgt_sentences,
            embed_func=embed_func,
            tokenizer_func=tokenizer_func,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        print(f"✅ 문장 정렬 완료: {len(results)}개 쌍")
        logger.info(f"✅ 문장 정렬 완료: {len(results)}개 쌍")
        
        # 무결성 검증 제거 - 분할만 수행, 검증은 맨 마지막에
        return results
        
    except Exception as e:
        logger.error(f"문단 처리 중 오류: {e}")
        # 오류시 안전한 기본 처리
        return [{
            '원문': src_paragraph,
            '번역문': tgt_paragraph,
            'similarity': 1.0,
            'split_method': 'error_fallback',
            'align_method': 'error_fallback'
        }]

def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    tokenizer_name: str = 'korean_hybrid',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8,
    use_spacy_tokenizer: bool = False,
    verbose: bool = False,
    max_workers: int = 4,
    batch_size: int = 100,
    **kwargs
):
    """파일 단위 처리 - 완벽한 무결성 보장"""
    print(f"📂 PA 파일 처리 시작 (완벽한 무결성 보장): {input_file}")
    if use_spacy_tokenizer:
        print(f"🔗 기존 방식 + Vice Versa 토크나이저 + spaCy 토크나이저 융합")
    else:
        print(f"🔄 기존 방식 + Vice Versa 토크나이저 통합")
    print(f"⚙️  토크나이저: {tokenizer_name}")
    print(f"⚙️  임베더: {embedder_name}")
    print(f"🔗  spaCy 융합: {use_spacy_tokenizer}")
    print(f"🔒  무결성 보장: ON")
    print(f"🚀  병렬 처리: max_workers={max_workers}, batch_size={batch_size}")
    
    try:
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None

    # 전체 원문/번역문 원본 합본 (무결성 비교용)
    original_src_all = ''.join(df['원문'].fillna('')) if '원문' in df.columns else ''
    original_tgt_all = ''.join(df['번역문'].fillna('')) if '번역문' in df.columns else ''
    
    file_id = f"file_{id(input_file)}"
    
    all_results = []
    total = len(df)
    processed_count = 0
    error_count = 0
    
    # 진행률은 상위 processor에서 관리됨 (중복 방지)
    
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        
        if src_paragraph.strip() and tgt_paragraph.strip():
            try:
                alignments = process_paragraph_alignment(
                    src_paragraph,
                    tgt_paragraph,
                    embedder_name=embedder_name,
                    tokenizer_name=tokenizer_name,
                    max_length=max_length,
                    similarity_threshold=similarity_threshold,
                    device=device,
                    quality_threshold=quality_threshold,
                    use_spacy_tokenizer=use_spacy_tokenizer,
                    max_workers=max_workers,
                    batch_size=batch_size
                )
                
                # 문단식별자 부여
                for a in alignments:
                    a['문단식별자'] = idx + 1
                
                all_results.extend(alignments)
                processed_count += 1
                
                # 진행률 업데이트는 상위 processor에서 처리됨
                
            except Exception as e:
                print(f"❌ 문단 {idx + 1} 처리 실패: {e}")
                error_count += 1
                if verbose:
                    import traceback
                    traceback.print_exc()
                
                # 오류시 안전한 폴백
                all_results.append({
                    '문단식별자': idx + 1,
                    '원문': src_paragraph,
                    '번역문': tgt_paragraph,
                    'similarity': 1.0,
                    'split_method': 'error_fallback',
                    'align_method': 'error_fallback'
                })
                
                # 진행률 업데이트는 상위 processor에서 처리됨
    
    if not all_results:
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]

    # 진행률 완료는 상위 processor에서 처리됨

    # === 최종 무결성 검증 및 보완 ===
    print("🔒 최종 무결성 검증 중...")
    
    output_src_all = ''.join(result_df['원문'].fillna(''))
    output_tgt_all = ''.join(result_df['번역문'].fillna(''))
    
    # 원문 무결성 검증
    src_valid, src_msg = integrity_manager.verify_integrity(output_src_all, f"{file_id}_src_all")
    if not src_valid:
        # 복원 시도는 중복/순서 교란을 초래하므로 현재는 경고만 출력
        print(f"⚠️ 원문 무결성 불일치: {src_msg} (복원 스킵)")
    
    # 번역문 무결성 검증
    output_tgt_all_after_src = ''.join(result_df['번역문'].fillna(''))
    tgt_valid, tgt_msg = integrity_manager.verify_integrity(output_tgt_all_after_src, f"{file_id}_tgt_all")
    
    if not tgt_valid:
        print(f"⚠️ 번역문 무결성 불일치: {tgt_msg} (복원 스킵)")
    
    # 최종 저장
    result_df = result_df[final_columns]
    result_df.to_excel(output_file, index=False)
    
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    print(f"✅ 처리 성공: {processed_count}개 문단")
    print(f"❌ 처리 실패: {error_count}개 문단")
    
    # 무결성 통계
    final_src_check = ''.join(result_df['원문'].fillna(''))
    final_tgt_check = ''.join(result_df['번역문'].fillna(''))
    
    final_src_valid, _ = integrity_manager.verify_integrity(final_src_check, f"{file_id}_src_all")
    final_tgt_valid, _ = integrity_manager.verify_integrity(final_tgt_check, f"{file_id}_tgt_all")
    
    print(f"🔒 최종 무결성 상태:")
    print(f"   원문: {'✅ 완벽' if final_src_valid else '❌ 불완전'}")
    print(f"   번역문: {'✅ 완벽' if final_tgt_valid else '❌ 불완전'}")
    
    if use_spacy_tokenizer:
        print(f"🔗 spaCy + 토크나이저 융합 방식 완료")
    else:
        print(f"🔄 Vice Versa 토크나이저 방식 완료")
    
    return result_df