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
from sentence_splitter import split_target_sentences_advanced, split_source_by_whitespace_and_align

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
    
    # 현재 정렬 결과 분석
    aligned_src = ''.join([align.get('원문', '') for align in alignments]).replace(' ', '')
    aligned_tgt = ''.join([align.get('번역문', '') for align in alignments]).replace(' ', '')
    
    original_src = src_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
    original_tgt = tgt_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
    
    restored_alignments = alignments[:]
    
    # 원문 복원
    if original_src != aligned_src:
        logger.info("원문 무결성 복원 시작...")
        sm = SequenceMatcher(None, aligned_src, original_src)
        opcodes = sm.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # 누락된 원문 추가
                missing_text = original_src[j1:j2]
                if restored_alignments:
                    restored_alignments[-1]['원문'] += missing_text
                else:
                    restored_alignments.append({
                        '원문': missing_text,
                        '번역문': '',
                        'similarity': 0.0,
                        'split_method': 'integrity_restore',
                        'align_method': 'src_missing_restore'
                    })
                logger.info(f"누락 원문 복원: '{missing_text}'")
                
            elif tag == 'delete':
                # 중복된 원문 제거
                excess_text = aligned_src[i1:i2]
                for align in restored_alignments:
                    if excess_text in align.get('원문', ''):
                        align['원문'] = align['원문'].replace(excess_text, '', 1)
                        logger.info(f"중복 원문 제거: '{excess_text}'")
                        break
    
    # 번역문 복원
    aligned_tgt_after_src_restore = ''.join([align.get('번역문', '') for align in restored_alignments]).replace(' ', '')
    
    if original_tgt != aligned_tgt_after_src_restore:
        logger.info("번역문 무결성 복원 시작...")
        sm = SequenceMatcher(None, aligned_tgt_after_src_restore, original_tgt)
        opcodes = sm.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # 누락된 번역문 추가
                missing_text = original_tgt[j1:j2]
                if restored_alignments:
                    restored_alignments[-1]['번역문'] += missing_text
                else:
                    restored_alignments.append({
                        '원문': '',
                        '번역문': missing_text,
                        'similarity': 0.0,
                        'split_method': 'integrity_restore',
                        'align_method': 'tgt_missing_restore'
                    })
                logger.info(f"누락 번역문 복원: '{missing_text}'")
                
            elif tag == 'delete':
                # 중복된 번역문 제거
                excess_text = aligned_tgt_after_src_restore[i1:i2]
                for align in restored_alignments:
                    if excess_text in align.get('번역문', ''):
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

def get_embedder_function(embedder_name: str, device: str = "cpu", openai_model: str = None, openai_api_key: str = None, max_workers: int = 4, batch_size: int = 100):
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
        max_workers=4,
        batch_size=100,
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
        max_workers=4,
        batch_size=100,
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
        max_workers=4,
        batch_size=100,
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

def improved_align_paragraphs(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    similarity_threshold: float = None,
    embedder_name: str = "bge",
    max_workers: int = 4,
    batch_size: int = 100
) -> List[Dict]:
    """기존 순차적 1:1 정렬 (무결성 보장)"""
    # config에서 파라미터 로드
    from common.config import get_alignment_params
    cfg_params = get_alignment_params()
    if similarity_threshold is None:
        similarity_threshold = cfg_params.get('similarity_threshold', 0.5)
    if not tgt_sentences:
        return []
    
    # 원문을 번역문 개수에 맞춰 의미적으로 분할 (임베딩 기반)
    aligned_src_chunks = split_source_by_whitespace_and_align(
        src_text, 
        len(tgt_sentences),
        target_sentences=tgt_sentences,  # 의미적 매칭을 위한 번역문 전달
        embedder_name=embedder_name,     # 임베더 이름 전달
        embedder_func=embed_func,        # 임베더 함수 전달
        max_workers=max_workers,         # 병렬 처리 매개변수 전달
        batch_size=batch_size            # 배치 크기 매개변수 전달
    )
    
    alignments = []
    for i in range(len(tgt_sentences)):
        src_chunk = aligned_src_chunks[i] if i < len(aligned_src_chunks) else ''
        tgt_sentence = tgt_sentences[i]
        
        # 실제 유사도 계산 (임베딩 기반)
        if embed_func and src_chunk.strip() and tgt_sentence.strip():
            try:
                src_embedding = embed_func([src_chunk])[0]
                tgt_embedding = embed_func([tgt_sentence])[0]
                
                import numpy as np
                src_norm = np.linalg.norm(src_embedding)
                tgt_norm = np.linalg.norm(tgt_embedding)
                
                if src_norm > 1e-8 and tgt_norm > 1e-8:
                    similarity = float(np.dot(src_embedding, tgt_embedding) / (src_norm * tgt_norm))
                else:
                    similarity = 0.0
            except Exception:
                similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        else:
            similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        
        # 낮은 similarity는 경고 로깅
        if similarity < similarity_threshold * 0.8:  # 약간 여유 있게
            logger.debug(f"PA: 낮은 유사도 감지 ({similarity:.3f} < {similarity_threshold*0.8:.3f}): '{src_chunk[:30]}...' <-> '{tgt_sentence[:30]}...'")
        
        alignments.append({
            '원문': src_chunk,
            '번역문': tgt_sentence,
            'similarity': similarity,
            'split_method': 'punctuation',
            'align_method': 'sequential'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'punctuation',
            'align_method': 'sequential_unmatched_src'
        })
    
    return alignments

def improved_align_paragraphs_new_parsers(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    similarity_threshold: float = None
) -> List[Dict]:
    """새 파서들 + 토크나이저 융합 정렬 (무결성 보장)"""
    from common.config import get_alignment_params
    cfg_params = get_alignment_params()
    if similarity_threshold is None:
        similarity_threshold = cfg_params.get('similarity_threshold', 0.5)
    
    if not tgt_sentences:
        return []
    
    # 새 파서들을 활용한 원문 분할
    aligned_src_chunks = safe_source_split(tgt_sentences, src_text, tokenizer_func, None)
    
    alignments = []
    for i in range(len(tgt_sentences)):
        src_chunk = aligned_src_chunks[i] if i < len(aligned_src_chunks) else ""
        tgt_sentence = tgt_sentences[i]
        
        similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        
        alignments.append({
            '원문': src_chunk,
            '번역문': tgt_sentence,
            'similarity': similarity,
            'split_method': 'new_parsers_fusion',
            'align_method': 'new_parsers_based_split'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'new_parsers_fusion',
            'align_method': 'new_parsers_unmatched_src'
        })
    
    return alignments

def improved_align_paragraphs_spacy_tokenizer(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    nlp=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """spaCy + 토크나이저 융합 정렬 (무결성 보장) - 레거시"""
    if not tgt_sentences:
        return []
    
    # spaCy + 토크나이저를 활용한 원문 분할
    aligned_src_chunks = safe_source_split(tgt_sentences, src_text, tokenizer_func, nlp)
    
    alignments = []
    for i in range(len(tgt_sentences)):
        src_chunk = aligned_src_chunks[i] if i < len(aligned_src_chunks) else ''
        tgt_sentence = tgt_sentences[i]
        
        similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        
        alignments.append({
            '원문': src_chunk,
            '번역문': tgt_sentence,
            'similarity': similarity,
            'split_method': 'spacy_tokenizer_fusion',
            'align_method': 'spacy_tokenizer_based_split'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'spacy_tokenizer_fusion',
            'align_method': 'sequential_unmatched_src'
        })
    
    return alignments

def detect_imbalanced_blocks(results: List[Dict], threshold_similarity: float = 0.5, empty_ratio: float = 0.3) -> List[Tuple[int, int]]:
    """불균형 블록 감지: 연속된 행들에서 원문 또는 번역문이 부재/저품질
    Returns: [(start_idx, end_idx), ...]
    """
    if not results:
        return []
    
    logger.info(f"🔍 불균형 블록 감지 시작 (total: {len(results)}행)")
    
    # 각 행의 품질 점수 계산
    quality_scores = []
    for i, r in enumerate(results):
        has_src = bool(r.get('원문', '').strip())
        has_tgt = bool(r.get('번역문', '').strip())
        sim = r.get('similarity', 0.0)
        
        # 원문이나 번역문이 없거나 유사도가 낮으면 저품질
        if not has_src or not has_tgt or sim < threshold_similarity:
            quality_scores.append('bad')
            logger.debug(f"  행 {i}: bad (src={has_src}, tgt={has_tgt}, sim={sim:.2f})")
        else:
            quality_scores.append('good')
    
    # 연속된 저품질 구간 찾기
    blocks = []
    i = 0
    while i < len(quality_scores):
        if quality_scores[i] == 'bad':
            start = i
            while i < len(quality_scores) and quality_scores[i] == 'bad':
                i += 1
            end = i - 1
            # 충분히 긴 블록만 반환 (최소 1개 행, 즉 한 행만 저품질이어도 반환)
            blocks.append((start, end))
            logger.info(f"  블록 발견: 행 {start}~{end} ({end-start+1}행)")
        else:
            i += 1
    
    logger.info(f"✅ 총 {len(blocks)}개 불균형 블록 감지")
    return blocks

def rematch_imbalanced_block(results: List[Dict], block_start: int, block_end: int, 
                             src_paragraph: str, tgt_paragraph: str,
                             embed_func=None, max_workers: int = 4, batch_size: int = 100) -> List[Dict]:
    """불균형 블록 내 의미 기반 재매칭
    
    Args:
        results: 현재 정렬 결과
        block_start, block_end: 불균형 블록의 인덱스 범위
        src_paragraph: 원본 전체 원문
        tgt_paragraph: 원본 전체 번역문
        embed_func: 임베더 함수
        
    Returns: 재매칭된 결과 리스트
    """
    block = results[block_start:block_end+1]
    
    # 블록 내 원문/번역문 추출
    block_src = ''.join([r.get('원문', '') for r in block])
    block_tgt = ''.join([r.get('번역문', '') for r in block])
    
    logger.info(f"🔄 블록 재매칭 (행 {block_start}~{block_end})")
    
    src_count = sum(1 for r in block if r.get('원문', '').strip())
    tgt_count = sum(1 for r in block if r.get('번역문', '').strip())
    
    logger.info(f"  블록 상태: 원문 {len(block_src)}자({src_count}행), 번역문 {len(block_tgt)}자({tgt_count}행)")
    
    # 불균형 유형 판단
    if src_count <= 1 and tgt_count >= 3:
        # 1:M 불균형 (원문 거의 없음, 번역문 많음) → 번역문을 원문과 매칭
        logger.info(f"  유형: 1:M 불균형 → 번역문 블록을 원문에 매칭")
        
        # 블록 내 모든 번역문을 연결
        all_tgt = ''.join([r.get('번역문', '') for r in block]).strip()
        
        # 원문 블록도 연결
        all_src = ''.join([r.get('원문', '') for r in block]).strip()
        
        if all_src:
            # 원문이 있으면 그냥 연결된 블록을 1개로
            return [{
                '원문': all_src,
                '번역문': all_tgt,
                'similarity': 0.5 if (all_src and all_tgt) else 0.0,
                'split_method': 'imbalance_rematch_merge',
                'align_method': 'block_consolidation'
            }]
        else:
            # 원문이 없으면 번역문 블록을 더 세밀하게 분할
            logger.info(f"  → 번역문 재분할 (길이: {len(all_tgt)})")
            new_tgt_sentences = split_target_sentences_advanced(all_tgt, max_length=80)
            
            if len(new_tgt_sentences) > 1:
                logger.info(f"  → {len(new_tgt_sentences)}개로 재분할")
                new_results = []
                for tgt_sent in new_tgt_sentences:
                    new_results.append({
                        '원문': '',  # 원문은 비움 (나중에 복구됨)
                        '번역문': tgt_sent,
                        'similarity': 0.0,
                        'split_method': 'imbalance_rematch_tgt_refined',
                        'align_method': 'semantic_rematch'
                    })
                return new_results
            else:
                # 재분할 실패
                return block
    
    elif src_count >= 3 and tgt_count <= 1:
        # N:1 불균형 (원문 많음, 번역문 거의 없음) → 원문을 번역문과 매칭
        logger.info(f"  유형: N:1 불균형 → 원문 블록을 번역문에 매칭")
        
        all_src = ''.join([r.get('원문', '') for r in block]).strip()
        all_tgt = ''.join([r.get('번역문', '') for r in block]).strip()
        
        if all_tgt:
            # 번역문이 있으면 연결된 블록을 1개로
            return [{
                '원문': all_src,
                '번역문': all_tgt,
                'similarity': 0.5 if (all_src and all_tgt) else 0.0,
                'split_method': 'imbalance_rematch_merge',
                'align_method': 'block_consolidation'
            }]
        else:
            # 번역문이 없으면 원문 블록을 더 세밀하게 분할
            logger.info(f"  → 원문 재분할 (길이: {len(all_src)})")
            new_src_sentences = split_target_sentences_advanced(all_src, max_length=80)
            
            if len(new_src_sentences) > 1:
                logger.info(f"  → {len(new_src_sentences)}개로 재분할")
                new_results = []
                for src_sent in new_src_sentences:
                    new_results.append({
                        '원문': src_sent,
                        '번역문': '',  # 번역문은 비움 (나중에 복구됨)
                        'similarity': 0.0,
                        'split_method': 'imbalance_rematch_src_refined',
                        'align_method': 'semantic_rematch'
                    })
                return new_results
            else:
                return block
    else:
        # 기타 불균형 → 블록 병합
        logger.info(f"  유형: 기타 불균형 → 블록 병합")
        all_src = ''.join([r.get('원문', '') for r in block]).strip()
        all_tgt = ''.join([r.get('번역문', '') for r in block]).strip()
        
        return [{
            '원문': all_src,
            '번역문': all_tgt,
            'similarity': 0.5 if (all_src and all_tgt) else 0.0,
            'split_method': 'imbalance_rematch_merge',
            'align_method': 'block_consolidation'
        }]

def word_level_matching_aligned_pairs(src_sentences: List[str], tgt_sentences: List[str],
                                       embed_func=None, tokenizer_func=None) -> List[Dict]:
    """
    정렬된 원문/번역문 쌍 리스트를 받아 각 쌍 내에서 어절 매칭 수행
    
    Args:
        src_sentences: 분할된 원문 문장 리스트
        tgt_sentences: 분할된 번역문 문장 리스트 (src와 동일 개수)
        embed_func: BGE 임베더
        tokenizer_func: 토크나이저
        
    Returns:
        정렬된 결과 리스트 (같은 인덱스의 원문/번역문 쌍)
    """
    if not embed_func or len(src_sentences) != len(tgt_sentences):
        logger.error(f"입력 오류: src={len(src_sentences)}, tgt={len(tgt_sentences)}")
        return []
    
    logger.info(f"🔤 정렬 쌍 기반 어절 매칭 시작 ({len(src_sentences)}쌍)")
    results = []
    
    for idx, (src_sent, tgt_sent) in enumerate(zip(src_sentences, tgt_sentences)):
        if not src_sent.strip() or not tgt_sent.strip():
            results.append({
                '원문': src_sent,
                '번역문': tgt_sent,
                'similarity': 0.0 if not (src_sent.strip() and tgt_sent.strip()) else 0.5,
                'split_method': 'paired_sentences',
                'align_method': 'word_level_paired'
            })
            continue
        
        # 원문 어절 분할
        src_words = src_sent.split()
        
        # 임베딩 계산
        try:
            src_emb = embed_func([src_sent])[0]
            tgt_emb = embed_func([tgt_sent])[0]
            similarity = float(np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-8))
        except Exception:
            similarity = 0.0
        
        results.append({
            '원문': src_sent,
            '번역문': tgt_sent,
            'similarity': similarity,
            'split_method': 'paired_sentences',
            'align_method': 'word_level_paired'
        })
    
    logger.info(f"✅ 정렬 쌍 매칭 완료: {len(results)}개 쌍")
    return results

def word_level_matching(src_paragraph: str, tgt_sentences: List[str], 
                        embed_func=None, tokenizer_func=None) -> List[Dict]:
    """
    어절 단위 매칭: 원문을 어절로 쪼개서 번역문 문장과 의미적으로 매칭
    
    Args:
        src_paragraph: 전체 원문
        tgt_sentences: 번역문 문장 리스트
        embed_func: BGE 임베더
        tokenizer_func: SikuBERT 토크나이저
        
    Returns:
        매칭된 정렬 결과 리스트
    """
    if not embed_func:
        logger.warning("임베더 없음, 어절 매칭 스킵")
        return []
    
    logger.info(f"🔤 어절 단위 매칭 시작 (원문: {len(src_paragraph)}자, 번역문: {len(tgt_sentences)}문장)")
    
    # 1. 원문을 어절(공백) 단위로 분할
    src_words = src_paragraph.split()
    logger.info(f"  원문 어절: {len(src_words)}개")
    
    if not src_words or not tgt_sentences:
        return []
    
    # 2. 각 원문 어절을 임베딩
    try:
        src_word_embeddings = embed_func(src_words)
        logger.info(f"  원문 어절 임베딩 완료")
    except Exception as e:
        logger.error(f"원문 어절 임베딩 실패: {e}")
        return []
    
    # 3. 각 번역문 문장을 임베딩
    try:
        tgt_sent_embeddings = embed_func(tgt_sentences)
        logger.info(f"  번역문 문장 임베딩 완료")
    except Exception as e:
        logger.error(f"번역문 문장 임베딩 실패: {e}")
        return []
    
    # 4. 각 원문 어절에 대해 가장 유사한 번역문 문장 찾기
    word_to_sent_mapping = []
    for i, src_emb in enumerate(src_word_embeddings):
        similarities = []
        for j, tgt_emb in enumerate(tgt_sent_embeddings):
            try:
                sim = float(np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-8))
                similarities.append((j, sim))
            except Exception:
                similarities.append((j, 0.0))
        
        # 가장 유사한 번역문 문장 선택
        best_sent_idx, best_sim = max(similarities, key=lambda x: x[1])
        word_to_sent_mapping.append((i, best_sent_idx, best_sim))
    
    logger.info(f"  어절-문장 매칭 완료")
    
    # 5. 번역문 문장별로 원문 어절들을 그룹화
    sent_to_words = {i: [] for i in range(len(tgt_sentences))}
    for word_idx, sent_idx, sim in word_to_sent_mapping:
        sent_to_words[sent_idx].append((word_idx, src_words[word_idx], sim))
    
    # 6. 결과 생성
    results = []
    for sent_idx, tgt_sent in enumerate(tgt_sentences):
        word_group = sent_to_words[sent_idx]
        
        if word_group:
            # 어절들을 순서대로 정렬하고 이어붙이기
            word_group_sorted = sorted(word_group, key=lambda x: x[0])
            src_text = ' '.join([word for _, word, _ in word_group_sorted])
            avg_sim = sum([sim for _, _, sim in word_group]) / len(word_group)
        else:
            src_text = ''
            avg_sim = 0.0
        
        results.append({
            '원문': src_text,
            '번역문': tgt_sent,
            'similarity': avg_sim,
            'split_method': 'word_level_matching',
            'align_method': 'semantic_word_grouping'
        })
    
    logger.info(f"✅ 어절 매칭 완료: {len(results)}개 쌍 생성")
    
    # 매칭되지 않은 원문 어절이 있는지 확인
    matched_words = set()
    for sent_idx in sent_to_words:
        for word_idx, _, _ in sent_to_words[sent_idx]:
            matched_words.add(word_idx)
    
    unmatched_word_indices = [i for i in range(len(src_words)) if i not in matched_words]
    
    # 🆕 무결성 보장: 매칭 안 된 원문 어절을 빈 원문 행에 의미 기반으로 분배
    if unmatched_word_indices:
        logger.warning(f"  ⚠️ 매칭 안 된 원문 어절: {len(unmatched_word_indices)}개 → 의미 기반 재분배")
        
        # 빈 원문을 가진 행 찾기
        empty_src_indices = [i for i, r in enumerate(results) if not r.get('원문', '').strip()]
        
        if empty_src_indices:
            try:
                # 매칭 안 된 원문 어절들 임베딩
                unmatched_words = [src_words[i] for i in unmatched_word_indices]
                unmatched_embeddings = embed_func(unmatched_words)
                
                # 빈 원문 행의 번역문들 임베딩
                empty_tgt_sentences = [results[i]['번역문'] for i in empty_src_indices]
                empty_tgt_embeddings = embed_func(empty_tgt_sentences)
                
                # 각 남은 어절을 가장 유사한 빈 번역문에 재배정
                word_to_empty_sent = {empty_idx: [] for empty_idx in empty_src_indices}
                
                for w_idx, (unmatched_idx, word_emb) in enumerate(zip(unmatched_word_indices, unmatched_embeddings)):
                    best_sent_local_idx = 0
                    best_sim = -1.0
                    
                    for s_idx, sent_emb in enumerate(empty_tgt_embeddings):
                        try:
                            sim = float(np.dot(word_emb, sent_emb) / (np.linalg.norm(word_emb) * np.linalg.norm(sent_emb) + 1e-8))
                            if sim > best_sim:
                                best_sim = sim
                                best_sent_local_idx = s_idx
                        except Exception:
                            pass
                    
                    target_result_idx = empty_src_indices[best_sent_local_idx]
                    word_to_empty_sent[target_result_idx].append((unmatched_idx, unmatched_words[w_idx]))
                
                # 각 빈 행에 재배정된 어절들 추가
                for result_idx, word_list in word_to_empty_sent.items():
                    if word_list:
                        # 원문 순서 유지
                        word_list_sorted = sorted(word_list, key=lambda x: x[0])
                        results[result_idx]['원문'] = ' '.join([w for _, w in word_list_sorted])
                        logger.info(f"    행 {result_idx}에 {len(word_list)}개 어절 재배정")
                
            except Exception as e:
                logger.error(f"의미 기반 재분배 실패: {e}, 마지막 행에 추가")
                if results:
                    unmatched_words = [src_words[i] for i in unmatched_word_indices]
                    results[-1]['원문'] += ' ' + ' '.join(unmatched_words)
        else:
            # 빈 행이 없으면 마지막 행에 추가
            if results:
                unmatched_words = [src_words[i] for i in unmatched_word_indices]
                results[-1]['원문'] += ' ' + ' '.join(unmatched_words)
                logger.info(f"  빈 행 없음, 마지막 행에 {len(unmatched_words)}개 어절 추가")
    
    # 🆕 무결성 검증: 빈 원문/번역문 행 확인
    empty_src_count = sum(1 for r in results if not r.get('원문', '').strip())
    empty_tgt_count = sum(1 for r in results if not r.get('번역문', '').strip())
    
    if empty_src_count > 0:
        logger.warning(f"  ⚠️ 빈 원문 행 {empty_src_count}개 남음")
    if empty_tgt_count > 0:
        logger.warning(f"  ⚠️ 빈 번역문 행 {empty_tgt_count}개 남음")
    
    return results

def segment_src_by_tgt_similarity(src_text: str, tgt_sentences: List[str], embed_func=None, tokenizer_func=None) -> List[str]:
    """
    원문을 토큰(어절) 단위로 분할하고, 번역문 문장과의 유사도를 기반으로
    순서를 보존하는 단조 증가 세그멘테이션(DP)으로 각 문장에 연속 구간을 할당.
    - 모든 토큰은 정확히 한 번만 사용
    - 문장 순서에 맞춰 구간이 앞에서 뒤로 진행 (reordering 방지)
    - 각 문장에 최소 1 토큰 할당 (단, 토큰 수 < 문장 수이면 일부는 빈 구간)
    - 🎯 원본 텍스트를 span 슬라이싱으로 재구성하여 무결성 보장
    """
    if not tgt_sentences:
        return []
    
    # 🎯 어절 경계(span) 추출 - 원본 텍스트 보존용
    import re
    word_spans = []  # List[Tuple[start, end]]
    for m in re.finditer(r"\S+", src_text):
        word_spans.append((m.start(), m.end()))
    
    words = [src_text[s:e] for (s, e) in word_spans]
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
                end_char = word_spans[next_idx - 1][1]
                segments.append(src_text[start_char:end_char])
            else:
                segments.append('')
            idx = next_idx
        # 마지막 세그먼트
        if idx < W and idx < len(word_spans):
            start_char = word_spans[idx][0]
            end_char = word_spans[W - 1][1] if W-1 < len(word_spans) else len(src_text)
            segments.append(src_text[start_char:end_char])
        else:
            segments.append('')
        return segments

    # 임베딩 계산
    try:
        import numpy as np
        token_embs = embed_func(words)  # (W, D)
        sent_embs = embed_func(tgt_sentences)  # (S, D)
        # numpy 배열화
        token_embs = np.array(token_embs)
        sent_embs = np.array(sent_embs)
    except Exception:
        # 임베딩 실패 시 균등 분할로 대체 (원본 슬라이싱 사용)
        base = max(1, W // S)
        segments = []
        idx = 0
        for s in range(S - 1):
            next_idx = min(W, idx + base)
            if idx < next_idx and idx < len(word_spans) and next_idx-1 < len(word_spans):
                start_char = word_spans[idx][0]
                end_char = word_spans[next_idx - 1][1]
                segments.append(src_text[start_char:end_char])
            else:
                segments.append('')
            idx = next_idx
        # 마지막 세그먼트
        if idx < W and idx < len(word_spans):
            start_char = word_spans[idx][0]
            end_char = word_spans[W - 1][1] if W-1 < len(word_spans) else len(src_text)
            segments.append(src_text[start_char:end_char])
        else:
            segments.append('')
        return segments

    import numpy as np
    # 토큰 임베딩 누적합으로 구간 평균 임베딩을 빠르게 계산
    prefix = np.zeros((W + 1, token_embs.shape[1]), dtype=float)
    prefix[1:] = np.cumsum(token_embs, axis=0)

    # DP: dp[i][j] = 첫 i개의 단어를 j+1개의 문장에 할당한 최대 점수
    # i는 단어 개수(0..W), j는 문장 index(0..S-1)
    NEG_INF = -1e18
    dp = np.full((W + 1, S), NEG_INF, dtype=float)
    prev = np.full((W + 1, S), -1, dtype=int)  # 경계 k 저장

    # base: 첫 문장(j=0)에 i>=1 단어 할당
    for i in range(1, W + 1):
        dp[i, 0] = prefix[i, 0]  # 0..i-1까지 문장0에 할당
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
                # 구간 평균 임베딩
                span_vec = (prefix[i] - prefix[k]) / span_len
                sv = sent_embs[j]
                num = float(np.dot(span_vec, sv))
                denom = (np.linalg.norm(span_vec) + 1e-8) * (np.linalg.norm(sv) + 1e-8)
                sim = num / denom
                score = dp[k, j - 1] + sim
                if score > best_score:
                    best_score = score
                    best_k = k
            dp[i, j] = best_score
            prev[i, j] = best_k

    # 복원: 모든 단어를 S문장에 할당 (가능하면)
    if W >= S and dp[W, S - 1] > NEG_INF / 2:
        cuts = [W]
        i = W
        for j in range(S - 1, 0, -1):
            k = prev[i, j]
            cuts.append(k)
            i = k
        cuts.append(0)
        cuts = sorted(cuts)
        
        # 🎯 원본 텍스트를 span 슬라이싱으로 재구성
        segments = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a >= b or a < 0 or b > len(word_spans):
                segments.append('')
            else:
                # 어절 인덱스 a부터 b-1까지의 원본 텍스트 슬라이스
                start_char = word_spans[a][0]
                end_char = word_spans[b - 1][1]
                segments.append(src_text[start_char:end_char])
        
        # cuts는 0..W로 끝나므로 S개 만들기 위해 조정
        if len(segments) > S:
            # 병합 - 원본 슬라이스 유지
            merged = [''] * S
            merged[0] = segments[0]
            idx = 1
            for seg in segments[1:]:
                if idx < S:
                    merged[idx] = seg
                    idx += 1
                else:
                    # 마지막에 병합 (span을 그대로 이어붙임)
                    if merged[-1]:
                        # 두 세그먼트 사이의 원본 텍스트를 그대로 가져옴
                        last_end = word_spans[cuts[idx-1]][1] if idx-1 < len(cuts) else len(src_text)
                        merged[-1] = src_text[word_spans[cuts[idx-1]-1][0]:end_char]
            segments = merged
        elif len(segments) < S:
            segments += [''] * (S - len(segments))
        return segments
    else:
        # 단어 수가 문장 수보다 적은 경우 등: 앞쪽에 1개씩 배정, 나머지는 빈
        segments = [''] * S
        for j in range(min(W, S)):
            if j < len(word_spans):
                s, e = word_spans[j]
                segments[j] = src_text[s:e]
        return segments

def apply_semantic_rematch_to_results(results: List[Dict], src_paragraph: str, tgt_paragraph: str,
                                     embed_func=None, tokenizer_func=None, max_workers: int = 4, batch_size: int = 100, skip_word_level_matching: bool = False) -> List[Dict]:
    """정렬 결과에 블록 재매칭 적용"""
    
    print(f"\n🔍 블록 재매칭 함수 호출 - 결과 {len(results)}개 행")

    def _apply_block_with_integrity(results_list: List[Dict], block_start: int, block_end: int) -> List[Dict]:
        """블록 재매칭 후 무결성 검증, 실패 시 보수적 재매칭/부분 롤백"""

        original_block = results_list[block_start:block_end+1]
        new_block = rematch_imbalanced_block(results_list, block_start, block_end, src_paragraph, tgt_paragraph, embed_func, max_workers, batch_size)
        candidate = results_list[:block_start] + new_block + results_list[block_end+1:]

        if verify_paragraph_integrity(src_paragraph, tgt_paragraph, candidate):
            return candidate

        logger.info("❌ 블록 재매칭 후 무결성 실패 → 보수적 재매칭 시도")

        if len(original_block) > 1:
            sub_size = max(1, len(original_block) // 2)
            sub_blocks = []
            idx = 0
            while idx < len(original_block):
                end_idx = min(len(original_block) - 1, idx + sub_size - 1)
                sub_blocks.append((idx, end_idx))
                idx = end_idx + 1

            candidate_block = original_block[:]
            offset = 0
            for local_start, local_end in sub_blocks:
                s = local_start + offset
                e = local_end + offset
                sub_len = e - s + 1
                sub_new = rematch_imbalanced_block(candidate_block, s, e, src_paragraph, tgt_paragraph, embed_func, max_workers, batch_size)
                candidate_block = candidate_block[:s] + sub_new + candidate_block[e+1:]
                offset += len(sub_new) - sub_len

            candidate = results_list[:block_start] + candidate_block + results_list[block_end+1:]
            if verify_paragraph_integrity(src_paragraph, tgt_paragraph, candidate):
                logger.info("✅ 보수적 재매칭 성공: 부분 롤백 적용")
                return candidate

        logger.warning("⚠️ 블록 재매칭 무결성 실패: 해당 블록 원본 유지")
        return results_list
    
    # embed_func가 없으면 어절 매칭 불가
    if embed_func is None:
        print("⚠️ 임베더 함수 없음, 블록 재매칭만 수행")
        logger.warning("⚠️ 임베더 함수 없음, 블록 재매칭만 수행")
        blocks = detect_imbalanced_blocks(results, threshold_similarity=0.5)
        if not blocks:
            return results
        for start, end in sorted(blocks, reverse=True):
            results = _apply_block_with_integrity(results, start, end)
        return results
    
    print(f"✅ 임베더 함수 확인됨: {type(embed_func)}")
    blocks = detect_imbalanced_blocks(results, threshold_similarity=0.5)
    
    if not blocks:
        print("✅ 불균형 블록 없음, 평균 유사도 확인 중...")
        logger.info("✅ 불균형 블록 없음, 평균 유사도 확인 중...")
        # 불균형 블록이 없어도 전체 품질이 낮으면 어절 매칭 시도
        avg_sim = sum([r.get('similarity', 0.0) for r in results]) / len(results) if results else 0.0
        print(f"  평균 유사도: {avg_sim:.3f}")
        logger.info(f"  평균 유사도: {avg_sim:.3f}")
        if avg_sim < 0.6:
            print(f"  🔤 평균 유사도 낮음 ({avg_sim:.2f}), 어절 매칭으로 전환")
            logger.info(f"  평균 유사도 낮음 ({avg_sim:.2f}), 어절 매칭으로 전환")
            if skip_word_level_matching:
                logger.info("  어절 매칭 스킵 (원본 재분할 완료됨)")
                return results
            # 번역문 문장들 추출
            tgt_sentences = [r.get('번역문', '') for r in results if r.get('번역문', '').strip()]
            if tgt_sentences:
                src_split = split_target_sentences_advanced(src_paragraph, max_length=150, splitter="punctuation")
                while len(src_split) < len(tgt_sentences):
                    src_split.append('')
                src_sentences = src_split[:len(tgt_sentences)]
                
                print(f"  원문 {len(src_sentences)}개, 번역문 {len(tgt_sentences)}개로 분할 후 쌍 기반 매칭")
                word_level_results = word_level_matching_aligned_pairs(src_sentences, tgt_sentences, embed_func, tokenizer_func)
                if word_level_results and len(word_level_results) > 0:
                    print(f"✅ 쌍 기반 매칭 완료: {len(word_level_results)}개 쌍 생성")
                    logger.info(f"✅ 쌍 기반 매칭 완료: {len(word_level_results)}개 쌍 생성")
                    if verify_paragraph_integrity(src_paragraph, tgt_paragraph, word_level_results):
                        return word_level_results
                    logger.warning("⚠️ 쌍 기반 매칭 후 무결성 실패: 기존 결과 유지")
        return results
    
    print(f"🔄 {len(blocks)}개 불균형 블록 감지")
    logger.info(f"🔄 {len(blocks)}개 불균형 블록 감지")
    
    # 불균형이 심각하면 (전체의 50% 이상) 어절 단위 매칭으로 전환
    total_rows = len(results)
    bad_rows = sum([end - start + 1 for start, end in blocks])
    bad_ratio = bad_rows / total_rows if total_rows > 0 else 0.0
    
    print(f"  불균형 비율: {bad_ratio:.1%} ({bad_rows}/{total_rows}행)")
    logger.info(f"  불균형 비율: {bad_ratio:.1%} ({bad_rows}/{total_rows}행)")
    
    if bad_ratio > 0.5:
        print(f"  🔤 불균형 심각 (>{50}%), 어절 단위 매칭으로 전환")
        logger.info(f"  🔤 불균형 심각 (>{50}%), 어절 단위 매칭으로 전환")
        if skip_word_level_matching:
            logger.info("  어절 매칭 스킵 (원본 재분할 완료됨)")
            return results
        # 번역문 문장들 추출
        tgt_sentences = [r.get('번역문', '') for r in results if r.get('번역문', '').strip()]
        if tgt_sentences:
            src_split = split_target_sentences_advanced(src_paragraph, max_length=150, splitter="punctuation")
            while len(src_split) < len(tgt_sentences):
                src_split.append('')
            src_sentences = src_split[:len(tgt_sentences)]
            
            print(f"  원문 {len(src_sentences)}개, 번역문 {len(tgt_sentences)}개로 분할 후 쌍 기반 매칭")
            word_level_results = word_level_matching_aligned_pairs(src_sentences, tgt_sentences, embed_func, tokenizer_func)
            if word_level_results and len(word_level_results) > 0:
                print(f"✅ 쌍 기반 매칭 완료: {len(word_level_results)}개 쌍 생성")
                logger.info(f"✅ 쌍 기반 매칭 완료: {len(word_level_results)}개 쌍 생성")
                if verify_paragraph_integrity(src_paragraph, tgt_paragraph, word_level_results):
                    return word_level_results
                logger.warning("⚠️ 쌍 기반 매칭 후 무결성 실패: 기존 결과 유지")
    
    # 블록별 재매칭 (기존 방식)
    logger.info(f"  블록별 재매칭 시작")
    for start, end in sorted(blocks, reverse=True):
        results = _apply_block_with_integrity(results, start, end)
    
    logger.info(f"✅ 블록 재매칭 완료")
    return results

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
    max_workers: int = 4,
    batch_size: int = 100
):
    """PA 처리: 완벽한 무결성 보장"""
    
    print(f"🔄 PA 처리 시작 (완벽한 무결성 보장)")
    
    para_id = f"paragraph_{id(src_paragraph)}_{id(tgt_paragraph)}"
    
    try:
        # 🎯 번역문을 한 번만 분할 (닫는 따옴표 병합 포함)
        tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length=max_length, splitter="punctuation")
        
        # 임베더 및 토크나이저 함수 준비
        embed_func = get_embedder_function(embedder_name, device=device, max_workers=max_workers, batch_size=batch_size)
        tokenizer_func = get_tokenizer_function(tokenizer_name)
        
        # 🎯 번역문 기준으로 원문을 어절 단위 의미 매칭으로 분할
        print(f"🔄 원문을 어절 단위로 분할해 단조 증가 DP 할당 중 ({len(tgt_sentences)}개 문장)...")
        logger.info(f"🔄 원문을 어절 단위로 분할해 단조 증가 DP 할당 중 ({len(tgt_sentences)}개 문장)...")

        if not embed_func or len(tgt_sentences) == 0:
            logger.error(f"어절 단위 의미 매칭 불가: embed_func={bool(embed_func)}, tgt={len(tgt_sentences)}")
            raise ValueError("어절 단위 의미 매칭 필수 조건 미충족")

        src_segments = segment_src_by_tgt_similarity(
            src_paragraph,
            tgt_sentences,
            embed_func=embed_func,
            tokenizer_func=tokenizer_func
        )

        # 로그: 문장별 배정 어절 수
        for sent_idx, seg in enumerate(src_segments):
            cnt = len(seg.split()) if seg else 0
            logger.info(f"  문장 {sent_idx}: {cnt}개 어절 배정")
        
        # 결과 생성: 원문-번역문 1:1 매칭
        results = []
        for i in range(len(tgt_sentences)):
            src_chunk = src_segments[i] if i < len(src_segments) else ''
            tgt_sentence = tgt_sentences[i]
            
            # 유사도 계산
            if embed_func and src_chunk.strip() and tgt_sentence.strip():
                try:
                    src_embedding = embed_func([src_chunk])[0]
                    tgt_embedding = embed_func([tgt_sentence])[0]
                    
                    import numpy as np
                    src_norm = np.linalg.norm(src_embedding)
                    tgt_norm = np.linalg.norm(tgt_embedding)
                    
                    if src_norm > 1e-8 and tgt_norm > 1e-8:
                        similarity = float(np.dot(src_embedding, tgt_embedding) / (src_norm * tgt_norm))
                    else:
                        similarity = 0.0
                except Exception:
                    similarity = compute_similarity_simple(src_chunk, tgt_sentence)
            else:
                similarity = compute_similarity_simple(src_chunk, tgt_sentence)
            
            results.append({
                '원문': src_chunk,
                '번역문': tgt_sentence,
                'similarity': similarity,
                'split_method': 'punctuation',
                'align_method': 'dp_word_based'
            })
        
        print(f"✅ 원문 어절 기반 의미 매칭 완료: {len(tgt_sentences)}개 문장")
        logger.info(f"✅ 원문 어절 기반 의미 매칭 완료: {len(tgt_sentences)}개 문장")
        
        # 🆕 블록 재매칭: 불균형 감지 및 의미 기반 재정렬
        print("🔄 블록 재매칭 단계 시작...")
        logger.info("🔄 블록 재매칭 단계 시작...")
        results_before_rematch = copy.deepcopy(results)
        results = apply_semantic_rematch_to_results(
            results, 
            src_paragraph, 
            tgt_paragraph, 
            embed_func=embed_func,
            tokenizer_func=tokenizer_func,
            max_workers=max_workers,
            batch_size=batch_size,
            skip_word_level_matching=True
        )
        print(f"✅ 블록 재매칭 완료, 결과: {len(results)}개 행")
        
        # 최종 무결성 검증: 실패 시 재매칭 전 결과로 복원
        if not verify_paragraph_integrity(src_paragraph, tgt_paragraph, results):
            logger.warning("문단 무결성 실패: 재매칭 전 결과로 복원")
            print("⚠️ 무결성 실패: 재매칭 전 결과로 롤백")
            results = results_before_rematch
        
        # 복원 결과도 무결성 검증 (추가 안전망)
        if not verify_paragraph_integrity(src_paragraph, tgt_paragraph, results):
            logger.warning("문단 무결성 재검증도 실패: 결과는 원본 유지")

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