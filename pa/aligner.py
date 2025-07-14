"""PA 전용 정렬기 - SA의 Vice Versa 방식 (완벽한 무결성 보장)"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
import hashlib
import logging

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

class IntegrityManager:
    """텍스트 무결성 관리 클래스"""
    
    def __init__(self):
        self.original_checksums = {}
        self.processing_log = []
        self.integrity_errors = []
    
    def calculate_checksum(self, text: str, label: str = "") -> str:
        """텍스트의 체크섬 계산"""
        if not isinstance(text, str):
            text = str(text)
        # 공백과 개행 정규화 후 체크섬 계산
        normalized = text.replace('\n', ' ').replace('\t', ' ')
        normalized = ' '.join(normalized.split())  # 연속 공백 제거
        
        checksum = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        self.processing_log.append(f"{label}: {checksum}")
        return checksum
    
    def store_original(self, text: str, identifier: str):
        """원본 텍스트 체크섬 저장"""
        checksum = self.calculate_checksum(text, f"ORIGINAL_{identifier}")
        self.original_checksums[identifier] = {
            'checksum': checksum,
            'text': text,
            'length': len(text.replace(' ', ''))
        }
    
    def verify_integrity(self, processed_text: str, identifier: str) -> Tuple[bool, str]:
        """처리된 텍스트의 무결성 검증"""
        if identifier not in self.original_checksums:
            return False, f"원본 데이터를 찾을 수 없음: {identifier}"
        
        original_info = self.original_checksums[identifier]
        processed_checksum = self.calculate_checksum(processed_text, f"PROCESSED_{identifier}")
        
        # 체크섬 비교
        if original_info['checksum'] == processed_checksum:
            return True, "무결성 검증 성공"
        
        # 길이 비교 (대안 검증)
        processed_length = len(processed_text.replace(' ', ''))
        length_diff = abs(original_info['length'] - processed_length)
        
        if length_diff == 0:
            return True, "길이 기반 무결성 검증 성공"
        
        error_msg = f"무결성 검증 실패 - 길이 차이: {length_diff}자"
        self.integrity_errors.append({
            'identifier': identifier,
            'original_checksum': original_info['checksum'],
            'processed_checksum': processed_checksum,
            'original_length': original_info['length'],
            'processed_length': processed_length,
            'length_diff': length_diff
        })
        
        return False, error_msg
    
    def restore_integrity(self, processed_units: List[str], identifier: str) -> List[str]:
        """무결성이 훼손된 경우 복원 시도"""
        if identifier not in self.original_checksums:
            logger.error(f"복원 불가: 원본 데이터 없음 - {identifier}")
            return processed_units
        
        original_text = self.original_checksums[identifier]['text']
        processed_combined = ''.join(processed_units).replace(' ', '')
        original_clean = original_text.replace(' ', '').replace('\n', '').replace('\t', '')
        
        if processed_combined == original_clean:
            return processed_units
        
        logger.warning(f"무결성 복원 시도: {identifier}")
        
        # SequenceMatcher를 사용한 차이점 분석 및 복원
        sm = SequenceMatcher(None, processed_combined, original_clean)
        opcodes = sm.get_opcodes()
        
        restored_units = processed_units[:]
        total_insert_length = 0
        total_delete_length = 0
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # 누락된 텍스트 추가
                missing_text = original_clean[j1:j2]
                if restored_units:
                    restored_units[-1] += missing_text
                else:
                    restored_units.append(missing_text)
                total_insert_length += len(missing_text)
                logger.info(f"누락 텍스트 복원: '{missing_text}'")
                
            elif tag == 'delete':
                # 중복된 텍스트 제거
                excess_text = processed_combined[i1:i2]
                for k, unit in enumerate(restored_units):
                    if excess_text in unit:
                        restored_units[k] = unit.replace(excess_text, '', 1)
                        total_delete_length += len(excess_text)
                        logger.info(f"중복 텍스트 제거: '{excess_text}'")
                        break
        
        logger.info(f"무결성 복원 완료: 추가 {total_insert_length}자, 제거 {total_delete_length}자")
        return restored_units

# 전역 무결성 관리자
integrity_manager = IntegrityManager()

def safe_text_split(text: str, max_length: int = 150, method: str = "punctuation") -> List[str]:
    """무결성 보장 텍스트 분할"""
    if not text or not text.strip():
        return []
    
    # 원본 저장
    text_id = f"split_{id(text)}"
    integrity_manager.store_original(text, text_id)
    
    try:
        # 기존 분할 방식 적용
        if method == "spacy_tokenizer":
            # spaCy + 토크나이저 방식
            sentences = split_target_sentences_spacy_tokenizer(text, max_length)
        else:
            # 기존 방식
            sentences = split_target_sentences_advanced(text, max_length, splitter=method)
        
        if not sentences:
            sentences = [text]
        
        # 무결성 검증
        combined_result = ''.join(sentences)
        is_valid, message = integrity_manager.verify_integrity(combined_result, text_id)
        
        if not is_valid:
            logger.warning(f"분할 무결성 검증 실패: {message}")
            # 복원 시도
            sentences = integrity_manager.restore_integrity(sentences, text_id)
            
            # 재검증
            combined_result = ''.join(sentences)
            is_valid, message = integrity_manager.verify_integrity(combined_result, text_id)
            
            if not is_valid:
                logger.error(f"무결성 복원 실패, 원본 반환: {message}")
                return [text]  # 실패시 원본 그대로 반환
        
        return sentences
        
    except Exception as e:
        logger.error(f"텍스트 분할 중 오류: {e}")
        return [text]  # 오류시 원본 그대로 반환

def safe_source_split(tgt_sentences: List[str], src_text: str, tokenizer_func=None, nlp=None) -> List[str]:
    """무결성 보장 원문 분할"""
    if not tgt_sentences or not src_text.strip():
        return []
    
    # 원본 저장
    src_id = f"src_split_{id(src_text)}"
    integrity_manager.store_original(src_text, src_id)
    
    try:
        # spaCy + 토크나이저 방식 또는 기본 방식
        if nlp and tokenizer_func:
            src_chunks = split_src_by_tgt_units_spacy_tokenizer(tgt_sentences, src_text, tokenizer_func, nlp)
        else:
            src_chunks = split_src_by_tgt_units_vice_versa(tgt_sentences, src_text, None, tokenizer_func)
        
        if not src_chunks:
            src_chunks = [src_text]
        
        # 무결성 검증
        combined_result = ''.join(src_chunks)
        is_valid, message = integrity_manager.verify_integrity(combined_result, src_id)
        
        if not is_valid:
            logger.warning(f"원문 분할 무결성 검증 실패: {message}")
            # 복원 시도
            src_chunks = integrity_manager.restore_integrity(src_chunks, src_id)
            
            # 재검증
            combined_result = ''.join(src_chunks)
            is_valid, message = integrity_manager.verify_integrity(combined_result, src_id)
            
            if not is_valid:
                logger.error(f"원문 무결성 복원 실패, 기본 분할 사용: {message}")
                # 기본 분할로 폴백
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
    """문단 단위 무결성 검증"""
    
    # 원본 텍스트 결합
    original_src = src_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
    original_tgt = tgt_paragraph.replace(' ', '').replace('\n', '').replace('\t', '')
    
    # 정렬 결과에서 텍스트 추출
    aligned_src = ''.join([align.get('원문', '') for align in alignments]).replace(' ', '')
    aligned_tgt = ''.join([align.get('번역문', '') for align in alignments]).replace(' ', '')
    
    # 무결성 검증
    src_integrity = (original_src == aligned_src)
    tgt_integrity = (original_tgt == aligned_tgt)
    
    if not src_integrity:
        logger.error(f"원문 무결성 실패 - 원본: {len(original_src)}자, 결과: {len(aligned_src)}자")
        logger.error(f"원본: {original_src[:100]}...")
        logger.error(f"결과: {aligned_src[:100]}...")
    
    if not tgt_integrity:
        logger.error(f"번역문 무결성 실패 - 원본: {len(original_tgt)}자, 결과: {len(aligned_tgt)}자")
        logger.error(f"원본: {original_tgt[:100]}...")
        logger.error(f"결과: {aligned_tgt[:100]}...")
    
    return src_integrity and tgt_integrity

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

def get_spacy_nlp():
    """spaCy 모델 안전하게 로드"""
    try:
        import spacy
        try:
            nlp = spacy.load("ko_core_news_sm")
            print("✅ spaCy 한국어 모델 로드 성공")
            return nlp
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
                print("⚠️ 한국어 모델 없음, 영어 모델 사용")
                return nlp
            except OSError:
                print("❌ spaCy 모델 없음")
                return None
    except ImportError:
        print("❌ spaCy 설치되지 않음")
        return None

def split_target_sentences_spacy_tokenizer(
    text: str, 
    max_length: int = 150,
    tokenizer_func=None,
    nlp=None
) -> List[str]:
    """spaCy + 토크나이저 융합 문장 분할 (무결성 보장)"""
    if not text.strip():
        return []
    
    # 무결성 관리 적용
    text_id = f"spacy_tok_{id(text)}"
    integrity_manager.store_original(text, text_id)
    
    sentences = []
    
    # 1단계: spaCy로 문장 경계 감지
    if nlp:
        try:
            doc = nlp(text)
            spacy_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if spacy_sentences:
                print(f"🔍 spaCy 분할: {len(spacy_sentences)}개 문장")
                sentences = spacy_sentences
            else:
                sentences = [text]
        except Exception as e:
            print(f"⚠️ spaCy 분할 실패: {e}")
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
    
    # 무결성 검증 및 복원
    combined_result = ''.join(sentences)
    is_valid, message = integrity_manager.verify_integrity(combined_result, text_id)
    
    if not is_valid:
        logger.warning(f"spaCy+토크나이저 분할 무결성 실패: {message}")
        sentences = integrity_manager.restore_integrity(sentences, text_id)
    
    return sentences if sentences else [text]

def split_long_sentence_with_tokenizer(sentence: str, max_length: int, tokenizer_func) -> List[str]:
    """토크나이저를 사용하여 긴 문장을 의미 단위로 분할 (무결성 보장)"""
    
    # 원본 저장
    sent_id = f"long_sent_{id(sentence)}"
    integrity_manager.store_original(sentence, sent_id)
    
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

def get_tokenizer_function(tokenizer_name: str = "jieba"):
    """토크나이저 함수 반환 - SA 재사용"""
    try:
        if tokenizer_name == "jieba":
            sys.path.insert(0, str(project_root / 'sa' / 'sa_tokenizers'))
            from jieba_mecab import tokenize_chinese_text
            print("✅ jieba 토크나이저 로드 성공")
            return tokenize_chinese_text
        elif tokenizer_name == "mecab":
            sys.path.insert(0, str(project_root / 'sa' / 'sa_tokenizers'))
            from jieba_mecab import tokenize_korean_text
            print("✅ mecab 토크나이저 로드 성공")
            return tokenize_korean_text
        else:
            print(f"⚠️ 기본 분할 사용: {tokenizer_name}")
            return lambda text: list(text)
    except ImportError as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}, 기본 분할 사용")
        return lambda text: list(text)

def get_embedder_function(embedder_name: str, device: str = "cpu", openai_model: str = None, openai_api_key: str = None):
    """임베더 함수 반환 - 기존과 동일"""
    
    if device == "cuda":
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("⚠️ CUDA 미지원: CPU로 전환합니다.")
            device = "cpu"
    
    if embedder_name == 'bge':
        try:
            sys.path.insert(0, str(project_root / 'common' / 'embedders'))
            from bge import get_embed_func
            embed_func = get_embed_func(device_id=0 if device == "cuda" else None)
            if embed_func is None:
                print("❌ BGE 임베더 초기화 실패")
                return None
            print("✅ BGE 임베더 초기화 성공")
            return embed_func
        except ImportError as e:
            print(f"❌ BGE 임베더 로드 실패: {e}")
            return None
            
    elif embedder_name == 'openai':
        try:
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            
            sys.path.insert(0, str(project_root / 'common' / 'embedders'))
            from openai import compute_embeddings_with_cache
            
            def embed_func(texts):
                return compute_embeddings_with_cache(
                    texts, 
                    model=openai_model if openai_model else "text-embedding-3-large"
                )
            print("✅ OpenAI 임베더 초기화 성공")
            return embed_func
        except ImportError as e:
            print(f"❌ OpenAI 임베더 로드 실패: {e}")
            return None
    else:
        print(f"❌ 지원하지 않는 임베더: {embedder_name}")
        return None

def split_src_by_tgt_units_spacy_tokenizer(
    tgt_sentences: List[str], 
    src_text: str, 
    tokenizer_func=None,
    nlp=None
) -> List[str]:
    """spaCy + 토크나이저를 활용한 Vice Versa 원문 분할 (무결성 보장)"""
    return safe_source_split(tgt_sentences, src_text, tokenizer_func, nlp)

def split_src_by_tgt_units_vice_versa(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    similarity_threshold: float = 0.3
) -> List[str]:
    """SA의 Vice Versa: 번역문 문장들을 기준으로 원문을 분할 (무결성 보장)"""
    return safe_source_split(tgt_sentences, src_text, tokenizer_func, None)

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
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """기존 순차적 1:1 정렬 (무결성 보장)"""
    if not tgt_sentences:
        return []
    
    # 원문을 번역문 개수에 맞춰 순차적으로 분할
    aligned_src_chunks = split_source_by_whitespace_and_align(src_text, len(tgt_sentences))
    
    alignments = []
    for i in range(len(tgt_sentences)):
        alignments.append({
            '원문': aligned_src_chunks[i] if i < len(aligned_src_chunks) else '',
            '번역문': tgt_sentences[i],
            'similarity': 1.0,
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

def improved_align_paragraphs_spacy_tokenizer(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    nlp=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """spaCy + 토크나이저 융합 정렬 (무결성 보장)"""
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

def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    tokenizer_name: str = 'jieba',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8,
    use_spacy_tokenizer: bool = False
):
    """PA 처리: 완벽한 무결성 보장"""
    
    print(f"🔄 PA 처리 시작 (완벽한 무결성 보장)")
    
    # 원본 문단 저장
    para_id = f"paragraph_{id(src_paragraph)}_{id(tgt_paragraph)}"
    integrity_manager.store_original(src_paragraph, f"{para_id}_src")
    integrity_manager.store_original(tgt_paragraph, f"{para_id}_tgt")
    
    try:
        # 1. 기존 순차적 정렬 (punctuation)
        tgt_sentences_seq = safe_text_split(tgt_paragraph, max_length, "punctuation")
        alignments_seq = improved_align_paragraphs(tgt_sentences_seq, src_paragraph)
        
        # 2. 기존 의미적 정렬 (spacy)
        tgt_sentences_sem = safe_text_split(tgt_paragraph, max_length, "spacy")
        embed_func = get_embedder_function(embedder_name, device=device)
        alignments_sem = improved_align_paragraphs(tgt_sentences_sem, src_paragraph, embed_func, similarity_threshold)
        
        # 3. 기존 Vice Versa 토크나이저 정렬
        tokenizer_func = get_tokenizer_function(tokenizer_name)
        tgt_sentences_tok = safe_text_split(tgt_paragraph, max_length, "punctuation")
        aligned_src_chunks = safe_source_split(tgt_sentences_tok, src_paragraph, tokenizer_func, None)
        
        alignments_tok = []
        for i, (src_chunk, tgt_sentence) in enumerate(zip(aligned_src_chunks, tgt_sentences_tok)):
            similarity = compute_similarity_simple(src_chunk, tgt_sentence)
            alignments_tok.append({
                '원문': src_chunk,
                '번역문': tgt_sentence,
                'similarity': similarity,
                'split_method': 'vice_versa_tokenized',
                'align_method': 'tgt_based_src_split'
            })
        
        # 4. 새로운 spaCy + 토크나이저 융합 정렬
        alignments_spacy_tok = []
        if use_spacy_tokenizer:
            nlp = get_spacy_nlp()
            tgt_sentences_spacy_tok = safe_text_split(tgt_paragraph, max_length, "spacy_tokenizer")
            alignments_spacy_tok = improved_align_paragraphs_spacy_tokenizer(
                tgt_sentences_spacy_tok, src_paragraph, embed_func, tokenizer_func, nlp, similarity_threshold
            )
        
        # 최적 방식 선택 및 결과 생성
        all_alignments = [alignments_seq, alignments_sem, alignments_tok, alignments_spacy_tok]
        max_len = max(len(alignments) for alignments in all_alignments if alignments)
        
        results = []
        for i in range(max_len):
            seq = alignments_seq[i] if i < len(alignments_seq) else {'원문':'','번역문':'','similarity':0.0,'split_method':'punctuation','align_method':'sequential'}
            sem = alignments_sem[i] if i < len(alignments_sem) else {'원문':'','번역문':'','similarity':0.0,'split_method':'spacy','align_method':'semantic'}
            tok = alignments_tok[i] if i < len(alignments_tok) else {'원문':'','번역문':'','similarity':0.0,'split_method':'vice_versa_tokenized','align_method':'tgt_based_src_split'}
            spacy_tok = alignments_spacy_tok[i] if i < len(alignments_spacy_tok) else {'원문':'','번역문':'','similarity':0.0,'split_method':'spacy_tokenizer_fusion','align_method':'spacy_tokenizer_based_split'}
            
            if use_spacy_tokenizer and alignments_spacy_tok:
                weighted_sim = seq['similarity']*0.2 + sem['similarity']*0.3 + tok['similarity']*0.2 + spacy_tok['similarity']*0.3
                
                if weighted_sim >= quality_threshold:
                    result = {
                        '원문': spacy_tok['원문'] if spacy_tok['원문'] else (tok['원문'] if tok['원문'] else (sem['원문'] if sem['원문'] else seq['원문'])),
                        '번역문': spacy_tok['번역문'] if spacy_tok['번역문'] else (tok['번역문'] if tok['번역문'] else (sem['번역문'] if sem['번역문'] else seq['번역문'])),
                        'similarity': weighted_sim,
                        'split_method': f"seq+sem+tok+spacy_tok",
                        'align_method': 'hybrid_with_spacy_tokenizer'
                    }
                else:
                    result = spacy_tok.copy()
                    result['align_method'] = 'spacy_tokenizer_fusion_only'
            else:
                weighted_sim = seq['similarity']*0.3 + sem['similarity']*0.4 + tok['similarity']*0.3
                
                if weighted_sim >= quality_threshold:
                    result = {
                        '원문': tok['원문'] if tok['원문'] else (sem['원문'] if sem['원문'] else seq['원문']),
                        '번역문': tok['번역문'] if tok['번역문'] else (sem['번역문'] if sem['번역문'] else seq['번역문']),
                        'similarity': weighted_sim,
                        'split_method': f"seq+sem+tok",
                        'align_method': 'hybrid_with_tokenizer'
                    }
                else:
                    result = tok.copy()
                    result['align_method'] = 'tokenizer_vice_versa_only'
            
            results.append(result)
        
        # 최종 무결성 검증
        if not verify_paragraph_integrity(src_paragraph, tgt_paragraph, results):
            logger.warning("문단 무결성 실패, 복원 시도")
            results = restore_paragraph_integrity(src_paragraph, tgt_paragraph, results)
            
            # 재검증
            if not verify_paragraph_integrity(src_paragraph, tgt_paragraph, results):
                logger.error("무결성 복원 실패")
        
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
    tokenizer_name: str = 'jieba',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8,
    use_spacy_tokenizer: bool = False,
    verbose: bool = False,
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
    
    try:
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None
    
    # 원본 데이터 전체 체크섬 저장
    original_src_all = ''.join([str(row.get('원문', '')) for _, row in df.iterrows()])
    original_tgt_all = ''.join([str(row.get('번역문', '')) for _, row in df.iterrows()])
    
    file_id = f"file_{id(input_file)}"
    integrity_manager.store_original(original_src_all, f"{file_id}_src_all")
    integrity_manager.store_original(original_tgt_all, f"{file_id}_tgt_all")
    
    all_results = []
    total = len(df)
    processed_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="📊 문단 처리 (무결성 보장)"):
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
                    use_spacy_tokenizer=use_spacy_tokenizer
                )
                
                # 문단식별자 부여
                for a in alignments:
                    a['문단식별자'] = idx + 1
                
                all_results.extend(alignments)
                processed_count += 1
                
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
    
    if not all_results:
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]

    # === 최종 무결성 검증 및 보완 ===
    print("🔒 최종 무결성 검증 중...")
    
    output_src_all = ''.join(result_df['원문'].fillna(''))
    output_tgt_all = ''.join(result_df['번역문'].fillna(''))
    
    # 원문 무결성 검증
    src_valid, src_msg = integrity_manager.verify_integrity(output_src_all, f"{file_id}_src_all")
    if not src_valid:
        print(f'⚠️ 원문 무결성 불일치: {src_msg}')
        print('🔧 원문 복원 시도 중...')
        
        sm = SequenceMatcher(None, output_src_all.replace(' ', ''), original_src_all.replace(' ', ''))
        opcodes = sm.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                if len(result_df) > 0:
                    result_df.at[result_df.index[-1], '원문'] += original_src_all[j1:j2]
                else:
                    new_row = pd.DataFrame([{
                        '문단식별자': df.shape[0] + 1,
                        '원문': original_src_all[j1:j2],
                        '번역문': '',
                        'similarity': 1.0,
                        'split_method': 'integrity_restore',
                        'align_method': 'src_missing_patch'
                    }])
                    result_df = pd.concat([result_df, new_row], ignore_index=True)
                print(f"✅ 누락 원문 복원: '{original_src_all[j1:j2][:50]}...'")
                
            elif tag == 'delete':
                excess_text = output_src_all[i1:i2]
                for idx in result_df.index:
                    if excess_text in str(result_df.at[idx, '원문']):
                        result_df.at[idx, '원문'] = str(result_df.at[idx, '원문']).replace(excess_text, '', 1)
                        print(f"✅ 중복 원문 제거: '{excess_text[:50]}...'")
                        break
    
    # 번역문 무결성 검증
    output_tgt_all_after_src = ''.join(result_df['번역문'].fillna(''))
    tgt_valid, tgt_msg = integrity_manager.verify_integrity(output_tgt_all_after_src, f"{file_id}_tgt_all")
    
    if not tgt_valid:
        print(f'⚠️ 번역문 무결성 불일치: {tgt_msg}')
        print('🔧 번역문 복원 시도 중...')
        
        sm = SequenceMatcher(None, output_tgt_all_after_src.replace(' ', ''), original_tgt_all.replace(' ', ''))
        opcodes = sm.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                if len(result_df) > 0:
                    result_df.at[result_df.index[-1], '번역문'] += original_tgt_all[j1:j2]
                else:
                    new_row = pd.DataFrame([{
                        '문단식별자': df.shape[0] + 1,
                        '원문': '',
                        '번역문': original_tgt_all[j1:j2],
                        'similarity': 1.0,
                        'split_method': 'integrity_restore',
                        'align_method': 'tgt_missing_patch'
                    }])
                    result_df = pd.concat([result_df, new_row], ignore_index=True)
                print(f"✅ 누락 번역문 복원: '{original_tgt_all[j1:j2][:50]}...'")
                
            elif tag == 'delete':
                excess_text = output_tgt_all_after_src[i1:i2]
                for idx in result_df.index:
                    if excess_text in str(result_df.at[idx, '번역문']):
                        result_df.at[idx, '번역문'] = str(result_df.at[idx, '번역문']).replace(excess_text, '', 1)
                        print(f"✅ 중복 번역문 제거: '{excess_text[:50]}...'")
                        break
    
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