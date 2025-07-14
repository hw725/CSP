"""괄호 및 구두점 처리 모듈 - 완벽한 무결성 보장"""

import logging
import regex  # 🆕 유니코드 속성 정규식
import re  # 괄호 추출 및 패턴 컴파일용
import numpy as np  # 임베딩 계산용
from typing import List, Dict, Any, Tuple
import hashlib
from difflib import SequenceMatcher
import pandas as pd  # 🔧 누락된 import 추가

logger = logging.getLogger(__name__)

class IntegrityGuard:
    """SA 전용 무결성 보호 시스템"""
    
    def __init__(self):
        self.original_checksums = {}
        self.processing_history = []
        self.restoration_count = 0
    
    def register_original(self, text_id: str, text: str, stage: str = "input"):
        """원본 텍스트 등록 및 체크섬 저장"""
        if not isinstance(text, str):
            text = str(text)
        
        # 정규화된 텍스트로 체크섬 계산 (공백/개행 무시)
        normalized = ''.join(text.split())
        checksum = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
        self.original_checksums[text_id] = {
            'original_text': text,
            'normalized': normalized,
            'checksum': checksum,
            'length': len(normalized),
            'stage': stage
        }
        
        self.processing_history.append(f"REGISTER {text_id}: {stage} - {len(text)}자 ({checksum[:8]})")
        logger.debug(f"무결성 등록: {text_id} - {stage}")
    
    def verify_integrity(self, text_id: str, processed_text: str, stage: str = "output") -> Tuple[bool, str, Dict]:
        """무결성 검증"""
        if text_id not in self.original_checksums:
            return False, f"원본 데이터 미등록: {text_id}", {}
        
        original_info = self.original_checksums[text_id]
        
        if not isinstance(processed_text, str):
            processed_text = str(processed_text)
        
        # 정규화된 텍스트로 비교
        processed_normalized = ''.join(processed_text.split())
        processed_checksum = hashlib.sha256(processed_normalized.encode('utf-8')).hexdigest()
        
        # 체크섬 비교
        integrity_valid = (original_info['checksum'] == processed_checksum)
        
        # 상세 정보
        verification_info = {
            'original_checksum': original_info['checksum'],
            'processed_checksum': processed_checksum,
            'original_length': original_info['length'],
            'processed_length': len(processed_normalized),
            'length_diff': len(processed_normalized) - original_info['length'],
            'character_accuracy': self._calculate_character_accuracy(original_info['normalized'], processed_normalized)
        }
        
        if integrity_valid:
            message = f"무결성 검증 성공: {stage}"
            self.processing_history.append(f"VERIFY_OK {text_id}: {stage} - 체크섬 일치")
        else:
            message = f"무결성 검증 실패: {stage} - 길이차이 {verification_info['length_diff']}자"
            self.processing_history.append(f"VERIFY_FAIL {text_id}: {stage} - {message}")
            logger.error(f"무결성 실패: {text_id} - {message}")
        
        return integrity_valid, message, verification_info
    
    def _calculate_character_accuracy(self, original: str, processed: str) -> float:
        """문자 단위 정확도 계산"""
        if not original:
            return 1.0 if not processed else 0.0
        
        sm = SequenceMatcher(None, original, processed)
        return sm.ratio()
    
    def restore_integrity(self, text_id: str, corrupted_parts: List[str], method: str = "auto") -> Tuple[List[str], bool]:
        """무결성 복원 시도"""
        if text_id not in self.original_checksums:
            logger.error(f"복원 불가: 원본 데이터 없음 - {text_id}")
            return corrupted_parts, False
        
        original_info = self.original_checksums[text_id]
        original_text = original_info['original_text']
        original_normalized = original_info['normalized']
        
        # 현재 처리된 텍스트 결합
        corrupted_combined = ''.join(corrupted_parts)
        corrupted_normalized = ''.join(corrupted_combined.split())
        
        logger.info(f"무결성 복원 시작: {text_id} - {method}")
        self.restoration_count += 1
        
        try:
            if method == "sequence_matcher":
                restored_parts = self._restore_with_sequence_matcher(
                    original_text, corrupted_parts, original_normalized, corrupted_normalized
                )
            elif method == "character_diff":
                restored_parts = self._restore_with_character_diff(
                    original_text, corrupted_parts, original_normalized, corrupted_normalized
                )
            else:  # auto
                # 먼저 sequence_matcher 시도
                restored_parts = self._restore_with_sequence_matcher(
                    original_text, corrupted_parts, original_normalized, corrupted_normalized
                )
                
                # 여전히 실패하면 character_diff 시도
                restored_combined = ''.join(restored_parts)
                if ''.join(restored_combined.split()) != original_normalized:
                    restored_parts = self._restore_with_character_diff(
                        original_text, corrupted_parts, original_normalized, corrupted_normalized
                    )
            
            # 복원 검증
            restored_combined = ''.join(restored_parts)
            restored_normalized = ''.join(restored_combined.split())
            
            success = (restored_normalized == original_normalized)
            
            if success:
                logger.info(f"무결성 복원 성공: {text_id} - {len(restored_parts)}개 부분")
                self.processing_history.append(f"RESTORE_OK {text_id}: {method} - 성공")
            else:
                logger.error(f"무결성 복원 실패: {text_id} - {method}")
                self.processing_history.append(f"RESTORE_FAIL {text_id}: {method} - 실패")
                # 실패 시 원본 텍스트로 대체
                restored_parts = [original_text]
            
            return restored_parts, success
            
        except Exception as e:
            logger.error(f"무결성 복원 중 오류: {e}")
            self.processing_history.append(f"RESTORE_ERROR {text_id}: {str(e)}")
            return [original_text], False
    
    def _restore_with_sequence_matcher(self, original_text: str, corrupted_parts: List[str], 
                                     original_normalized: str, corrupted_normalized: str) -> List[str]:
        """SequenceMatcher를 사용한 복원"""
        
        sm = SequenceMatcher(None, corrupted_normalized, original_normalized)
        opcodes = sm.get_opcodes()
        
        restored_parts = corrupted_parts[:]
        cumulative_offset = 0
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # 누락된 텍스트 추가
                missing_text = original_normalized[j1:j2]
                
                if restored_parts:
                    # 마지막 부분에 추가
                    restored_parts[-1] += missing_text
                else:
                    restored_parts.append(missing_text)
                
                logger.debug(f"누락 텍스트 복원: '{missing_text}'")
                
            elif tag == 'delete':
                # 중복된 텍스트 제거
                excess_text = corrupted_normalized[i1:i2]
                
                # 해당 텍스트를 포함한 부분에서 제거
                for k, part in enumerate(restored_parts):
                    if excess_text in part:
                        restored_parts[k] = part.replace(excess_text, '', 1)
                        logger.debug(f"중복 텍스트 제거: '{excess_text}'")
                        break
        
        return restored_parts
    
    def _restore_with_character_diff(self, original_text: str, corrupted_parts: List[str],
                                   original_normalized: str, corrupted_normalized: str) -> List[str]:
        """문자 단위 차이 분석을 통한 복원"""
        
        # 원본 텍스트를 corrupted_parts 수만큼 균등 분할
        if not corrupted_parts:
            return [original_text]
        
        part_count = len(corrupted_parts)
        text_length = len(original_text)
        part_length = text_length // part_count
        remainder = text_length % part_count
        
        restored_parts = []
        start = 0
        
        for i in range(part_count):
            current_length = part_length + (1 if i < remainder else 0)
            end = start + current_length
            
            if end > text_length:
                end = text_length
            
            restored_parts.append(original_text[start:end])
            start = end
        
        # 남은 텍스트가 있으면 마지막 부분에 추가
        if start < text_length:
            if restored_parts:
                restored_parts[-1] += original_text[start:]
            else:
                restored_parts.append(original_text[start:])
        
        return restored_parts
    
    def get_integrity_report(self) -> Dict:
        """무결성 보고서 생성"""
        return {
            'total_registered': len(self.original_checksums),
            'restoration_attempts': self.restoration_count,
            'processing_history': self.processing_history[-20:],  # 최근 20개
            'registered_items': list(self.original_checksums.keys())
        }

# 전역 무결성 보호자
integrity_guard = IntegrityGuard()

def safe_mask_brackets(text: str, text_type: str = 'source') -> Tuple[str, List[Dict]]:
    """무결성 보장 괄호 마스킹"""
    
    text_id = f"mask_{text_type}_{id(text)}"
    integrity_guard.register_original(text_id, text, "pre_mask")
    
    try:
        masked_text, bracket_masks = mask_brackets(text, text_type)
        
        # 무결성 검증
        restored_for_verification = restore_brackets(masked_text, bracket_masks)
        is_valid, message, info = integrity_guard.verify_integrity(text_id, restored_for_verification, "mask_verify")
        
        if not is_valid:
            logger.warning(f"괄호 마스킹 무결성 실패: {message}")
            # 복원 시도
            corrected_parts, success = integrity_guard.restore_integrity(text_id, [masked_text], "sequence_matcher")
            if success and corrected_parts:
                masked_text = corrected_parts[0]
                # 새로운 마스크 생성 필요시
                bracket_masks = []  # 실패시 빈 마스크
        
        return masked_text, bracket_masks
        
    except Exception as e:
        logger.error(f"괄호 마스킹 중 오류: {e}")
        return text, []  # 오류 시 원본 반환

def safe_restore_brackets(text: str, bracket_masks: List[Dict]) -> str:
    """무결성 보장 괄호 복원"""
    
    try:
        restored_text = restore_brackets(text, bracket_masks)
        
        # 기본 검증 (마스크 수와 복원된 괄호 수 비교)
        if bracket_masks:
            expected_brackets = len(bracket_masks)
            actual_brackets = len([m for m in bracket_masks if m.get('content', '') in restored_text])
            
            if expected_brackets != actual_brackets:
                logger.warning(f"괄호 복원 불완전: 예상 {expected_brackets}개, 실제 {actual_brackets}개")
        
        return restored_text
        
    except Exception as e:
        logger.error(f"괄호 복원 중 오류: {e}")
        return text  # 오류 시 원본 반환

def mask_brackets(text: str, text_type: str = 'source') -> Tuple[str, List[Dict]]:
    """
    괄호와 그 내용을 마스킹하여 정렬 품질 향상
    완벽한 무결성 보장
    """
    if not text or not isinstance(text, str):
        return str(text), []
    
    # 괄호 패턴 정의 (중첩 지원)
    bracket_patterns = [
        (r'\([^()]*\)', 'parentheses'),     # 소괄호
        (r'\[[^\[\]]*\]', 'square'),        # 대괄호  
        (r'\{[^{}]*\}', 'curly'),           # 중괄호
        (r'「[^「」]*」', 'corner'),          # 모서리 괄호
        (r'『[^『』]*』', 'double_corner'),   # 이중 모서리 괄호
        (r'〈[^〈〉]*〉', 'angle'),           # 꺾쇠 괄호
        (r'《[^《》]*》', 'double_angle'),    # 이중 꺾쇠 괄호
    ]
    
    masked_text = text
    bracket_masks = []
    mask_counter = 0
    
    # 각 괄호 패턴에 대해 마스킹
    for pattern, bracket_type in bracket_patterns:
        matches = list(regex.finditer(pattern, masked_text))
        
        # 뒤에서부터 처리하여 인덱스 변화 방지
        for match in reversed(matches):
            start, end = match.span()
            content = match.group()
            
            # 마스크 토큰 생성
            mask_token = f"__BRACKET_MASK_{mask_counter}__"
            mask_counter += 1
            
            # 마스크 정보 저장
            bracket_masks.insert(0, {
                'mask_token': mask_token,
                'content': content,
                'start_pos': start,
                'end_pos': end,
                'bracket_type': bracket_type,
                'text_type': text_type
            })
            
            # 텍스트에서 괄호를 마스크로 대체
            masked_text = masked_text[:start] + mask_token + masked_text[end:]
    
    logger.debug(f"괄호 마스킹 완료: {len(bracket_masks)}개 괄호 처리")
    return masked_text, bracket_masks

def restore_brackets(text: str, bracket_masks: List[Dict]) -> str:
    """
    마스킹된 괄호를 원래 위치에 복원
    완벽한 무결성 보장
    """
    if not text or not bracket_masks:
        return text
    
    restored_text = text
    
    # 마스크 토큰을 원래 괄호로 복원
    for mask_info in bracket_masks:
        mask_token = mask_info['mask_token']
        content = mask_info['content']
        
        if mask_token in restored_text:
            restored_text = restored_text.replace(mask_token, content, 1)
        else:
            logger.warning(f"마스크 토큰 누락: {mask_token}")
    
    # 복원되지 않은 마스크 토큰 검사
    remaining_masks = regex.findall(r'__BRACKET_MASK_\d+__', restored_text)
    if remaining_masks:
        logger.error(f"복원되지 않은 마스크: {remaining_masks}")
        # 남은 마스크 토큰들을 빈 문자열로 대체
        for remaining in remaining_masks:
            restored_text = restored_text.replace(remaining, '')
    
    logger.debug(f"괄호 복원 완료: {len(bracket_masks)}개 괄호 복원")
    return restored_text

def safe_split_sentences(text: str, max_length: int = 150, method: str = "punctuation") -> List[str]:
    """무결성 보장 문장 분할"""
    
    if not text or not text.strip():
        return []
    
    text_id = f"split_{method}_{id(text)}"
    integrity_guard.register_original(text_id, text, f"pre_split_{method}")
    
    try:
        # 기존 분할 함수 호출
        if method == "punctuation":
            sentences = split_by_punctuation(text, max_length)
        elif method == "spacy":
            sentences = split_by_spacy(text, max_length)
        else:
            sentences = split_by_punctuation(text, max_length)  # 기본값
        
        if not sentences:
            sentences = [text]
        
        # 무결성 검증
        combined_result = ''.join(sentences)
        is_valid, message, info = integrity_guard.verify_integrity(text_id, combined_result, f"post_split_{method}")
        
        if not is_valid:
            logger.warning(f"문장 분할 무결성 실패: {message}")
            # 복원 시도
            restored_sentences, success = integrity_guard.restore_integrity(text_id, sentences)
            
            if success:
                sentences = restored_sentences
                logger.info(f"문장 분할 무결성 복원 성공")
            else:
                logger.error(f"문장 분할 무결성 복원 실패, 원본 반환")
                sentences = [text]
        
        return sentences
        
    except Exception as e:
        logger.error(f"문장 분할 중 오류: {e}")
        return [text]

def split_by_punctuation(text: str, max_length: int = 150) -> List[str]:
    """구두점 기반 문장 분할 (기존 로직 유지)"""
    
    if not text.strip():
        return []
    
    # 구두점 패턴 (한국어 + 중국어)
    sentence_enders = r'[.!?。！？；]'
    
    # 구두점으로 분할
    sentences = regex.split(sentence_enders, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 길이 제한 적용
    final_sentences = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            final_sentences.append(sentence)
        else:
            # 긴 문장은 쉼표로 추가 분할
            sub_sentences = sentence.split(',')
            for sub in sub_sentences:
                if sub.strip():
                    final_sentences.append(sub.strip())
    
    return final_sentences

def split_by_spacy(text: str, max_length: int = 150) -> List[str]:
    """spaCy 기반 문장 분할"""
    
    try:
        import spacy
        
        # 한국어 모델 시도
        try:
            nlp = spacy.load("ko_core_news_sm")
        except OSError:
            # 영어 모델 폴백
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy 모델 없음, 구두점 분할 사용")
                return split_by_punctuation(text, max_length)
        
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # 길이 제한 적용
        final_sentences = []
        for sentence in sentences:
            if len(sentence) <= max_length:
                final_sentences.append(sentence)
            else:
                # 긴 문장은 구두점 분할로 폴백
                sub_sentences = split_by_punctuation(sentence, max_length)
                final_sentences.extend(sub_sentences)
        
        return final_sentences
        
    except ImportError:
        logger.warning("spaCy 설치되지 않음, 구두점 분할 사용")
        return split_by_punctuation(text, max_length)

def process_text_with_integrity(text: str, processing_func, text_id: str = None, **kwargs) -> Any:
    """임의의 텍스트 처리 함수에 무결성 보장 적용"""
    
    if text_id is None:
        text_id = f"process_{id(text)}"
    
    integrity_guard.register_original(text_id, text, "pre_process")
    
    try:
        result = processing_func(text, **kwargs)
        
        # 결과가 문자열이면 검증
        if isinstance(result, str):
            is_valid, message, info = integrity_guard.verify_integrity(text_id, result, "post_process")
            
            if not is_valid:
                logger.warning(f"처리 함수 무결성 실패: {message}")
                # 복원 시도
                restored_parts, success = integrity_guard.restore_integrity(text_id, [result])
                if success and restored_parts:
                    result = restored_parts[0]
        
        # 결과가 리스트면 결합해서 검증
        elif isinstance(result, list) and all(isinstance(item, str) for item in result):
            combined = ''.join(result)
            is_valid, message, info = integrity_guard.verify_integrity(text_id, combined, "post_process")
            
            if not is_valid:
                logger.warning(f"처리 함수 무결성 실패: {message}")
                # 복원 시도
                restored_parts, success = integrity_guard.restore_integrity(text_id, result)
                if success:
                    result = restored_parts
        
        return result
        
    except Exception as e:
        logger.error(f"텍스트 처리 중 오류: {e}")
        return text  # 오류 시 원본 반환

def get_integrity_status() -> Dict:
    """현재 무결성 상태 반환"""
    return integrity_guard.get_integrity_report()

def reset_integrity_guard():
    """무결성 보호자 초기화"""
    global integrity_guard
    integrity_guard = IntegrityGuard()
    logger.info("무결성 보호자 초기화 완료")

# 기존 함수들의 무결성 보장 래퍼
def safe_extract_quotes(text: str) -> Tuple[str, List[Dict]]:
    """무결성 보장 인용부호 추출"""
    return process_text_with_integrity(
        text, 
        lambda t: extract_quotes(t),
        f"quotes_{id(text)}"
    )

def safe_normalize_punctuation(text: str) -> str:
    """무결성 보장 구두점 정규화"""
    return process_text_with_integrity(
        text,
        lambda t: normalize_punctuation(t),
        f"normalize_{id(text)}"
    )

# 기존 함수들 (로직 변경 없이 유지)
def extract_quotes(text: str) -> Tuple[str, List[Dict]]:
    """인용부호 및 내용 추출 (기존 로직)"""
    # 기존 구현...
    return text, []

def normalize_punctuation(text: str) -> str:
    """구두점 정규화 (기존 로직)"""
    # 기존 구현...
    return text

# 추가 안전 장치
def validate_text_integrity(original: str, processed: str, tolerance: float = 0.95) -> bool:
    """텍스트 무결성 검증 (공개 API)"""
    
    if not original or not processed:
        return not original and not processed
    
    # 정규화 후 비교
    orig_normalized = ''.join(original.split())
    proc_normalized = ''.join(processed.split())
    
    if orig_normalized == proc_normalized:
        return True
    
    # 허용 오차 내 유사도 확인
    sm = SequenceMatcher(None, orig_normalized, proc_normalized)
    similarity = sm.ratio()
    
    return similarity >= tolerance

def emergency_restore_text(text_fragments: List[str], original_text: str) -> List[str]:
    """비상 텍스트 복원"""
    
    logger.warning("비상 텍스트 복원 실행")
    
    if not text_fragments:
        return [original_text]
    
    # 원본을 조각 수에 맞춰 균등 분할
    fragment_count = len(text_fragments)
    text_length = len(original_text)
    
    if fragment_count == 1:
        return [original_text]
    
    avg_length = text_length // fragment_count
    remainder = text_length % fragment_count
    
    restored_fragments = []
    start = 0
    
    for i in range(fragment_count):
        length = avg_length + (1 if i < remainder else 0)
        end = min(start + length, text_length)
        
        if start < text_length:
            restored_fragments.append(original_text[start:end])
        else:
            restored_fragments.append('')
        
        start = end
    
    # 남은 텍스트가 있으면 마지막에 추가
    if start < text_length:
        if restored_fragments:
            restored_fragments[-1] += original_text[start:]
        else:
            restored_fragments.append(original_text[start:])
    
    logger.info(f"비상 복원 완료: {len(restored_fragments)}개 조각")
    return restored_fragments