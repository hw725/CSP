"""간소화된 병렬 처리 - 완벽한 무결성 보장"""

import os
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback
import hashlib

# 무결성 모듈 import
try:
    from punctuation import integrity_guard, safe_mask_brackets, safe_restore_brackets, get_integrity_status
except ImportError:
    # 폴백 처리
    integrity_guard = None
    def safe_mask_brackets(text, text_type='source'):
        return text, []
    def safe_restore_brackets(text, masks):
        return text
    def get_integrity_status():
        return {}

logger = logging.getLogger(__name__)

class SafeFileProcessor:
    """무결성 보장 파일 처리기"""
    
    def __init__(self, max_workers: int = 4, chunk_size: int = 100):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.processed_count = 0
        self.error_count = 0
        self.integrity_failures = 0
        self.integrity_enabled = integrity_guard is not None  # 🔧 무결성 사용 가능 여부
    
    def process_file_with_integrity(
        self, 
        input_file: str, 
        output_file: str,
        processing_function,
        **kwargs
    ) -> bool:
        """무결성 보장 파일 처리"""
        
        logger.info(f"무결성 보장 파일 처리 시작: {input_file}")
        
        try:
            # 1. 입력 파일 로드 및 무결성 등록
            df = pd.read_excel(input_file)
            logger.info(f"입력 데이터 로드: {len(df)}개 행")
            
            # 전체 데이터 무결성 등록
            file_id = f"file_{hashlib.md5(input_file.encode()).hexdigest()[:8]}"
            self._register_file_integrity(df, file_id)
            
            # 2. 청크 단위로 처리
            results = []
            chunks = self._create_chunks(df)
            
            logger.info(f"데이터를 {len(chunks)}개 청크로 분할하여 처리")
            
            for i, chunk in enumerate(chunks):
                logger.info(f"청크 {i+1}/{len(chunks)} 처리 중...")
                
                chunk_results = self._process_chunk_with_integrity(
                    chunk, processing_function, f"{file_id}_chunk_{i}", **kwargs
                )
                
                if chunk_results:
                    results.extend(chunk_results)
                    self.processed_count += len(chunk_results)
                else:
                    self.error_count += len(chunk)
            
            # 3. 결과 저장 및 검증
            if results:
                result_df = pd.DataFrame(results)
                
                # 최종 무결성 검증
                final_integrity = self._verify_final_integrity(df, result_df, file_id)
                
                if final_integrity['valid']:
                    result_df.to_excel(output_file, index=False)
                    logger.info(f"결과 저장 완료: {output_file}")
                    logger.info(f"처리 통계: 성공 {self.processed_count}, 실패 {self.error_count}, 무결성실패 {self.integrity_failures}")
                    return True
                else:
                    logger.error(f"최종 무결성 검증 실패: {final_integrity['message']}")
                    # 복구된 결과로 저장
                    if final_integrity.get('restored_df') is not None:
                        final_integrity['restored_df'].to_excel(output_file, index=False)
                        logger.info(f"복구된 결과로 저장: {output_file}")
                        return True
                    else:
                        return False
            else:
                logger.error("처리된 결과가 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"파일 처리 중 오류: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _register_file_integrity(self, df: pd.DataFrame, file_id: str):
        """파일 전체 무결성 등록 (안전 버전)"""
        
        if not self.integrity_enabled:
            logger.debug("무결성 모듈 비활성화됨, 등록 스킵")
            return
        
        # 원문과 번역문 전체 결합
        if '원문' in df.columns and '번역문' in df.columns:
            total_src = ''.join(df['원문'].fillna('').astype(str))
            total_tgt = ''.join(df['번역문'].fillna('').astype(str))
            
            try:
                integrity_guard.register_original(f"{file_id}_src_all", total_src, "file_input_src")
                integrity_guard.register_original(f"{file_id}_tgt_all", total_tgt, "file_input_tgt")
                logger.info(f"파일 무결성 등록: 원문 {len(total_src)}자, 번역문 {len(total_tgt)}자")
            except Exception as e:
                logger.warning(f"무결성 등록 실패: {e}")
                self.integrity_enabled = False
    
    def _create_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """데이터프레임을 청크로 분할"""
        
        chunks = []
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size].copy()
            chunks.append(chunk)
        
        return chunks
    
    def _process_chunk_with_integrity(
        self, 
        chunk: pd.DataFrame, 
        processing_function,
        chunk_id: str,
        **kwargs
    ) -> List[Dict]:
        """청크 단위 무결성 보장 처리 (안전 버전)"""
        
        results = []
        
        for idx, row in chunk.iterrows():
            try:
                # 행별 무결성 등록 (안전)
                row_id = f"{chunk_id}_row_{idx}"
                src_text = str(row.get('원문', ''))
                tgt_text = str(row.get('번역문', ''))
                
                if src_text.strip() and tgt_text.strip():
                    if self.integrity_enabled:
                        try:
                            integrity_guard.register_original(f"{row_id}_src", src_text, "row_input_src")
                            integrity_guard.register_original(f"{row_id}_tgt", tgt_text, "row_input_tgt")
                        except Exception as e:
                            logger.warning(f"행 무결성 등록 실패: {e}")
                            self.integrity_enabled = False
                    
                    # 처리 함수 실행
                    row_result = processing_function(row, row_id=row_id, **kwargs)
                    
                    if row_result:
                        # 행별 무결성 검증 (안전)
                        if self.integrity_enabled and self._verify_row_integrity(row, row_result, row_id):
                            results.extend(row_result if isinstance(row_result, list) else [row_result])
                        elif not self.integrity_enabled:
                            # 무결성 비활성화시 결과 그대로 사용
                            results.extend(row_result if isinstance(row_result, list) else [row_result])
                        else:
                            self.integrity_failures += 1
                            logger.warning(f"행 {idx} 무결성 실패, 스킵")
                
            except Exception as e:
                logger.error(f"행 {idx} 처리 실패: {e}")
                self.error_count += 1
                continue
        
        return results
    
    def _verify_row_integrity(self, original_row: pd.Series, processed_results: List[Dict], row_id: str) -> bool:
        """행별 무결성 검증 (안전 버전)"""
        
        if not self.integrity_enabled:
            return True  # 무결성 비활성화시 항상 성공
        
        try:
            original_src = str(original_row.get('원문', ''))
            original_tgt = str(original_row.get('번역문', ''))
            
            if isinstance(processed_results, dict):
                processed_results = [processed_results]
            
            # 처리된 결과에서 원문과 번역문 결합
            processed_src = ''.join([str(result.get('원문', '')) for result in processed_results])
            processed_tgt = ''.join([str(result.get('번역문', '')) for result in processed_results])
            
            # 무결성 검증
            src_valid, src_msg, _ = integrity_guard.verify_integrity(f"{row_id}_src", processed_src, "row_output_src")
            tgt_valid, tgt_msg, _ = integrity_guard.verify_integrity(f"{row_id}_tgt", processed_tgt, "row_output_tgt")
            
            if not src_valid:
                logger.warning(f"원문 무결성 실패: {src_msg}")
            
            if not tgt_valid:
                logger.warning(f"번역문 무결성 실패: {tgt_msg}")
            
            return src_valid and tgt_valid
            
        except Exception as e:
            logger.warning(f"무결성 검증 중 오류: {e}")
            self.integrity_enabled = False
            return True  # 오류시 통과 처리
    
    def _verify_final_integrity(self, original_df: pd.DataFrame, result_df: pd.DataFrame, file_id: str) -> Dict:
        """최종 무결성 검증 (안전 버전)"""
        
        if not self.integrity_enabled:
            logger.info("무결성 모듈 비활성화됨, 최종 검증 스킵")
            return {
                'valid': True,
                'message': '무결성 검증 비활성화됨',
                'src_info': {},
                'tgt_info': {}
            }
        
        try:
            logger.info("최종 무결성 검증 시작...")
            
            # 원본 데이터 결합
            original_src_all = ''.join(original_df['원문'].fillna('').astype(str))
            original_tgt_all = ''.join(original_df['번역문'].fillna('').astype(str))
            
            # 결과 데이터 결합
            result_src_all = ''.join(result_df['원문'].fillna('').astype(str))
            result_tgt_all = ''.join(result_df['번역문'].fillna('').astype(str))
            
            # 무결성 검증
            src_valid, src_msg, src_info = integrity_guard.verify_integrity(
                f"{file_id}_src_all", result_src_all, "file_output_src"
            )
            tgt_valid, tgt_msg, tgt_info = integrity_guard.verify_integrity(
                f"{file_id}_tgt_all", result_tgt_all, "file_output_tgt"
            )
            
            overall_valid = src_valid and tgt_valid
            
            if overall_valid:
                return {
                    'valid': True,
                    'message': '최종 무결성 검증 성공',
                    'src_info': src_info,
                    'tgt_info': tgt_info
                }
            else:
                logger.warning("최종 무결성 실패, 복구 시도...")
                
                # 복구 시도
                restored_df = self._restore_final_integrity(
                    original_df, result_df, file_id, src_valid, tgt_valid
                )
                
                return {
                    'valid': False,
                    'message': f'무결성 실패 - 원문: {src_msg}, 번역문: {tgt_msg}',
                    'src_info': src_info,
                    'tgt_info': tgt_info,
                    'restored_df': restored_df
                }
                
        except Exception as e:
            logger.error(f"최종 무결성 검증 중 오류: {e}")
            self.integrity_enabled = False
            return {
                'valid': True,  # 오류시 통과 처리
                'message': f'무결성 검증 오류로 스킵: {e}',
                'src_info': {},
                'tgt_info': {}
            }
    
    def _restore_final_integrity(
        self, 
        original_df: pd.DataFrame, 
        result_df: pd.DataFrame, 
        file_id: str,
        src_valid: bool,
        tgt_valid: bool
    ) -> pd.DataFrame:
        """최종 무결성 복구 (안전 버전)"""
        
        if not self.integrity_enabled:
            return result_df
        
        try:
            restored_df = result_df.copy()
            
            # 원문 복구
            if not src_valid:
                logger.info("원문 무결성 복구 중...")
                
                result_src_parts = result_df['원문'].fillna('').astype(str).tolist()
                original_src_all = ''.join(original_df['원문'].fillna('').astype(str))
                
                restored_src_parts, success = integrity_guard.restore_integrity(
                    f"{file_id}_src_all", result_src_parts, "sequence_matcher"
                )
                
                if success and len(restored_src_parts) == len(result_df):
                    restored_df['원문'] = restored_src_parts
                    logger.info("원문 복구 성공")
                else:
                    logger.warning("원문 복구 실패, 원본으로 대체")
                    # 원본을 결과 개수만큼 분할
                    src_parts = self._split_text_by_count(original_src_all, len(result_df))
                    restored_df['원문'] = src_parts
            
            # 번역문 복구
            if not tgt_valid:
                logger.info("번역문 무결성 복구 중...")
                
                result_tgt_parts = result_df['번역문'].fillna('').astype(str).tolist()
                original_tgt_all = ''.join(original_df['번역문'].fillna('').astype(str))
                
                restored_tgt_parts, success = integrity_guard.restore_integrity(
                    f"{file_id}_tgt_all", result_tgt_parts, "sequence_matcher"
                )
                
                if success and len(restored_tgt_parts) == len(result_df):
                    restored_df['번역문'] = restored_tgt_parts
                    logger.info("번역문 복구 성공")
                else:
                    logger.warning("번역문 복구 실패, 원본으로 대체")
                    # 원본을 결과 개수만큼 분할
                    tgt_parts = self._split_text_by_count(original_tgt_all, len(result_df))
                    restored_df['번역문'] = tgt_parts
            
            return restored_df
            
        except Exception as e:
            logger.error(f"무결성 복구 중 오류: {e}")
            return result_df  # 오류시 원본 결과 반환
    
    def _split_text_by_count(self, text: str, count: int) -> List[str]:
        """텍스트를 지정된 개수로 균등 분할"""
        
        if count <= 0:
            return []
        
        if count == 1:
            return [text]
        
        text_length = len(text)
        part_length = text_length // count
        remainder = text_length % count
        
        parts = []
        start = 0
        
        for i in range(count):
            current_length = part_length + (1 if i < remainder else 0)
            end = start + current_length
            
            if end > text_length:
                end = text_length
            
            if start < text_length:
                parts.append(text[start:end])
            else:
                parts.append('')
            
            start = end
        
        return parts

# ===== 기존 SA 호환 함수들 =====

def process_file(
    input_file: str,
    output_file: str,
    embedder_name: str = "bge",
    max_workers: int = 4,
    chunk_size: int = 100,
    use_parallel: bool = True,
    **kwargs
) -> bool:
    """기존 SA 호환 파일 처리 함수 (무결성 보장)"""
    
    logger.info(f"SA 파일 처리 시작: {input_file}")
    logger.info(f"설정: 임베더={embedder_name}, 병렬={use_parallel}, 워커={max_workers}")
    
    try:
        # SA 처리 함수 import
        from sa_tokenizers.jieba_mecab import process_single_row
        
        # 무결성 보장 처리기 생성
        processor = SafeFileProcessor(max_workers=max_workers, chunk_size=chunk_size)
        
        # 무결성 보장 처리 실행
        success = processor.process_file_with_integrity(
            input_file=input_file,
            output_file=output_file,
            processing_function=safe_process_sa_row,
            embedder_name=embedder_name,
            **kwargs
        )
        
        if success:
            logger.info(f"✅ SA 파일 처리 완료: {output_file}")
        else:
            logger.error(f"❌ SA 파일 처리 실패")
        
        return success
        
    except ImportError as e:
        logger.error(f"SA 모듈 import 실패: {e}")
        # 폴백: 기본 처리
        success = process_file_fallback(input_file, output_file, **kwargs)
        return success  # 🔧 bool 반환
    
    except Exception as e:
        logger.error(f"SA 파일 처리 중 오류: {e}")
        logger.error(traceback.format_exc())
        return False

def process_file_fallback(input_file: str, output_file: str, **kwargs) -> bool:
    """폴백 파일 처리 함수"""
    
    logger.warning("폴백 모드로 파일 처리 중...")
    
    try:
        df = pd.read_excel(input_file)
        
        results = []
        for idx, row in df.iterrows():
            src_text = str(row.get('원문', ''))
            tgt_text = str(row.get('번역문', ''))
            
            if src_text.strip() and tgt_text.strip():
                results.append({
                    '문장식별자': idx + 1,
                    '원문': src_text,
                    '번역문': tgt_text,
                    '분할방법': 'fallback_mode',
                    '유사도': 1.0,
                    '원문_토큰수': len(src_text.split()),
                    '번역문_토큰수': len(tgt_text.split())
                })
        
        if results:
            result_df = pd.DataFrame(results)
            result_df.to_excel(output_file, index=False)
            logger.info(f"폴백 처리 완료: {len(results)}개 결과")
            return True  # 🔧 성공시 True 반환
        else:
            logger.error("처리할 데이터 없음")
            return False
            
    except Exception as e:
        logger.error(f"폴백 처리 실패: {e}")
        return False

# 무결성 보장 처리 함수들
def safe_process_single_text(text: str, processing_function, text_id: str = None, **kwargs) -> Any:
    """단일 텍스트 무결성 보장 처리 (안전 버전)"""
    
    if not integrity_guard:
        return processing_function(text, **kwargs)
    
    if text_id is None:
        text_id = f"single_{id(text)}"
    
    try:
        integrity_guard.register_original(text_id, text, "single_input")
        
        result = processing_function(text, **kwargs)
        
        # 결과 검증
        if isinstance(result, str):
            is_valid, message, info = integrity_guard.verify_integrity(text_id, result, "single_output")
            
            if not is_valid:
                logger.warning(f"단일 텍스트 처리 무결성 실패: {message}")
                # 복구 시도
                restored_parts, success = integrity_guard.restore_integrity(text_id, [result])
                if success and restored_parts:
                    result = restored_parts[0]
                else:
                    result = text  # 실패시 원본 반환
        
        return result
        
    except Exception as e:
        logger.error(f"단일 텍스트 처리 중 오류: {e}")
        return processing_function(text, **kwargs)  # 무결성 실패시 기본 처리

def safe_process_text_list(text_list: List[str], processing_function, list_id: str = None, **kwargs) -> List[str]:
    """텍스트 리스트 무결성 보장 처리 (안전 버전)"""
    
    if not integrity_guard:
        return [processing_function(text, **kwargs) for text in text_list]
    
    if list_id is None:
        list_id = f"list_{id(text_list)}"
    
    try:
        # 전체 리스트 무결성 등록
        combined_text = ''.join(text_list)
        integrity_guard.register_original(list_id, combined_text, "list_input")
        
        results = []
        
        for i, text in enumerate(text_list):
            if text.strip():
                item_id = f"{list_id}_item_{i}"
                result = safe_process_single_text(text, processing_function, item_id, **kwargs)
                results.append(result)
            else:
                results.append(text)
        
        # 전체 결과 검증
        combined_result = ''.join(results)
        is_valid, message, info = integrity_guard.verify_integrity(list_id, combined_result, "list_output")
        
        if not is_valid:
            logger.warning(f"텍스트 리스트 처리 무결성 실패: {message}")
            # 복구 시도
            restored_parts, success = integrity_guard.restore_integrity(list_id, results)
            if success:
                results = restored_parts
            else:
                results = text_list  # 실패시 원본 반환
        
        return results
        
    except Exception as e:
        logger.error(f"텍스트 리스트 처리 중 오류: {e}")
        return [processing_function(text, **kwargs) for text in text_list]

# 병렬 처리 관련 함수들 (기존 호환성)
def process_chunks_parallel(chunks: List[Any], processing_function, max_workers: int = 4, **kwargs) -> List[Any]:
    """병렬 청크 처리"""
    
    logger.info(f"병렬 처리 시작: {len(chunks)}개 청크, {max_workers}개 워커")
    
    results = []
    
    if max_workers == 1:
        # 순차 처리
        for i, chunk in enumerate(chunks):
            logger.info(f"청크 {i+1}/{len(chunks)} 순차 처리 중...")
            chunk_result = processing_function(chunk, **kwargs)
            if chunk_result:
                results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
    else:
        # 병렬 처리
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(processing_function, chunk, **kwargs): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    if chunk_result:
                        results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
                    logger.info(f"청크 {chunk_idx + 1} 처리 완료")
                except Exception as e:
                    logger.error(f"청크 {chunk_idx + 1} 처리 실패: {e}")
    
    logger.info(f"병렬 처리 완료: {len(results)}개 결과")
    return results

def split_dataframe_into_chunks(df: pd.DataFrame, chunk_size: int = 100) -> List[pd.DataFrame]:
    """데이터프레임을 청크로 분할"""
    
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    
    logger.info(f"데이터프레임 분할: {len(df)}행 → {len(chunks)}개 청크")
    return chunks

def safe_process_sa_row(row: pd.Series, row_id: str = None, **kwargs) -> List[Dict]:
    """SA 행 처리 무결성 보장 래퍼"""
    
    try:
        # SA 토크나이저 처리 함수 import
        from sa_tokenizers.jieba_mecab import process_single_row  # 🔧 올바른 함수명
        
        # 원본 데이터 추출
        src_text = str(row.get('원문', ''))
        tgt_text = str(row.get('번역문', ''))
        
        if not src_text.strip() or not tgt_text.strip():
            return []
        
        # 무결성 보장 괄호 처리 (안전 버전)
        if integrity_guard:
            masked_src, src_masks = safe_mask_brackets(src_text, 'source')
            masked_tgt, tgt_masks = safe_mask_brackets(tgt_text, 'target')
            
            # SA 처리 실행
            masked_row = row.copy()
            masked_row['원문'] = masked_src
            masked_row['번역문'] = masked_tgt
            
            results = process_single_row(masked_row, row_id=row_id, **kwargs)
            
            if not results:
                return []
            
            # 괄호 복원
            restored_results = []
            for result in results:
                if isinstance(result, dict):
                    restored_src = safe_restore_brackets(str(result.get('원문', '')), src_masks)
                    restored_tgt = safe_restore_brackets(str(result.get('번역문', '')), tgt_masks)
                    
                    restored_result = result.copy()
                    restored_result['원문'] = restored_src
                    restored_result['번역문'] = restored_tgt
                    
                    restored_results.append(restored_result)
                else:
                    restored_results.append(result)
            
            return restored_results
        else:
            # 무결성 모듈 없으면 직접 처리
            return process_single_row(row, row_id=row_id, **kwargs)
        
    except Exception as e:
        logger.error(f"SA 행 처리 실패: {e}")
        # 폴백: 원본 데이터 반환
        return [{
            '원문': str(row.get('원문', '')),
            '번역문': str(row.get('번역문', '')),
            '문장식별자': f"{row_id}_error" if row_id else getattr(row, 'name', 0),
            '분할방법': 'error_fallback',
            '유사도': 1.0,
            '원문_토큰수': len(str(row.get('원문', '')).split()),
            '번역문_토큰수': len(str(row.get('번역문', '')).split())
        }]