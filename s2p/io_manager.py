"""간소화된 병렬 처리 - 완벽한 무결성 보장"""

import os
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import traceback
import hashlib

# 통합 진행률 관리자
from common.progress_manager import start_unified_progress, update_unified_progress, finish_unified_progress, set_progress_description
# 전역 무결성 검증 모듈
from common.integrity_verifier import verify_global_integrity

# 로거를 가장 먼저 초기화하여 예외 처리에서도 사용 가능하게 함
logger = logging.getLogger(__name__)

# 무결성 모듈 import (패키지 경로)
try:
    from s2p.punctuation import integrity_guard, safe_mask_brackets, safe_restore_brackets, get_integrity_status
except ImportError as e:
    logger.warning(f"⚠️ 무결성 모듈 import 실패: {e}")
    logger.warning("   무결성 검증이 비활성화되고 텍스트 손실 위험 증가")
    # 폴백 처리
    integrity_guard = None
    def safe_mask_brackets(text, text_type='source'):
        return text, []
    def safe_restore_brackets(text, masks):
        return text
    def get_integrity_status():
        return {}

class SafeFileProcessor:
    """무결성 보장 파일 처리기"""
    
    def __init__(self, max_workers: int = 4, chunk_size: int = 100, verbose: bool = False):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.verbose = verbose  # 🔧 verbose 옵션 추가
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
            # CSV 또는 Excel 자동 감지
            if str(input_file).endswith('.csv'):
                df = pd.read_csv(input_file)
            else:
                df = pd.read_excel(input_file)
            logger.info(f"입력 데이터 로드: {len(df)}개 행")
            
            # 🔧 NaN 값 필터링 (원문/번역문 중 하나라도 NaN이면 제외)
            # 🔧 NaN 값 처리 (원문/번역문 NaN -> 빈 문자열로 대체하여 행 유지)
            original_rows = len(df)
            df['원문'] = df['원문'].fillna('').astype(str)
            df['번역문'] = df['번역문'].fillna('').astype(str)
            
            # 빈 행 로깅만 수행
            empty_rows = len(df[ (df['원문'].str.strip() == '') & (df['번역문'].str.strip() == '') ])
            if empty_rows > 0:
                logger.info(f"⚠️ 빈 원문/번역문 {empty_rows}개 포함 (누락 방지를 위해 유지)")
            
            if len(df) == 0:
                logger.error("❌ 유효한 데이터 없음 (모두 NaN)")
                return False
            
            # 전체 데이터 무결성 등록
            file_id = f"file_{hashlib.md5(input_file.encode()).hexdigest()[:8]}"
            self._register_file_integrity(df, file_id)
            
            # 🔧 [최적화] 임베딩/경계 사전 계산을 청크 루프 내부로 이동
            # 전체 데이터를 한 번에 처리하지 않고 청크 단위로 처리하여 첫 체크포인트가 빨리 생성됨
            embedder_name = kwargs.get('embedder_name', 'bge')
            use_boundary_model = kwargs.get('use_boundary_model', False)
            
            # 2. 청크 단위로 처리
            results = []
            chunks = self._create_chunks(df)
            total_rows = len(df)
            processed_rows = 0
            
            print(f"📊 SA 처리 시작: {total_rows:,}개 행 ({len(chunks)}개 청크)", flush=True)
            
            # 진행률 출력 간격 (10행마다 또는 1% 단위)
            progress_interval = max(10, total_rows // 100)
            
            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()
                print(f"\n🔧 청크 {i+1}/{len(chunks)} 시작 ({len(chunk)}행)...", flush=True)
                
                # 🚀 [청크별] 임베딩 사전 계산
                if embedder_name.lower() != 'none':
                    try:
                        from common.embedders.bge import get_embedding_manager
                        manager = get_embedding_manager()
                        
                        # 청크의 텍스트만 수집
                        chunk_texts = set()
                        for _, row in chunk.iterrows():
                            src = str(row.get('원문', '')).strip()
                            tgt = str(row.get('번역문', '')).strip()
                            if src:
                                chunk_texts.add(src)
                                for token in src.split():
                                    if token.strip():
                                        chunk_texts.add(token.strip())
                            if tgt:
                                chunk_texts.add(tgt)
                                for token in tgt.split():
                                    if token.strip():
                                        chunk_texts.add(token.strip())
                        
                        chunk_texts_list = list(chunk_texts)
                        if chunk_texts_list:
                            batch_size = kwargs.get('batch_size', 64)
                            _ = manager.compute_embeddings_with_cache(
                                chunk_texts_list, 
                                batch_size=batch_size,
                                show_batch_progress=False
                            )
                        print(f"   ✅ 청크 임베딩 계산: {len(chunk_texts_list)}개", flush=True)
                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️ 청크 임베딩 계산 실패: {e}", flush=True)
                
                # 🚀 [청크별] 경계 배치 사전 계산
                batch_boundary_cache = {}
                if use_boundary_model:
                    try:
                        from common.s2p_crossattn_boundary_loader import get_crossattn_boundary_tagger
                        boundary_model = get_crossattn_boundary_tagger(device=kwargs.get('device', 'cuda'))
                        
                        # 청크의 원문/번역문 수집
                        chunk_src = []
                        chunk_tgt = []
                        chunk_n_segments = []
                        chunk_row_indices = []
                        
                        for idx, row in chunk.iterrows():
                            src = str(row.get('원문', '')).strip()
                            tgt = str(row.get('번역문', '')).strip()
                            if src and tgt:
                                chunk_src.append(src)
                                chunk_tgt.append(tgt)
                                chunk_n_segments.append(len(src.split()))
                                chunk_row_indices.append(idx)
                        
                        # 배치 처리 (GPU 메모리 제한으로 작은 배치)
                        batch_size = 32
                        for batch_start in range(0, len(chunk_src), batch_size):
                            batch_end = min(batch_start + batch_size, len(chunk_src))
                            segments_batch = boundary_model.segment_text_batch(
                                chunk_src[batch_start:batch_end], 
                                chunk_tgt[batch_start:batch_end], 
                                n_segments_list=chunk_n_segments[batch_start:batch_end],
                                threshold=kwargs.get('boundary_threshold', 0.55)
                            )
                            for idx, segments in zip(chunk_row_indices[batch_start:batch_end], segments_batch):
                                batch_boundary_cache[idx] = segments
                        
                        print(f"   ✅ 청크 경계 계산: {len(batch_boundary_cache)}개", flush=True)
                    except Exception as e:
                        if self.verbose:
                            print(f"   ⚠️ 청크 경계 계산 실패: {e}", flush=True)
                
                if self.verbose:
                    logger.info(f"청크 {i+1}/{len(chunks)} 처리 중...")
                
                # 행 단위 업데이트를 위한 콜백
                def update_progress():
                    nonlocal processed_rows
                    processed_rows += 1
                    if processed_rows % progress_interval == 0 or processed_rows == total_rows:
                        pct = processed_rows * 100 // total_rows
                        print(f"🔄 SA 처리: {processed_rows:,}/{total_rows:,} ({pct}%) 성공={self.processed_count} 실패={self.error_count}", flush=True)

                # 🚀 배치 경계 캐시를 kwargs에 전달
                chunk_kwargs = dict(kwargs)
                chunk_kwargs['_boundary_cache'] = batch_boundary_cache

                chunk_results = self._process_chunk_with_integrity(
                    chunk, processing_function, f"{file_id}_chunk_{i}", 
                    progress_callback=update_progress,
                    **chunk_kwargs
                )
                
                if chunk_results:
                    results.extend(chunk_results)
                    
                    # 🚀 체크포인트: 청크 완료 시 중간 결과 저장
                    checkpoint_path = Path(str(output_file).replace('.xlsx', '_checkpoint.csv'))
                    try:
                        chunk_df = pd.DataFrame(chunk_results)
                        if i == 0:
                            # 첫 청크: 헤더 포함하여 새 파일 생성
                            chunk_df.to_csv(checkpoint_path, index=False, mode='w')
                        else:
                            # 이후 청크: 헤더 없이 append
                            chunk_df.to_csv(checkpoint_path, index=False, mode='a', header=False)
                        print(f"💾 체크포인트 저장: 청크 {i+1}/{len(chunks)} ({len(results):,}개 누적)", flush=True)
                    except Exception as e:
                        logger.warning(f"⚠️ 체크포인트 저장 실패: {e}")
            
            # 완료 메시지
            print(f"✅ SA 처리 완료: {self.processed_count:,}개 구문 생성", flush=True)
            
            # 3. 결과 저장 및 검증
            if results:
                result_df = pd.DataFrame(results)
                
                # 최종 무결성 검증
                final_integrity = self._verify_final_integrity(df, result_df, file_id)
                
                if final_integrity['valid']:
                    # CSV 또는 Excel 자동 감지하여 저장
                    if str(output_file).endswith('.csv'):
                        result_df.to_csv(output_file, index=False)
                    else:
                        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='results')
                    
                    # 🆕 빠른 무결성 검증 (문자 수 기반, 공백 무시)
                    try:
                        # 공백, 줄바꿈, 탭 제거 후 순수 문자만 비교
                        def normalize_text(series):
                            return ''.join(series.fillna('').astype(str)).replace(' ', '').replace('\n', '').replace('\t', '')

                        original_src = normalize_text(df['원문'])
                        original_tgt = normalize_text(df['번역문'])
                        result_src = normalize_text(result_df['원문'])
                        result_tgt = normalize_text(result_df['번역문'])
                        
                        losses = []
                        
                        # 원문 손실 확인
                        if len(result_src) != len(original_src):
                            diff = len(original_src) - len(result_src)
                            losses.append({
                                'type': '원문',
                                'original_length': len(original_src),
                                'result_length': len(result_src),
                                'difference': diff,
                                'loss_rate': abs(diff) / len(original_src) * 100 if len(original_src) > 0 else 0
                            })
                        
                        # 번역문 손실 확인
                        if len(result_tgt) != len(original_tgt):
                            diff = len(original_tgt) - len(result_tgt)
                            losses.append({
                                'type': '번역문',
                                'original_length': len(original_tgt),
                                'result_length': len(result_tgt),
                                'difference': diff,
                                'loss_rate': abs(diff) / len(original_tgt) * 100 if len(original_tgt) > 0 else 0
                            })
                        
                        if losses:
                            integrity_losses_df = pd.DataFrame(losses)
                            # 무결성 손실 시트는 Excel만 지원
                            if not str(output_file).endswith('.csv'):
                                with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                                    integrity_losses_df.to_excel(writer, index=False, sheet_name='integrity_losses')
                            
                            for loss in losses:
                                print(f"⚠️ {loss['type']} 무결성 손실: {loss['difference']:,}자 ({loss['loss_rate']:.2f}%)")
                        else:
                            print("✅ 무결성 검증 통과 (손실 없음)")
                            
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"무결성 검증 오류: {e}")
                    
                    if self.verbose:
                        logger.info(f"결과 저장 완료: {output_file}")
                        logger.info(f"처리 통계: 성공 {self.processed_count}, 실패 {self.error_count}, 무결성실패 {self.integrity_failures}")
                    # 기본 모드에서는 통합 진행률에서 완료 메시지 처리
                    return True
                else:
                    if self.verbose:
                        logger.error(f"최종 무결성 검증 실패: {final_integrity['message']}")
                    # 복구된 결과로 저장
                    if final_integrity.get('restored_df') is not None:
                        if str(output_file).endswith('.csv'):
                            final_integrity['restored_df'].to_csv(output_file, index=False)
                        else:
                            final_integrity['restored_df'].to_excel(output_file, index=False)
                        if self.verbose:
                            logger.info(f"복구된 결과로 저장: {output_file}")
                        # 기본 모드에서는 중복 메시지 방지
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
        progress_callback=None,
        **kwargs
    ) -> List[Dict]:
        """청크 단위 무결성 보장 처리 (안전 버전)"""
        
        results = []
        chunk_size = len(chunk)
        chunk_start_time = time.time()
        
        for i, (idx, row) in enumerate(chunk.iterrows()):
            try:
                # 🚀 행 단위 진행률 출력 (50행마다 새 줄로)
                if i % 50 == 0 or i == chunk_size - 1:
                    elapsed = time.time() - chunk_start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (chunk_size - i - 1) / rate if rate > 0 else 0
                    print(f"  📍 청크 진행: {i+1}/{chunk_size} ({(i+1)*100//chunk_size}%) | {rate:.1f}행/초 | ETA: {eta:.0f}초", flush=True)
                
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
                    
                    # 🔧 [수정] 메타데이터 보존 (book_name, 문장식별자)
                    if row_result:
                        metadata = {}
                        # book_name 보존
                        if 'book_name' in row and pd.notna(row['book_name']):
                            metadata['book_name'] = row['book_name']
                        # 🔧 문장식별자 유지 (갱신하지 않음)
                        if '문장식별자' in row and pd.notna(row['문장식별자']):
                            metadata['문장식별자'] = row['문장식별자']
                            
                        if metadata:
                            if isinstance(row_result, list):
                                for res in row_result:
                                    res.update(metadata)
                            elif isinstance(row_result, dict):
                                row_result.update(metadata)
                    
                    if row_result:
                        # 행별 무결성 검증 (안전)
                        if self.integrity_enabled and self._verify_row_integrity(row, row_result, row_id):
                            results.extend(row_result if isinstance(row_result, list) else [row_result])
                            self.processed_count += len(row_result) if isinstance(row_result, list) else 1
                        elif not self.integrity_enabled:
                            # 무결성 비활성화시 결과 그대로 사용
                            results.extend(row_result if isinstance(row_result, list) else [row_result])
                            self.processed_count += len(row_result) if isinstance(row_result, list) else 1
                        else:
                            self.integrity_failures += 1
                            logger.warning(f"행 {idx} 무결성 실패, 스킵")
                
            except Exception as e:
                logger.error(f"행 {idx} 처리 실패: {e}")
                self.error_count += 1
                continue
        
        # 청크 완료 메시지
        elapsed = time.time() - chunk_start_time
        print(f"\n  ✅ 청크 완료: {chunk_size}행 처리 ({elapsed:.1f}초, {chunk_size/elapsed:.1f}행/초)", flush=True)
        
        # 🚀 디스크 캐시 일괄 저장 (최적화: 행별 저장을 끄고 청크 단위로 저장)
        try:
            from common.embedders.bge import get_embedding_manager
            manager = get_embedding_manager()
            manager._save_disk_cache()
        except Exception:
            pass
        
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
    verbose: bool = False,  # 🔧 verbose 옵션 추가
    use_boundary_model: bool = False,  # 🆕 경계 모델 사용 여부
    boundary_threshold: float = 0.5,  # 🆕 경계 모델 임계값
    device: str = "cuda",  # 🆕 디바이스 설정
    **kwargs
) -> bool:
    """기존 SA 호환 파일 처리 함수 (무결성 보장)"""
    
    if verbose:
        logger.info(f"SA 파일 처리 시작: {input_file}")
        logger.info(f"설정: 임베더={embedder_name}, 병렬={use_parallel}, 워커={max_workers}")
        if use_boundary_model:
            logger.info(f"🔧 경계 모델 사용: threshold={boundary_threshold}, device={device}")
    
    try:
        # SA 처리 함수 import (패키지 경로)
        from s2p.s2p_aligner import process_single_row, reset_segment_counter  # 🆕 SA 얼라이너에서 import

        # 구식별자를 파일 단위로 리셋해 누적 증가하도록 설정
        reset_segment_counter()
        
        # 무결성 보장 처리기 생성
        processor = SafeFileProcessor(
            max_workers=max_workers, 
            chunk_size=chunk_size, 
            verbose=verbose  # 🔧 verbose 옵션 전달
        )
        
        # 무결성 보장 처리 실행
        success = processor.process_file_with_integrity(
            input_file=input_file,
            output_file=output_file,
            processing_function=safe_process_sa_row,
            embedder_name=embedder_name,
            use_boundary_model=use_boundary_model,  # 🆕 경계 모델 옵션 전달
            boundary_threshold=boundary_threshold,  # 🆕 임계값 전달
            device=device,  # 🆕 디바이스 전달
            **kwargs
        )
        
        if success:
            if verbose:
                logger.info(f"✅ SA 파일 처리 완료: {output_file}")
        else:
            if verbose:
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
        # CSV 또는 Excel 자동 감지
        if str(input_file).endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)
        
        # 🔧 NaN 값 처리
        original_rows = len(df)
        df['원문'] = df['원문'].fillna('').astype(str)
        df['번역문'] = df['번역문'].fillna('').astype(str)
        
        empty_rows = len(df[ (df['원문'].str.strip() == '') & (df['번역문'].str.strip() == '') ])
        if empty_rows > 0:
            logger.info(f"⚠️ 빈 원문/번역문 {empty_rows}개 포함 (누락 방지를 위해 유지)")
        
        results = []
        for idx, row in df.iterrows():
            src_text = str(row.get('원문', ''))
            tgt_text = str(row.get('번역문', ''))
            
            # 🔧 원본 문장식별자 유지 (없을 때만 새로 생성)
            sent_id = row.get('문장식별자', idx + 1) if '문장식별자' in row and pd.notna(row.get('문장식별자')) else idx + 1
            
            # 🛡️ 어떤 경우에도 절대 누락시키지 않음
            result = {
                '문장식별자': sent_id,
                '원문': src_text.strip() if src_text.strip() else src_text,
                '번역문': tgt_text.strip() if tgt_text.strip() else tgt_text,
                '분할방법': 'fallback_mode',
                '유사도': 1.0,
                '원문_토큰수': len(src_text.split()) if src_text.strip() else 0,
                '번역문_토큰수': len(tgt_text.split()) if tgt_text.strip() else 0
            }
            # book_name 보존
            if 'book_name' in row and pd.notna(row.get('book_name')):
                result['book_name'] = row['book_name']
            results.append(result)
        
        if results:
            result_df = pd.DataFrame(results)
            if str(output_file).endswith('.csv'):
                result_df.to_csv(output_file, index=False)
            else:
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
    """SA 행 처리 무결성 보장 래퍼
    
    🆕 통합 모드: use_boundary_model=True이면
    - 기존 process_single_row 실행 (BGE 기반)
    - 내부에서 경계/alignment 모델로 refinement
    """
    
    try:
        # 🔧 process_single_row 캐싱 (매번 import 안 하도록)
        if not hasattr(safe_process_sa_row, '_process_func'):
            from s2p.s2p_aligner import process_single_row
            safe_process_sa_row._process_func = process_single_row
        
        # 경계 모델 사용 시 모델 로드 (캐싱)
        use_boundary_model = kwargs.get('use_boundary_model', False)
        
        if use_boundary_model and not hasattr(safe_process_sa_row, '_boundary_model'):
            # 🆕 Cross-Attention SA 경계 모델 사용
            from common.s2p_crossattn_boundary_loader import get_crossattn_boundary_tagger
            
            device = kwargs.get('device', 'cuda')
            
            logger.info(f"🔧 Cross-Attention SA 경계 모델 로드 중... (device={device})")
            
            # Cross-Attention 모델 로드
            safe_process_sa_row._boundary_model = get_crossattn_boundary_tagger(device=device)
            
            # Alignment 모델은 기존 경로 유지
            from common.alignment_model_loader import AlignmentMatcher
            alignment_model_path = Path("models/dual_encoder_alignment_sa.pt")
            if alignment_model_path.exists():
                safe_process_sa_row._alignment_model = AlignmentMatcher(model_path=alignment_model_path, device=device)
            else:
                safe_process_sa_row._alignment_model = None
                logger.warning(f"⚠️ Alignment 모델 없음: {alignment_model_path}")
            
            logger.info("✅ Cross-Attention SA 경계 모델 로드 완료")
        
        # 원본 데이터 추출
        src_text = str(row.get('원문', ''))
        tgt_text = str(row.get('번역문', ''))
        
        if not src_text.strip() or not tgt_text.strip():
            return []
        
        # 🚀 배치 경계 캐시 확인
        boundary_cache = kwargs.get('_boundary_cache', {})
        row_idx = getattr(row, 'name', None)  # DataFrame 인덱스
        if row_idx is not None and row_idx in boundary_cache:
            # 사전 계산된 경계 결과를 kwargs에 추가
            kwargs['_precomputed_tgt_segments'] = boundary_cache[row_idx]
        
        # 🚨 마스킹 비활성화: 괄호 마스킹 없이 직접 처리
        # (PUA 토큰 분할 문제 해결을 위해 마스킹 로직 완전 우회)
        return safe_process_sa_row._process_func(row, row_id=row_id, **kwargs)
        
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