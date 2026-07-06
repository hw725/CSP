"""
XML 파이프라인 프로세서 — PA/SA 정렬 파이프라인 오케스트레이션

순수 XML 추출은 xml_extractor 패키지 참조.
이 모듈은 PA/SA subprocess 실행, 정확도 평가, 유사도 분석을 담당.
"""

import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import sqlite3
import json
import time
from datetime import datetime

# xml_extractor에서 추출 기능 import
from xml_extractor.xml_processor import XMLProcessor, XMLPair, create_xml_pair_from_directory

logger = logging.getLogger(__name__)


class XMLPipelineProcessor:
    """XML 파이프라인 전체 처리 클래스"""
    
    def __init__(self, results_dir: str = "xml_pipeline_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 연결
        self.db_path = self.results_dir / "xml_pipeline_results.db"
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            # 데이터베이스 디렉토리 권한 확인
            if not self.results_dir.exists():
                self.results_dir.mkdir(parents=True, exist_ok=True)
                
            # 데이터베이스 파일 권한 테스트
            test_db_path = self.db_path
            if test_db_path.exists():
                logger.info(f"기존 데이터베이스 발견: {test_db_path}")
            else:
                logger.info(f"새 데이터베이스 생성: {test_db_path}")
            
            with sqlite3.connect(str(test_db_path)) as conn:
                # XML 쌍 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS xml_pairs (
                        pair_id TEXT PRIMARY KEY,
                        name TEXT,
                        original_file TEXT,
                        translation_file TEXT,
                        result_folder TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'created'
                    )
                """)
                
                # 파이프라인 결과 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair_id TEXT,
                        stage TEXT,
                        status TEXT,
                        processing_time REAL,
                        accuracy_score REAL,
                        notes TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (pair_id) REFERENCES xml_pairs (pair_id)
                    )
                """)
                
                conn.commit()
                
                # 테이블 생성 확인
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                logger.info(f"데이터베이스 테이블 생성 완료: {[t[0] for t in tables]}")
                
        except Exception as e:
            error_msg = f"데이터베이스 초기화 실패: {e}\n경로: {self.db_path}\n권한 문제일 가능성이 있습니다."
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    def _compute_global_text_integrity(self, ground_truth_file: str, prediction_file: str, orig_xml: str = None, trans_xml: str = None) -> Dict[str, float]:
        """전역 텍스트 무결성 계산 (accuracy_evaluator 로직 적용) - 분리된 XML 파일 지원"""
        try:
            # 파일 로드
            gt_df = pd.read_excel(ground_truth_file, engine='openpyxl')
            pred_df = pd.read_excel(prediction_file, engine='openpyxl')
            
            # 컬럼 탐지
            gt_src_col, gt_tgt_col = self._detect_source_target_cols(gt_df)
            pred_src_col, pred_tgt_col = self._detect_source_target_cols(pred_df)
            
            # 🔧 분리된 XML 파일이 있는 경우 직접 처리 - 비활성화
            # if orig_xml and trans_xml and Path(orig_xml).exists() and Path(trans_xml).exists():
            #     print("🔍 분리된 XML 파일에서 직접 전역 무결성 계산")
            #     return self._compute_global_integrity_from_xml_pair(orig_xml, trans_xml, pred_df, pred_src_col, pred_tgt_col)
            
            if not all([gt_src_col, gt_tgt_col, pred_src_col, pred_tgt_col]):
                print("⚠️ 전역 무결성: 컬럼 탐지 실패")
                return {
                    'global_source_text_similarity': 0.0,
                    'global_target_text_similarity': 0.0,
                    'global_source_text_match': 0.0,
                    'global_target_text_match': 0.0,
                    'error': '컬럼 탐지 실패'
                }
            
            # 전역 텍스트 결합 (공백 없이)
            def combine_text(df: pd.DataFrame, col: str) -> str:
                try:
                    return ''.join(df[col].astype(str).fillna(''))
                except Exception:
                    return ''
            
            gt_src_all = combine_text(gt_df, gt_src_col)
            gt_tgt_all = combine_text(gt_df, gt_tgt_col)
            pred_src_all = combine_text(pred_df, pred_src_col)
            pred_tgt_all = combine_text(pred_df, pred_tgt_col)
            
            # 유사도 및 일치도 계산
            import difflib
            
            # 원문 유사도
            src_similarity = difflib.SequenceMatcher(None, gt_src_all, pred_src_all).ratio()
            src_match = 1.0 if gt_src_all == pred_src_all else 0.0
            
            # 번역문 유사도  
            tgt_similarity = difflib.SequenceMatcher(None, gt_tgt_all, pred_tgt_all).ratio()
            tgt_match = 1.0 if gt_tgt_all == pred_tgt_all else 0.0
            
            # 길이 차이 계산
            src_len_gt = len(gt_src_all)
            src_len_pred = len(pred_src_all)
            tgt_len_gt = len(gt_tgt_all)
            tgt_len_pred = len(pred_tgt_all)
            
            # diff 상세 분석
            src_diff = self._analyze_text_diff(gt_src_all, pred_src_all)
            tgt_diff = self._analyze_text_diff(gt_tgt_all, pred_tgt_all)
            
            return {
                'global_source_len_gt': src_len_gt,
                'global_source_len_pred': src_len_pred,
                'global_source_delta': src_len_pred - src_len_gt,
                'global_target_len_gt': tgt_len_gt,
                'global_target_len_pred': tgt_len_pred,
                'global_target_delta': tgt_len_pred - tgt_len_gt,
                'global_source_text_similarity': float(src_similarity),
                'global_target_text_similarity': float(tgt_similarity),
                'global_source_text_match': float(src_match),
                'global_target_text_match': float(tgt_match),
                'global_source_ops_insert': int(src_diff['insert']),
                'global_source_ops_delete': int(src_diff['delete']),
                'global_source_ops_replace': int(src_diff['replace']),
                'global_source_first_diff_index': int(src_diff['first_diff_a_idx']),
                'global_source_first_diff_context_gt': src_diff['first_diff_a_ctx'],
                'global_source_first_diff_context_pred': src_diff['first_diff_b_ctx'],
                'global_target_ops_insert': int(tgt_diff['insert']),
                'global_target_ops_delete': int(tgt_diff['delete']),
                'global_target_ops_replace': int(tgt_diff['replace']),
                'global_target_first_diff_index': int(tgt_diff['first_diff_a_idx']),
                'global_target_first_diff_context_gt': tgt_diff['first_diff_a_ctx'],
                'global_target_first_diff_context_pred': tgt_diff['first_diff_b_ctx'],
            }
            
        except Exception as e:
            print(f"❌ 전역 무결성 계산 오류: {e}")
            return {
                'global_source_text_similarity': 0.0,
                'global_target_text_similarity': 0.0,
                'global_source_text_match': 0.0,
                'global_target_text_match': 0.0,
                'error': str(e)
            }
    
    def _detect_source_target_cols(self, df: pd.DataFrame) -> Tuple[str, str]:
        """원문/번역문 컬럼 탐지"""
        source_col = None
        target_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if source_col is None and ('원문' in str(col) or 'source' in col_lower or 'original' in col_lower):
                source_col = col
            if target_col is None and ('번역문' in str(col) or '번역' in str(col) or 'target' in col_lower or 'translation' in col_lower):
                target_col = col
                
        return source_col, target_col
    
    def _analyze_text_diff(self, text_a: str, text_b: str) -> Dict[str, any]:
        """텍스트 차이 상세 분석 - 공백 외 차이점 중점 분석"""
        import difflib
        import re
        
        # [, ], - 문자 제거 (비교 전 정규화)
        text_a = re.sub(r'[\[\-\]]', '', text_a)
        text_b = re.sub(r'[\[\-\]]', '', text_b)
        
        sm = difflib.SequenceMatcher(a=text_a, b=text_b)
        insert_count = 0
        delete_count = 0
        replace_count = 0
        first_a_idx = None
        first_b_idx = None
        
        # 공백이 아닌 차이점들을 저장할 리스트
        non_whitespace_diffs = []
        whitespace_only_diffs = []
        
        print(f"\n🔍 텍스트 차이 상세 분석:")
        print(f"   원본 길이: {len(text_a):,}자")
        print(f"   비교 대상 길이: {len(text_b):,}자")
        
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
                
            if first_a_idx is None:
                first_a_idx, first_b_idx = i1, j1
                
            # 차이점 텍스트 추출
            diff_text_a = text_a[i1:i2] if i1 < i2 else ""
            diff_text_b = text_b[j1:j2] if j1 < j2 else ""
            
            # 공백만인지 확인
            is_whitespace_only = (
                re.match(r'^\s*$', diff_text_a) and re.match(r'^\s*$', diff_text_b)
            ) or (
                diff_text_a.replace(' ', '') == diff_text_b.replace(' ', '') and 
                diff_text_a != diff_text_b
            )
            
            if tag == 'insert':
                insert_count += (j2 - j1)
                if not is_whitespace_only and diff_text_b.strip():
                    non_whitespace_diffs.append({
                        'type': 'INSERT',
                        'position': j1,
                        'content': repr(diff_text_b[:50]),  # 처음 50자만
                        'context_before': text_b[max(0, j1-20):j1],
                        'context_after': text_b[j2:min(len(text_b), j2+20)]
                    })
                elif is_whitespace_only:
                    whitespace_only_diffs.append({'type': 'INSERT_WS', 'content': repr(diff_text_b)})
                    
            elif tag == 'delete':
                delete_count += (i2 - i1)
                if not is_whitespace_only and diff_text_a.strip():
                    non_whitespace_diffs.append({
                        'type': 'DELETE',
                        'position': i1,
                        'content': repr(diff_text_a[:50]),  # 처음 50자만
                        'context_before': text_a[max(0, i1-20):i1],
                        'context_after': text_a[i2:min(len(text_a), i2+20)]
                    })
                elif is_whitespace_only:
                    whitespace_only_diffs.append({'type': 'DELETE_WS', 'content': repr(diff_text_a)})
                    
            elif tag == 'replace':
                replace_count += max(i2 - i1, j2 - j1)
                if not is_whitespace_only:
                    non_whitespace_diffs.append({
                        'type': 'REPLACE',
                        'position': i1,
                        'original': repr(diff_text_a[:50]),
                        'replaced': repr(diff_text_b[:50]),
                        'context_before': text_a[max(0, i1-20):i1],
                        'context_after': text_a[i2:min(len(text_a), i2+20)]
                    })
                elif is_whitespace_only:
                    whitespace_only_diffs.append({
                        'type': 'REPLACE_WS', 
                        'original': repr(diff_text_a), 
                        'replaced': repr(diff_text_b)
                    })
        
            # 비공백 차이점 로깅 및 파일 저장
            diff_log_content = []
            diff_log_content.append(f"텍스트 차이 분석 결과 ({len(text_a):,}자 vs {len(text_b):,}자)")
            diff_log_content.append("=" * 60)
            
            if non_whitespace_diffs:
                print(f"\n⚠️  공백 외 차이점 발견 ({len(non_whitespace_diffs)}개):")
                diff_log_content.append(f"\n⚠️ 공백 외 차이점 발견 ({len(non_whitespace_diffs)}개):")
                
                for i, diff in enumerate(non_whitespace_diffs):
                    if diff['type'] == 'INSERT':
                        msg = f"   {i+1}. 삽입 위치 {diff['position']}: {diff['content']}"
                        context_msg = f"      컨텍스트: ...{diff['context_before']}{diff['context_after']}..."
                        if i < 10:  # 콘솔에는 10개만
                            print(msg)
                            print(context_msg)
                        diff_log_content.append(msg)
                        diff_log_content.append(context_msg)
                    elif diff['type'] == 'DELETE':
                        msg = f"   {i+1}. 삭제 위치 {diff['position']}: {diff['content']}"
                        context_msg = f"      컨텍스트: ...{diff['context_before']}{diff['context_after']}..."
                        if i < 10:  # 콘솔에는 10개만
                            print(msg)
                            print(context_msg)
                        diff_log_content.append(msg)
                        diff_log_content.append(context_msg)
                    elif diff['type'] == 'REPLACE':
                        msg = f"   {i+1}. 변경 위치 {diff['position']}: {diff['original']} → {diff['replaced']}"
                        context_msg = f"      컨텍스트: ...{diff['context_before']}{diff['context_after']}..."
                        if i < 10:  # 콘솔에는 10개만
                            print(msg)
                            print(context_msg)
                        diff_log_content.append(msg)
                        diff_log_content.append(context_msg)
                
                if len(non_whitespace_diffs) > 10:
                    remaining = len(non_whitespace_diffs) - 10
                    print(f"   ... 외 {remaining}개 차이점 더 있음")
                    diff_log_content.append(f"   ... 외 {remaining}개 차이점 더 있음")
            else:
                print(f"✅ 공백 외 차이점 없음 - 차이는 모두 공백/띄어쓰기 관련")
                diff_log_content.append(f"✅ 공백 외 차이점 없음 - 차이는 모두 공백/띄어쓰기 관련")
            
            # 공백 차이점 요약
            if whitespace_only_diffs:
                print(f"📝 공백 차이점: {len(whitespace_only_diffs)}개 (요약 생략)")
                diff_log_content.append(f"📝 공백 차이점: {len(whitespace_only_diffs)}개")
                
                # 공백 차이점도 파일에만 저장
                if len(whitespace_only_diffs) > 0:
                    diff_log_content.append("\n공백 관련 차이점:")
                    for i, diff in enumerate(whitespace_only_diffs[:20]):  # 최대 20개
                        diff_log_content.append(f"   {i+1}. {diff['type']}: {diff.get('content', 'N/A')}")
                    if len(whitespace_only_diffs) > 20:
                        diff_log_content.append(f"   ... 외 {len(whitespace_only_diffs)-20}개 더")
            
            # 차이점 로그를 전역 변수에 저장 (나중에 파일로 저장하기 위해)
            if not hasattr(self, '_diff_logs'):
                self._diff_logs = []
            self._diff_logs.append('\n'.join(diff_log_content))        # 첫 차이점 컨텍스트 추출
        def get_context(text: str, idx: int, width: int = 20) -> str:
            if idx is None:
                return ''
            start = max(0, idx - width)
            end = min(len(text), idx + width)
            return text[start:end]
        
        return {
            'insert': insert_count,
            'delete': delete_count,
            'replace': replace_count,
            'first_diff_a_idx': -1 if first_a_idx is None else first_a_idx,
            'first_diff_b_idx': -1 if first_b_idx is None else first_b_idx,
            'first_diff_a_ctx': get_context(text_a, first_a_idx),
            'first_diff_b_ctx': get_context(text_b, first_b_idx),
            'non_whitespace_diffs_count': len(non_whitespace_diffs),
            'whitespace_only_diffs_count': len(whitespace_only_diffs),
            'non_whitespace_diffs': non_whitespace_diffs[:20]  # 최대 20개까지 저장
        }
    
    def _compute_global_integrity_from_xml_pair(self, orig_xml: str, trans_xml: str, pred_df: pd.DataFrame, pred_src_col: str, pred_tgt_col: str) -> Dict[str, float]:
        """분리된 XML 파일 쌍에서 직접 전역 무결성 계산"""
        try:
            import xml.etree.ElementTree as ET
            
            print(f"🔍 XML 쌍에서 전역 텍스트 추출:")
            print(f"   원문 XML: {Path(orig_xml).name}")
            print(f"   번역문 XML: {Path(trans_xml).name}")
            
            # 원문 XML에서 전체 텍스트 추출
            orig_tree = ET.parse(orig_xml)
            orig_root = orig_tree.getroot()
            orig_elements = orig_root.findall('.//원문')
            
            orig_all_text = ""
            for orig_elem in orig_elements:
                orig_text = XMLProcessor._join_w_texts(orig_elem)
                # 추가 정규식 적용 (이중 보장)
                import re
                orig_text = re.sub(r'[\[\-\]]', '', orig_text)
                orig_all_text += orig_text
            
            # 번역문 XML에서 전체 텍스트 추출
            trans_tree = ET.parse(trans_xml)
            trans_root = trans_tree.getroot()
            trans_elements = trans_root.findall('.//번역문')
            
            trans_all_text = ""
            for trans_elem in trans_elements:
                trans_text = XMLProcessor._join_w_texts(trans_elem)
                # 추가 정규식 적용 (이중 보장)
                import re
                trans_text = re.sub(r'[\[\-\]]', '', trans_text)
                trans_all_text += trans_text
            
            # 텍스트 정제 함수 정의
            def clean_text_for_comparison(text: str) -> str:
                """텍스트 비교용 정제: [ ] - 부호 제거"""
                import re
                # [ ] - 부호 제거
                text = re.sub(r'[\[\-\]]', '', text)
                return text
            
            # 텍스트 정제 ([ ] - 부호 제거)
            orig_all_text = clean_text_for_comparison(orig_all_text)
            trans_all_text = clean_text_for_comparison(trans_all_text)
            
            print(f"📊 XML 전역 텍스트:")
            print(f"   원문 길이: {len(orig_all_text):,}자")
            print(f"   번역문 길이: {len(trans_all_text):,}자")
            
            # 예측 데이터에서 전체 텍스트 추출
            
            def combine_text(df: pd.DataFrame, col: str) -> str:
                try:
                    if col and col in df.columns:
                        combined = ''.join(df[col].astype(str).fillna(''))
                        return clean_text_for_comparison(combined)
                    return ''
                except Exception:
                    return ''
            
            pred_src_all = combine_text(pred_df, pred_src_col)
            pred_tgt_all = combine_text(pred_df, pred_tgt_col)
            
            print(f"📊 예측 전역 텍스트:")
            print(f"   예측 원문 길이: {len(pred_src_all):,}자")
            print(f"   예측 번역문 길이: {len(pred_tgt_all):,}자")
            
            # 유사도 및 일치도 계산
            import difflib
            
            # 원문 유사도
            src_similarity = difflib.SequenceMatcher(None, orig_all_text, pred_src_all).ratio()
            src_match = 1.0 if orig_all_text == pred_src_all else 0.0
            
            # 번역문 유사도  
            tgt_similarity = difflib.SequenceMatcher(None, trans_all_text, pred_tgt_all).ratio()
            tgt_match = 1.0 if trans_all_text == pred_tgt_all else 0.0
            
            # 길이 차이 계산
            src_len_gt = len(orig_all_text)
            src_len_pred = len(pred_src_all)
            tgt_len_gt = len(trans_all_text)
            tgt_len_pred = len(pred_tgt_all)
            
            # diff 상세 분석
            src_diff = self._analyze_text_diff(orig_all_text, pred_src_all)
            tgt_diff = self._analyze_text_diff(trans_all_text, pred_tgt_all)
            
            print(f"🎯 전역 무결성 결과:")
            print(f"   원문 유사도: {src_similarity:.3f}")
            print(f"   번역문 유사도: {tgt_similarity:.3f}")
            
            # 공백 외 차이점 요약 (안전 호출)
            try:
                self._log_non_whitespace_diff_summary("원문", src_diff)
                self._log_non_whitespace_diff_summary("번역문", tgt_diff)
            except AttributeError as ae:
                print(f"⚠️ 차이점 요약 출력 건너뜀: {ae}")
            except Exception as ee:
                print(f"⚠️ 차이점 요약 중 오류: {ee}")
            
            return {
                'global_source_len_gt': src_len_gt,
                'global_source_len_pred': src_len_pred,
                'global_source_delta': src_len_pred - src_len_gt,
                'global_target_len_gt': tgt_len_gt,
                'global_target_len_pred': tgt_len_pred,
                'global_target_delta': tgt_len_pred - tgt_len_gt,
                'global_source_text_similarity': float(src_similarity),
                'global_target_text_similarity': float(tgt_similarity),
                'global_source_text_match': float(src_match),
                'global_target_text_match': float(tgt_match),
                'global_source_ops_insert': int(src_diff['insert']),
                'global_source_ops_delete': int(src_diff['delete']),
                'global_source_ops_replace': int(src_diff['replace']),
                'global_source_first_diff_index': int(src_diff['first_diff_a_idx']),
                'global_source_first_diff_context_gt': src_diff['first_diff_a_ctx'],
                'global_source_first_diff_context_pred': src_diff['first_diff_b_ctx'],
                'global_target_ops_insert': int(tgt_diff['insert']),
                'global_target_ops_delete': int(tgt_diff['delete']),
                'global_target_ops_replace': int(tgt_diff['replace']),
                'global_target_first_diff_index': int(tgt_diff['first_diff_a_idx']),
                'global_target_first_diff_context_gt': tgt_diff['first_diff_a_ctx'],
                'global_target_first_diff_context_pred': tgt_diff['first_diff_b_ctx'],
            }
            
        except Exception as e:
            print(f"❌ XML 쌍 전역 무결성 계산 오류: {e}")
            return {
                'global_source_text_similarity': 0.0,
                'global_target_text_similarity': 0.0,
                'global_source_text_match': 0.0,
                'global_target_text_match': 0.0,
                'error': f'XML 쌍 처리 오류: {e}'
            }
    
    def _basic_accuracy_comparison_with_integrity(self, xml_phrase_file: str, sa_phrase_file: str, global_integrity: Dict) -> Dict[str, any]:
        """AccuracyEvaluator가 없을 때의 기본 비교 (전역 무결성 포함)"""
        try:
            xml_df = pd.read_excel(xml_phrase_file, engine='openpyxl')
            sa_df = pd.read_excel(sa_phrase_file, engine='openpyxl')
            
            xml_count = len(xml_df)
            sa_count = len(sa_df)
            
            # 길이 기반 기본 정확도
            length_accuracy = min(sa_count / xml_count, 1.0) if xml_count > 0 else 0.0
            
            # 전역 무결성 점수 반영
            source_integrity = global_integrity.get('global_source_text_similarity', 0.0)
            target_integrity = global_integrity.get('global_target_text_similarity', 0.0)
            
            # 종합 점수 (길이 정확도와 무결성의 가중평균)
            combined_score = (length_accuracy * 0.3 + source_integrity * 0.4 + target_integrity * 0.3)
            
            return {
                'status': 'success',
                'accuracy_score': combined_score,
                'length_based_accuracy': length_accuracy,
                'global_integrity': global_integrity,
                'xml_phrase_count': xml_count,
                'sa_result_count': sa_count
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'accuracy_score': 0.0,
                'global_integrity': global_integrity
            }
    
    def _run_xml_level_similarity_analysis(self, xml_file: str, pa_output: str, sa_output: str, accuracy_dir: Path) -> Dict[str, any]:
        """XML 레벨 유사도 분석 실행"""
        try:
            print("🔍 XML 레벨 유사도 분석 시작...")
            
            # XMLLevelSimilarityCalculator import 시도 (다중 경로 지원)
            try:
                from .xml_level_similarity import XMLLevelSimilarityCalculator
                print("✅ XMLLevelSimilarityCalculator 모듈 로딩 성공 (상대 경로)")
            except ImportError:
                try:
                    from xml_pipeline.xml_level_similarity import XMLLevelSimilarityCalculator
                    print("✅ XMLLevelSimilarityCalculator 모듈 로딩 성공 (절대 경로)")
                except ImportError:
                    try:
                        from xml_level_similarity import XMLLevelSimilarityCalculator
                        print("✅ XMLLevelSimilarityCalculator 모듈 로딩 성공 (직접 경로)")
                    except ImportError:
                        # 추가적인 경로 시도
                        import sys
                        import os
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        sys.path.insert(0, current_dir)
                        try:
                            from xml_level_similarity import XMLLevelSimilarityCalculator
                            print("✅ XMLLevelSimilarityCalculator 모듈 로딩 성공 (sys.path 추가)")
                        except ImportError:
                            print("❌ XMLLevelSimilarityCalculator를 찾을 수 없습니다. XML 레벨 분석을 건너뜁니다.")
                            print(f"   현재 디렉토리: {current_dir}")
                            print(f"   sys.path: {sys.path[:3]}...")
                            return {
                                'error': 'XMLLevelSimilarityCalculator 모듈을 찾을 수 없음',
                                'pa_analysis': {'error': '모듈 없음'},
                                'sa_analysis': {'error': '모듈 없음'},
                                'sliding_window_score': 0.0,
                                'lcs_score': 0.0
                            }
            
            # XML 레벨 유사도 계산기 초기화
            calculator = XMLLevelSimilarityCalculator(use_embeddings=True)
            
            results = {
                'pa_analysis': {'error': 'PA 파일 없음'},
                'sa_analysis': {'error': 'SA 파일 없음'},
                'sliding_window_score': 0.0,
                'lcs_score': 0.0
            }
            
            # � 번역문 XML 파일 추정 (PA/SA 공통 사용)
            xml_translation_file = None
            xml_path = Path(xml_file)
            if '원문' in xml_path.name:
                translation_name = xml_path.name.replace('원문', '번역문')
                xml_translation_file = str(xml_path.parent / translation_name)
                if not Path(xml_translation_file).exists():
                    xml_translation_file = None
                    print(f"⚠️ 번역문 XML 파일을 찾을 수 없습니다: {xml_translation_file}")
                else:
                    print(f"✅ 번역문 XML 파일 확인: {xml_translation_file}")
            
            # PA 분석 (문장 단위)
            if pa_output and Path(pa_output).exists():
                print("📊 PA XML 레벨 분석 수행")
                try:
                    
                    pa_analysis = calculator.calculate_pa_similarity(xml_file, pa_output, xml_translation_file)
                    results['pa_analysis'] = pa_analysis
                    
                    # PA 점수 추출
                    if 'avg_similarity' in pa_analysis:
                        results['sliding_window_score'] = pa_analysis['avg_similarity']
                    
                    print(f"✅ PA 분석 완료: {pa_analysis.get('xml_unit_count', 0)}개 XML 단위 vs {pa_analysis.get('result_row_count', 0)}개 결과")
                except Exception as e:
                    print(f"⚠️ PA 분석 실패: {e}")
                    results['pa_analysis'] = {'error': f'PA 분석 실패: {e}'}
            
            # SA 분석 (구 단위)
            if sa_output and Path(sa_output).exists():
                print("📊 SA XML 레벨 분석 수행")
                try:
                    # 번역문 XML 파일을 SA에도 전달 (이미 위에서 추정함)
                    sa_analysis = calculator.calculate_sa_similarity(xml_file, sa_output, xml_translation_file)
                    results['sa_analysis'] = sa_analysis
                    
                    # SA 점수 추출
                    if 'avg_similarity' in sa_analysis:
                        results['lcs_score'] = sa_analysis['avg_similarity']
                    
                    print(f"✅ SA 분석 완료: {sa_analysis.get('xml_unit_count', 0)}개 XML 단위 vs {sa_analysis.get('result_row_count', 0)}개 결과")
                except Exception as e:
                    print(f"⚠️ SA 분석 실패: {e}")
                    results['sa_analysis'] = {'error': f'SA 분석 실패: {e}'}
            
            # 종합 점수 계산
            pa_score = results['sliding_window_score']
            sa_score = results['lcs_score']
            
            if pa_score > 0 and sa_score > 0:
                combined_score = (pa_score + sa_score) / 2
            elif pa_score > 0:
                combined_score = pa_score
            elif sa_score > 0:
                combined_score = sa_score
            else:
                combined_score = 0.0
            
            results['combined_xml_level_score'] = combined_score
            
            print(f"🎯 XML 레벨 유사도: PA({pa_score:.3f}) + SA({sa_score:.3f}) = 종합({combined_score:.3f})")
            
            # 💡 종합 유사도 분석 (전역 무결성 포함) - PA 파일이 없어도 실행
            if sa_output and Path(sa_output).exists():
                try:
                    print("🎯 종합 유사도 분석 (전역 무결성 포함) 시작...")
                    
                    # 전역 무결성 데이터 계산 (SA 파일 기반)
                    global_integrity = None
                    if xml_translation_file:
                        try:
                            global_integrity = self._compute_global_text_integrity(
                                xml_translation_file, sa_output, xml_file, xml_translation_file
                            )
                        except Exception as e:
                            print(f"⚠️ 전역 무결성 계산 실패: {e}")
                    
                    # PA 파일 경로 처리 (없으면 빈 문자열)
                    pa_file_path = pa_output if pa_output and Path(pa_output).exists() else ""
                    
                    comprehensive_results = calculator.calculate_comprehensive_similarity(
                        xml_file=xml_file,
                        pa_result_file=pa_file_path,
                        sa_result_file=sa_output,
                        xml_translation_file=xml_translation_file,
                        global_integrity=global_integrity
                    )
                    results['comprehensive_analysis'] = comprehensive_results
                    print("✅ 종합 유사도 분석 완료 (전역 무결성 포함)")
                except Exception as e:
                    print(f"⚠️ 종합 유사도 분석 실패: {e}")
                    results['comprehensive_analysis'] = {'error': f'종합 분석 실패: {e}'}
            
            # 결과를 파일로 저장
            xml_level_report_file = accuracy_dir / "xml_level_similarity.json"
            with open(xml_level_report_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            return results
            
        except Exception as e:
            error_msg = f"XML 레벨 분석 오류: {e}"
            print(f"❌ {error_msg}")
            # 상세 오류 정보 출력
            import traceback
            print("❌ XML 레벨 분석 상세 오류:")
            traceback.print_exc()
            
            # 오류가 발생해도 빈 결과 파일 생성
            error_results = {
                'error': error_msg,
                'pa_analysis': {'error': error_msg},
                'sa_analysis': {'error': error_msg},
                'sliding_window_score': 0.0,
                'lcs_score': 0.0,
                'combined_xml_level_score': 0.0
            }
            
            # 오류 결과도 파일로 저장
            try:
                xml_level_report_file = accuracy_dir / "xml_level_similarity.json"
                with open(xml_level_report_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(error_results, f, indent=2, ensure_ascii=False, default=str)
                print(f"💾 오류 결과 저장: {xml_level_report_file}")
            except Exception as save_error:
                print(f"❌ 오류 결과 저장 실패: {save_error}")
                
            return error_results
    
    def add_xml_pair(self, xml_pair: XMLPair):
        """XML 쌍을 데이터베이스에 추가"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO xml_pairs 
                (pair_id, name, original_file, translation_file, status)
                VALUES (?, ?, ?, ?, 'registered')
            """, (xml_pair.id, xml_pair.name, xml_pair.original_path, xml_pair.translation_path))
            conn.commit()
    
    def get_xml_pairs(self) -> List[Dict[str, any]]:
        """데이터베이스에서 모든 XML 쌍 조회"""
        with sqlite3.connect(self.db_path) as conn:
            # Row factory 설정으로 딕셔너리 형태로 결과 반환
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT pair_id, name, original_file, translation_file, 
                       result_folder, created_at, status
                FROM xml_pairs 
                ORDER BY created_at DESC
            """)
            results = []
            for row in cursor:
                results.append({
                    'pair_id': row['pair_id'],
                    'name': row['name'],
                    'original_file': row['original_file'],
                    'translation_file': row['translation_file'],
                    'result_folder': row['result_folder'],
                    'created_at': row['created_at'],
                    'status': row['status']
                })
            return results
    
    def process_xml_pair_pipeline(self, xml_pair: XMLPair) -> Dict[str, any]:
        """XML 쌍 전체 파이프라인 처리"""
        
        start_time = time.time()
        
        # XMLPair 객체에서 정보 추출
        pair_id = xml_pair.id
        original_xml = xml_pair.original_path
        translation_xml = xml_pair.translation_path
        
        # 결과 폴더 생성
        result_folder = self.results_dir / f"{pair_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_folder.mkdir(exist_ok=True)
        
        print(f"🚀 XML 파이프라인 시작: {pair_id}")
        print(f"📁 결과 폴더: {result_folder}")
        
        # 데이터베이스에 쌍 등록
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO xml_pairs 
                (pair_id, name, original_file, translation_file, result_folder, status)
                VALUES (?, ?, ?, ?, ?, 'processing')
            """, (pair_id, xml_pair.name, original_xml, translation_xml, str(result_folder)))
            conn.commit()
        
        results = {
            'pair_id': pair_id,
            'result_folder': str(result_folder),
            'stages': {},
            'errors': []
        }
        
        try:
            # 1단계: 문단병렬 생성
            print("📋 1단계: 문단병렬 데이터 생성 중...")
            stage_start = time.time()
            
            paragraph_data = XMLProcessor.extract_paragraph_data(original_xml, translation_xml)
            paragraph_dir = result_folder / "paragraph"
            paragraph_dir.mkdir(exist_ok=True)
            paragraph_file = paragraph_dir / "paragraph_parallel.xlsx"
            paragraph_data.to_excel(paragraph_file, index=False)
            
            stage_time = time.time() - stage_start
            results['stages']['paragraph'] = {
                'status': 'success',
                'time': stage_time,
                'output_file': str(paragraph_file),
                'row_count': len(paragraph_data)
            }
            self._record_stage_result(pair_id, 'paragraph', 'success', stage_time)
            print(f"✅ 문단병렬 완료: {len(paragraph_data)}개 쌍 ({stage_time:.1f}초)")
            
            # 2단계: PA 처리
            print("🔄 2단계: PA 처리 중...")
            stage_start = time.time()
            
            pa_result = self._run_pa_process(paragraph_file, result_folder, xml_pair)
            stage_time = time.time() - stage_start
            results['stages']['pa'] = pa_result
            results['stages']['pa']['time'] = stage_time
            self._record_stage_result(pair_id, 'pa', pa_result['status'], stage_time)
            print(f"✅ PA 처리 완료 ({stage_time:.1f}초)")
            
            # 3단계: 문장병렬 비교
            print("📊 3단계: 문장병렬 비교 중...")
            stage_start = time.time()
            
            sentence_comparison = self._compare_sentence_alignment(
                original_xml, translation_xml, pa_result.get('output_file'), result_folder
            )
            stage_time = time.time() - stage_start
            results['stages']['sentence_comparison'] = sentence_comparison
            results['stages']['sentence_comparison']['time'] = stage_time
            self._record_stage_result(pair_id, 'sentence_comparison', sentence_comparison['status'], stage_time)
            print(f"✅ 문장병렬 비교 완료 ({stage_time:.1f}초)")
            
            # 4단계: SA 처리
            print("🔄 4단계: SA 처리 중...")
            stage_start = time.time()
            
            sa_result = self._run_sa_process(sentence_comparison.get('sa_input_file'), result_folder, xml_pair)
            stage_time = time.time() - stage_start
            results['stages']['sa'] = sa_result
            results['stages']['sa']['time'] = stage_time
            self._record_stage_result(pair_id, 'sa', sa_result['status'], stage_time)
            print(f"✅ SA 처리 완료 ({stage_time:.1f}초)")
            
            # 5단계: 구병렬 비교 및 정확도 분석
            print("📈 5단계: 정확도 분석 중...")
            stage_start = time.time()
            
            accuracy_result = self._analyze_accuracy(
                original_xml, translation_xml, sa_result.get('output_file'), result_folder,
                pa_output_file=pa_result.get('output_file')  # PA 결과 파일 전달
            )
            stage_time = time.time() - stage_start
            results['stages']['accuracy'] = accuracy_result
            results['stages']['accuracy']['time'] = stage_time
            
            accuracy_score = accuracy_result.get('accuracy_score', 0.0)
            self._record_stage_result(pair_id, 'accuracy', accuracy_result['status'], stage_time, accuracy_score)
            print(f"✅ 정확도 분석 완료: {accuracy_score:.1%} ({stage_time:.1f}초)")
            
            # 매칭 정확도 지표 설명
            if accuracy_result.get('status') == 'success':
                try:
                    self._explain_matching_accuracy_metrics(accuracy_result)
                except Exception as e:
                    print(f"⚠️ 매칭 정확도 지표 설명 건너뜀: {e}")
            
            # 전체 결과 요약 생성
            total_time = time.time() - start_time
            summary = self._generate_pipeline_summary(results, total_time)
            
            # 요약 파일 저장
            summary_file = result_folder / "pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # 텍스트 요약도 저장
            text_summary = self._generate_text_summary(summary)
            text_summary_file = result_folder / "pipeline_summary.txt"
            with open(text_summary_file, 'w', encoding='utf-8') as f:
                f.write(text_summary)
            
            # 데이터베이스 상태 업데이트
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE xml_pairs SET status = 'completed' WHERE pair_id = ?
                """, (pair_id,))
                conn.commit()
            
            print(f"🎉 파이프라인 완료! 총 소요시간: {total_time:.1f}초")
            return summary
            
        except Exception as e:
            error_msg = f"파이프라인 오류: {e}"
            results['errors'].append(error_msg)
            
            # 데이터베이스 상태 업데이트
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE xml_pairs SET status = 'failed' WHERE pair_id = ?
                """, (pair_id,))
                conn.commit()
            
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _run_pa_process(self, input_file: str, result_folder: Path, xml_pair: XMLPair = None) -> Dict[str, any]:
        """PA 프로세스 실행"""
        try:
            pa_dir = result_folder / "pa_results"
            pa_dir.mkdir(exist_ok=True)
            pa_output = pa_dir / "pa_output.xlsx"
            
            # PA 모듈 실행 - 절대 경로 사용
            import subprocess
            import sys
            
            # 현재 작업 디렉토리 확인 및 PA 모듈 경로 설정
            current_dir = Path.cwd()
            pa_main_path = current_dir / "pa" / "main.py"
            
            # PA main.py 파일 존재 확인
            if not pa_main_path.exists():
                return {
                    'status': 'failed',
                    'error': f"PA main.py를 찾을 수 없습니다: {pa_main_path}",
                    'stdout': f"현재 디렉토리: {current_dir}"
                }
            
            # 입력 파일 존재 확인
            if not Path(input_file).exists():
                return {
                    'status': 'failed',
                    'error': f"입력 파일을 찾을 수 없습니다: {input_file}",
                    'stdout': ''
                }
            
            # 🎯 PA 실행 - 원래 방식으로 복원
            base_cmd = [
                sys.executable, str(pa_main_path),
                str(input_file), 
                str(pa_output),
                "--embedder", "bge",
                "--max-workers", "2"
            ]
            
            # 최적화 패치 적용
            try:
                from xml_optimization_patch import build_optimized_pa_cmd
                cmd = build_optimized_pa_cmd(xml_pair, str(input_file), str(pa_output), base_cmd)
            except ImportError:
                cmd = base_cmd  # 패치가 없으면 기본 명령어 사용
            
            print(f"PA 실행 명령어: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if pa_output.exists():
                    return {
                        'status': 'success',
                        'output_file': str(pa_output),
                        'stdout': result.stdout
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': f"PA 출력 파일이 생성되지 않았습니다: {pa_output}",
                        'stdout': result.stdout
                    }
            else:
                return {
                    'status': 'failed',
                    'error': f"PA 실행 실패 (exit code: {result.returncode})\n{result.stderr}",
                    'stdout': result.stdout
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': f"PA 프로세스 예외: {str(e)}"
            }
    
    def _run_sa_process(self, input_file: str, result_folder: Path, xml_pair: XMLPair = None) -> Dict[str, any]:
        """SA 프로세스 실행"""
        try:
            sa_dir = result_folder / "sa_results"
            sa_dir.mkdir(exist_ok=True)
            sa_output = sa_dir / "sa_output.xlsx"
            
            # SA 모듈 실행 - 절대 경로 사용
            import subprocess
            import sys
            
            # 현재 작업 디렉토리 확인 및 SA 모듈 경로 설정
            current_dir = Path.cwd()
            sa_main_path = current_dir / "sa" / "main.py"
            
            # SA main.py 파일 존재 확인
            if not sa_main_path.exists():
                return {
                    'status': 'failed',
                    'error': f"SA main.py를 찾을 수 없습니다: {sa_main_path}",
                    'stdout': f"현재 디렉토리: {current_dir}"
                }
            
            # 입력 파일 존재 확인
            if not Path(input_file).exists():
                return {
                    'status': 'failed',
                    'error': f"입력 파일을 찾을 수 없습니다: {input_file}",
                    'stdout': ''
                }
            
            # 🎯 SA 실행 - 원래 방식으로 복원
            base_cmd = [
                sys.executable, str(sa_main_path),
                str(input_file),
                str(sa_output),
                "--embedder", "bge",
                "--max-workers", "2"
            ]
            
            # 최적화 패치 적용
            try:
                from xml_optimization_patch import build_optimized_sa_cmd
                cmd = build_optimized_sa_cmd(xml_pair, str(input_file), str(sa_output), base_cmd)
            except ImportError:
                cmd = base_cmd  # 패치가 없으면 기본 명령어 사용
            
            print(f"SA 실행 명령어: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if sa_output.exists():
                    return {
                        'status': 'success', 
                        'output_file': str(sa_output),
                        'stdout': result.stdout
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': f"SA 출력 파일이 생성되지 않았습니다: {sa_output}",
                        'stdout': result.stdout
                    }
            else:
                return {
                    'status': 'failed',
                    'error': f"SA 실행 실패 (exit code: {result.returncode})\n{result.stderr}",
                    'stdout': result.stdout
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': f"SA 프로세스 예외: {str(e)}"
            }
    
    def _compare_sentence_alignment(self, orig_xml: str, trans_xml: str, pa_output: str, result_folder: Path) -> Dict[str, any]:
        """문장병렬 비교 (XML 정답 vs PA 결과)"""
        try:
            sentence_dir = result_folder / "sentence"
            sentence_dir.mkdir(exist_ok=True)
            
            # XML에서 정답 문장병렬 추출
            xml_sentences = XMLProcessor.extract_sentence_data(orig_xml, trans_xml)
            xml_sentence_file = sentence_dir / "sentence_truth.xlsx"
            xml_sentences.to_excel(xml_sentence_file, index=False)
            
            # PA 결과에서 문장 추출 (PA는 문단을 문장으로 분할함)
            if pa_output and Path(pa_output).exists():
                pa_data = pd.read_excel(pa_output, engine='openpyxl')
                # PA 결과를 문장 형식으로 변환
                sentence_from_pa = pa_data.copy()
                sentence_from_pa['문장식별자'] = range(1, len(sentence_from_pa) + 1)
                
                pa_sentence_file = sentence_dir / "sentence_from_pa.xlsx"
                sentence_from_pa.to_excel(pa_sentence_file, index=False)
                
                # SA 입력용 파일 (XML 정답 사용)
                sa_input_file = xml_sentence_file
            else:
                pa_sentence_file = None
                sa_input_file = xml_sentence_file
            
            return {
                'status': 'success',
                'xml_truth_file': str(xml_sentence_file),
                'pa_result_file': str(pa_sentence_file) if pa_sentence_file else None,
                'sa_input_file': str(sa_input_file),
                'xml_sentence_count': len(xml_sentences)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _analyze_accuracy(self, orig_xml: str, trans_xml: str, sa_output: str, result_folder: Path, pa_output_file: str = None) -> Dict[str, any]:
        """고도화된 정확도 분석 - 전역 무결성 체크와 XML 레벨 분석 통합"""
        try:
            accuracy_dir = result_folder / "accuracy"
            accuracy_dir.mkdir(exist_ok=True)
            
            print("🔍 정확도 분석 시작 (전역 무결성 + XML 레벨 분석)...")
            
            # XML에서 정답 구병렬 추출
            xml_phrases = XMLProcessor.extract_phrase_data(orig_xml, trans_xml)
            xml_phrase_file = accuracy_dir / "phrase_truth.xlsx"
            xml_phrases.to_excel(xml_phrase_file, index=False)
            
            print(f"📊 XML 정답 구: {len(xml_phrases)}개")
            
            # 🔧 문장 단위 Truth 파일 생성 (PA 비교용)
            sentence_dir = result_folder / "sentence"
            sentence_dir.mkdir(exist_ok=True)
            xml_sentence_file = sentence_dir / "sentence_truth.xlsx"
            
            # 문장 단위 파일이 없으면 생성
            if not xml_sentence_file.exists():
                xml_sentences = XMLProcessor.extract_sentence_data(orig_xml, trans_xml)
                xml_sentences.to_excel(xml_sentence_file, index=False)
            
            # 전역 무결성 체크 실행
            print("🔍 전역 무결성 체크 시작...")
            try:
                global_integrity = self._compute_global_text_integrity(
                    xml_phrase_file, sa_output, orig_xml, trans_xml
                )
            except Exception as e:
                print(f"⚠️ 전역 무결성 체크 실패: {e}")
                global_integrity = {
                    'error': str(e),
                    'global_source_text_similarity': 0.0,
                    'global_target_text_similarity': 0.0,
                    'global_source_len_gt': 0,
                    'global_target_len_gt': 0,
                    'global_source_len_pred': 0,
                    'global_target_len_pred': 0,
                    'global_source_ops_replace': 0,
                    'global_target_ops_replace': 0
                }
            
            # XMLLevelSimilarityCalculator에 전역 무결성 데이터 설정
            try:
                from .xml_level_similarity import XMLLevelSimilarityCalculator
                # 임시 인스턴스를 만들어서 global_integrity 설정
                temp_calculator = XMLLevelSimilarityCalculator(use_embeddings=False)
                temp_calculator.global_integrity = {
                    'original_similarity': global_integrity.get('global_source_text_similarity', 0),
                    'translation_similarity': global_integrity.get('global_target_text_similarity', 0),
                    'original_length_xml': global_integrity.get('global_source_len_gt', 0),
                    'translation_length_xml': global_integrity.get('global_target_len_gt', 0),
                    'original_length_predicted': global_integrity.get('global_source_len_pred', 0),
                    'translation_length_predicted': global_integrity.get('global_target_len_pred', 0),
                    'original_differences': {
                        'has_content_differences': global_integrity.get('global_source_ops_replace', 0) > 0,
                        'non_space_differences': []
                    },
                    'translation_differences': {
                        'has_content_differences': global_integrity.get('global_target_ops_replace', 0) > 0, 
                        'non_space_differences': []
                    }
                }
                print("✅ XMLLevelSimilarityCalculator에 전역 무결성 데이터 설정 완료")
            except Exception as e:
                print(f"⚠️ XMLLevelSimilarityCalculator 설정 실패: {e}")
            
            # 기본 무결성 체크
            basic_accuracy_score = 0.0
            comparison_results = {}
            
            if sa_output and Path(sa_output).exists():
                sa_data = pd.read_excel(sa_output, engine='openpyxl')
                
                # 기본 정확도 계산 (길이 기반)
                xml_count = len(xml_phrases)
                sa_count = len(sa_data)
                
                if xml_count > 0:
                    basic_accuracy_score = min(sa_count / xml_count, 1.0)
                
                comparison_results = {
                    'xml_phrase_count': xml_count,
                    'sa_result_count': sa_count,
                    'length_based_accuracy': basic_accuracy_score,
                    'global_integrity': global_integrity
                }
            
            # XML 레벨 유사도 분석 실행
            try:
                xml_level_analysis = self._run_xml_level_similarity_analysis(
                    orig_xml, pa_output_file, sa_output, accuracy_dir
                )
            except Exception as e:
                print(f"⚠️ XML 레벨 분석 실패: {e}")
                xml_level_analysis = {'error': str(e)}
            
            # 차이점 로그 파일 저장
            try:
                self._save_diff_logs(accuracy_dir)
            except Exception as e:
                print(f"⚠️ 차이점 로그 저장 실패: {e}")
                # 기본 로그 파일 생성
                try:
                    diff_log_file = accuracy_dir / "text_differences_detail.txt"
                    with open(diff_log_file, 'w', encoding='utf-8') as f:
                        f.write("차이점 로그 생성 실패\n")
                        f.write(f"오류: {e}\n")
                        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                except Exception as save_e:
                    print(f"⚠️ 기본 로그 파일 생성도 실패: {save_e}")
            
            # 최종 정확도 점수 계산 (PA/SA 매칭 성능 중심)
            final_accuracy_score = basic_accuracy_score
            
            if xml_level_analysis:
                # PA 매칭 성능 점수 추출
                pa_score = 0.0
                sa_score = 0.0
                
                if 'pa_analysis' in xml_level_analysis:
                    pa_data = xml_level_analysis['pa_analysis']
                    # PA: F1 점수를 주요 지표로 사용 (60%) + 평균 유사도 (40%)
                    pa_f1 = pa_data.get('f1_score', 0.0)
                    pa_avg_sim = pa_data.get('avg_similarity', 0.0)
                    pa_score = (pa_f1 * 0.6 + pa_avg_sim * 0.4)
                    
                if 'sa_analysis' in xml_level_analysis:
                    sa_data = xml_level_analysis['sa_analysis']
                    # SA: F1 점수를 주요 지표로 사용 (60%) + 평균 유사도 (40%)
                    sa_f1 = sa_data.get('f1_score', 0.0)
                    sa_avg_sim = sa_data.get('avg_similarity', 0.0)
                    sa_score = (sa_f1 * 0.6 + sa_avg_sim * 0.4)
                
                # 전역 무결성 점수 (낮은 가중치 적용)
                source_integrity = global_integrity.get('global_source_text_similarity', 0.0)
                target_integrity = global_integrity.get('global_target_text_similarity', 0.0)
                integrity_score = (source_integrity + target_integrity) / 2
                
                # 최종 점수 계산: PA(45%) + SA(45%) + 무결성(10%)
                if pa_score > 0 and sa_score > 0:
                    final_accuracy_score = (pa_score * 0.45 + sa_score * 0.45 + integrity_score * 0.1)
                    print(f"🎯 최종 정확도: PA({pa_score:.3f}) + SA({sa_score:.3f}) + 무결성({integrity_score:.3f}) = {final_accuracy_score:.3f}")
                elif pa_score > 0:
                    # SA가 없으면 PA 중심으로 계산
                    final_accuracy_score = (pa_score * 0.8 + integrity_score * 0.2)
                    print(f"🎯 최종 정확도: PA({pa_score:.3f}) + 무결성({integrity_score:.3f}) = {final_accuracy_score:.3f}")
                elif sa_score > 0:
                    # PA가 없으면 SA 중심으로 계산
                    final_accuracy_score = (sa_score * 0.8 + integrity_score * 0.2)
                    print(f"🎯 최종 정확도: SA({sa_score:.3f}) + 무결성({integrity_score:.3f}) = {final_accuracy_score:.3f}")
                else:
                    # PA/SA 둘 다 없으면 무결성만
                    final_accuracy_score = integrity_score
                    print(f"🎯 최종 정확도: 무결성만({integrity_score:.3f}) = {final_accuracy_score:.3f}")
            
            # 간소화된 분석 결과
            accuracy_report = {
                'accuracy_score': final_accuracy_score,
                'comparison_results': comparison_results,
                'global_integrity': global_integrity,
                'xml_level_analysis': xml_level_analysis,
                'xml_phrase_file': str(xml_phrase_file),
                'sa_result_file': sa_output,
                'timestamp': datetime.now().isoformat()
            }
            
            # 메인 리포트 파일 저장
            report_file = accuracy_dir / "accuracy_report.json"
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(accuracy_report, f, indent=2, ensure_ascii=False, default=str)
                print(f"✅ accuracy_report.json 저장 완료")
            except Exception as e:
                print(f"⚠️ JSON 보고서 저장 실패: {e}")
                # 최소한의 JSON 보고서라도 생성
                try:
                    minimal_report = {
                        'accuracy_score': final_accuracy_score,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'xml_phrase_count': len(xml_phrases) if 'xml_phrases' in locals() else 0
                    }
                    with open(report_file, 'w', encoding='utf-8') as f:
                        json.dump(minimal_report, f, indent=2, ensure_ascii=False)
                    print(f"✅ 최소 JSON 보고서 저장 완료")
                except Exception as save_e:
                    print(f"❌ 최소 JSON 보고서 저장도 실패: {save_e}")
            
            print(f"📊 정확도 분석 완료: {final_accuracy_score:.1%}")
            
            return {
                'status': 'success',
                'accuracy_score': final_accuracy_score,
                'report_file': str(report_file),
                'xml_phrase_count': len(xml_phrases),
                'comparison_results': comparison_results,
                'global_integrity': global_integrity,
                'xml_level_analysis': xml_level_analysis
            }
            
        except Exception as e:
            error_msg = f"정확도 분석 오류: {e}"
            print(f"❌ {error_msg}")
            return {
                'status': 'failed',
                'error': error_msg,
                'accuracy_score': 0.0,
                'global_integrity': {'error': str(e)},
                'xml_level_analysis': {'error': str(e)}
            }
    
    
    def _record_stage_result(self, pair_id: str, stage: str, status: str, processing_time: float, accuracy_score: float = None):
        """단계 결과 데이터베이스 기록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO pipeline_results 
                    (pair_id, stage, status, processing_time, accuracy_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (pair_id, stage, status, processing_time, accuracy_score))
                conn.commit()
        except Exception as e:
            logger.error(f"단계 결과 기록 실패: {e}")
    
    def _generate_pipeline_summary(self, results: Dict, total_time: float) -> Dict:
        """파이프라인 요약 생성"""
        summary = {
            'pair_id': results['pair_id'],
            'total_processing_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'success' if not results['errors'] else 'partial_failure',
            'stages': {},
            'errors': results['errors']
        }
        
        for stage_name, stage_data in results['stages'].items():
            summary['stages'][stage_name] = {
                'status': stage_data.get('status', 'unknown'),
                'processing_time': stage_data.get('time', 0),
                'accuracy_score': stage_data.get('accuracy_score'),
                'output_files': [
                    f for f in [
                        stage_data.get('output_file'),
                        stage_data.get('report_file')
                    ] if f
                ]
            }
        
        return summary
    
    def _generate_text_summary(self, summary: Dict) -> str:
        """텍스트 요약 생성"""
        lines = []
        lines.append(f"XML 파이프라인 처리 결과 요약")
        lines.append(f"=" * 50)
        lines.append(f"쌍 ID: {summary['pair_id']}")
        lines.append(f"처리 시간: {summary['timestamp']}")
        lines.append(f"총 소요 시간: {summary['total_processing_time']:.1f}초")
        lines.append(f"전체 상태: {summary['overall_status']}")
        lines.append("")
        
        lines.append("단계별 결과:")
        lines.append("-" * 30)
        
        for stage, data in summary['stages'].items():
            lines.append(f"{stage}:")
            lines.append(f"  상태: {data['status']}")
            lines.append(f"  처리시간: {data['processing_time']:.1f}초")
            if data.get('accuracy_score'):
                lines.append(f"  정확도: {data['accuracy_score']:.1%}")
            lines.append("")
        
        if summary['errors']:
            lines.append("오류 목록:")
            lines.append("-" * 20)
            for error in summary['errors']:
                lines.append(f"- {error}")
        
        return "\n".join(lines)
    
    def get_recent_results(self, limit: int = 10) -> List[Dict]:
        """최근 처리 결과 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pair_id, name, status, created_at, result_folder
                    FROM xml_pairs
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'pair_id': row[0],
                        'name': row[1],
                        'status': row[2],
                        'created_at': row[3],
                        'result_folder': row[4]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"최근 결과 조회 실패: {e}")
            return []
    
    def get_pair_details(self, pair_id: str) -> Optional[Dict]:
        """특정 쌍의 상세 결과 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 쌍 기본 정보
                cursor = conn.execute("""
                    SELECT * FROM xml_pairs WHERE pair_id = ?
                """, (pair_id,))
                
                pair_row = cursor.fetchone()
                if not pair_row:
                    return None
                
                # 단계별 결과
                cursor = conn.execute("""
                    SELECT stage, status, processing_time, accuracy_score, timestamp
                    FROM pipeline_results
                    WHERE pair_id = ?
                    ORDER BY timestamp
                """, (pair_id,))
                
                stages = []
                for row in cursor.fetchall():
                    stages.append({
                        'stage': row[0],
                        'status': row[1],
                        'processing_time': row[2],
                        'accuracy_score': row[3],
                        'timestamp': row[4]
                    })
                
                return {
                    'pair_info': {
                        'pair_id': pair_row[0],
                        'name': pair_row[1],
                        'original_file': pair_row[2],
                        'translation_file': pair_row[3],
                        'result_folder': pair_row[4],
                        'created_at': pair_row[5],
                        'status': pair_row[6]
                    },
                    'stages': stages
                }
                
        except Exception as e:
            logger.error(f"쌍 상세 조회 실패: {e}")
            return None

    def _log_non_whitespace_diff_summary(self, text_type: str, diff_result: Dict) -> None:
        """공백 외 차이점 요약 로깅"""
        non_ws_count = diff_result.get('non_whitespace_diffs_count', 0)
        ws_count = diff_result.get('whitespace_only_diffs_count', 0)
        
        if non_ws_count > 0:
            print(f"⚠️  {text_type} 공백 외 차이점: {non_ws_count}개")
            non_ws_diffs = diff_result.get('non_whitespace_diffs', [])
            for i, diff in enumerate(non_ws_diffs[:3]):  # 상위 3개만 표시
                if diff['type'] == 'INSERT':
                    print(f"   • [삽입] {diff['content']}")
                elif diff['type'] == 'DELETE':
                    print(f"   • [삭제] {diff['content']}")
                elif diff['type'] == 'REPLACE':
                    print(f"   • [변경] {diff['original']} → {diff['replaced']}")
            if len(non_ws_diffs) > 3:
                print(f"   ... 외 {len(non_ws_diffs)-3}개 더")
        else:
            print(f"✅ {text_type}: 공백 외 차이점 없음")
        
        if ws_count > 0:
            print(f"   📝 공백 차이점: {ws_count}개")

    def _save_diff_logs(self, accuracy_dir: Path) -> None:
        """차이점 로그를 별도 파일로 저장"""
        if hasattr(self, '_diff_logs') and self._diff_logs:
            try:
                diff_log_file = accuracy_dir / "text_differences_detail.txt"
                with open(diff_log_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("텍스트 차이점 상세 분석 로그\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for i, log_content in enumerate(self._diff_logs, 1):
                        f.write(f"[분석 {i}]\n")
                        f.write(log_content)
                        f.write("\n\n" + "─" * 60 + "\n\n")
                
                print(f"📋 차이점 상세 로그 저장: {diff_log_file}")
                # 로그 초기화
                self._diff_logs = []
            except Exception as e:
                print(f"⚠️ 차이점 로그 저장 실패: {e}")

    def _explain_matching_accuracy_metrics(self, analysis_result: Dict) -> None:
        """매칭 정확도 지표들 설명"""
        print("\n📊 매칭 정확도 지표 설명:")
        print("  🎯 전체 정확도 = PA 점수(45%) + SA 점수(45%) + 무결성(10%)")
        print("  📈 PA/SA 점수 = F1 점수(60%) + 평균 유사도(40%)")
        print("  ✅ F1 점수 = 2 × (정밀도 × 재현율) / (정밀도 + 재현율)")
        print("  📏 평균 유사도 = 매칭된 문장/구 간 의미적 유사도 평균")
        print("  🔍 무결성 = XML 원본과 처리 결과 간 텍스트 일치도")


if __name__ == "__main__":
    # 테스트용 코드
    processor = XMLPipelineProcessor()
    
    # 샘플 XML 파일로 테스트
    orig_xml = "sources/jti_4c0231-[역주]당송팔대가문초증공1_원문_x-C2018.xml"
    trans_xml = "sources/jti_4c0231-[역주]당송팔대가문초증공1_번역문_x-C2018.xml"
    
    if Path(orig_xml).exists() and Path(trans_xml).exists():
        print("🧪 테스트 실행 중...")
        # XMLPair 객체 생성
        xml_pair = XMLPair(
            pair_id="test_001",
            original_path=orig_xml,
            translation_path=trans_xml
        )
        result = processor.process_xml_pair_pipeline(xml_pair)
        print("✅ 테스트 완료!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("❌ 테스트 파일을 찾을 수 없습니다.")


class XMLPipelineProcessorHelper:
    """XMLPipelineProcessor를 위한 헬퍼 메서드들"""
    
    @staticmethod
    def log_non_whitespace_diff_summary(text_type: str, diff_result: Dict) -> None:
        """공백 외 차이점 요약 로깅"""
        non_ws_count = diff_result.get('non_whitespace_diffs_count', 0)
        ws_count = diff_result.get('whitespace_only_diffs_count', 0)
        
        if non_ws_count > 0:
            print(f"⚠️  {text_type} 공백 외 차이점: {non_ws_count}개")
            non_ws_diffs = diff_result.get('non_whitespace_diffs', [])
            for i, diff in enumerate(non_ws_diffs[:3]):  # 상위 3개만 표시
                if diff['type'] == 'INSERT':
                    print(f"   • [삽입] {diff['content']}")
                elif diff['type'] == 'DELETE':
                    print(f"   • [삭제] {diff['content']}")
                elif diff['type'] == 'REPLACE':
                    print(f"   • [변경] {diff['original']} → {diff['replaced']}")
            if len(non_ws_diffs) > 3:
                print(f"   ... 외 {len(non_ws_diffs)-3}개 더")
        else:
            print(f"✅ {text_type}: 공백 외 차이점 없음")
        
        if ws_count > 0:
            print(f"   📝 공백 차이점: {ws_count}개")

    def _explain_matching_accuracy_metrics(self, analysis_result: Dict) -> None:
        """매칭 정확도 지표들 설명"""
        if 'xml_level_analysis' not in analysis_result:
            return
        
        xml_analysis = analysis_result['xml_level_analysis']
        
        print(f"\n📊 매칭 정확도 지표 설명:")
        print(f"=" * 50)
        
        for level_key in ['pa_analysis', 'sa_analysis']:
            if level_key not in xml_analysis:
                continue
                
            level_data = xml_analysis[level_key]
            level_name = "문단병렬(PA)" if level_key == 'pa_analysis' else "문장병렬(SA)"
            
            print(f"\n🔍 {level_name} 분석:")
            print(f"   XML 단위 수: {level_data.get('xml_unit_count', 0):,}개")
            print(f"   결과 행 수: {level_data.get('result_row_count', 0):,}개")
            print(f"   매칭된 쌍: {level_data.get('matched_pairs', 0):,}개")
            
            precision = level_data.get('precision', 0) * 100
            recall = level_data.get('recall', 0) * 100
            f1 = level_data.get('f1_score', 0) * 100
            accuracy = level_data.get('accuracy', 0) * 100
            avg_sim = level_data.get('avg_similarity', 0) * 100
            
            print(f"\n   📈 정확도 지표:")
            print(f"   • Precision (정밀도): {precision:.1f}%")
            print(f"     ➜ 결과에서 매칭된 것 중 실제로 올바른 매칭의 비율")
            print(f"     ➜ 높을수록 '잘못 매칭된 것'이 적음")
            
            print(f"\n   • Recall (재현율): {recall:.1f}%") 
            print(f"     ➜ 실제 정답 중에서 시스템이 찾아낸 것의 비율")
            print(f"     ➜ 높을수록 '놓친 것'이 적음")
            
            print(f"\n   • F1 Score: {f1:.1f}%")
            print(f"     ➜ Precision과 Recall의 조화평균 (균형 지표)")
            print(f"     ➜ 전반적인 매칭 성능을 나타냄")
            
            print(f"\n   • Accuracy (완전일치율): {accuracy:.1f}%")
            print(f"     ➜ 전체 중 완전히 정확한 매칭의 비율")
            print(f"     ➜ ⚠️ 원문↔번역문에서는 당연히 매우 낮음 (한자↔한글)")
            
            print(f"\n   • Average Similarity (평균 유사도): {avg_sim:.1f}%")
            print(f"     ➜ 매칭된 쌍들의 텍스트 유사도 평균")
            print(f"     ➜ 원↔번역 관계에서 40-60%가 정상 범위")
            
            # 성능 해석
            if f1 >= 90:
                grade = "🟢 우수"
            elif f1 >= 70:
                grade = "🟡 양호"
            elif f1 >= 50:
                grade = "🟠 보통"
            else:
                grade = "🔴 개선필요"
                
            print(f"\n   🎯 {level_name} 매칭 성능: {grade} (F1: {f1:.1f}%)")
            
            # 성능 해석 가이드
            if precision >= 85 and recall >= 85:
                print(f"   ✅ 매칭 시스템이 올바르게 작동하고 있습니다")
                if avg_sim < 60:
                    print(f"   💡 낮은 유사도는 원문↔번역문 특성상 정상입니다")
            elif precision < 70 or recall < 70:
                print(f"   ⚠️ 매칭 알고리즘 개선이 필요할 수 있습니다")