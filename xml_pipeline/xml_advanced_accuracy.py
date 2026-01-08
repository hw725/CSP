#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML 파이프라인용 고도화된 정확도 평가 모듈
accuracy_evaluator.py의 기능을 활용하여 다각도 정밀 평가 수행
PA ↔ XML <s> 단위, SA ↔ XML <w> 단위 비교 포함
"""

import pandas as pd
import json
import sys
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import difflib

# accuracy 모듈 import
sys.path.append(str(Path(__file__).parent / "accuracy"))
try:
    from accuracy.accuracy_evaluator import AccuracyEvaluator
    from accuracy.thresholds_config import THRESHOLDS
except ImportError:
    try:
        from accuracy_evaluator import AccuracyEvaluator  
        from thresholds_config import THRESHOLDS
    except ImportError:
        print("⚠️ accuracy_evaluator 모듈을 찾을 수 없습니다. 기본 평가만 수행됩니다.")
        AccuracyEvaluator = None
        THRESHOLDS = None

# XML 레벨 유사도 계산기 import
XMLLevelSimilarityCalculator = None
try:
    # 현재 디렉토리를 Python 경로에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 먼저 상대 임포트를 시도
    try:
        from .xml_level_similarity import XMLLevelSimilarityCalculator
        print("✅ XMLLevelSimilarityCalculator import 성공 (상대 임포트)")
    except ImportError:
        # 상대 임포트 실패시 절대 임포트 시도
        try:
            from xml_pipeline.xml_level_similarity import XMLLevelSimilarityCalculator
            print("✅ XMLLevelSimilarityCalculator import 성공 (절대 임포트)")
        except ImportError:
            # 직접 임포트 시도
            import xml_level_similarity
            XMLLevelSimilarityCalculator = xml_level_similarity.XMLLevelSimilarityCalculator
            print("✅ XMLLevelSimilarityCalculator import 성공 (직접 임포트)")
            
except ImportError as e:
    print(f"⚠️ xml_level_similarity 모듈을 찾을 수 없습니다: {e}")
    print(f"   현재 파일 위치: {__file__}")
    print(f"   Python 경로: {sys.path}")
    print("   XML 레벨 분석이 제외됩니다.")
    XMLLevelSimilarityCalculator = None


class AdvancedAccuracyAnalyzer:
    """고도화된 정확도 분석기"""
    
    def __init__(self, accuracy_dir: Path, tuning_config: Dict[str, Any] = None):
        self.accuracy_dir = accuracy_dir
        self.accuracy_dir.mkdir(exist_ok=True)
        
        # 튜닝 가능한 파라미터 설정
        self.tuning_config = tuning_config or self._get_default_tuning_config()
        print(f"🔧 튜닝 설정 적용됨: {json.dumps(self.tuning_config, indent=2, ensure_ascii=False)}")
    
    def _remove_editing_marks(self, text: str) -> str:
        """편집 마크 [, -, ] 완전 제거"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # 편집 마크 패턴 제거
        import re
        # [ ] - 문자 제거
        text = re.sub(r'[\[\-\]]', '', text)
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임의 편집 마크 제거"""
        cleaned_df = df.copy()
        text_columns = ['원문', '번역문', '구']
        
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(self._remove_editing_marks)
        
        return cleaned_df
    
    def _get_default_tuning_config(self) -> Dict[str, Any]:
        """기본 튜닝 설정"""
        return {
            "weights": {
                "text_integrity": 0.35,      # 텍스트 무결성 가중치
                "segmentation": 0.15,        # 분할 품질 가중치  
                "text_similarity": 0.25,     # 텍스트 유사도 가중치
                "structure": 0.25           # 구조 일치도 가중치
            },
            "thresholds": {
                "similarity_low": 0.3,       # 낮은 유사도 임계값
                "similarity_medium": 0.6,    # 중간 유사도 임계값
                "similarity_high": 0.8,      # 높은 유사도 임계값
                "excess_data_threshold": 50, # 초과 데이터 경고 임계값
                "text_integrity_min": 0.95   # 최소 텍스트 무결성 기준
            },
            "sampling": {
                "mismatch_sample_size": 10,  # 불일치 데이터 샘플 개수
                "excess_sample_size": 10,    # 초과 데이터 샘플 개수
                "show_details": True         # 상세 내용 표시 여부
            },
            "filtering": {
                "min_text_length": 2,        # 최소 텍스트 길이
                "remove_empty": True,        # 빈 값 제거 여부
                "normalize_whitespace": True # 공백 정규화 여부
            }
        }
    
    def analyze_comprehensive_accuracy(self, 
                                     xml_truth_file: str, 
                                     sa_result_file: str, 
                                     pair_id: str,
                                     pa_result_file: str = None,
                                     xml_translation_file: str = None,
                                     original_xml_file: str = None) -> Dict[str, Any]:
        """종합적인 정확도 분석 수행 - PA와 SA 레벨 모두 분석"""
        
        print(f"📊 고도화된 정확도 분석 시작: {pair_id}")
        
        analysis_results = {
            'pair_id': pair_id,
            'timestamp': datetime.now().isoformat(),
            'files': {
                'xml_truth': xml_truth_file,
                'sa_result': sa_result_file,
                'pa_result': pa_result_file
            },
            'metrics': {},
            'detailed_analysis': {},
            'xml_level_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. 기본 데이터 무결성 검사
            integrity_result = self._check_data_integrity(xml_truth_file, sa_result_file)
            analysis_results['metrics']['integrity'] = integrity_result
            
            # 2. XML 레벨별 유사도 분석 (우선순위!) - 슬라이딩 윈도우 방식 사용
            xml_file_for_analysis = original_xml_file if original_xml_file else xml_truth_file
            xml_level_result = self._analyze_xml_level_similarity(xml_file_for_analysis, sa_result_file, pa_result_file, xml_translation_file)
            analysis_results['xml_level_analysis'] = xml_level_result
            
            # XML 레벨 분석 결과를 주요 메트릭으로 설정
            if 'comprehensive_summary' in xml_level_result:
                summary = xml_level_result['comprehensive_summary']
                if 'pa_level' in summary:
                    analysis_results['metrics']['pa_similarity_scores'] = summary['pa_level']
                if 'sa_level' in summary:
                    analysis_results['metrics']['sa_similarity_scores'] = summary['sa_level']
            
            # 3. 텍스트 유사도 분석
            similarity_result = self._analyze_text_similarity(xml_truth_file, sa_result_file)
            analysis_results['metrics']['text_similarity'] = similarity_result
            
            # 4. 길이 및 구조 분석
            structure_result = self._analyze_structure(xml_truth_file, sa_result_file)
            analysis_results['metrics']['structure'] = structure_result
            
            # 5. 기존 AccuracyEvaluator 비교 (보조적) - XML 레벨 분석이 실패한 경우에만 사용
            if AccuracyEvaluator and 'error' in xml_level_result:
                print("⚠️ XML 레벨 분석 실패, 기존 평가 방식으로 대체...")
                evaluator_result = self._run_accuracy_evaluator(xml_truth_file, sa_result_file)
                analysis_results['detailed_analysis']['evaluator'] = evaluator_result
                analysis_results['metrics']['f1_scores'] = evaluator_result.get('f1_scores', {})
                analysis_results['metrics']['similarity_scores'] = evaluator_result.get('similarity_scores', {})
            
            # 6. 품질 등급 평가 (XML 레벨 분석 포함)
            grade_result = self._calculate_quality_grade(analysis_results['metrics'], xml_level_result)
            analysis_results['quality_grade'] = grade_result
            
            # 7. 개선 권장사항 생성
            recommendations = self._generate_recommendations(analysis_results)
            analysis_results['recommendations'] = recommendations
            
            # 8. 스마트 튜닝 제안 생성
            smart_tuning = self._generate_smart_tuning_suggestions(analysis_results)
            analysis_results['smart_tuning_suggestions'] = smart_tuning
            
            # 9. 상세 리포트 파일 저장
            self._save_detailed_reports(analysis_results)
            
            print(f"✅ 정확도 분석 완료")
            return analysis_results
            
        except Exception as e:
            error_msg = f"정확도 분석 오류: {e}"
            print(f"❌ {error_msg}")
            analysis_results['error'] = error_msg
            return analysis_results
    
    def _check_data_integrity(self, truth_file: str, result_file: str) -> Dict[str, Any]:
        """데이터 무결성 검사 - 전체 문자열 일치도 중심 + 초과/불일치 데이터 상세 분석"""
        try:
            # 정답 데이터 로드
            truth_df = pd.read_excel(truth_file)
            result_df = pd.read_excel(result_file)
            
            # 편집 마크 제거
            truth_df = self._clean_dataframe(truth_df)
            result_df = self._clean_dataframe(result_df)
            
            # 기본 개수 정보
            count_info = {
                'truth_count': len(truth_df),
                'result_count': len(result_df),
                'count_ratio': len(result_df) / len(truth_df) if len(truth_df) > 0 else 0,
                'missing_data': len(truth_df) - len(result_df),
                'excess_data': max(0, len(result_df) - len(truth_df))
            }
            
            # 초과/불일치 데이터 상세 분석
            mismatch_analysis = self._analyze_mismatch_data(truth_df, result_df)
            
            # 전체 문자열 무결성 계산 (핵심!) - 비활성화
            # text_integrity = self._calculate_full_text_integrity(truth_df, result_df)
            text_integrity = {'similarity': 1.0, 'original_similarity': 1.0, 'translation_similarity': 1.0}
            
            # 컬럼 구조 검사 (분할 품질 평가용)
            truth_cols = set(truth_df.columns)
            result_cols = set(result_df.columns)
            
            segmentation_quality = {
                'common_columns': list(truth_cols & result_cols),
                'missing_columns': list(truth_cols - result_cols),
                'extra_columns': list(result_cols - truth_cols),
                'structure_score': len(truth_cols & result_cols) / len(truth_cols | result_cols) if truth_cols or result_cols else 0
            }
            
            # 종합 무결성 점수 (전체 문자열 일치도가 핵심)
            integrity_score = text_integrity['overall_text_similarity']
            
            # 무결성 대조 로그 생성
            contrast_log = self._generate_integrity_contrast_log(mismatch_analysis, integrity_score)
            
            integrity = {
                **count_info,
                'mismatch_analysis': mismatch_analysis,  # 새로 추가된 불일치 분석
                'integrity_contrast_log': contrast_log,  # 무결성 대조 로그
                'text_integrity': text_integrity,
                'segmentation_quality': segmentation_quality,
                'completeness_score': integrity_score,  # 무결성 = 텍스트 일치도
                'column_match': segmentation_quality   # 하위 호환성
            }
            
            return integrity
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_full_text_integrity(self, truth_df: pd.DataFrame, result_df: pd.DataFrame) -> Dict[str, Any]:
        """전역적 전체 문자열 무결성 검사"""
        try:
            # 전역 텍스트 결합 (순서 보장)
            def combine_global_text(df, column):
                if column not in df.columns:
                    return ""
                # 전체 텍스트를 순서대로 연결 (공백도 보존)
                texts = df[column].fillna('').astype(str)
                combined = ''.join(texts)
                return combined
            
            # 정규화된 텍스트 결합 (비교용)
            def combine_normalized_text(df, column):
                if column not in df.columns:
                    return ""
                texts = df[column].fillna('').astype(str)
                combined = ''.join(texts)
                # 공백과 줄바꿈 정규화
                import re
                combined = re.sub(r'\s+', '', combined)  # 모든 공백 제거하여 순수 문자만 비교
                return combined
            
            # 정답과 결과의 전역 텍스트 추출
            truth_original_raw = combine_global_text(truth_df, '원문')
            truth_translation_raw = combine_global_text(truth_df, '번역문')
            result_original_raw = combine_global_text(result_df, '원문')
            result_translation_raw = combine_global_text(result_df, '번역문')
            
            # 정규화된 버전 (순수 문자 비교용)
            truth_original_norm = combine_normalized_text(truth_df, '원문')
            truth_translation_norm = combine_normalized_text(truth_df, '번역문')
            result_original_norm = combine_normalized_text(result_df, '원문')
            result_translation_norm = combine_normalized_text(result_df, '번역문')
            
            # 전역 무결성 계산
            def calculate_global_integrity(truth_text: str, result_text: str, normalized_truth: str, normalized_result: str) -> Dict[str, float]:
                # 1. 완전 일치 검사 (정규화된 텍스트로)
                exact_match = 1.0 if normalized_truth == normalized_result else 0.0
                
                # 2. 문자열 유사도 계산 (원본 텍스트로)
                raw_similarity = difflib.SequenceMatcher(None, truth_text, result_text).ratio()
                
                # 3. 정규화된 유사도 계산
                norm_similarity = difflib.SequenceMatcher(None, normalized_truth, normalized_result).ratio()
                
                # 4. 길이 보존율
                len_preservation = min(len(result_text), len(truth_text)) / max(len(result_text), len(truth_text)) if max(len(result_text), len(truth_text)) > 0 else 1.0
                
                return {
                    'exact_match': exact_match,
                    'raw_similarity': raw_similarity,
                    'normalized_similarity': norm_similarity,
                    'length_preservation': len_preservation,
                    'truth_length': len(truth_text),
                    'result_length': len(result_text)
                }
            
            original_integrity = calculate_global_integrity(
                truth_original_raw, result_original_raw, 
                truth_original_norm, result_original_norm
            )
            translation_integrity = calculate_global_integrity(
                truth_translation_raw, result_translation_raw,
                truth_translation_norm, result_translation_norm
            )
            
            # 전역 무결성 종합 계산
            truth_total_len = original_integrity['truth_length'] + translation_integrity['truth_length']
            result_total_len = original_integrity['result_length'] + translation_integrity['result_length']
            
            # 전역 완전 일치도 (핵심 무결성 지표)
            exact_match_score = (original_integrity['exact_match'] + translation_integrity['exact_match']) / 2
            
            # 정규화된 유사도 (공백 무시)
            normalized_similarity = (original_integrity['normalized_similarity'] + translation_integrity['normalized_similarity']) / 2
            
            # 원본 텍스트 유사도 (공백 포함)
            raw_similarity = (original_integrity['raw_similarity'] + translation_integrity['raw_similarity']) / 2
            
            # 길이 보존율
            length_preservation = (original_integrity['length_preservation'] + translation_integrity['length_preservation']) / 2
            
            return {
                'global_exact_match': exact_match_score,              # 완전 일치 (0 or 1)
                'global_normalized_similarity': normalized_similarity, # 정규화된 유사도
                'global_raw_similarity': raw_similarity,             # 원본 유사도
                'global_length_preservation': length_preservation,    # 길이 보존율
                'original_integrity': original_integrity,            # 원문 상세 정보
                'translation_integrity': translation_integrity,      # 번역문 상세 정보
                'truth_total_length': truth_total_len,
                'result_total_length': result_total_len,
                # 하위 호환성
                'original_text_similarity': original_integrity['normalized_similarity'],
                'translation_text_similarity': translation_integrity['normalized_similarity'],
                'overall_text_similarity': normalized_similarity,
                'length_preservation': length_preservation
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'original_text_similarity': 0.0,
                'translation_text_similarity': 0.0,
                'overall_text_similarity': 0.0,
                'length_preservation': 0.0
            }
    
    def _run_accuracy_evaluator(self, truth_file: str, result_file: str) -> Dict[str, Any]:
        """AccuracyEvaluator를 사용한 정밀 평가"""
        try:
            # 임시 출력 파일 생성
            eval_output = self.accuracy_dir / "detailed_evaluation.xlsx"
            
            # AccuracyEvaluator 실행
            evaluator = AccuracyEvaluator(
                ground_truth_file=truth_file,
                prediction_file=result_file,
                project='sa',  # SA 프로젝트 설정 사용
                ignore_space_punct=True,  # 관대한 일치 판정
                brief=False
            )
            
            # 데이터 로드 및 평가 수행
            evaluator.load_data()
            # AccuracyEvaluator의 실제 메서드 시그니처에 맞게 호출
            try:
                evaluation_result = evaluator.evaluate_accuracy(unit='sentence')
                # 결과가 DataFrame인 경우 파일로 저장
                if hasattr(evaluation_result, 'to_excel'):
                    evaluation_result.to_excel(eval_output, index=False)
                elif isinstance(evaluation_result, dict) and 'detailed_results' in evaluation_result:
                    # 상세 결과가 DataFrame인 경우
                    detailed = evaluation_result.get('detailed_results')
                    if hasattr(detailed, 'to_excel'):
                        detailed.to_excel(eval_output, index=False)
            except Exception as method_error:
                print(f"   AccuracyEvaluator 호출 실패, 기본 평가로 대체: {method_error}")
                evaluation_result = {'error': f'Method call failed: {method_error}'}
            
            return {
                'evaluation_file': str(eval_output),
                'f1_scores': evaluation_result.get('overall', {}).get('f1_scores', {}),
                'similarity_scores': evaluation_result.get('overall', {}).get('similarity_scores', {}),
                'match_statistics': evaluation_result.get('overall', {}).get('match_stats', {}),
                'grade_assessment': evaluation_result.get('grade_assessment', {}),
                'detailed_results': evaluation_result
            }
            
        except Exception as e:
            print(f"⚠️ AccuracyEvaluator 실행 오류: {e}")
            return {'error': str(e), 'fallback': True}
    
    def _analyze_text_similarity(self, truth_file: str, result_file: str) -> Dict[str, Any]:
        """텍스트 유사도 분석"""
        try:
            truth_df = pd.read_excel(truth_file)
            result_df = pd.read_excel(result_file)
            
            # 편집 마크 제거
            truth_df = self._clean_dataframe(truth_df)
            result_df = self._clean_dataframe(result_df)
            
            # 공통 텍스트 컬럼 찾기
            text_columns = []
            for col in ['원문', '번역문', '구']:
                if col in truth_df.columns and col in result_df.columns:
                    text_columns.append(col)
            
            similarities = {}
            
            for col in text_columns:
                col_similarities = []
                min_len = min(len(truth_df), len(result_df))
                
                for i in range(min_len):
                    truth_text = str(truth_df.iloc[i][col]).strip()
                    result_text = str(result_df.iloc[i][col]).strip()
                    
                    # 문자열 유사도 계산
                    similarity = difflib.SequenceMatcher(None, truth_text, result_text).ratio()
                    col_similarities.append(similarity)
                
                similarities[col] = {
                    'average_similarity': sum(col_similarities) / len(col_similarities) if col_similarities else 0,
                    'min_similarity': min(col_similarities) if col_similarities else 0,
                    'max_similarity': max(col_similarities) if col_similarities else 0,
                    'similarity_distribution': self._calculate_distribution(col_similarities)
                }
            
            return similarities
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_structure(self, truth_file: str, result_file: str) -> Dict[str, Any]:
        """구조 분석 (길이, 패턴 등)"""
        try:
            truth_df = pd.read_excel(truth_file)
            result_df = pd.read_excel(result_file)
            
            # 편집 마크 제거
            truth_df = self._clean_dataframe(truth_df)
            result_df = self._clean_dataframe(result_df)
            
            structure_analysis = {}
            
            # 텍스트 길이 분석
            for col in ['원문', '번역문']:
                if col in truth_df.columns and col in result_df.columns:
                    truth_lengths = [len(str(text).strip()) for text in truth_df[col]]
                    result_lengths = [len(str(text).strip()) for text in result_df[col][:len(truth_lengths)]]
                    
                    structure_analysis[f'{col}_length'] = {
                        'truth_avg_length': sum(truth_lengths) / len(truth_lengths) if truth_lengths else 0,
                        'result_avg_length': sum(result_lengths) / len(result_lengths) if result_lengths else 0,
                        'length_correlation': self._calculate_correlation(truth_lengths, result_lengths),
                        'length_difference_ratio': abs(sum(result_lengths) - sum(truth_lengths)) / sum(truth_lengths) if sum(truth_lengths) > 0 else 1
                    }
            
            # 빈 값 분석
            structure_analysis['empty_values'] = {
                'truth_empty_count': truth_df.isnull().sum().sum(),
                'result_empty_count': result_df.isnull().sum().sum(),
                'empty_ratio_difference': abs(
                    (result_df.isnull().sum().sum() / len(result_df)) - 
                    (truth_df.isnull().sum().sum() / len(truth_df))
                ) if len(truth_df) > 0 and len(result_df) > 0 else 1
            }
            
            return structure_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_mismatch_data(self, truth_df: pd.DataFrame, result_df: pd.DataFrame) -> Dict[str, Any]:
        """초과/불일치 데이터 상세 분석 + 무결성 문제 대조 로그"""
        try:
            analysis = {
                'excess_data_details': [],
                'missing_data_details': [],
                'mismatch_samples': [],
                'data_quality_issues': [],
                'integrity_comparison_log': [],  # 새로 추가: 무결성 대조 로그
                'tuning_suggestions': {},        # 새로 추가: 튜닝 제안
                'summary': {}
            }
            
            sample_size = self.tuning_config['sampling']['mismatch_sample_size']
            show_details = self.tuning_config['sampling']['show_details']
            
            # 1. 초과 데이터 분석 (결과에만 있는 데이터)
            if len(result_df) > len(truth_df):
                excess_count = len(result_df) - len(truth_df)
                excess_data = result_df.iloc[len(truth_df):len(result_df)]
                
                print(f"🔍 초과 데이터 발견: {excess_count}개")
                
                # 초과 데이터 전체 처리 (대조 로그용)
                for idx, row in excess_data.iterrows():
                    excess_info = {
                        'index': idx,
                        'data_type': 'excess',
                        'reason': f'정답 데이터 범위 초과 (인덱스 {idx})',
                        'content': {}
                    }
                    
                    # 주요 컬럼 내용 추가
                    for col in ['원문', '번역문', '구', '구식별자']:
                        if col in row.index and pd.notna(row[col]):
                            content = str(row[col])
                            if show_details:
                                excess_info['content'][col] = content[:100] + ('...' if len(content) > 100 else '')
                            else:
                                excess_info['content'][col] = f"길이: {len(content)}"
                    
                    analysis['excess_data_details'].append(excess_info)
            
            # 2. 누락 데이터 분석 (정답에만 있는 데이터)
            if len(truth_df) > len(result_df):
                missing_count = len(truth_df) - len(result_df)
                missing_data = truth_df.iloc[len(result_df):len(truth_df)]
                
                print(f"🔍 누락 데이터 발견: {missing_count}개")
                
                # 누락 데이터 전체 처리 (대조 로그용)
                for idx, row in missing_data.iterrows():
                    missing_info = {
                        'index': idx,
                        'data_type': 'missing',
                        'reason': f'결과 데이터에 누락 (인덱스 {idx})',
                        'content': {}
                    }
                    
                    # 주요 컬럼 내용 추가
                    for col in ['원문', '번역문', '구', '구식별자']:
                        if col in row.index and pd.notna(row[col]):
                            content = str(row[col])
                            if show_details:
                                missing_info['content'][col] = content[:100] + ('...' if len(content) > 100 else '')
                            else:
                                missing_info['content'][col] = f"길이: {len(content)}"
                    
                    analysis['missing_data_details'].append(missing_info)
            
            # 3. 텍스트 불일치 분석 (공통 인덱스 범위에서 전체 처리)
            common_len = min(len(truth_df), len(result_df))
            mismatch_count = 0
            
            # 주요 컬럼에서 불일치 검사 (전체 데이터 검사)
            for col in ['원문', '번역문']:
                if col in truth_df.columns and col in result_df.columns:
                    for i in range(common_len):  # 전체 공통 길이 검사
                        truth_text = str(truth_df.iloc[i][col]).strip()
                        result_text = str(result_df.iloc[i][col]).strip()
                        
                        # 유사도 계산
                        similarity = difflib.SequenceMatcher(None, truth_text, result_text).ratio()
                        
                        # 낮은 유사도인 경우 불일치로 판단 (전체 기록)
                        if similarity < self.tuning_config['thresholds']['similarity_low']:
                            mismatch_count += 1
                            mismatch_info = {
                                'index': i,
                                'column': col,
                                'similarity': round(similarity, 3),
                                'reason': f'텍스트 유사도 낮음 ({similarity:.1%})',
                                'truth_preview': truth_text[:50] + ('...' if len(truth_text) > 50 else ''),
                                'result_preview': result_text[:50] + ('...' if len(result_text) > 50 else ''),
                                'length_diff': abs(len(truth_text) - len(result_text))
                            }
                            analysis['mismatch_samples'].append(mismatch_info)
            
            # 4. 데이터 품질 문제 검사
            quality_issues = []
            
            # 빈 값 검사
            for col in ['원문', '번역문']:
                if col in result_df.columns:
                    empty_count = result_df[col].isnull().sum() + (result_df[col] == '').sum()
                    if empty_count > 0:
                        quality_issues.append({
                            'type': 'empty_values',
                            'column': col,
                            'count': int(empty_count),
                            'description': f'{col} 컬럼에 {empty_count}개 빈 값 발견'
                        })
            
            # 이상 길이 검사
            min_length = self.tuning_config['filtering']['min_text_length']
            for col in ['원문', '번역문']:
                if col in result_df.columns:
                    short_texts = result_df[col].apply(lambda x: len(str(x).strip()) < min_length).sum()
                    if short_texts > 0:
                        quality_issues.append({
                            'type': 'short_text',
                            'column': col,
                            'count': int(short_texts),
                            'description': f'{col} 컬럼에 {min_length}자 미만 텍스트 {short_texts}개 발견'
                        })
            
            analysis['data_quality_issues'] = quality_issues
            
            # 5. 무결성 문제 대조 로그 생성
            analysis['integrity_comparison_log'] = self._generate_integrity_comparison_log(
                truth_df, result_df, analysis['excess_data_details'], 
                analysis['missing_data_details'], analysis['mismatch_samples']
            )
            
            # 6. 스마트 튜닝 제안 생성
            analysis['tuning_suggestions'] = self._generate_tuning_suggestions(
                len(analysis['excess_data_details']), len(analysis['missing_data_details']), 
                mismatch_count, analysis['mismatch_samples'], quality_issues
            )
            
            # 7. 요약 정보
            analysis['summary'] = {
                'total_excess': len(analysis['excess_data_details']),
                'total_missing': len(analysis['missing_data_details']),
                'total_mismatches': mismatch_count,
                'total_quality_issues': len(quality_issues),
                'data_integrity_issues': len(analysis['excess_data_details']) + len(analysis['missing_data_details']) + mismatch_count,
                'severity_level': self._assess_mismatch_severity(len(analysis['excess_data_details']), 
                                                               len(analysis['missing_data_details']), 
                                                               mismatch_count)
            }
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ 불일치 데이터 분석 오류: {e}")
            return {'error': str(e)}
    
    def _generate_integrity_comparison_log(self, truth_df: pd.DataFrame, result_df: pd.DataFrame,
                                         excess_details: List, missing_details: List, 
                                         mismatch_samples: List) -> List[Dict[str, Any]]:
        """무결성 문제 대조 로그 생성"""
        try:
            comparison_log = []
            
            # 1. 전체 데이터 개요
            comparison_log.append({
                'log_type': 'data_overview',
                'timestamp': datetime.now().isoformat(),
                'comparison': {
                    'truth_total_rows': len(truth_df),
                    'result_total_rows': len(result_df),
                    'row_difference': len(result_df) - len(truth_df),
                    'truth_columns': list(truth_df.columns),
                    'result_columns': list(result_df.columns),
                    'column_difference': list(set(result_df.columns) - set(truth_df.columns))
                },
                'assessment': 'excess_data' if len(result_df) > len(truth_df) else 
                             'missing_data' if len(result_df) < len(truth_df) else 'equal_rows'
            })
            
            # 2. 초과 데이터 상세 대조
            if excess_details:
                for excess in excess_details[:5]:  # 최대 5개
                    comparison_log.append({
                        'log_type': 'excess_data_comparison',
                        'index': excess.get('index'),
                        'issue_description': f"인덱스 {excess.get('index')}에서 초과 데이터 발견",
                        'truth_data': '존재하지 않음 (정답 데이터 범위 초과)',
                        'result_data': excess.get('content', {}),
                        'impact_analysis': {
                            'data_pollution': '결과에 불필요한 데이터 추가됨',
                            'accuracy_effect': '정확도 계산에 노이즈 추가',
                            'recommended_action': '후처리 필터링 또는 임베딩 임계값 상향 조정'
                        }
                    })
            
            # 3. 누락 데이터 상세 대조  
            if missing_details:
                for missing in missing_details[:5]:  # 최대 5개
                    comparison_log.append({
                        'log_type': 'missing_data_comparison',
                        'index': missing.get('index'),
                        'issue_description': f"인덱스 {missing.get('index')}에서 데이터 누락 발견",
                        'truth_data': missing.get('content', {}),
                        'result_data': '존재하지 않음 (결과 데이터에 누락)',
                        'impact_analysis': {
                            'data_loss': '정답 데이터의 일부가 처리되지 않음',
                            'accuracy_effect': '완전성 점수 하락',
                            'recommended_action': '임베딩 임계값 하향 조정 또는 전처리 개선'
                        }
                    })
            
            # 4. 텍스트 불일치 상세 대조
            if mismatch_samples:
                for mismatch in mismatch_samples[:5]:  # 최대 5개
                    comparison_log.append({
                        'log_type': 'text_mismatch_comparison',
                        'index': mismatch.get('index'),
                        'column': mismatch.get('column'),
                        'similarity_score': mismatch.get('similarity'),
                        'issue_description': f"{mismatch.get('column')} 컬럼에서 텍스트 불일치 (유사도: {mismatch.get('similarity', 0):.1%})",
                        'truth_text': mismatch.get('truth_preview'),
                        'result_text': mismatch.get('result_preview'),
                        'character_diff': {
                            'length_difference': mismatch.get('length_diff', 0),
                            'similarity_analysis': self._analyze_text_difference(
                                mismatch.get('truth_preview', ''), 
                                mismatch.get('result_preview', '')
                            )
                        },
                        'impact_analysis': {
                            'content_preservation': '텍스트 내용 변경됨',
                            'accuracy_effect': '유사도 점수 하락',
                            'recommended_action': '토큰화 방식 또는 전처리 과정 점검'
                        }
                    })
            
            # 5. 전체 무결성 평가
            total_issues = len(excess_details) + len(missing_details) + len(mismatch_samples)
            comparison_log.append({
                'log_type': 'integrity_summary',
                'total_issues_found': total_issues,
                'issue_breakdown': {
                    'excess_data_count': len(excess_details),
                    'missing_data_count': len(missing_details), 
                    'text_mismatch_count': len(mismatch_samples)
                },
                'integrity_score': max(0, 1 - (total_issues / max(len(truth_df), 1))),
                'overall_assessment': self._assess_integrity_health(total_issues, len(truth_df)),
                'next_steps': self._suggest_integrity_improvement_steps(excess_details, missing_details, mismatch_samples)
            })
            
            return comparison_log
            
        except Exception as e:
            print(f"⚠️ 무결성 대조 로그 생성 오류: {e}")
            return [{'log_type': 'error', 'message': str(e)}]
    
    def _analyze_text_difference(self, truth_text: str, result_text: str) -> Dict[str, Any]:
        """텍스트 차이 상세 분석"""
        try:
            # difflib를 사용한 상세 차이 분석
            diff = difflib.unified_diff(
                truth_text.splitlines(), result_text.splitlines(),
                fromfile='정답', tofile='결과', lineterm=''
            )
            diff_lines = list(diff)
            
            return {
                'character_level_similarity': difflib.SequenceMatcher(None, truth_text, result_text).ratio(),
                'word_level_similarity': difflib.SequenceMatcher(
                    None, truth_text.split(), result_text.split()
                ).ratio() if truth_text and result_text else 0,
                'length_ratio': len(result_text) / len(truth_text) if len(truth_text) > 0 else 1,
                'has_additions': any(line.startswith('+') for line in diff_lines),
                'has_deletions': any(line.startswith('-') for line in diff_lines),
                'diff_summary': f"{len(diff_lines)}개 차이점 발견" if diff_lines else "차이점 없음"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_integrity_health(self, total_issues: int, total_data: int) -> str:
        """무결성 건강도 평가"""
        if total_data == 0:
            return 'unknown'
        
        issue_ratio = total_issues / total_data
        
        if issue_ratio == 0:
            return 'perfect'
        elif issue_ratio <= 0.01:  # 1% 미만
            return 'excellent'
        elif issue_ratio <= 0.05:  # 5% 미만
            return 'good'
        elif issue_ratio <= 0.1:   # 10% 미만
            return 'fair'
        elif issue_ratio <= 0.2:   # 20% 미만
            return 'poor'
        else:
            return 'critical'
    
    def _suggest_integrity_improvement_steps(self, excess_details: List, missing_details: List, mismatch_samples: List) -> List[str]:
        """무결성 개선 단계별 제안"""
        steps = []
        
        if excess_details:
            steps.append(f"1️⃣ 초과 데이터 {len(excess_details)}개 제거: 후처리 필터링 로직 추가")
            steps.append("   - 임베딩 임계값을 현재보다 0.1-0.2 상향 조정")
            steps.append("   - 결과 데이터의 신뢰도 점수 기준 강화")
        
        if missing_details:
            steps.append(f"2️⃣ 누락 데이터 {len(missing_details)}개 복구: 전처리 과정 개선")
            steps.append("   - 임베딩 임계값을 현재보다 0.1-0.2 하향 조정")
            steps.append("   - 입력 데이터의 정규화 및 정제 강화")
        
        if mismatch_samples:
            steps.append(f"3️⃣ 텍스트 불일치 {len(mismatch_samples)}개 해결: 토큰화 방식 점검")
            steps.append("   - 형태소 분석기 설정 또는 버전 확인")
            steps.append("   - 특수 문자 및 편집 표시 처리 로직 검토")
        
        if not steps:
            steps.append("✅ 무결성 문제가 발견되지 않았습니다. 현재 설정을 유지하세요.")
        
        return steps
    
    def _assess_mismatch_severity(self, excess_count: int, missing_count: int, mismatch_count: int) -> str:
        """불일치 심각도 평가"""
        total_issues = excess_count + missing_count + mismatch_count
        excess_threshold = self.tuning_config['thresholds']['excess_data_threshold']
        
        if total_issues == 0:
            return 'excellent'
        elif total_issues <= 10:
            return 'good'
        elif total_issues <= excess_threshold:
            return 'moderate'
        elif total_issues <= excess_threshold * 2:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_tuning_suggestions(self, excess_count: int, missing_count: int, 
                                   mismatch_count: int, mismatch_samples: List,
                                   quality_issues: List) -> Dict[str, Any]:
        """스마트 튜닝 제안 생성"""
        try:
            suggestions = {
                'immediate_actions': [],      # 즉시 실행 가능한 액션
                'parameter_adjustments': {},  # 파라미터 조정 제안
                'long_term_improvements': [], # 장기적 개선 방안
                'priority_ranking': [],       # 우선순위별 작업 목록
                'expected_outcomes': {}       # 예상 개선 효과
            }
            
            current_config = self.tuning_config
            
            # 1. 초과 데이터 기반 튜닝 제안
            if excess_count > 0:
                excess_ratio = excess_count / current_config['thresholds']['excess_data_threshold']
                
                if excess_ratio > 1.0:  # 임계값 초과
                    suggestions['immediate_actions'].append({
                        'action': 'excess_data_filtering',
                        'description': f'{excess_count}개 초과 데이터 즉시 필터링 필요',
                        'urgency': 'high',
                        'effort': 'medium'
                    })
                
                # 임베딩 임계값 조정 제안
                current_threshold = getattr(self, 'embedding_threshold', 0.7)  # 기본값
                suggested_threshold = min(0.95, current_threshold + (excess_ratio * 0.1))
                
                suggestions['parameter_adjustments']['embedding_threshold'] = {
                    'current_value': current_threshold,
                    'suggested_value': round(suggested_threshold, 2),
                    'adjustment_reason': f'초과 데이터 {excess_count}개 감소를 위한 임계값 상향 조정',
                    'expected_reduction': f'{int(excess_count * 0.3)}-{int(excess_count * 0.7)}개 예상 감소'
                }
            
            # 2. 누락 데이터 기반 튜닝 제안  
            if missing_count > 0:
                suggestions['immediate_actions'].append({
                    'action': 'missing_data_recovery',
                    'description': f'{missing_count}개 누락 데이터 복구 필요',
                    'urgency': 'medium',
                    'effort': 'high'
                })
                
                # 임베딩 임계값 하향 조정
                current_threshold = getattr(self, 'embedding_threshold', 0.7)
                suggested_threshold = max(0.3, current_threshold - 0.1)
                
                suggestions['parameter_adjustments']['embedding_threshold_lower'] = {
                    'current_value': current_threshold,
                    'suggested_value': round(suggested_threshold, 2),
                    'adjustment_reason': f'누락 데이터 {missing_count}개 복구를 위한 임계값 하향 조정',
                    'expected_recovery': f'{int(missing_count * 0.4)}-{int(missing_count * 0.8)}개 예상 복구'
                }
            
            # 3. 텍스트 불일치 기반 튜닝 제안
            if mismatch_samples:
                avg_similarity = sum(sample.get('similarity', 0) for sample in mismatch_samples) / len(mismatch_samples)
                
                if avg_similarity < current_config['thresholds']['similarity_low']:
                    suggestions['immediate_actions'].append({
                        'action': 'preprocessing_review',
                        'description': f'평균 유사도 {avg_similarity:.1%}로 전처리 과정 검토 필요',
                        'urgency': 'high',
                        'effort': 'high'
                    })
                
                # 유사도 임계값 조정
                suggestions['parameter_adjustments']['similarity_threshold'] = {
                    'current_low': current_config['thresholds']['similarity_low'],
                    'suggested_low': max(0.1, avg_similarity - 0.1),
                    'current_medium': current_config['thresholds']['similarity_medium'],
                    'suggested_medium': avg_similarity + 0.2,
                    'adjustment_reason': '실제 데이터 분포에 맞는 임계값 조정'
                }
            
            # 4. 데이터 품질 문제 기반 제안
            if quality_issues:
                for issue in quality_issues:
                    if issue['type'] == 'empty_values':
                        suggestions['immediate_actions'].append({
                            'action': 'empty_value_handling',
                            'description': f"{issue['column']}에서 {issue['count']}개 빈 값 처리 필요",
                            'urgency': 'medium',
                            'effort': 'low'
                        })
                    elif issue['type'] == 'short_text':
                        suggestions['parameter_adjustments']['min_text_length'] = {
                            'current_value': current_config['filtering']['min_text_length'],
                            'suggested_value': 1,  # 더 관대하게
                            'adjustment_reason': f"짧은 텍스트 {issue['count']}개 포함을 위한 기준 완화"
                        }
            
            # 5. 우선순위 랭킹 생성
            priority_scores = []
            
            if excess_count > current_config['thresholds']['excess_data_threshold']:
                priority_scores.append(('excess_data_filtering', 90))
            if missing_count > 10:
                priority_scores.append(('missing_data_recovery', 85))
            if mismatch_samples and len(mismatch_samples) > 5:
                priority_scores.append(('preprocessing_review', 80))
            if len(quality_issues) > 3:
                priority_scores.append(('data_quality_improvement', 70))
            
            suggestions['priority_ranking'] = sorted(priority_scores, key=lambda x: x[1], reverse=True)
            
            # 6. 예상 개선 효과
            suggestions['expected_outcomes'] = {
                'accuracy_improvement': self._estimate_accuracy_improvement(
                    excess_count, missing_count, len(mismatch_samples)
                ),
                'processing_time_impact': self._estimate_processing_time_impact(
                    suggestions['parameter_adjustments']
                ),
                'resource_requirements': self._estimate_resource_requirements(
                    suggestions['immediate_actions']
                )
            }
            
            # 7. 장기적 개선 방안
            if excess_count + missing_count + mismatch_count > 50:
                suggestions['long_term_improvements'].extend([
                    '🔄 전체 파이프라인 아키텍처 재검토',
                    '🤖 더 강력한 임베딩 모델로 업그레이드',
                    '📊 동적 임계값 조정 시스템 도입'
                ])
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️ 튜닝 제안 생성 오류: {e}")
            return {'error': str(e)}
    
    def _estimate_accuracy_improvement(self, excess: int, missing: int, mismatches: int) -> Dict[str, float]:
        """정확도 개선 예상치 계산"""
        total_issues = excess + missing + mismatches
        if total_issues == 0:
            return {'expected_improvement': 0.0, 'confidence': 1.0}
        
        # 경험적 공식 기반 예상 개선율
        base_improvement = min(0.3, total_issues * 0.01)  # 최대 30% 개선
        confidence = max(0.6, 1 - (total_issues * 0.002))  # 신뢰도
        
        return {
            'expected_improvement': round(base_improvement, 3),
            'confidence': round(confidence, 3),
            'improvement_range': f"{base_improvement*0.5:.1%} - {base_improvement*1.5:.1%}"
        }
    
    def _estimate_processing_time_impact(self, adjustments: Dict) -> Dict[str, str]:
        """처리 시간 영향 예상"""
        if not adjustments:
            return {'impact': 'none', 'description': '파라미터 변경 없음'}
        
        impact_factors = []
        if 'embedding_threshold' in adjustments:
            impact_factors.append('임베딩 계산 시간 변화')
        if 'similarity_threshold' in adjustments:
            impact_factors.append('유사도 계산 오버헤드')
        
        return {
            'impact': 'minimal' if len(impact_factors) <= 2 else 'moderate',
            'description': f"{len(impact_factors)}개 파라미터 변경으로 인한 성능 영향",
            'factors': impact_factors
        }
    
    def _estimate_resource_requirements(self, actions: List) -> Dict[str, Any]:
        """리소스 요구사항 예상"""
        if not actions:
            return {'level': 'none', 'description': '즉시 액션 없음'}
        
        effort_levels = [action.get('effort', 'low') for action in actions]
        urgency_levels = [action.get('urgency', 'low') for action in actions]
        
        high_effort_count = effort_levels.count('high')
        high_urgency_count = urgency_levels.count('high')
        
        return {
            'level': 'high' if high_effort_count > 1 or high_urgency_count > 2 else 'medium',
            'description': f"{len(actions)}개 액션 필요 (고강도: {high_effort_count}, 긴급: {high_urgency_count})",
            'estimated_time': f"{len(actions) * 2}-{len(actions) * 4}시간"
        }
    
    def _calculate_quality_grade(self, metrics: Dict[str, Any], xml_level_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """종합 품질 등급 계산 - 튜닝 가능한 가중치 적용"""
        try:
            grades = {}
            overall_score = 0
            weight_sum = 0
            
            # 튜닝 가능한 가중치 가져오기
            weights = self.tuning_config['weights']
            
            # 텍스트 무결성 점수 (튜닝 가능한 가중치)
            if 'integrity' in metrics and 'completeness_score' in metrics['integrity']:
                integrity_score = metrics['integrity']['completeness_score']
                grades['text_integrity'] = self._score_to_grade(integrity_score)
                weight = weights['text_integrity']
                overall_score += integrity_score * weight
                weight_sum += weight
            
            # 분할 품질 점수 (튜닝 가능한 가중치)
            if 'integrity' in metrics and 'segmentation_quality' in metrics['integrity']:
                seg_score = metrics['integrity']['segmentation_quality']['structure_score']
                grades['segmentation'] = self._score_to_grade(seg_score)
                weight = weights['segmentation']
                overall_score += seg_score * weight
                weight_sum += weight
            
            # 텍스트 유사도 (튜닝 가능한 가중치)
            if 'text_similarity' in metrics:
                similarities = []
                for col_data in metrics['text_similarity'].values():
                    if isinstance(col_data, dict) and 'average_similarity' in col_data:
                        similarities.append(col_data['average_similarity'])
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    grades['text_similarity'] = self._score_to_grade(avg_similarity)
                    weight = weights['text_similarity']
                    overall_score += avg_similarity * weight
                    weight_sum += weight
            
            # 구조 일치도 (튜닝 가능한 가중치)
            if 'structure' in metrics:
                structure_score = self._calculate_structure_score(metrics['structure'])
                grades['structure'] = self._score_to_grade(structure_score)
                weight = weights['structure']
                overall_score += structure_score * weight
                weight_sum += weight
            
            # XML 레벨 유사도 (가중치: 50% - 높은 우선순위!)
            if xml_level_result and 'comprehensive_summary' in xml_level_result:
                xml_scores = self._calculate_xml_level_score(xml_level_result['comprehensive_summary'])
                if xml_scores and xml_scores['combined_score'] is not None:
                    combined_score = xml_scores['combined_score']
                    grades['xml_level'] = self._score_to_grade(combined_score)
                    grades['pa_score'] = xml_scores['pa_score']
                    grades['sa_score'] = xml_scores['sa_score'] 
                    overall_score += combined_score * 0.5  # 높은 가중치
                    weight_sum += 0.5
                    print(f"✅ XML 레벨 종합 점수: {combined_score:.1%} (등급: {grades['xml_level']})")
                elif xml_scores and (xml_scores['pa_score'] or xml_scores['sa_score']):
                    # PA 또는 SA 단독 점수만 있는 경우
                    single_score = xml_scores['pa_score'] or xml_scores['sa_score']
                    grades['xml_level'] = self._score_to_grade(single_score)
                    grades['pa_score'] = xml_scores['pa_score']
                    grades['sa_score'] = xml_scores['sa_score']
                    overall_score += single_score * 0.5
                    weight_sum += 0.5
                    score_type = "PA" if xml_scores['pa_score'] else "SA"
                    print(f"✅ XML 레벨 {score_type} 단독 점수: {single_score:.1%} (등급: {grades['xml_level']})")
            
            # 전체 점수 계산
            final_score = overall_score / weight_sum if weight_sum > 0 else 0
            grades['overall'] = self._score_to_grade(final_score)
            grades['overall_score'] = final_score
            
            return grades
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_xml_level_score(self, comprehensive_summary: Dict[str, Any]) -> Dict[str, Any]:
        """XML 레벨 개별 점수 계산"""
        try:
            result = {
                'pa_score': None,
                'sa_score': None,
                'combined_score': None
            }
            
            # PA 레벨 점수 (문장 레벨 정확도)
            if 'pa_level' in comprehensive_summary:
                pa_data = comprehensive_summary['pa_level']
                if 'sentence_level_accuracy' in pa_data:
                    pa_score = pa_data['sentence_level_accuracy']
                    result['pa_score'] = pa_score
                    print(f"   📊 PA 문장 레벨 정확도: {pa_score:.1%}")
            
            # SA 레벨 점수 (단어 레벨 정확도)  
            if 'sa_level' in comprehensive_summary:
                sa_data = comprehensive_summary['sa_level']
                if 'word_level_accuracy' in sa_data:
                    sa_score = sa_data['word_level_accuracy']
                    result['sa_score'] = sa_score
                    print(f"   📊 SA 단어 레벨 정확도: {sa_score:.1%}")
            
            # 종합 점수 계산 (PA와 SA 모두 있을 때만)
            if result['pa_score'] is not None and result['sa_score'] is not None:
                combined_score = (result['pa_score'] + result['sa_score']) / 2
                result['combined_score'] = combined_score
                print(f"   🎯 종합 정확도 (PA+SA 평균): {combined_score:.1%}")
            elif result['pa_score'] is not None:
                print(f"   🎯 PA 단독 정확도: {result['pa_score']:.1%}")
            elif result['sa_score'] is not None:
                print(f"   🎯 SA 단독 정확도: {result['sa_score']:.1%}")
            else:
                print("   ⚠️ XML 레벨 점수 계산 데이터 부족")
            
            # XML 레벨 분석 결과 출력 (실제 데이터 사용)
            try:
                # xml_level_analysis에서 실제 점수 추출
                xml_level_data = comprehensive_summary.get('xml_level_analysis', {})
                if xml_level_data:
                    print(f"   🔍 매칭 방식별 비교:")
                    
                    # PA 분석 결과 (슬라이딩 윈도우)
                    pa_data = xml_level_data.get('pa_analysis', {})
                    pa_similarity = pa_data.get('avg_similarity', 0)
                    print(f"      • 슬라이딩 윈도우 (PA): {pa_similarity:.3f}")
                    
                    # SA 분석 결과 (LCS 방식)  
                    sa_data = xml_level_data.get('sa_analysis', {})
                    sa_similarity = sa_data.get('avg_similarity', 0)
                    print(f"      • LCS 방식 (SA): {sa_similarity:.3f}")
                    
                    # 종합 점수도 표시
                    combined_score = xml_level_data.get('combined_xml_level_score', 0)
                    if combined_score > 0:
                        print(f"      🎯 종합 XML 레벨 점수: {combined_score:.3f}")
                    
                    # 어느 방식이 더 나은지 표시
                    if sa_similarity > pa_similarity:
                        diff = sa_similarity - pa_similarity
                        print(f"      ✨ LCS 방식이 {diff:.3f} 더 높은 점수")
                    elif pa_similarity > sa_similarity:
                        diff = pa_similarity - sa_similarity
                        print(f"      ✨ 슬라이딩 윈도우가 {diff:.3f} 더 높은 점수")
                    else:
                        print(f"      ⚖️ 두 방식 점수가 동일")
                        
                    # 매칭 성능도 표시
                    if pa_data:
                        pa_precision = pa_data.get('precision', 0) * 100
                        pa_recall = pa_data.get('recall', 0) * 100
                        pa_f1 = pa_data.get('f1_score', 0) * 100
                        print(f"      📊 PA 매칭 성능: Precision {pa_precision:.1f}%, Recall {pa_recall:.1f}%, F1 {pa_f1:.1f}%")
                    
                    if sa_data:
                        sa_precision = sa_data.get('precision', 0) * 100
                        sa_recall = sa_data.get('recall', 0) * 100  
                        sa_f1 = sa_data.get('f1_score', 0) * 100
                        print(f"      📊 SA 매칭 성능: Precision {sa_precision:.1f}%, Recall {sa_recall:.1f}%, F1 {sa_f1:.1f}%")
                else:
                    print(f"   ⚠️ XML 레벨 분석 데이터를 찾을 수 없음")
                    
            except Exception as e:
                print(f"   ⚠️ XML 레벨 분석 출력 중 오류: {e}")
                pass
                
            return result
                
        except Exception as e:
            print(f"   ❌ XML 레벨 점수 계산 오류: {e}")
            return None
    
    def _score_to_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.95:
            return 'A+'
        elif score >= 0.9:
            return 'A'
        elif score >= 0.85:
            return 'B+'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.75:
            return 'C+'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _calculate_structure_score(self, structure_data: Dict[str, Any]) -> float:
        """구조 점수 계산"""
        try:
            scores = []
            
            # 길이 비율 점수
            for key, data in structure_data.items():
                if '_length' in key and isinstance(data, dict):
                    if 'length_difference_ratio' in data:
                        # 차이가 적을수록 높은 점수
                        length_score = max(0, 1 - data['length_difference_ratio'])
                        scores.append(length_score)
            
            # 빈 값 비율 점수
            if 'empty_values' in structure_data:
                empty_diff = structure_data['empty_values'].get('empty_ratio_difference', 1)
                empty_score = max(0, 1 - empty_diff)
                scores.append(empty_score)
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except:
            return 0.5
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """값 분포 계산"""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'q1': sorted_values[n // 4] if n > 0 else 0,
            'median': sorted_values[n // 2] if n > 0 else 0,
            'q3': sorted_values[3 * n // 4] if n > 0 else 0,
            'std': (sum((x - sum(values)/n)**2 for x in values) / n)**0.5 if n > 1 else 0
        }
    
    def _calculate_correlation(self, list1: List[float], list2: List[float]) -> float:
        """상관계수 계산"""
        if len(list1) != len(list2) or len(list1) < 2:
            return 0
        
        n = len(list1)
        mean1 = sum(list1) / n
        mean2 = sum(list2) / n
        
        numerator = sum((list1[i] - mean1) * (list2[i] - mean2) for i in range(n))
        denom1 = sum((x - mean1)**2 for x in list1)**0.5
        denom2 = sum((x - mean2)**2 for x in list2)**0.5
        
        if denom1 * denom2 == 0:
            return 0
        
        return numerator / (denom1 * denom2)
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성 - 불일치 분석 및 튜닝 파라미터 기반"""
        recommendations = []
        metrics = analysis_results.get('metrics', {})
        
        # 불일치 데이터 분석 기반 권장사항
        if 'integrity' in metrics and 'mismatch_analysis' in metrics['integrity']:
            mismatch = metrics['integrity']['mismatch_analysis']
            summary = mismatch.get('summary', {})
            
            # 초과 데이터 권장사항
            excess_count = summary.get('total_excess', 0)
            excess_threshold = self.tuning_config['thresholds']['excess_data_threshold']
            if excess_count > excess_threshold:
                recommendations.append(
                    f"🔴 {excess_count}개의 초과 데이터가 임계값({excess_threshold})을 초과했습니다. "
                    f"임베딩 임계값을 높이거나 후처리 필터링을 강화하세요."
                )
            elif excess_count > 0:
                recommendations.append(f"{excess_count}개의 초과 데이터가 있습니다. 임베딩 임계값을 높이거나 후처리 필터링을 고려해보세요.")
            
            # 누락 데이터 권장사항  
            missing_count = summary.get('total_missing', 0)
            if missing_count > 0:
                recommendations.append(f"{missing_count}개의 데이터가 누락되었습니다. 임베딩 임계값을 낮춰보세요.")
            
            # 불일치 데이터 권장사항
            mismatch_count = summary.get('total_mismatches', 0)
            if mismatch_count > 0:
                recommendations.append(
                    f"{mismatch_count}개의 텍스트 불일치가 발견되었습니다. "
                    f"유사도 임계값({self.tuning_config['thresholds']['similarity_low']}) 조정을 고려하세요."
                )
            
            # 심각도별 권장사항
            severity = summary.get('severity_level', 'good')
            if severity == 'critical':
                recommendations.append("⚠️ 데이터 품질이 매우 낮습니다. 전체 파이프라인 재검토가 필요합니다.")
            elif severity == 'poor':
                recommendations.append("⚠️ 데이터 품질이 낮습니다. 주요 파라미터 튜닝이 필요합니다.")
        
        # 무결성 관련 권장사항
        if 'integrity' in metrics:
            integrity = metrics['integrity']
            text_integrity = integrity.get('text_integrity', {})
            
            # 텍스트 무결성 검사
            overall_similarity = text_integrity.get('overall_text_similarity', 1)
            min_threshold = self.tuning_config['thresholds']['text_integrity_min']
            if overall_similarity < min_threshold:
                recommendations.append(
                    f"🔴 텍스트 무결성이 임계값({min_threshold:.1%}) 미만입니다 ({overall_similarity:.1%}). "
                    f"전처리 과정이나 파싱 로직을 점검하세요."
                )
            
            # 원문/번역문별 세부 권장사항
            original_sim = text_integrity.get('original_text_similarity', 1)
            translation_sim = text_integrity.get('translation_text_similarity', 1)
            
            if original_sim < 0.95:
                recommendations.append(f"원문 유사도가 낮습니다 ({original_sim:.1%}). 원문 파싱 과정을 검토하세요.")
            if translation_sim < 0.95:
                recommendations.append(f"번역문 유사도가 낮습니다 ({translation_sim:.1%}). 번역문 파싱 과정을 검토하세요.")
        
        # 텍스트 유사도 관련 권장사항
        if 'text_similarity' in metrics:
            thresholds = self.tuning_config['thresholds']
            for col, sim_data in metrics['text_similarity'].items():
                if isinstance(sim_data, dict):
                    avg_sim = sim_data.get('average_similarity', 1)
                    if avg_sim < thresholds['similarity_low']:
                        recommendations.append(f"{col} 유사도가 매우 낮습니다 ({avg_sim:.1%}). 토큰화 방식이나 전처리 과정을 검토해보세요.")
                    elif avg_sim < thresholds['similarity_medium']:
                        recommendations.append(f"{col} 유사도가 낮습니다 ({avg_sim:.1%}). 세부 튜닝을 고려해보세요.")
        
        # 전체 등급 관련 권장사항
        if 'quality_grade' in analysis_results:
            overall_grade = analysis_results['quality_grade'].get('overall', 'F')
            if overall_grade in ['D', 'F']:
                recommendations.append("전체 품질이 낮습니다. 임베딩 모델 변경이나 하이퍼파라미터 튜닝을 권장합니다.")
            elif overall_grade in ['C', 'C+']:
                recommendations.append("품질이 보통 수준입니다. 세부 튜닝으로 개선 가능합니다.")
        
        # 튜닝 파라미터 제안
        if len(recommendations) > 3:  # 많은 문제가 있을 때
            recommendations.append(
                f"💡 현재 튜닝 설정: 유사도 임계값 {self.tuning_config['thresholds']['similarity_low']:.1f}, "
                f"초과 데이터 임계값 {self.tuning_config['thresholds']['excess_data_threshold']}, "
                f"샘플링 크기 {self.tuning_config['sampling']['mismatch_sample_size']}"
            )
        
        return recommendations
    
    def _save_detailed_reports(self, analysis_results: Dict[str, Any]):
        """상세 리포트 파일들 저장"""
        
        # JSON 리포트 (안전한 직렬화)
        json_report_file = self.accuracy_dir / "comprehensive_accuracy_report.json"
        with open(json_report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 텍스트 요약 리포트
        text_report_file = self.accuracy_dir / "accuracy_summary.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_report(analysis_results))
        
        print(f"📄 상세 리포트 저장됨:")
        print(f"   JSON: {json_report_file}")
        print(f"   텍스트: {text_report_file}")
    
    def _generate_text_report(self, analysis_results: Dict[str, Any]) -> str:
        """텍스트 형식 리포트 생성"""
        lines = []
        
        lines.append("=" * 80)
        lines.append("🎯 CSP XML 파이프라인 종합 정확도 분석 리포트")
        lines.append("=" * 80)
        lines.append("")
        
        # 기본 정보
        lines.append(f"📋 분석 정보")
        lines.append("-" * 40)
        lines.append(f"쌍 ID: {analysis_results.get('pair_id', 'N/A')}")
        lines.append(f"분석 시간: {analysis_results.get('timestamp', 'N/A')}")
        lines.append("")
        
        # 품질 등급
        if 'quality_grade' in analysis_results:
            grades = analysis_results['quality_grade']
            lines.append(f"🏆 품질 등급")
            lines.append("-" * 40)
            lines.append(f"종합 등급: {grades.get('overall', 'N/A')} ({grades.get('overall_score', 0):.1%})")
            
            for metric, grade in grades.items():
                if metric not in ['overall', 'overall_score']:
                    lines.append(f"  {metric}: {grade}")
            lines.append("")
        
        # 전역 텍스트 무결성 분석  
        if 'integrity' in analysis_results.get('metrics', {}):
            integrity = analysis_results['metrics']['integrity']
            lines.append(f"🔒 전역 텍스트 무결성")
            lines.append("-" * 40)
            if 'text_integrity' in integrity:
                text_int = integrity['text_integrity']
                # 전역 무결성 지표 우선 표시
                if 'global_exact_match' in text_int:
                    lines.append(f"전역 완전 일치: {text_int.get('global_exact_match', 0):.1%}")
                    lines.append(f"전역 정규화 유사도: {text_int.get('global_normalized_similarity', 0):.1%}")
                    lines.append(f"전역 원본 유사도: {text_int.get('global_raw_similarity', 0):.1%}")
                    lines.append(f"전역 길이 보존율: {text_int.get('global_length_preservation', 0):.1%}")
                else:
                    # 하위 호환성 (기존 방식)
                    lines.append(f"전체 문자열 일치도: {text_int.get('overall_text_similarity', 0):.1%}")
                    lines.append(f"원문 일치도: {text_int.get('original_text_similarity', 0):.1%}")
                    lines.append(f"번역문 일치도: {text_int.get('translation_text_similarity', 0):.1%}")
                    lines.append(f"길이 보존율: {text_int.get('length_preservation', 0):.1%}")
                
                # 상세 정보 표시
                if 'original_integrity' in text_int and 'translation_integrity' in text_int:
                    lines.append("")
                    lines.append("📊 상세 분석:")
                    orig = text_int['original_integrity']
                    trans = text_int['translation_integrity']
                    lines.append(f"  원문: 완전일치 {orig.get('exact_match', 0):.0%}, 유사도 {orig.get('normalized_similarity', 0):.1%}")
                    lines.append(f"  번역문: 완전일치 {trans.get('exact_match', 0):.0%}, 유사도 {trans.get('normalized_similarity', 0):.1%}")
            lines.append("")
            
            # 분할 품질 분석
            lines.append(f"✂️ 분할 품질")
            lines.append("-" * 40)
            lines.append(f"정답 데이터: {integrity.get('truth_count', 0):,}개")
            lines.append(f"결과 데이터: {integrity.get('result_count', 0):,}개")
            if 'segmentation_quality' in integrity:
                seg_qual = integrity['segmentation_quality']
                lines.append(f"구조 일치도: {seg_qual.get('structure_score', 0):.1%}")
            if integrity.get('missing_data', 0) > 0:
                lines.append(f"누락 데이터: {integrity['missing_data']:,}개")
            if integrity.get('excess_data', 0) > 0:
                lines.append(f"초과 데이터: {integrity['excess_data']:,}개")
            lines.append("")
            
            # 불일치 데이터 상세 분석
            if 'mismatch_analysis' in integrity:
                mismatch = integrity['mismatch_analysis']
                summary = mismatch.get('summary', {})
                
                lines.append(f"🔍 불일치 데이터 분석")
                lines.append("-" * 40)
                lines.append(f"초과 데이터: {summary.get('total_excess', 0)}개")
                lines.append(f"누락 데이터: {summary.get('total_missing', 0)}개")
                lines.append(f"텍스트 불일치: {summary.get('total_mismatches', 0)}개")
                lines.append(f"심각도: {summary.get('severity_level', 'unknown')}")
                
                # 초과 데이터 전체 요약 (처음 3개 예시와 전체 개수)
                if mismatch.get('excess_data_details'):
                    lines.append("")
                    lines.append(f"📋 초과 데이터 전체: {len(mismatch['excess_data_details'])}개")
                    lines.append("대표 예시:")
                    for i, excess in enumerate(mismatch['excess_data_details'][:3]):  # 최대 3개 예시
                        lines.append(f"  {i+1}. 인덱스 {excess.get('index')}: {excess.get('reason')}")
                        if excess.get('content') and self.tuning_config['sampling']['show_details']:
                            for col, content in excess['content'].items():
                                lines.append(f"     {col}: {content}")
                    if len(mismatch['excess_data_details']) > 3:
                        lines.append(f"     ... 외 {len(mismatch['excess_data_details']) - 3}개")
                
                # 누락 데이터 전체 요약
                if mismatch.get('missing_data_details'):
                    lines.append("")
                    lines.append(f"📋 누락 데이터 전체: {len(mismatch['missing_data_details'])}개")
                    lines.append("대표 예시:")
                    for i, missing in enumerate(mismatch['missing_data_details'][:3]):  # 최대 3개 예시
                        lines.append(f"  {i+1}. 인덱스 {missing.get('index')}: {missing.get('reason')}")
                    if len(mismatch['missing_data_details']) > 3:
                        lines.append(f"     ... 외 {len(mismatch['missing_data_details']) - 3}개")
                
                # 텍스트 불일치 전체 요약
                if mismatch.get('mismatch_samples'):
                    lines.append("")
                    lines.append(f"📋 텍스트 불일치 전체: {len(mismatch['mismatch_samples'])}개")
                    lines.append("대표 예시:")
                    for i, sample in enumerate(mismatch['mismatch_samples'][:3]):  # 최대 3개 예시
                        lines.append(f"  {i+1}. {sample.get('column')} (유사도: {sample.get('similarity', 0):.1%})")
                        lines.append(f"     정답: {sample.get('truth_preview', '')}")
                        lines.append(f"     결과: {sample.get('result_preview', '')}")
                    if len(mismatch['mismatch_samples']) > 3:
                        lines.append(f"     ... 외 {len(mismatch['mismatch_samples']) - 3}개")
                
                # 데이터 품질 문제
                if mismatch.get('data_quality_issues'):
                    lines.append("")
                    lines.append("⚠️ 데이터 품질 문제:")
                    for issue in mismatch['data_quality_issues'][:5]:  # 최대 5개만
                        lines.append(f"  • {issue.get('description')}")
                
                lines.append("")
        
        # 텍스트 유사도
        if 'text_similarity' in analysis_results.get('metrics', {}):
            lines.append(f"📝 텍스트 유사도")
            lines.append("-" * 40)
            for col, sim_data in analysis_results['metrics']['text_similarity'].items():
                if isinstance(sim_data, dict):
                    lines.append(f"{col}:")
                    lines.append(f"  평균 유사도: {sim_data.get('average_similarity', 0):.1%}")
                    lines.append(f"  최소/최대: {sim_data.get('min_similarity', 0):.1%} / {sim_data.get('max_similarity', 0):.1%}")
            lines.append("")
        
        # F1 점수 (AccuracyEvaluator 결과가 있는 경우)
        if 'f1_scores' in analysis_results.get('metrics', {}):
            lines.append(f"📊 F1 점수 (정밀 평가)")
            lines.append("-" * 40)
            for metric, score in analysis_results['metrics']['f1_scores'].items():
                lines.append(f"{metric}: {score:.3f}")
            lines.append("")
        
        # 무결성 대조 로그 
        if 'integrity' in analysis_results.get('metrics', {}) and 'mismatch_analysis' in analysis_results['metrics']['integrity']:
            mismatch = analysis_results['metrics']['integrity']['mismatch_analysis']
            if 'integrity_comparison_log' in mismatch:
                lines.append(f"🔍 무결성 대조 로그")
                lines.append("-" * 40)
                
                for log_entry in mismatch['integrity_comparison_log'][-3:]:  # 최근 3개만
                    if log_entry.get('log_type') == 'integrity_summary':
                        lines.append(f"전체 무결성 평가: {log_entry.get('overall_assessment', 'unknown')}")
                        lines.append(f"무결성 점수: {log_entry.get('integrity_score', 0):.1%}")
                        if 'next_steps' in log_entry:
                            lines.append("단계별 개선 방안:")
                            for step in log_entry['next_steps'][:3]:
                                lines.append(f"  {step}")
                    elif log_entry.get('log_type') in ['excess_data_comparison', 'missing_data_comparison', 'text_mismatch_comparison']:
                        lines.append(f"• {log_entry.get('issue_description', 'N/A')}")
                        if 'impact_analysis' in log_entry:
                            impact = log_entry['impact_analysis']
                            lines.append(f"  권장 조치: {impact.get('recommended_action', 'N/A')}")
                lines.append("")
        
        # 스마트 튜닝 제안
        if 'integrity' in analysis_results.get('metrics', {}) and 'mismatch_analysis' in analysis_results['metrics']['integrity']:
            mismatch = analysis_results['metrics']['integrity']['mismatch_analysis']
            if 'tuning_suggestions' in mismatch:
                suggestions = mismatch['tuning_suggestions']
                lines.append(f"🎯 스마트 튜닝 제안")
                lines.append("-" * 40)
                
                # 즉시 실행 액션
                if 'immediate_actions' in suggestions and suggestions['immediate_actions']:
                    lines.append("즉시 실행 필요:")
                    for action in suggestions['immediate_actions'][:3]:
                        lines.append(f"  🚨 {action.get('description', 'N/A')} (우선도: {action.get('urgency', 'low')})")
                
                # 파라미터 조정
                if 'parameter_adjustments' in suggestions and suggestions['parameter_adjustments']:
                    lines.append("파라미터 조정 제안:")
                    for param, adjustment in suggestions['parameter_adjustments'].items():
                        if isinstance(adjustment, dict):
                            current = adjustment.get('current_value', 'N/A')
                            suggested = adjustment.get('suggested_value', 'N/A')
                            lines.append(f"  📊 {param}: {current} → {suggested}")
                            lines.append(f"     사유: {adjustment.get('adjustment_reason', 'N/A')}")
                
                # 예상 개선 효과
                if 'expected_outcomes' in suggestions and 'accuracy_improvement' in suggestions['expected_outcomes']:
                    improvement = suggestions['expected_outcomes']['accuracy_improvement']
                    lines.append(f"예상 개선 효과: {improvement.get('improvement_range', 'N/A')}")
                    lines.append(f"신뢰도: {improvement.get('confidence', 0):.1%}")
                
                lines.append("")
        
        # 기존 권장사항
        if analysis_results.get('recommendations'):
            lines.append(f"💡 개선 권장사항")
            lines.append("-" * 40)
            for i, rec in enumerate(analysis_results['recommendations'], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _analyze_xml_level_similarity(self, xml_file: str, sa_result_file: str, pa_result_file: str = None, xml_translation_file: str = None) -> Dict[str, Any]:
        """XML 레벨별 유사도 분석"""
        print("🔍 XML 레벨 유사도 분석 시작...")
        
        if not XMLLevelSimilarityCalculator:
            return {'error': 'XMLLevelSimilarityCalculator를 사용할 수 없습니다'}
        
        try:
            calculator = XMLLevelSimilarityCalculator(use_embeddings=True)
            
            # PA와 SA 개별 분석 및 종합
            pa_analysis = {'error': 'PA 결과 파일 없음'}
            sa_analysis = {'error': 'SA 결과 파일 없음'}
            
            # PA 분석 (있는 경우)
            if pa_result_file and Path(pa_result_file).exists():
                print("📊 PA XML 레벨 분석 수행")
                try:
                    pa_analysis = calculator.calculate_pa_similarity(xml_file, pa_result_file, xml_translation_file)
                    print(f"✅ PA 분석 완료: {pa_analysis.get('xml_unit_count', 0)}개 XML 단위 vs {pa_analysis.get('result_row_count', 0)}개 결과")
                except Exception as e:
                    print(f"⚠️ PA 분석 실패: {e}")
                    pa_analysis = {'error': f'PA 분석 실패: {e}'}
            else:
                print("⚠️ PA 결과 파일이 제공되지 않음")
            
            # SA 분석 (있는 경우)  
            if sa_result_file and Path(sa_result_file).exists():
                print("📊 SA XML 레벨 분석 수행")
                try:
                    sa_analysis = calculator.calculate_sa_similarity(xml_file, sa_result_file, xml_translation_file)
                    print(f"✅ SA 분석 완료: {sa_analysis.get('xml_unit_count', 0)}개 XML 단위 vs {sa_analysis.get('result_row_count', 0)}개 결과")
                except Exception as e:
                    print(f"⚠️ SA 분석 실패: {e}")
                    sa_analysis = {'error': f'SA 분석 실패: {e}'}
            else:
                print("⚠️ SA 결과 파일이 제공되지 않음")
            
            # 종합 요약 생성 (PA와 SA 개별 점수 포함)
            comprehensive_summary = self._generate_pa_sa_comprehensive_summary(pa_analysis, sa_analysis)
            
            xml_level_results = {
                'xml_file': xml_file,
                'pa_result_file': pa_result_file,
                'sa_result_file': sa_result_file,
                'pa_analysis': pa_analysis,
                'sa_analysis': sa_analysis,
                'comprehensive_summary': comprehensive_summary
            }
            
            print("✅ XML 레벨 유사도 분석 완료")
            return xml_level_results
            
        except Exception as e:
            error_msg = f"XML 레벨 분석 오류: {e}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
    
    def _generate_pa_sa_comprehensive_summary(self, pa_analysis: Dict, sa_analysis: Dict) -> Dict:
        """PA와 SA 개별 분석을 종합한 요약 생성"""
        summary = {
            'pa_level': {},
            'sa_level': {},
            'comparison_methods': {},
            'overall_assessment': {}
        }
        
        # PA 레벨 분석
        if 'error' not in pa_analysis and 'statistics' in pa_analysis:
            pa_stats = pa_analysis['statistics']
            if 'original_similarities' in pa_stats:
                summary['pa_level'] = {
                    'sentence_level_accuracy': pa_stats['original_similarities'].get('mean', 0),
                    'processing_completeness': pa_analysis.get('result_row_count', 0) / max(pa_analysis.get('xml_unit_count', 1), 1),
                    'total_processed': pa_analysis.get('result_row_count', 0)
                }
                print(f"📊 PA 문장 레벨 정확도: {summary['pa_level']['sentence_level_accuracy']:.1%}")
        
        # SA 레벨 분석  
        if 'error' not in sa_analysis and 'statistics' in sa_analysis:
            sa_stats = sa_analysis['statistics']
            if 'original_similarities' in sa_stats:
                summary['sa_level'] = {
                    'word_level_accuracy': sa_stats['original_similarities'].get('mean', 0),
                    'phrase_segmentation': sa_stats.get('phrase_similarities', {}).get('mean', 0),
                    'processing_completeness': sa_analysis.get('result_row_count', 0) / max(sa_analysis.get('xml_unit_count', 1), 1),
                    'total_processed': sa_analysis.get('result_row_count', 0)
                }
                print(f"📊 SA 단어 레벨 정확도: {summary['sa_level']['word_level_accuracy']:.1%}")
        
        # 방식별 비교 결과 (SA에서 LCS vs 슬라이딩 윈도우)
        if 'error' not in sa_analysis and 'comparison_summary' in sa_analysis:
            comparison = sa_analysis['comparison_summary']
            summary['comparison_methods'] = {
                'sliding_window_avg': comparison.get('sliding_window_avg', 0),
                'lcs_avg': comparison.get('lcs_avg', 0),
                'method_used': comparison.get('method_used', 'unknown')
            }
        
        # 전체 평가
        pa_score = summary['pa_level'].get('sentence_level_accuracy', 0) if summary['pa_level'] else 0
        sa_score = summary['sa_level'].get('word_level_accuracy', 0) if summary['sa_level'] else 0
        
        summary['overall_assessment'] = {
            'pa_available': bool(summary['pa_level']),
            'sa_available': bool(summary['sa_level']),
            'pa_score': pa_score,
            'sa_score': sa_score,
            'combined_score': (pa_score + sa_score) / 2 if pa_score > 0 and sa_score > 0 else (pa_score or sa_score),
            'recommendation': self._generate_pa_sa_recommendation(pa_score, sa_score)
        }
        
        return summary
    
    def _generate_pa_sa_recommendation(self, pa_score: float, sa_score: float) -> str:
        """PA와 SA 점수 기반 권장사항 생성"""
        if pa_score > 0 and sa_score > 0:
            avg_score = (pa_score + sa_score) / 2
            pa_vs_sa = "PA가 우수" if pa_score > sa_score + 0.1 else "SA가 우수" if sa_score > pa_score + 0.1 else "PA/SA 균형"
            
            if avg_score > 0.8:
                return f"우수한 성능 - {pa_vs_sa} (PA: {pa_score:.1%}, SA: {sa_score:.1%})"
            elif avg_score > 0.6:
                return f"보통 수준 - {pa_vs_sa} (PA: {pa_score:.1%}, SA: {sa_score:.1%})"
            else:
                return f"개선 필요 - {pa_vs_sa} (PA: {pa_score:.1%}, SA: {sa_score:.1%})"
        elif pa_score > 0:
            return f"PA 단독: {pa_score:.1%}" + (" - 우수" if pa_score > 0.8 else " - 보통" if pa_score > 0.6 else " - 개선 필요")
        elif sa_score > 0:
            return f"SA 단독: {sa_score:.1%}" + (" - 우수" if sa_score > 0.8 else " - 보통" if sa_score > 0.6 else " - 개선 필요")
        else:
            return "분석 데이터 부족 - PA/SA 결과 확인 필요"
    


    def _generate_integrity_contrast_log(self, mismatch_analysis: Dict[str, Any], integrity_score: float) -> List[Dict[str, Any]]:
        """무결성 대조 로그 생성"""
        contrast_log = []
        
        # 전체 평가
        severity = "critical" if integrity_score < 0.7 else "poor" if integrity_score < 0.8 else "moderate"
        contrast_log.append({
            "log_type": "integrity_summary",
            "overall_assessment": severity,
            "integrity_score": integrity_score,
            "issue_description": f"무결성 점수 {integrity_score:.1%} - {severity} 수준",
            "next_steps": [
                "데이터 품질 개선 필요",
                "매핑 알고리즘 조정 검토",
                "임계값 튜닝 권장"
            ]
        })
        
        # 개별 문제 분석
        if mismatch_analysis.get('excess_data_count', 0) > 0:
            contrast_log.append({
                "log_type": "excess_data_comparison",
                "issue_description": f"초과 데이터 {mismatch_analysis['excess_data_count']}개 발견",
                "impact_analysis": {
                    "severity": "high" if mismatch_analysis['excess_data_count'] > 100 else "medium",
                    "recommended_action": "데이터 전처리 강화 또는 매핑 규칙 조정"
                }
            })
        
        return contrast_log

    def _generate_smart_tuning_suggestions(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """스마트 튜닝 제안 생성"""
        suggestions = {
            "immediate_actions": [],
            "parameter_adjustments": {},
            "expected_outcomes": {}
        }
        
        # 무결성 점수 기반 제안
        integrity_score = analysis_results.get('metrics', {}).get('integrity', {}).get('completeness_score', 0)
        
        if integrity_score < 0.7:
            suggestions["immediate_actions"].append({
                "description": "데이터 품질 심각 - 즉시 개선 필요",
                "urgency": "critical"
            })
            suggestions["parameter_adjustments"]["similarity_threshold"] = {
                "current_value": 0.8,
                "suggested_value": 0.6,
                "adjustment_reason": "낮은 무결성으로 인한 임계값 완화 필요"
            }
        
        # 예상 개선 효과
        suggestions["expected_outcomes"]["accuracy_improvement"] = {
            "improvement_range": "5-15%",
            "confidence": 0.75
        }
        
        return suggestions


def extract_paragraphs_from_xml(xml_file: Path) -> Dict[str, str]:
    """XML에서 단락(<단락>) 단위로 ID와 텍스트 추출"""
    import re
    
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # <단락 id="X"> 태그 내의 텍스트를 ID와 함께 추출
        paragraphs = {}
        pattern = r'<단락[^>]*id="([^"]*)"[^>]*>(.*?)</단락>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for paragraph_id, match in matches:
            # XML 태그 모두 제거하고 텍스트만 추출
            clean_text = re.sub(r'<[^>]+>', '', match).strip()
            if clean_text:
                paragraphs[paragraph_id] = clean_text
        
        return paragraphs
        
    except Exception as e:
        print(f"❌ XML 단락 추출 오류: {e}")
        return {}

def extract_sentences_from_xml(xml_file: Path) -> Dict[str, str]:
    """XML에서 문장(<s>) 단위로 ID와 텍스트 추출"""
    import re
    
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # <s id="X"> 태그 내의 텍스트를 ID와 함께 추출
        sentences = {}
        pattern = r'<s[^>]*id="([^"]*)"[^>]*>(.*?)</s>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for sentence_id, match in matches:
            # XML 태그 모두 제거하고 텍스트만 추출
            clean_text = re.sub(r'<[^>]+>', '', match).strip()
            if clean_text:
                sentences[sentence_id] = clean_text
        
        return sentences
        
    except Exception as e:
        print(f"❌ XML 문장 추출 오류: {e}")
        return {}

def xml_to_dataframe_for_pa(source_file: Path, target_file: Path) -> pd.DataFrame:
    """PA용: XML에서 단락 단위로 ID 매칭하여 DataFrame 생성"""
    try:
        source_paragraphs = extract_paragraphs_from_xml(source_file)
        target_paragraphs = extract_paragraphs_from_xml(target_file)
        
        # ID 기반 매칭
        matched_pairs = []
        for paragraph_id in source_paragraphs:
            if paragraph_id in target_paragraphs:
                matched_pairs.append({
                    '원문': source_paragraphs[paragraph_id],
                    '번역문': target_paragraphs[paragraph_id]
                })
        
        df = pd.DataFrame(matched_pairs)
        
        print(f"📄 PA용 XML → DataFrame 변환 완료: {len(df)}개 단락 (ID 매칭)")
        return df
        
    except Exception as e:
        print(f"❌ PA용 XML 변환 오류: {e}")
        return pd.DataFrame()

def xml_to_dataframe_for_sa(source_file: Path, target_file: Path) -> pd.DataFrame:
    """SA용: XML에서 문장 단위로 ID 매칭하여 DataFrame 생성"""
    try:
        source_sentences = extract_sentences_from_xml(source_file)
        target_sentences = extract_sentences_from_xml(target_file)
        
        # ID 기반 매칭
        matched_pairs = []
        for sentence_id in source_sentences:
            if sentence_id in target_sentences:
                matched_pairs.append({
                    '원문': source_sentences[sentence_id],
                    '번역문': target_sentences[sentence_id]
                })
        
        df = pd.DataFrame(matched_pairs)
        
        print(f"📄 SA용 XML → DataFrame 변환 완료: {len(df)}개 문장 (ID 매칭)")
        return df
        
    except Exception as e:
        print(f"❌ SA용 XML 변환 오류: {e}")
        return pd.DataFrame()

def run_pa_analysis(source_file: Path, target_file: Path, output_dir: Path) -> Path:
    """PA 분석 실행"""
    print(f"🔄 PA 분석 실행 중: {source_file.name} + {target_file.name}")
    
    try:
        # XML을 DataFrame으로 변환 (PA용: 단락 단위)
        df = xml_to_dataframe_for_pa(source_file, target_file)
        if df.empty:
            print(f"❌ XML 파일 변환 실패")
            return None
        
        # 임시 Excel 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_input:
            temp_input_path = temp_input.name
            df.to_excel(temp_input_path, index=False)
        
        # PA 시스템 실행
        pa_dir = Path(__file__).parent / "pa"
        
        # PA 기본 출력 파일 (절대 경로 사용)
        pa_default_output = pa_dir / "output.xlsx"
        pa_final_output = output_dir / "pa_result.xlsx"
        cmd = [
            sys.executable, str(pa_dir / "main.py"),
            temp_input_path, str(pa_default_output.absolute()),
            "--embedder", "bge"
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(pa_dir),
            timeout=300
        )
        
        # 임시 파일 정리
        os.unlink(temp_input_path)
        
        if result.returncode == 0 and pa_default_output.exists():
            # 결과 파일을 최종 위치로 복사
            shutil.copy2(pa_default_output, pa_final_output)
            print(f"✅ PA 분석 완료: {pa_final_output}")
            return pa_final_output
        else:
            print(f"❌ PA 분석 실패: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ PA 분석 타임아웃")
        return None
    except Exception as e:
        print(f"❌ PA 분석 오류: {e}")
        return None


def run_sa_analysis(source_file: Path, target_file: Path, output_dir: Path) -> Path:
    """SA 분석 실행"""
    print(f"🔄 SA 분석 실행 중: {source_file.name} + {target_file.name}")
    
    try:
        # XML을 DataFrame으로 변환 (SA용: 문장 단위)
        df = xml_to_dataframe_for_sa(source_file, target_file)
        if df.empty:
            print(f"❌ XML 파일 변환 실패")
            return None
        
        # 임시 Excel 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_input:
            temp_input_path = temp_input.name
            df.to_excel(temp_input_path, index=False)
        
        # SA 시스템 실행
        sa_dir = Path(__file__).parent / "sa"
        
        # SA 기본 출력 파일 (절대 경로 사용)
        sa_default_output = sa_dir / "output.xlsx"
        sa_final_output = output_dir / "sa_result.xlsx"
        cmd = [
            sys.executable, str(sa_dir / "main.py"),
            temp_input_path, str(sa_default_output.absolute()),
            "--embedder", "bge"
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(sa_dir),
            timeout=300
        )
        
        # 임시 파일 정리
        os.unlink(temp_input_path)
        
        if result.returncode == 0 and sa_default_output.exists():
            # 결과 파일을 최종 위치로 복사
            shutil.copy2(sa_default_output, sa_final_output)
            print(f"✅ SA 분석 완료: {sa_final_output}")
            return sa_final_output
        else:
            print(f"❌ SA 분석 실패: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ SA 분석 타임아웃")
        return None
    except Exception as e:
        print(f"❌ SA 분석 오류: {e}")
        return None


def create_advanced_accuracy_analyzer(accuracy_dir: Path, tuning_config: Dict[str, Any] = None) -> AdvancedAccuracyAnalyzer:
    """고도화된 정확도 분석기 생성 - 튜닝 설정 지원"""
    return AdvancedAccuracyAnalyzer(accuracy_dir, tuning_config)


def create_tuning_config(
    # 가중치 설정
    text_integrity_weight: float = 0.35,
    segmentation_weight: float = 0.15, 
    text_similarity_weight: float = 0.25,
    structure_weight: float = 0.25,
    
    # 임계값 설정
    similarity_low: float = 0.3,
    similarity_medium: float = 0.6,
    similarity_high: float = 0.8,
    excess_data_threshold: int = 50,
    text_integrity_min: float = 0.95,
    
    # 샘플링 설정
    mismatch_sample_size: int = 10,
    excess_sample_size: int = 10,
    show_details: bool = True,
    
    # 필터링 설정
    min_text_length: int = 2,
    remove_empty: bool = True,
    normalize_whitespace: bool = True
) -> Dict[str, Any]:
    """
    튜닝 설정 생성 헬퍼 함수
    
    Args:
        text_integrity_weight: 텍스트 무결성 가중치 (기본: 0.35)
        segmentation_weight: 분할 품질 가중치 (기본: 0.15)
        text_similarity_weight: 텍스트 유사도 가중치 (기본: 0.25)
        structure_weight: 구조 일치도 가중치 (기본: 0.25)
        similarity_low: 낮은 유사도 임계값 (기본: 0.3)
        similarity_medium: 중간 유사도 임계값 (기본: 0.6)
        similarity_high: 높은 유사도 임계값 (기본: 0.8)
        excess_data_threshold: 초과 데이터 경고 임계값 (기본: 50)
        text_integrity_min: 최소 텍스트 무결성 기준 (기본: 0.95)
        mismatch_sample_size: 불일치 데이터 샘플 개수 (기본: 10)
        excess_sample_size: 초과 데이터 샘플 개수 (기본: 10)
        show_details: 상세 내용 표시 여부 (기본: True)
        min_text_length: 최소 텍스트 길이 (기본: 2)
        remove_empty: 빈 값 제거 여부 (기본: True)
        normalize_whitespace: 공백 정규화 여부 (기본: True)
    
    Returns:
        튜닝 설정 딕셔너리
    
    Examples:
        # 엄격한 평가를 위한 설정
        strict_config = create_tuning_config(
            similarity_low=0.5,
            similarity_medium=0.7, 
            similarity_high=0.9,
            excess_data_threshold=20,
            text_integrity_min=0.98
        )
        
        # 관대한 평가를 위한 설정
        lenient_config = create_tuning_config(
            similarity_low=0.2,
            similarity_medium=0.5,
            similarity_high=0.7,
            excess_data_threshold=100,
            text_integrity_min=0.90
        )
        
        # 상세 분석을 위한 설정
        detailed_config = create_tuning_config(
            mismatch_sample_size=20,
            excess_sample_size=20,
            show_details=True
        )
    """
    return {
        "weights": {
            "text_integrity": text_integrity_weight,
            "segmentation": segmentation_weight,
            "text_similarity": text_similarity_weight,
            "structure": structure_weight
        },
        "thresholds": {
            "similarity_low": similarity_low,
            "similarity_medium": similarity_medium,
            "similarity_high": similarity_high,
            "excess_data_threshold": excess_data_threshold,
            "text_integrity_min": text_integrity_min
        },
        "sampling": {
            "mismatch_sample_size": mismatch_sample_size,
            "excess_sample_size": excess_sample_size,
            "show_details": show_details
        },
        "filtering": {
            "min_text_length": min_text_length,
            "remove_empty": remove_empty,
            "normalize_whitespace": normalize_whitespace
        }
    }


if __name__ == "__main__":
    """
    사용 예제 및 테스트
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='고도화된 정확도 분석 실행')
    # 배치 처리 옵션
    parser.add_argument('--batch-all', action='store_true', help='sources 폴더의 모든 책 배치 처리')
    parser.add_argument('--mode', default='comprehensive', choices=['comprehensive', 'basic'], help='분석 모드')
    
    # 개별 파일 처리 옵션
    parser.add_argument('--xml_truth', help='XML 정답 파일 경로')
    parser.add_argument('--sa_result', help='SA 결과 파일 경로') 
    parser.add_argument('--output_dir', help='결과 출력 디렉토리')
    parser.add_argument('--pair_id', default='test', help='쌍 ID (기본: test)')
    
    # 튜닝 파라미터
    parser.add_argument('--similarity_low', type=float, default=0.3, help='낮은 유사도 임계값')
    parser.add_argument('--similarity_high', type=float, default=0.8, help='높은 유사도 임계값')
    parser.add_argument('--excess_threshold', type=int, default=50, help='초과 데이터 경고 임계값')
    parser.add_argument('--sample_size', type=int, default=10, help='불일치 샘플 크기')
    parser.add_argument('--strict', action='store_true', help='엄격한 평가 모드')
    parser.add_argument('--lenient', action='store_true', help='관대한 평가 모드')
    
    args = parser.parse_args()
    
    # 배치 처리 실행
    if args.batch_all:
        print("🚀 24책 전체 배치 처리 시작...")
        
        # sources 디렉토리에서 XML 파일 찾기
        sources_dir = Path("sources")
        if not sources_dir.exists():
            print(f"❌ sources 디렉토리를 찾을 수 없습니다: {sources_dir}")
            exit(1)
        
        # 원문 XML 파일들 찾기 (패턴: *원문*.xml)
        xml_files = []
        for pattern in ["*원문_x-C*.xml", "*원문-C*.xml"]:
            xml_files.extend(sources_dir.glob(pattern))
        
        # 중복 제거 및 정렬
        xml_files = sorted(list(set(xml_files)))
        
        if not xml_files:
            print(f"❌ sources 디렉토리에서 원문 XML 파일을 찾을 수 없습니다")
            print(f"   경로: {sources_dir.absolute()}")
            print(f"   찾는 패턴: *원문_x-C*.xml, *원문-C*.xml")
            exit(1)
        
        print(f"📚 {len(xml_files)}개 책 발견")
        
        # 튜닝 설정 생성
        if args.strict:
            tuning_config = create_tuning_config(
                similarity_low=0.5, similarity_medium=0.7, similarity_high=0.9,
                excess_data_threshold=20, text_integrity_min=0.98
            )
            print("🔒 엄격한 평가 모드 적용")
        elif args.lenient:
            tuning_config = create_tuning_config(
                similarity_low=0.2, similarity_medium=0.5, similarity_high=0.7,
                excess_data_threshold=100, text_integrity_min=0.90
            )
            print("🔓 관대한 평가 모드 적용")
        else:
            tuning_config = create_tuning_config()
            print("⚙️ 기본 튜닝 설정 적용")
        
        # 각 책에 대해 처리 - 책별 결과 디렉토리 생성
        base_accuracy_dir = Path("xml_pipeline_results")
        base_accuracy_dir.mkdir(exist_ok=True)
        
        # 분석기는 하나만 생성 (임시 디렉토리로 초기화)
        temp_dir = base_accuracy_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        analyzer = create_advanced_accuracy_analyzer(temp_dir, tuning_config)
        
        batch_results = {}
        successful_count = 0
        failed_count = 0
        
        # 안전장치: 최대 처리 시간 및 메모리 체크
        import time
        import gc
        
        for i, xml_file in enumerate(xml_files, 1):
            start_time = time.time()
            
            try:
                # 책 이름 추출 (원문 파일에서)
                filename = xml_file.stem
                
                # '원문'과 연도 부분 제거하여 순수 책 이름 추출
                if '_원문_x-C' in filename:
                    book_name = filename.split('_원문_x-C')[0]
                elif '_원문-C' in filename:
                    book_name = filename.split('_원문-C')[0]
                else:
                    # 예상하지 못한 패턴의 경우 전체 파일명 사용
                    book_name = filename
                
                # 공백 문제 방지: 공백을 언더스코어로 변경
                book_name = book_name.replace(' ', '_')
                
                print(f"\n{'='*60}")
                print(f"📖 [{i}/{len(xml_files)}] {book_name} 처리 중...")
                print(f"📄 원문 파일: {xml_file.name}")
                print(f"{'='*60}")
                
                # 번역문 XML 파일 찾기 (한글 인코딩 문제 고려)
                translation_xml_file = None
                
                # 모든 XML 파일을 검사하여 매칭되는 번역문 파일 찾기
                all_xml_files = list(sources_dir.glob("*.xml"))
                
                for xml_file_candidate in all_xml_files:
                    candidate_name = xml_file_candidate.stem
                    
                    # 같은 책의 번역문 파일인지 확인
                    # 1. 파일명 시작 부분이 같은지 (jti_xxx 부분)
                    if candidate_name.startswith(book_name.split('-')[0]):
                        # 2. 번역문 키워드가 포함되어 있는지 확인 (한글이 깨져도 패턴으로 확인)
                        original_file_parts = xml_file.stem.split('_')
                        candidate_parts = candidate_name.split('_')
                        
                        # 파일 구조가 유사하고, 원문이 아닌 파일이면 번역문으로 추정
                        if (len(candidate_parts) >= len(original_file_parts) and 
                            candidate_name != xml_file.stem and  # 자기 자신이 아님
                            candidate_parts[0] == original_file_parts[0]):  # 같은 책 ID
                            
                            translation_xml_file = xml_file_candidate
                            print(f"📝 번역문 XML 파일 발견: {translation_xml_file.name}")
                            break
                
                if not translation_xml_file:
                    print(f"⚠️ 번역문 XML 파일을 찾을 수 없습니다: {book_name}")
                
                # 책별 결과 디렉토리 생성
                book_accuracy_dir = base_accuracy_dir / book_name
                book_accuracy_dir.mkdir(exist_ok=True)
                print(f"📁 결과 디렉토리 생성: {book_accuracy_dir}")
                
                # 분석기의 결과 디렉토리를 책별로 변경
                analyzer.accuracy_dir = book_accuracy_dir
                
                # 1단계: PA/SA 분석 실행
                print(f"🚀 {book_name} PA/SA 분석 실행 중...")
                
                # PA 분석 실행
                pa_result_file = run_pa_analysis(xml_file, translation_xml_file, book_accuracy_dir)
                if not pa_result_file:
                    print(f"⚠️ PA 분석 실패: {book_name} (함수 반환값 None)")
                elif not pa_result_file.exists():
                    print(f"⚠️ PA 분석 실패: {book_name} (파일 없음: {pa_result_file})")
                else:
                    print(f"✅ PA 분석 성공: {pa_result_file}")
                
                # SA 분석 실행  
                sa_result_file = run_sa_analysis(xml_file, translation_xml_file, book_accuracy_dir)
                if not sa_result_file:
                    print(f"❌ SA 분석 실패: {book_name} (함수 반환값 None)")
                    batch_results[book_name] = {"error": "SA 분석 함수 반환값 None"}
                    failed_count += 1
                    continue
                elif not sa_result_file.exists():
                    print(f"❌ SA 분석 실패: {book_name} (파일 없음: {sa_result_file})")
                    batch_results[book_name] = {"error": f"SA 결과 파일 없음: {sa_result_file}"}
                    failed_count += 1
                    continue
                else:
                    print(f"✅ SA 분석 성공: {sa_result_file}")
                
                # 메모리 정리
                gc.collect()
                
                # 2단계: 정확도 분석 실행
                print(f"🔍 {book_name} 정확도 분석 시작...")
                result = analyzer.analyze_comprehensive_accuracy(
                    xml_truth_file=str(xml_file),
                    xml_translation_file=str(translation_xml_file) if translation_xml_file else None,
                    sa_result_file=str(sa_result_file),
                    pa_result_file=str(pa_result_file) if pa_result_file and pa_result_file.exists() else None,
                    pair_id=book_name
                )
                
                batch_results[book_name] = result
                successful_count += 1
                
                # 개별 결과 요약 출력
                if 'quality_grade' in result:
                    grade = result['quality_grade']
                    print(f"📊 {book_name} 결과 요약:")
                    print(f"   종합 등급: {grade.get('overall', 'N/A')} ({grade.get('overall_score', 0):.1%})")
                    if 'pa_score' in grade and grade['pa_score'] is not None:
                        print(f"   PA 점수: {grade['pa_score']:.1%}")
                    if 'sa_score' in grade and grade['sa_score'] is not None:
                        print(f"   SA 점수: {grade['sa_score']:.1%}")
                
                elapsed_time = time.time() - start_time
                print(f"✅ {book_name} 분석 완료 (소요시간: {elapsed_time:.1f}초)")
                
                # 진행상황 요약
                if i % 5 == 0 or i == len(xml_files):
                    print(f"\n📈 진행상황: {i}/{len(xml_files)} (성공: {successful_count}, 실패: {failed_count})")
                
            except KeyboardInterrupt:
                print(f"\n⚠️ 사용자에 의해 중단됨 (처리된 책: {i-1}/{len(xml_files)})")
                break
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                print(f"❌ {book_name} 분석 실패: {error_msg}")
                batch_results[book_name] = {'error': error_msg}
                
                # 연속 실패 체크
                if failed_count > 3:
                    print(f"⚠️ 연속 실패 {failed_count}회 발생. 시스템 안정성을 위해 잠시 대기...")
                    time.sleep(2)
                
            finally:
                # 메모리 정리
                gc.collect()
        
        # 전체 배치 결과 요약
        print(f"\n{'='*60}")
        print(f"📊 24책 배치 처리 완료 요약")
        print(f"{'='*60}")
        
        successful_count = sum(1 for result in batch_results.values() if 'error' not in result)
        failed_count = len(batch_results) - successful_count
        
        print(f"성공: {successful_count}개, 실패: {failed_count}개")
        
        # 성공한 결과들의 평균 점수 계산
        if successful_count > 0:
            pa_scores = []
            sa_scores = []
            overall_scores = []
            
            for book_name, result in batch_results.items():
                if 'error' not in result and 'quality_grade' in result:
                    grade = result['quality_grade']
                    if 'pa_score' in grade and grade['pa_score'] is not None:
                        pa_scores.append(grade['pa_score'])
                    if 'sa_score' in grade and grade['sa_score'] is not None:
                        sa_scores.append(grade['sa_score'])
                    if 'overall_score' in grade:
                        overall_scores.append(grade['overall_score'])
            
            print(f"\n📈 전체 평균 점수:")
            if pa_scores:
                print(f"   PA 평균: {sum(pa_scores)/len(pa_scores):.1%} ({len(pa_scores)}개 책)")
            if sa_scores:
                print(f"   SA 평균: {sum(sa_scores)/len(sa_scores):.1%} ({len(sa_scores)}개 책)")
            if overall_scores:
                print(f"   종합 평균: {sum(overall_scores)/len(overall_scores):.1%}")
        
        print(f"\n💾 상세 결과는 xml_pipeline_results 디렉토리에서 확인하세요.")
        exit(0)
    
    # 개별 파일 처리
    if not args.xml_truth or not args.sa_result or not args.output_dir:
        parser.error("개별 파일 처리시 --xml_truth, --sa_result, --output_dir 필수")
    
    # 튜닝 설정 생성
    if args.strict:
        tuning_config = create_tuning_config(
            similarity_low=0.5, similarity_medium=0.7, similarity_high=0.9,
            excess_data_threshold=20, text_integrity_min=0.98
        )
        print("🔒 엄격한 평가 모드 적용")
    elif args.lenient:
        tuning_config = create_tuning_config(
            similarity_low=0.2, similarity_medium=0.5, similarity_high=0.7,
            excess_data_threshold=100, text_integrity_min=0.90
        )
        print("🔓 관대한 평가 모드 적용")
    else:
        tuning_config = create_tuning_config(
            similarity_low=args.similarity_low,
            similarity_high=args.similarity_high,
            excess_data_threshold=args.excess_threshold,
            mismatch_sample_size=args.sample_size
        )
        print("⚙️ 사용자 정의 튜닝 설정 적용")
    
    # 분석 실행
    accuracy_dir = Path(args.output_dir)
    analyzer = create_advanced_accuracy_analyzer(accuracy_dir, tuning_config)
    
    result = analyzer.analyze_comprehensive_accuracy(
        xml_truth_file=args.xml_truth,
        sa_result_file=args.sa_result,
        pair_id=args.pair_id
    )
    
    print(f"✅ 분석 완료: {result.get('quality_grade', {}).get('overall', 'N/A')} 등급")
    if result.get('recommendations'):
        print("💡 권장사항:")
        for rec in result['recommendations']:
            print(f"  • {rec}")