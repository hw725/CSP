#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
관자 원문 분할 정확도 평가 스크립트
구병렬 정답 데이터와 output 예측 데이터를 비교하여 정확도를 산출
"""

import pandas as pd
import sys
import os
from collections import defaultdict
from typing import List, Dict, Tuple
import argparse
import difflib
# 임계값 설정 로드 (패키지/스크립트 실행 모두 호환)
try:
    from .thresholds_config import THRESHOLDS
except Exception:
    try:
        # 스크립트로 직접 실행되는 경우 상대 import 실패 가능
        from thresholds_config import THRESHOLDS
    except Exception:
        THRESHOLDS = None

class AccuracyEvaluator:
    def __init__(self, ground_truth_file: str, prediction_file: str, project: str | None = None, *, brief: bool = False, minimal_summary: bool = False, ignore_space_punct: bool = False, ignore_space_only: bool = False, ignore_brackets: bool = False, use_ko_particle_hint: bool = False, particle_weight: float = 0.15, max_dup_per_gt: int = 1, monotonic_alignment: bool = True):
        """
        Args:
            ground_truth_file: 정답 파일 (구병렬 기준)
            prediction_file: 예측 파일 (output01 등)
        """
        self.ground_truth_file = ground_truth_file
        self.prediction_file = prediction_file
        self.gt_data = None
        self.pred_data = None
        self.execution_log = []  # 실행 로그 저장용
        self.source_mismatch_details = []  # 원문 불일치 상세 정보
        self.brief = brief
        self.minimal_summary = minimal_summary
        # 관대 일치 모드: off | space | space_punct
        if bool(ignore_space_punct):
            self._lenient_mode = 'space_punct'
        elif bool(ignore_space_only):
            self._lenient_mode = 'space'
        else:
            self._lenient_mode = 'off'
        self._lenient_source_match = (self._lenient_mode != 'off')
        # [-텍스트] 패턴 무시 옵션
        self._ignore_brackets = bool(ignore_brackets)
    # 경고 옵션 (관대 일치이지만 엄격 불일치인 경우 경고)
        self._warn_lenient_mismatch = False
        # 한글 토씨(조사) 기반 번역문 경계 힌트 사용 여부/가중치
        self._use_ko_particle_hint = bool(use_ko_particle_hint)
        try:
            # 지연 로드: 실제 사용 시에만 초기화
            self._kiwi = None
        except Exception:
            self._kiwi = None
        # 0.0~1.0 사이의 적정 가중치 권장(기본 0.15)
        try:
            pw = float(particle_weight)
            self._particle_weight = max(0.0, min(1.0, pw))
        except Exception:
            self._particle_weight = 0.15
        # 임계값 설정
        self.project = (project or '').lower() or None
        self.thresholds = None
        if self.project and THRESHOLDS and self.project in THRESHOLDS:
            self.thresholds = THRESHOLDS[self.project]
        # 그룹(식별자) 라벨링 (문단/문장 자동 결정)
        self.group_id_col_name = None  # 사용된 식별자 컬럼명
        self.group_label = '문장'  # 기본: 문장, 문단식별자 검출 시 '문단'으로 교체
        # 전역 무결성 캐시
        self._global_integrity = None
        # 행 모드 자동 오프셋 보정 관련(실행 시 전달)
        self._row_shift_applied = 0
        self._row_shift_overlap = 0
        self._row_shift_improved = 0
        # 행 오프셋 보정 추가 통계 (요약/CSV에 포함)
        self._row_shift_zero_eq = 0
        self._row_shift_best_eq = 0
        self._row_shift_best_avg_sim = 0.0
        # 중복 매칭 및 순서 제약
        try:
            self._max_dup_per_gt = max(1, int(max_dup_per_gt))
        except Exception:
            self._max_dup_per_gt = 1
        self._monotonic_alignment = bool(monotonic_alignment)
        # 메트릭 한글 라벨 매핑 (CSV/엑셀에서 사용)
        self.metric_label_ko = {
            'avg_partial_match': '부분 일치율',
            'avg_source_partial_match': '원문 부분 일치율',
            'avg_target_partial_match': '번역문 부분 일치율',
            'avg_target_avg_similarity': '번역문 평균 유사도',
            'avg_target_text_similarity': '번역문 전체 유사도',
            'avg_source_text_similarity': '원문 전체 유사도',
            'avg_source_avg_similarity': '원문 평균 유사도',
            'avg_source_jaccard': '원문 Jaccard 유사도',
            'avg_target_accuracy': '번역문 정확도',
            'avg_f1_score': 'F1 점수',
            'avg_source_f1_score': '원문 F1',
            'avg_target_f1_score': '번역문 F1',
            'avg_precision': '정밀도',
            'avg_recall': '재현율',
            'avg_source_precision': '원문 정밀도',
            'avg_source_recall': '원문 재현율',
            'avg_target_precision': '번역문 정밀도',
            'avg_target_recall': '번역문 재현율',
            'avg_exact_match': '완전 일치율',
            'avg_text_match': '전체 텍스트 일치율',
            'avg_source_text_match': '원문 일치율',
            'avg_target_text_match': '번역문 일치율',
            'avg_segment_count_match': '세그먼트 수 일치율',
            'matching_rate': '매칭율',
            'matched_count': '매칭된 개수',
            'total_count': '총 개수',
            'total_gt_segments': '총 정답 세그먼트',
            'total_pred_segments': '총 예측 세그먼트',
            'total_matched_pairs': '총 매칭된 쌍',
            'total_correct_translation_pairs': '총 정확한 번역 쌍',
            'global_source_delta': '원문 길이 차이',
            'global_target_delta': '번역 길이 차이',
            'global_source_ops_insert': '원문 삽입',
            'global_source_ops_delete': '원문 삭제',
            'global_source_ops_replace': '원문 치환',
            'global_target_ops_insert': '번역 삽입',
            'global_target_ops_delete': '번역 삭제',
            'global_target_ops_replace': '번역 치환',
            'row_shift_applied': '행 오프셋 적용',
            'row_shift_overlap': '행 오프셋 겹침',
            'row_shift_improved': '행 오프셋 개선',
            'row_shift_zero_eq': '행 오프셋 0 일치',
            'row_shift_best_eq': '행 오프셋 최고 일치',
            'row_shift_best_avg_sim': '행 오프셋 최고 유사도',
        }
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """문자열 유사도 계산 (SequenceMatcher 사용)"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _primary_match_basis_for_project(self) -> str:
        """프로젝트별 primary 매칭 기준.

        - PA: 원문 기준(source)
        - SA: 번역 기준(target)
        """
        if (self.project or '').lower() == 'sa':
            return 'target'
        return 'source'

    @staticmethod
    def _basis_label(match_basis: str) -> str:
        if match_basis == 'target':
            return '번역기준(target)'
        return '원문기준(source)'

    @staticmethod
    def _basis_suffix(match_basis: str) -> str:
        if match_basis == 'target':
            return '_번역기준'
        return '_원문기준'

    @staticmethod
    def _normalize_results_object(results: Dict[str, any]) -> Dict[str, any]:
        """호환성: 결과가 wrapper(primary/secondary)인지 단일 결과인지 통일."""
        if isinstance(results, dict) and 'primary' in results and isinstance(results.get('primary'), dict):
            return results
        return {
            'primary_basis': (results or {}).get('summary', {}).get('match_basis', 'source'),
            'primary': results,
        }
    
    def log_detailed_differences(self, text_gt: str, text_pred: str, output_file: str):
        """모든 차이점을 상세히 로깅하여 파일로 저장"""
        if not text_gt or not text_pred:
            return {"error": "빈 텍스트"}
        
        detailed_diffs = []
        non_space_diffs = []
        space_diffs = []
        
        matcher = difflib.SequenceMatcher(None, text_gt, text_pred)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                gt_text = text_gt[i1:i2]
                pred_text = text_pred[j1:j2]
                
                diff_entry = {
                    'operation': tag,
                    'position': i1,
                    'gt_text': gt_text,
                    'pred_text': pred_text,
                    'context_before': text_gt[max(0, i1-30):i1],
                    'context_after': text_gt[i2:i2+30]
                }
                
                detailed_diffs.append(diff_entry)
                
                # 공백 외 차이점 분류
                if gt_text.strip() != pred_text.strip():
                    non_space_diffs.append(diff_entry)
                else:
                    space_diffs.append(diff_entry)
        
        # 로그 출력
        print(f"\n🔍 상세 차이점 분석:")
        print(f"  전체 차이점: {len(detailed_diffs)}개")
        print(f"  공백 외 차이점: {len(non_space_diffs)}개")
        print(f"  공백 차이점: {len(space_diffs)}개")
        
        if non_space_diffs:
            print(f"\n📝 공백 외 차이점 발견 ({len(non_space_diffs)}개):")
            for i, diff in enumerate(non_space_diffs[:10], 1):  # 상위 10개만 출력
                op_name = {'delete': '삭제', 'insert': '삽입', 'replace': '교체'}.get(diff['operation'], diff['operation'])
                print(f"   {i}. [{op_name}] 위치 {diff['position']}: '{diff['gt_text']}' → '{diff['pred_text']}'")
                print(f"      컨텍스트: ...{diff['context_before']}[{op_name}]{diff['context_after']}...")
            
            if len(non_space_diffs) > 10:
                print(f"   ... 및 {len(non_space_diffs) - 10}개 추가")
        
        print(f"📝 공백 차이점: {len(space_diffs)}개 (요약 생략)")
        
        # 파일로 저장
        result = {
            'total_differences': len(detailed_diffs),
            'non_space_differences': len(non_space_diffs),
            'space_only_differences': len(space_diffs),
            'similarity_ratio': matcher.ratio(),
            'detailed_differences': detailed_diffs,
            'non_space_differences_list': non_space_diffs,
            'space_differences_list': space_diffs
        }
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
        
    def log(self, message: str):
        """실행 로그 기록"""
        print(message)
        self.execution_log.append(message)

    def grade_with_thresholds(self, overall: Dict[str, float]) -> Dict[str, any]:
        """프로젝트 임계값을 적용해 지표별/전체 등급 산출"""
        if not self.thresholds:
            return {}
        levels = self.thresholds['levels']
        metrics = self.thresholds['metrics']
        per_metric = {}
        # 등급 함수
        def label_for(value: float, lv: Dict[str, float]) -> str:
            if value is None:
                return 'below'
            if value >= lv['top']:
                return 'top'
            if value >= lv['recommended']:
                return 'recommended'
            if value >= lv['min']:
                return 'min'
            return 'below'
        # 지표별
        for m in metrics:
            v = overall.get(f'avg_{m}')
            lv_min = levels['min'].get(m, float('inf'))
            lv_rec = levels['recommended'].get(m, float('inf'))
            lv_top = levels['top'].get(m, float('inf'))
            lab = label_for(v, {'min': lv_min, 'recommended': lv_rec, 'top': lv_top})
            per_metric[m] = {'value': v, 'label': lab, 'min': lv_min, 'recommended': lv_rec, 'top': lv_top}
        # 전체 등급: 모든 핵심 지표의 최소 등급(보수적)으로 결정
        order = {'below': 0, 'min': 1, 'recommended': 2, 'top': 3}
        overall_label = min((order[per_metric[m]['label']] for m in metrics), default=0)
        # 역매핑
        inv = {v: k for k, v in order.items()}
        return {
            'project': self.project,
            'unit': self.thresholds.get('unit'),
            'per_metric': per_metric,
            'overall_label': inv.get(overall_label, 'below'),
        }
        
    def load_data(self):
        """데이터 파일들 로드"""
        try:
            self.log(f"정답 파일 로딩: {self.ground_truth_file}")
            self.gt_data = pd.read_excel(self.ground_truth_file)
            
            self.log(f"예측 파일 로딩: {self.prediction_file}")
            if self.prediction_file.endswith('.xlsx'):
                self.pred_data = pd.read_excel(self.prediction_file)
            else:
                self.pred_data = pd.read_csv(self.prediction_file)
                
            self.log(f"정답 데이터: {len(self.gt_data)}행")
            self.log(f"예측 데이터: {len(self.pred_data)}행")
            
        except Exception as e:
            self.log(f"데이터 로딩 오류: {e}")
            sys.exit(1)
    
    def normalize_text(self, text: str) -> str:
        """그룹화용 최소 정리: 앞뒤 공백만 제거, 내부 개행을 공백으로 치환 (내부 공백 보존)"""
        if pd.isna(text):
            return ""
        text = str(text).strip()
        # 내부 개행을 공백으로 치환 (내부 공백은 보존)
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text
    
    def group_by_sentence_id(self, data: pd.DataFrame) -> Dict[int, List[Dict[str, str]]]:
        """문장식별자별로 원문과 번역문을 그룹화"""
        grouped = defaultdict(list)
        
        sentence_col = None
        source_col = None
        target_col = None
        
        # 컬럼명 자동 감지
        for col in data.columns:
            if '문장식별자' in str(col) or 'sentence' in str(col).lower():
                sentence_col = col
            elif '문단식별자' in str(col) or 'paragraph' in str(col).lower():
                if sentence_col is None:  # 문장식별자가 없으면 문단식별자 사용
                    sentence_col = col
            elif '구식별자' in str(col) or 'phrase' in str(col).lower():
                if sentence_col is None:  # 문장식별자나 문단식별자가 없으면 구식별자 사용
                    sentence_col = col
            elif '원문' in str(col) or 'source' in str(col).lower() or 'original' in str(col).lower():
                source_col = col
            elif '번역문' in str(col) or 'target' in str(col).lower() or 'translation' in str(col).lower():
                target_col = col
        
        if sentence_col is None or source_col is None or target_col is None:
            self.log(f"필요한 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(data.columns)}")
            self.log(f"필요 컬럼: 문장식별자 또는 문단식별자 또는 구식별자, 원문, 번역문")
            sys.exit(1)
            
        # 라벨 결정: 문단/문장/구
        self.group_id_col_name = str(sentence_col)
        if ('문단식별자' in str(sentence_col)) or ('paragraph' in str(sentence_col).lower()):
            self.group_label = '문단'
        elif ('구식별자' in str(sentence_col)) or ('phrase' in str(sentence_col).lower()):
            self.group_label = '구'
        else:
            self.group_label = '문장'

        self.log(f"사용 컬럼: {self.group_label}식별자={sentence_col}, 원문={source_col}, 번역문={target_col}")
        
        for _, row in data.iterrows():
            sentence_id = int(row[sentence_col])
            source_text = self.normalize_text(row[source_col])
            target_text = self.normalize_text(row[target_col])
            
            if source_text or target_text:  # 빈 텍스트가 아닌 경우만
                grouped[sentence_id].append({
                    'source': source_text,
                    'target': target_text
                })
        
        return grouped

    def group_by_row(self, data: pd.DataFrame) -> Dict[int, List[Dict[str, str]]]:
        """행(로우) 단위로 1:1 매칭 그룹 구성: 각 행을 하나의 단위로 평가"""
        grouped = {}
        # 원문/번역문 컬럼 자동 감지(문장식별자 무시)
        source_col = None
        target_col = None
        for col in data.columns:
            if ('원문' in str(col)) or ('source' in str(col).lower()) or ('original' in str(col).lower()):
                source_col = col if source_col is None else source_col
            if ('번역문' in str(col)) or ('target' in str(col).lower()) or ('translation' in str(col).lower()):
                target_col = col if target_col is None else target_col
        if source_col is None or target_col is None:
            self.log(f"필요한 컬럼을 찾을 수 없습니다(행 단위). 사용 가능한 컬럼: {list(data.columns)}")
            self.log(f"필요 컬럼: 원문, 번역문")
            sys.exit(1)
        for idx, row in data.iterrows():
            sid = int(idx) + 1  # 1-based id
            source_text = self.normalize_text(row[source_col])
            target_text = self.normalize_text(row[target_col])
            grouped[sid] = [{'source': source_text, 'target': target_text}]
        return grouped

    def _detect_source_target_cols(self, data: pd.DataFrame) -> Tuple[str | None, str | None]:
        """원문/번역문 컬럼 자동 감지 (전역 무결성 체크용 간단 탐지)"""
        source_col = None
        target_col = None
        for col in data.columns:
            c = str(col).lower()
            if source_col is None and (('원문' in str(col)) or ('source' in c) or ('original' in c)):
                source_col = col
            if target_col is None and (('번역문' in str(col)) or ('target' in c) or ('translation' in c)):
                target_col = col
        return source_col, target_col

    def compute_global_text_integrity(self) -> Dict[str, float | int | bool]:
        """데이터 전체를 이어붙여 전역 텍스트 무결성(길이 Δ, 유사도, 일치 여부) 측정"""
        if self.gt_data is None or self.pred_data is None:
            return {}
        # 컬럼 탐지
        gt_src_col, gt_tgt_col = self._detect_source_target_cols(self.gt_data)
        pd_src_col, pd_tgt_col = self._detect_source_target_cols(self.pred_data)
        if not gt_src_col or not gt_tgt_col or not pd_src_col or not pd_tgt_col:
            return {}
        # 전역 결합 (공백 삽입 없이)
        def combine(df: pd.DataFrame, col: str) -> str:
            try:
                return ''.join(df[col].astype(str).fillna(''))
            except Exception:
                return ''
        gt_src_all = combine(self.gt_data, gt_src_col)
        gt_tgt_all = combine(self.gt_data, gt_tgt_col)
        pd_src_all = combine(self.pred_data, pd_src_col)
        pd_tgt_all = combine(self.pred_data, pd_tgt_col)
        # 문자 단위 diff 요약 계산
        import difflib
        def diff_summary(a: str, b: str):
            sm = difflib.SequenceMatcher(a=a, b=b)
            ins = 0
            dele = 0
            repl = 0
            first_a = None
            first_b = None
            
            # 🆕 모든 차이점 상세 로깅
            all_differences = []
            non_space_differences = []
            space_differences = []
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    continue
                    
                if first_a is None:
                    first_a, first_b = i1, j1
                    
                # 차이점 상세 정보 수집
                gt_text = a[i1:i2] if a else ""
                pred_text = b[j1:j2] if b else ""
                
                diff_detail = {
                    'operation': tag,
                    'position_gt': i1,
                    'position_pred': j1,
                    'gt_text': gt_text,
                    'pred_text': pred_text,
                    'context_before': a[max(0, i1-30):i1] if a else "",
                    'context_after': a[i2:i2+30] if a else ""
                }
                
                all_differences.append(diff_detail)
                
                # 공백 외 차이점 분류
                if gt_text.strip() != pred_text.strip():
                    non_space_differences.append(diff_detail)
                else:
                    space_differences.append(diff_detail)
                
                if tag == 'insert':
                    ins += (j2 - j1)
                elif tag == 'delete':
                    dele += (i2 - i1)
                elif tag == 'replace':
                    # 치환 길이는 두 구간 중 더 긴 길이로 근사
                    repl += max(i2 - i1, j2 - j1)
            
            # 첫 차이 스니펫
            def ctx(s: str, idx: int, width: int = 20) -> str:
                if idx is None:
                    return ''
                start = max(0, idx - width)
                end = min(len(s), idx + width)
                snippet = s[start:end]
                return snippet
            
            return {
                'insert': ins,
                'delete': dele,
                'replace': repl,
                'first_diff_a_idx': (-1 if first_a is None else first_a),
                'first_diff_b_idx': (-1 if first_b is None else first_b),
                'first_diff_a_ctx': ctx(a, first_a),
                'first_diff_b_ctx': ctx(b, first_b),
                # 🆕 완전한 차이점 상세 정보
                'all_differences': all_differences,
                'non_space_differences': non_space_differences, 
                'space_differences': space_differences,
                'total_diff_count': len(all_differences),
                'non_space_diff_count': len(non_space_differences),
                'space_diff_count': len(space_differences),
                'similarity_ratio': sm.ratio()
            }
        # 길이 및 유사도
        src_len_gt = len(gt_src_all)
        src_len_pd = len(pd_src_all)
        tgt_len_gt = len(gt_tgt_all)
        tgt_len_pd = len(pd_tgt_all)
        src_delta = src_len_gt - src_len_pd
        tgt_delta = tgt_len_gt - tgt_len_pd
        src_sim = self.calculate_text_similarity(gt_src_all, pd_src_all)
        tgt_sim = self.calculate_text_similarity(gt_tgt_all, pd_tgt_all)
        if self._lenient_source_match:
            src_eq = (self.normalize_for_matching(gt_src_all) == self.normalize_for_matching(pd_src_all))
            tgt_eq = (self.normalize_for_matching(gt_tgt_all) == self.normalize_for_matching(pd_tgt_all))
        else:
            src_eq = gt_src_all == pd_src_all
            tgt_eq = gt_tgt_all == pd_tgt_all
        # diff 요약
        src_diff = diff_summary(gt_src_all, pd_src_all)
        tgt_diff = diff_summary(gt_tgt_all, pd_tgt_all)
        # 결과
        integrity = {
            'global_source_len_gt': src_len_gt,
            'global_source_len_pred': src_len_pd,
            'global_source_delta': src_delta,
            'global_target_len_gt': tgt_len_gt,
            'global_target_len_pred': tgt_len_pd,
            'global_target_delta': tgt_delta,
            'global_source_text_similarity': src_sim,
            'global_target_text_similarity': tgt_sim,
            'global_source_text_match': float(1.0 if src_eq else 0.0),
            'global_target_text_match': float(1.0 if tgt_eq else 0.0),
            # 문자 단위 diff 요약 (원문)
            'global_source_ops_insert': int(src_diff['insert']),
            'global_source_ops_delete': int(src_diff['delete']),
            'global_source_ops_replace': int(src_diff['replace']),
            'global_source_first_diff_index': int(src_diff['first_diff_a_idx']),
            'global_source_first_diff_context_gt': src_diff['first_diff_a_ctx'],
            'global_source_first_diff_context_pred': src_diff['first_diff_b_ctx'],
            # 🆕 원문 상세 차이점 정보
            'global_source_detailed_differences': src_diff['all_differences'],
            'global_source_non_space_diffs': src_diff['non_space_differences'],
            'global_source_space_diffs': src_diff['space_differences'],
            'global_source_total_diff_count': src_diff['total_diff_count'],
            'global_source_non_space_count': src_diff['non_space_diff_count'],
            'global_source_space_count': src_diff['space_diff_count'],
            # 문자 단위 diff 요약 (번역)
            'global_target_ops_insert': int(tgt_diff['insert']),
            'global_target_ops_delete': int(tgt_diff['delete']),
            'global_target_ops_replace': int(tgt_diff['replace']),
            'global_target_first_diff_index': int(tgt_diff['first_diff_a_idx']),
            'global_target_first_diff_context_gt': tgt_diff['first_diff_a_ctx'],
            'global_target_first_diff_context_pred': tgt_diff['first_diff_b_ctx'],
            # 🆕 번역문 상세 차이점 정보
            'global_target_detailed_differences': tgt_diff['all_differences'],
            'global_target_non_space_diffs': tgt_diff['non_space_differences'],
            'global_target_space_diffs': tgt_diff['space_differences'],
            'global_target_total_diff_count': tgt_diff['total_diff_count'],
            'global_target_non_space_count': tgt_diff['non_space_diff_count'],
            'global_target_space_count': tgt_diff['space_diff_count'],
        }
        self._global_integrity = integrity
        return integrity
    
    def normalize_for_matching(self, text: str) -> str:
        """기존 호환용: 공백+구두점 제거 후 소문자 (deprecated, 내부 호환 유지)."""
        if pd.isna(text):
            return ""
        import re
        normalized = str(text).strip()
        normalized = re.sub(r'[\s\t\n\r]+', '', normalized)
        normalized = re.sub(r'[。，、；：！？""''「」『』（）〈〉《》【】〔〕，ㆍ·…‧‥]+', '', normalized)
        return normalized.lower()

    def _normalize_by_mode(self, text: str) -> str:
        """lenient 모드(off/space/space_punct)에 따른 비교용 정규화"""
        if pd.isna(text):
            return ""
        s = str(text).strip()
        
        # [-텍스트] 패턴 제거 (ignore_brackets 옵션)
        if getattr(self, '_ignore_brackets', False):
            import re
            s = re.sub(r'\[-[^\]]*\]', '', s)
        
        if self._lenient_mode == 'off':
            return s
        import re
        if self._lenient_mode == 'space':
            return re.sub(r'[\s\t\n\r]+', '', s)
        # space_punct
        s = re.sub(r'[\s\t\n\r]+', '', s)
        s = re.sub(r'[。，、；：！？""''「」『』（）〈〉《》【】〔〕，ㆍ·…‧‥]+', '', s)
        return s.lower()

    # normalize_by_policy: 사용처가 없어 제거함 (관대한 비교는 normalize_for_matching 사용)
    
    def tokenize_korean(self, text: str) -> List[str]:
        """한글 어절(어휘 단위) 토큰화"""
        # 공백으로 분리된 어절 단위로 토큰화
        # 한글, 한자, 영문, 숫자를 포함하여 공백/구두점으로 분리
        import re
        # 공백과 일부 구두점으로 분리 (숫자 포함)
        tokens = re.findall(r'[\w가-힣一-鿿]+', text)
        return tokens

    def calculate_token_level_matching(self, gt_segments: List[Dict[str, str]], pred_segments: List[Dict[str, str]]) -> List[Tuple[int, int, float, float]]:
        """토큰(어절) 레벨 n:m 매칭
        
        Returns:
            List of (gt_start_idx, gt_end_idx, pred_start_idx, pred_end_idx, source_overlap_ratio, translation_similarity)
            - GT 행 [gt_start_idx:gt_end_idx+1]가 Pred 행 [pred_start_idx:pred_end_idx+1]과 매칭됨
            - source_overlap_ratio: 어절 기준 겹침 비율 (Jaccard 유사도)
        """
        gt_sources = [seg['source'] for seg in gt_segments]
        gt_targets = [seg['target'] for seg in gt_segments]
        pred_sources = [seg['source'] for seg in pred_segments]
        pred_targets = [seg['target'] for seg in pred_segments]
        
        # 각 행을 어절로 토큰화
        gt_source_tokens = [self.tokenize_korean(src) for src in gt_sources]
        gt_target_tokens = [self.tokenize_korean(tgt) for tgt in gt_targets]
        pred_source_tokens = [self.tokenize_korean(src) for src in pred_sources]
        pred_target_tokens = [self.tokenize_korean(tgt) for tgt in pred_targets]
        
        gt_len = len(gt_segments)
        pred_len = len(pred_segments)
        
        # DP 기반 최적 n:m 매칭
        # dp[i][j] = 최대 누적 겹침 유사도 (GT [0:i] vs Pred [0:j])
        # 추적: 각 상태의 이전 상태를 기록하여 역추적
        dp = {}
        traceback = {}
        
        def get_overlap_ratio(gt_toks, pred_toks) -> float:
            """어절 집합 기준 Jaccard 유사도"""
            gt_set = set(gt_toks)
            pred_set = set(pred_toks)
            if not gt_set and not pred_set:
                return 1.0
            if not gt_set or not pred_set:
                return 0.0
            intersection = len(gt_set & pred_set)
            union = len(gt_set | pred_set)
            return intersection / union if union > 0 else 0.0
        
        # DP 초기화
        dp[(0, 0)] = 0.0
        traceback[(0, 0)] = None
        
        # DP 계산
        for i in range(gt_len + 1):
            for j in range(pred_len + 1):
                if i == 0 and j == 0:
                    continue
                
                best_score = -1.0
                best_prev = None
                
                # 1) 현재 상태 도달 방법
                # - (i-1, j)에서: GT 행 i-1 skip (매칭 안 함)
                if i > 0 and (i-1, j) in dp:
                    score = dp[(i-1, j)] - 0.5  # 페널티: 미매칭 GT
                    if score > best_score:
                        best_score = score
                        best_prev = (i-1, j)
                
                # - (i, j-1)에서: Pred 행 j-1 skip (매칭 안 함)
                if j > 0 and (i, j-1) in dp:
                    score = dp[(i, j-1)] - 0.5  # 페널티: 미생성 Pred
                    if score > best_score:
                        best_score = score
                        best_prev = (i, j-1)
                
                # - (i', j')에서: GT [i':i] vs Pred [j':j] 매칭 (모든 i', j' 시도)
                for i_prev in range(i):
                    for j_prev in range(j):
                        if (i_prev, j_prev) not in dp:
                            continue
                        
                        # GT [i_prev:i], Pred [j_prev:j]의 어절 겹침
                        gt_toks = []
                        for idx in range(i_prev, i):
                            gt_toks.extend(gt_source_tokens[idx])
                        
                        pred_toks = []
                        for idx in range(j_prev, j):
                            pred_toks.extend(pred_source_tokens[idx])
                        
                        overlap = get_overlap_ratio(gt_toks, pred_toks)
                        score = dp[(i_prev, j_prev)] + overlap
                        
                        if score > best_score:
                            best_score = score
                            best_prev = (i_prev, j_prev)
                
                if best_score >= 0:
                    dp[(i, j)] = best_score
                    traceback[(i, j)] = best_prev
        
        # 역추적: 최적 매칭 경로 복원
        matchings = []
        current = (gt_len, pred_len)
        
        while current and traceback.get(current):
            prev = traceback[current]
            i, j = current
            i_prev, j_prev = prev
            
            # 매칭 구간 추가
            if i > i_prev and j > j_prev:
                # GT [i_prev:i]와 Pred [j_prev:j] 매칭
                gt_toks = []
                for idx in range(i_prev, i):
                    gt_toks.extend(gt_source_tokens[idx])
                pred_toks = []
                for idx in range(j_prev, j):
                    pred_toks.extend(pred_source_tokens[idx])
                
                overlap = get_overlap_ratio(gt_toks, pred_toks)
                matchings.append((i_prev, i - 1, j_prev, j - 1, overlap))
            
            current = prev
        
        matchings.reverse()
        return matchings

    def calculate_sentence_accuracy(self, gt_segments: List[Dict[str, str]], pred_segments: List[Dict[str, str]], sentence_id: int, *, mismatch_container: List[Dict[str, any]] | None = None) -> Dict[str, float]:
        """단일 문장의 분할 정확도 계산 (토큰 레벨 n:m 매칭 + 번역문 평가)

        mismatch_container를 주입하면 원문 불일치 로그를 외부 리스트에 모아 메인/추가 평가를 분리할 수 있다.
        """
        if mismatch_container is None:
            mismatch_container = self.source_mismatch_details
        # 원문과 번역문 분리
        gt_sources = [seg['source'] for seg in gt_segments]
        gt_targets = [seg['target'] for seg in gt_segments]
        pred_sources = [seg['source'] for seg in pred_segments]
        pred_targets = [seg['target'] for seg in pred_segments]

        # 전체 텍스트 복원
        gt_source_full = "".join(gt_sources)
        gt_target_full = "".join(gt_targets)
        pred_source_full = "".join(pred_sources)
        pred_target_full = "".join(pred_targets)

        # 길이 및 Δ (입력-출력, 양수=손실, 음수=증가)
        source_text_len_gt = len(gt_source_full)
        source_text_len_pred = len(pred_source_full)
        target_text_len_gt = len(gt_target_full)
        target_text_len_pred = len(pred_target_full)
        source_text_len_delta = source_text_len_gt - source_text_len_pred
        target_text_len_delta = target_text_len_gt - target_text_len_pred

        # 🚨 원문 일치 여부 확인 및 불일치 로깅 (엄격/관대 병렬 계산)
        strict_source_eq = (gt_source_full == pred_source_full)
        lenient_source_eq = (self._normalize_by_mode(gt_source_full) == self._normalize_by_mode(pred_source_full))
        source_text_match = (lenient_source_eq if self._lenient_source_match else strict_source_eq)
        if self._lenient_source_match and self._warn_lenient_mismatch and (lenient_source_eq and not strict_source_eq):
            self.log(f"   ⚠️ 관대 일치이나 엄격 불일치(원문): 문장 {sentence_id}")
        if not source_text_match:
            self.log(f"⚠️  문장 {sentence_id}: 원문 불일치 감지!")
            self.log(f"   정답 원문: '{gt_source_full}'")
            self.log(f"   예측 원문: '{pred_source_full}'")
            # 원문이 다르면 길이 차이도 로깅
            len_diff = len(pred_source_full) - len(gt_source_full)
            if len_diff != 0:
                self.log(f"   길이 차이: {len_diff:+d} 글자")

            # 상세 정보 저장
            mismatch_container.append({
                'sentence_id': sentence_id,
                'gt_source': gt_source_full,
                'pred_source': pred_source_full,
                'length_diff': len_diff,
                'similarity': self.calculate_text_similarity(gt_source_full, pred_source_full)
            })

        # 번역문 일치 여부(엄격/관대)
        strict_target_eq = (gt_target_full == pred_target_full)
        lenient_target_eq = (self._normalize_by_mode(gt_target_full) == self._normalize_by_mode(pred_target_full))
        target_text_match = (lenient_target_eq if self._lenient_source_match else strict_target_eq)
        # 필요 시 경고(관대=일치, 엄격=불일치)
        if self._lenient_source_match and self._warn_lenient_mismatch and (lenient_target_eq and not strict_target_eq):
            self.log(f"   ⚠️ 관대 일치이나 엄격 불일치(번역): 문장 {sentence_id}")
        # 전체 텍스트 일치 (원문 + 번역문 모두)
        text_match = source_text_match and target_text_match

        # 세그먼트 수 일치 여부
        segment_count_match = len(gt_segments) == len(pred_segments)

        # 완전 일치 (순서와 내용 모두 일치)
        exact_match = gt_segments == pred_segments

        # 🎯 핵심: 토큰(어절) 레벨 n:m 매칭 (문장 경계 무시, 어절 기준 매칭)
        self.log(f"   🔄 문장 {sentence_id}: 토큰 레벨 n:m 매칭 시작...")
        
        # 토큰 레벨 매칭 수행
        token_matchings = self.calculate_token_level_matching(gt_segments, pred_segments)
        
        # 토큰 매칭 결과를 행 단위 정확도로 변환
        correct_translation_pairs = 0
        translation_similarities = []
        source_similarities = []
        matched_pairs = []
        total_matched_gt_segments = 0
        total_matched_pred_segments = 0
        
        for gt_start, gt_end, pred_start, pred_end, source_overlap in token_matchings:
            # GT [gt_start:gt_end+1]과 Pred [pred_start:pred_end+1]이 매칭됨
            gt_indices = list(range(gt_start, gt_end + 1))
            pred_indices = list(range(pred_start, pred_end + 1))
            
            total_matched_gt_segments += len(gt_indices)
            total_matched_pred_segments += len(pred_indices)
            
            # 매칭된 GT와 Pred 번역문을 합치기
            gt_target_matched = "".join([gt_targets[i] for i in gt_indices])
            pred_target_matched = "".join([pred_targets[i] for i in pred_indices])

            # 매칭된 GT와 Pred 원문도 합치기 (원문 평균 유사도 산출용)
            gt_source_matched = "".join([gt_sources[i] for i in gt_indices])
            pred_source_matched = "".join([pred_sources[i] for i in pred_indices])
            
            # 번역문 비교
            gt_target_norm = self._normalize_by_mode(gt_target_matched)
            pred_target_norm = self._normalize_by_mode(pred_target_matched)
            
            if gt_target_matched == pred_target_matched or gt_target_norm == pred_target_norm:
                correct_translation_pairs += 1
                translation_similarities.append(1.0)
            else:
                similarity = self.calculate_text_similarity(gt_target_matched, pred_target_matched)
                translation_similarities.append(similarity)
                if similarity >= 0.9:
                    correct_translation_pairs += 1

            # 원문 유사도(블록 단위)도 함께 계산
            gt_source_norm = self._normalize_by_mode(gt_source_matched)
            pred_source_norm = self._normalize_by_mode(pred_source_matched)
            if gt_source_matched == pred_source_matched or gt_source_norm == pred_source_norm:
                source_similarities.append(1.0)
            else:
                source_similarities.append(self.calculate_text_similarity(gt_source_matched, pred_source_matched))
            
            # 매칭 정보 저장 (n:m으로 확장)
            matched_pairs.append({
                'gt_indices': gt_indices,
                'pred_indices': pred_indices,
                'gt_segs': [gt_segments[i] for i in gt_indices],
                'pred_segs': [pred_segments[i] for i in pred_indices],
                'source_match_type': 'token_level_nm',
                'source_overlap': source_overlap
            })
            
            self.log(f"     ✅ GT [{gt_start+1}:{gt_end+1}] → Pred [{pred_start+1}:{pred_end+1}] (어절 겹침: {source_overlap:.3f})")

        # 3) 정확도 지표 계산 (토큰 레벨 n:m 매칭 기반)
        total_gt_segments = len(gt_segments)
        total_pred_segments = len(pred_segments)
        matched_pairs_count = len(matched_pairs)
        
        # 토큰 레벨 매칭: 행 커버리지
        # - Precision: 매칭된 모든 행 / 전체 GT 행
        # - Recall: 매칭된 모든 행 / 전체 Pred 행
        # 주의: matched_pairs가 n:m이므로, 실제 커버된 행의 비율로 계산
        matched_gt_idxs = set()
        matched_pred_idxs = set()
        for pair in matched_pairs:
            matched_gt_idxs.update(pair['gt_indices'])
            matched_pred_idxs.update(pair['pred_indices'])
        
        num_matched_gt = len(matched_gt_idxs)
        num_matched_pred = len(matched_pred_idxs)
        
        # Precision: 매칭된 GT 비율 (미매칭 GT 페널티)
        source_precision = num_matched_gt / total_gt_segments if total_gt_segments > 0 else 0.0
        # Recall: 매칭된 Pred 비율 (미생성/미사용 Pred 페널티)
        source_recall = num_matched_pred / total_pred_segments if total_pred_segments > 0 else 0.0
        # F1
        source_f1_score = (2 * source_precision * source_recall / (source_precision + source_recall)
                           if source_precision + source_recall > 0 else 0.0)

        # 번역문 정확도 (매칭된 쌍에서만 평가)
        if matched_pairs_count > 0:
            target_accuracy = correct_translation_pairs / matched_pairs_count
            target_precision = target_accuracy  # 매칭된 쌍에서의 번역문 정확도
            target_recall = correct_translation_pairs / total_gt_segments  # 전체 대비 올바른 번역 비율
            target_f1_score = (2 * target_precision * target_recall / (target_precision + target_recall)
                               if target_precision + target_recall > 0 else 0.0)
            # 번역문 평균 유사도 계산
            target_avg_similarity = sum(translation_similarities) / len(translation_similarities) if translation_similarities else 0.0
            # 원문 평균 유사도(블록 단위) 계산
            source_avg_similarity = sum(source_similarities) / len(source_similarities) if source_similarities else 0.0
        else:
            target_accuracy = 0.0
            target_precision = 0.0
            target_recall = 0.0
            target_f1_score = 0.0
            target_avg_similarity = 0.0
            source_avg_similarity = 0.0

        # 전체 F1 점수 (원문 매칭과 번역문 정확도의 조화평균)
        f1_score = (source_f1_score + target_f1_score) / 2

        # 🆕 부분 일치 계산 (토큰 기준)
        # 1) 어절 단위 유사도 (모든 토큰 매칭 고려)
        source_overlap_scores = [pair['source_overlap'] for pair in matched_pairs]
        source_avg_overlap = sum(source_overlap_scores) / len(source_overlap_scores) if source_overlap_scores else 0.0

        # 2) 전체 텍스트 유사도
        source_text_similarity = self.calculate_text_similarity(gt_source_full, pred_source_full)
        target_text_similarity = self.calculate_text_similarity(gt_target_full, pred_target_full)

        # 3) 토큰 레벨 Jaccard 유사도 (원문)
        gt_all_tokens = set()
        pred_all_tokens = set()
        for gt_src in gt_sources:
            gt_all_tokens.update(self.tokenize_korean(gt_src))
        for pred_src in pred_sources:
            pred_all_tokens.update(self.tokenize_korean(pred_src))
        
        source_jaccard = (len(gt_all_tokens.intersection(pred_all_tokens)) /
                          len(gt_all_tokens.union(pred_all_tokens))
                          if len(gt_all_tokens.union(pred_all_tokens)) > 0 else 0.0)

        # 4) 최종 부분 일치 점수 (토큰 겹침 우선)
        source_partial_match = (source_jaccard + source_text_similarity + source_avg_overlap) / 3
        target_partial_match = target_avg_similarity  # 매칭된 쌍에서의 번역문 유사도
        partial_match = (source_partial_match + target_partial_match) / 2

        # 🆕 원문과 번역문 쌍의 통합 유사도 계산
        combined_text_similarity = (source_text_similarity + target_text_similarity) / 2
        combined_avg_similarity = (source_avg_overlap + target_avg_similarity) / 2

        # 🆕 매칭 상세 정보 로깅
        unmatched_gt = total_gt_segments - num_matched_gt
        unmatched_pred = total_pred_segments - num_matched_pred
        if unmatched_gt > 0 or unmatched_pred > 0:
            self.log(f"   문장 {sentence_id} 토큰 레벨 매칭 결과:")
            self.log(f"     - GT 행: {num_matched_gt}/{total_gt_segments}개 매칭 ({unmatched_gt}개 미매칭)")
            self.log(f"     - Pred 행: {num_matched_pred}/{total_pred_segments}개 활용 ({unmatched_pred}개 미활용)")

        return {
            'text_match': float(text_match),
            'source_text_match': float(source_text_match),
            'target_text_match': float(target_text_match),
            # 참고: 엄격/관대 판정도 함께 기록(요약에서는 기본 제외)
            'source_text_match_strict': float(strict_source_eq),
            'source_text_match_lenient': float(lenient_source_eq),
            'target_text_match_strict': float(strict_target_eq),
            'target_text_match_lenient': float(lenient_target_eq),
            # 길이 및 Δ(행/그룹 단위)
            'source_text_len_gt': float(source_text_len_gt),
            'source_text_len_pred': float(source_text_len_pred),
            'source_text_len_delta': float(source_text_len_delta),
            'target_text_len_gt': float(target_text_len_gt),
            'target_text_len_pred': float(target_text_len_pred),
            'target_text_len_delta': float(target_text_len_delta),
            'segment_count_match': float(segment_count_match),
            'exact_match': float(exact_match),
            'partial_match': partial_match,
            'source_partial_match': source_partial_match,
            'target_partial_match': target_partial_match,
            'precision': (source_precision + target_precision) / 2,
            'source_precision': source_precision,
            'target_precision': target_precision,
            'recall': (source_recall + target_recall) / 2,
            'source_recall': source_recall,
            'target_recall': target_recall,
            'f1_score': f1_score,
            'source_f1_score': source_f1_score,
            'target_f1_score': target_f1_score,
            'gt_segments': len(gt_segments),
            'pred_segments': len(pred_segments),
            # 🆕 토큰 레벨 n:m 매칭 지표들
            'matched_pairs': matched_pairs_count,
            'matched_gt_count': num_matched_gt,
            'matched_pred_count': num_matched_pred,
            'correct_translation_pairs': correct_translation_pairs,
            'token_based_matches': matched_pairs_count,  # 토큰 레벨 n:m 매칭 쌍 수
            # 🆕 번역문 평가 지표들
            'target_accuracy': target_accuracy,
            'target_avg_similarity': target_avg_similarity,
            # 🆕 부분 일치 세부 지표들
            'source_jaccard': source_jaccard,
            'source_text_similarity': source_text_similarity,
            'target_text_similarity': target_text_similarity,
            'source_avg_overlap': source_avg_overlap,
            'source_avg_similarity': source_avg_similarity,
            # 🆕 원문과 번역문 쌍의 통합 유사도
            'combined_text_similarity': combined_text_similarity,
            'combined_avg_similarity': combined_avg_similarity
        }
    
    def find_best_alignment_by_source(self, gt_sources: List[str], pred_sources: List[str]) -> List[Tuple[int, int, float]]:
        """원문 기준 순서대로 최적 정렬 찾기 (PA 방식)"""
        self.log("🔄 원문 기준 순서대로 정렬 계산 중...")
        self.log("   💡 매칭 방식: 정답 원문 순서대로 가장 유사한 예측 원문 찾기")
        
        gt_len = len(gt_sources)
        pred_len = len(pred_sources)
        
        alignments = []
        used_pred = set()  # 이미 매칭된 예측 행 추적
        
        # 정답 원문을 순서대로 처리 (gt_sources[0], gt_sources[1], ...)
        for gt_idx, gt_source in enumerate(gt_sources):
            best_pred_idx = -1
            best_similarity = 0.0
            
            self.log(f"   🔍 정답 행 {gt_idx+1} 처리 중...")
            
            # 1단계: 현재 위치 근처에서 우선 검색 (±2 범위)
            # 순서가 크게 바뀌지 않을 가능성이 높으므로
            search_start = max(0, gt_idx - 2)
            search_end = min(pred_len, gt_idx + 3)
            
            for pred_idx in range(search_start, search_end):
                if pred_idx in used_pred:  # 이미 다른 정답과 매칭된 예측은 제외
                    continue
                    
                pred_source = pred_sources[pred_idx]
                similarity = self.calculate_text_similarity(gt_source, pred_source)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pred_idx = pred_idx
            
            # 2단계: 근처에서 충분히 유사한 것을 못 찾으면 전체 검색
            if best_similarity < 0.1:  # skip_threshold
                self.log(f"     📡 근처 검색 실패, 전체 범위 검색...")
                for pred_idx in range(pred_len):
                    if pred_idx in used_pred:  # 이미 매칭된 예측은 제외
                        continue
                        
                    pred_source = pred_sources[pred_idx]
                    similarity = self.calculate_text_similarity(gt_source, pred_source)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pred_idx = pred_idx
            
            # 3단계: 매칭 결과 저장
            if best_pred_idx >= 0 and best_similarity >= 0.1:  # skip_threshold
                # 원문-번역문 쌍이 함께 매칭됨
                alignments.append((gt_idx, best_pred_idx, best_similarity))
                used_pred.add(best_pred_idx)  # 매칭된 예측 행 마킹
                self.log(f"     ✅ 정답 {gt_idx+1} → 예측 {best_pred_idx+1} (원문 유사도: {best_similarity:.3f})")
            else:
                # 매칭 실패 - 해당 정답 원문-번역문 쌍에 대응하는 예측이 없음
                alignments.append((gt_idx, -1, 0.0))
                self.log(f"     ❌ 정답 {gt_idx+1} → 매칭 없음 (최고 유사도: {best_similarity:.3f})")
        
        matched_count = sum(1 for _, pred_idx, _ in alignments if pred_idx >= 0)
        self.log(f"✅ 원문 기준 순서대로 정렬 완료: {matched_count}/{gt_len} 쌍 매칭")
        self.log(f"   📋 매칭 원리: 각 정답 [원문,번역문] 쌍에 대해 원문이 가장 유사한 예측 [원문,번역문] 쌍 찾기")
        return alignments

    def find_best_alignment_by_source_with_dup(self, gt_sources, pred_sources, k=None, allow_dup=None, search_window=None) -> List[Tuple[int, int, float]]:
        """원문 기준 정렬(중복 허용 + 단조 제약 + 검색 윈도우)
        - 각 GT 인덱스에 대해 최대 k개의 예측을 순서 역행 없이 매칭
        - search_window: 각 GT 행 주변 ±N 범위 내에서만 검색 (None이면 전체)
        - 임계값 0.1 이상인 상위 후보를 선택
        
        Args:
            gt_sources: GT 원문 리스트 또는 (id, segments) 튜플 리스트
            pred_sources: Pred 원문 리스트 또는 (id, segments) 튜플 리스트
            k: 각 GT당 최대 매칭 개수 (None이면 _max_dup_per_gt 사용)
            allow_dup: 중복 매칭 허용 여부 (None이면 True)
            search_window: 검색 윈도우 크기 (None이면 제한 없음)
        """
        # 입력 타입 처리: 튜플 리스트면 원문 추출
        if gt_sources and isinstance(gt_sources[0], tuple):
            gt_texts = [segs[0]['source'] if segs else '' for _, segs in gt_sources]
        else:
            gt_texts = gt_sources
        
        if pred_sources and isinstance(pred_sources[0], tuple):
            pred_texts = [segs[0]['source'] if segs else '' for _, segs in pred_sources]
        else:
            pred_texts = pred_sources
        
        # 파라미터 설정
        if k is None:
            k = max(1, int(getattr(self, '_max_dup_per_gt', 1)))
        if allow_dup is None:
            allow_dup = True
        monotonic = bool(getattr(self, '_monotonic_alignment', True))
        
        self.log(f"🔄 원문 기준 정렬(중복 k={k}, 단조={monotonic}, 윈도우={search_window}) 계산 중...")
        gt_len = len(gt_texts)
        pred_len = len(pred_texts)
        alignments: List[Tuple[int, int, float]] = []
        used_pred = set()
        cur_pred_start = 0
        
        for gt_idx, gt_source in enumerate(gt_texts):
            # 검색 범위 결정
            if search_window is not None:
                # 윈도우 기반: gt_idx ±search_window 범위
                window_start = max(0, gt_idx - search_window)
                window_end = min(pred_len, gt_idx + search_window + 1)
                if monotonic:
                    # 단조 제약과 윈도우 제약 교집합
                    j_range = range(max(cur_pred_start, window_start), window_end)
                else:
                    j_range = range(window_start, window_end)
            else:
                # 윈도우 없음: 기존 로직
                j_range = range(cur_pred_start, pred_len) if monotonic else range(0, pred_len)
            
            candidates = []
            for pred_idx in j_range:
                if not allow_dup and pred_idx in used_pred:
                    continue
                sim = self.calculate_text_similarity(gt_source, pred_texts[pred_idx])
                if sim >= 0.1:  # skip threshold
                    candidates.append((pred_idx, sim))
            
            # 상위 k개 선택 (유사도 desc, 인덱스 asc 안정성)
            candidates.sort(key=lambda x: (-x[1], x[0]))
            selected = candidates[:k]
            
            if selected:
                for pred_idx, sim in selected:
                    alignments.append((gt_idx, pred_idx, sim))
                    if not allow_dup:
                        used_pred.add(pred_idx)
                if monotonic:
                    cur_pred_start = max(cur_pred_start, max(p for p, _ in selected) + 1)
            else:
                # 매칭 실패 기록
                alignments.append((gt_idx, -1, 0.0))
        
        matched_count = sum(1 for _, j, s in alignments if j >= 0 and s > 0)
        self.log(f"✅ 원문 기준 정렬 완료: 매칭 {matched_count}건 (GT {gt_len}, Pred {pred_len})")
        return alignments

    # ---- 토씨(조사) 힌트 확장 매칭 ----
    def _ensure_kiwi(self):
        """Kiwi 인스턴스를 지연 초기화. 실패 시 None 유지(힌트 미사용)."""
        if self._kiwi is not None:
            return self._kiwi
        try:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi()
        except Exception:
            self._kiwi = None
        return self._kiwi

    def _extract_ko_particles(self, text: str) -> List[str]:
        """한글 조사 후보를 추출. Kiwi 사용 가능 시 태그 기반, 불가 시 빈 리스트.
        - Sejong 태그 체계 기준: JKS/JKC/JKG/JKO/JKB/JKV/JKQ/JC/JX 등
        - 무결성 보존: 입력 텍스트는 읽기만 함
        """
        if not text:
            return []
        kiwi = self._ensure_kiwi()
        if kiwi is None:
            return []
        try:
            tokens = kiwi.tokenize(str(text))
            parts = []
            for tk in tokens:
                tag = getattr(tk, 'tag', '')
                if tag.startswith('JK') or tag in ('JC', 'JX'):
                    parts.append(getattr(tk, 'form', ''))
            return parts
        except Exception:
            return []

    def _particle_overlap_score(self, a: str, b: str) -> float:
        """번역문 두 텍스트의 조사 겹침 점수(Jaccard 유사도)."""
        A = set(self._extract_ko_particles(a))
        B = set(self._extract_ko_particles(b))
        if not A and not B:
            return 0.0
        union = len(A | B)
        if union == 0:
            return 0.0
        return len(A & B) / union

    def find_best_alignment_by_source_with_hint(self, gt_segments: List[Dict[str, str]], pred_segments: List[Dict[str, str]]) -> List[Tuple[int, int, float]]:
        """원문 유사도 + 번역문 조사 겹침을 가중 결합하여 정렬.
        - 기본 검색 창: ±2, 부족하면 전체 검색(기존 로직 유지)
        - 최종 유사도 = source_sim + w * particle_overlap, [0,1]에서 적절히 동작하도록 제한
        """
        self.log("🔄 원문+조사 힌트 기반 정렬 계산 중...")
        gt_sources = [seg['source'] for seg in gt_segments]
        pred_sources = [seg['source'] for seg in pred_segments]
        gt_targets = [seg['target'] for seg in gt_segments]
        pred_targets = [seg['target'] for seg in pred_segments]

        gt_len = len(gt_sources)
        pred_len = len(pred_sources)
        alignments: List[Tuple[int, int, float]] = []
        used_pred = set()

        for gt_idx, gt_source in enumerate(gt_sources):
            best_pred_idx = -1
            best_score = 0.0
            # 1) 근처 검색(±2)
            search_start = max(0, gt_idx - 2)
            search_end = min(pred_len, gt_idx + 3)
            for pred_idx in range(search_start, search_end):
                if pred_idx in used_pred:
                    continue
                base_sim = self.calculate_text_similarity(gt_source, pred_sources[pred_idx])
                hint = self._particle_overlap_score(gt_targets[gt_idx], pred_targets[pred_idx])
                score = max(0.0, min(1.0, base_sim + self._particle_weight * hint))
                if score > best_score:
                    best_score = score
                    best_pred_idx = pred_idx
            # 2) 전역 검색(근처에서 점수가 너무 낮으면)
            if best_score < 0.1:
                for pred_idx in range(pred_len):
                    if pred_idx in used_pred:
                        continue
                    base_sim = self.calculate_text_similarity(gt_source, pred_sources[pred_idx])
                    hint = self._particle_overlap_score(gt_targets[gt_idx], pred_targets[pred_idx])
                    score = max(0.0, min(1.0, base_sim + self._particle_weight * hint))
                    if score > best_score:
                        best_score = score
                        best_pred_idx = pred_idx
            # 3) 적용(임계값 0.1)
            if best_pred_idx >= 0 and best_score >= 0.1:
                alignments.append((gt_idx, best_pred_idx, best_score))
                used_pred.add(best_pred_idx)
                self.log(f"     ✅ 정답 {gt_idx+1} → 예측 {best_pred_idx+1} (합성 유사도: {best_score:.3f})")
            else:
                alignments.append((gt_idx, -1, 0.0))
                self.log(f"     ❌ 정답 {gt_idx+1} → 매칭 없음 (최고 합성 유사도: {best_score:.3f})")

        matched = sum(1 for _, j, _ in alignments if j >= 0)
        self.log(f"✅ 원문+조사 힌트 정렬 완료: {matched}/{gt_len} 쌍 매칭")
        return alignments
    
    def smart_match_sentences_by_source_only(self, all_gt_sentences: List[Tuple[int, List[Dict[str, str]]]], 
                                           all_pred_sentences: List[Tuple[int, List[Dict[str, str]]]]) -> List[Tuple[int, int, float]]:
        """원문 기준으로만 스마트 매칭 - 문장식별자 무시"""
        self.log("🔄 원문 기준 순수 매칭 계산 중...")
        self.log("   💡 문장식별자를 완전히 무시하고 원문 유사도로만 매칭")
        
        matches = []
        used_pred_ids = set()
        
        # 정답 문장들을 순서대로 처리
        for gt_id, gt_segments in all_gt_sentences:
            gt_source_full = "".join([seg['source'] for seg in gt_segments])
            
            best_pred_id = None
            best_similarity = 0.0
            
            # 사용되지 않은 모든 예측 문장들과 비교
            for pred_id, pred_segments in all_pred_sentences:
                if pred_id in used_pred_ids:
                    continue
                    
                pred_source_full = "".join([seg['source'] for seg in pred_segments])
                similarity = self.calculate_text_similarity(gt_source_full, pred_source_full)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pred_id = pred_id
            
            # 매칭 결과 저장 (임계값 0.1 이상일 때만)
            if best_pred_id is not None and best_similarity >= 0.1:
                matches.append((gt_id, best_pred_id, best_similarity))
                used_pred_ids.add(best_pred_id)
                self.log(f"  정답문장{gt_id} ↔ 예측문장{best_pred_id} (원문 유사도: {best_similarity:.3f})")
                
                if best_similarity < 0.8:  # 낮은 유사도 경고
                    self.log(f"    ⚠️ 낮은 유사도 매칭")
            else:
                self.log(f"  정답문장{gt_id} → 매칭 실패 (최고 유사도: {best_similarity:.3f})")
        
        self.log(f"✅ 원문 기준 순수 매칭 완료: {len(matches)}개 쌍 매칭")
        return matches

    def smart_match_sentences_by_target_only(self, all_gt_sentences: List[Tuple[int, List[Dict[str, str]]]],
                                           all_pred_sentences: List[Tuple[int, List[Dict[str, str]]]]) -> List[Tuple[int, int, float]]:
        """번역문 기준 스마트 매칭 - 문장식별자 무시"""
        self.log("🔄 번역문 기준 순수 매칭 계산 중...")
        self.log("   💡 문장식별자를 무시하고 번역문 유사도로만 매칭")

        matches = []
        used_pred_ids = set()

        for gt_id, gt_segments in all_gt_sentences:
            gt_target_full = "".join([seg['target'] for seg in gt_segments])

            best_pred_id = None
            best_similarity = 0.0

            for pred_id, pred_segments in all_pred_sentences:
                if pred_id in used_pred_ids:
                    continue

                pred_target_full = "".join([seg['target'] for seg in pred_segments])
                similarity = self.calculate_text_similarity(gt_target_full, pred_target_full)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pred_id = pred_id

            if best_pred_id is not None and best_similarity >= 0.1:
                matches.append((gt_id, best_pred_id, best_similarity))
                used_pred_ids.add(best_pred_id)
                self.log(f"  정답문장{gt_id} ↔ 예측문장{best_pred_id} (번역 유사도: {best_similarity:.3f})")
                if best_similarity < 0.8:
                    self.log("    ⚠️ 낮은 유사도 매칭(번역 기준)")
            else:
                self.log(f"  정답문장{gt_id} → 매칭 실패 (번역 기준 최고 유사도: {best_similarity:.3f})")

        self.log(f"✅ 번역문 기준 순수 매칭 완료: {len(matches)}개 쌍 매칭")
        return matches
    
    def _build_matches(self, gt_grouped: Dict[int, List[Dict[str, str]]], pred_grouped: Dict[int, List[Dict[str, str]]], unit: str, match_on: str = 'source') -> Tuple[List[Tuple[int, int, float]], Dict[str, any]]:
        """매칭 전략을 캡슐화하여 재사용 (원문/번역 기준 선택)."""
        row_shift_meta = {
            'row_shift_applied': 0,
            'row_shift_overlap': 0,
            'row_shift_improved': 0,
            'row_shift_zero_eq': getattr(self, '_row_shift_zero_eq', 0),
            'row_shift_best_eq': getattr(self, '_row_shift_best_eq', 0),
            'row_shift_best_avg_sim': getattr(self, '_row_shift_best_avg_sim', 0.0),
        }

        match_label = '원문' if match_on == 'source' else '번역문'
        if unit == 'row':
            if (self.project == 'pa') or getattr(self, '_detect_row_shift', False) or match_on == 'target':
                if match_on == 'source':
                    self.log("\n• 🎯 PA/행 모드: 원문 기준 스마트 매칭 시작...")
                    self.log("   💡 문장 분할 차이를 고려하여 원문 유사도로 최적 매칭")
                    matcher = self.smart_match_sentences_by_source_only
                else:
                    self.log("\n• 🎯 행 모드: 번역문 기준 스마트 매칭 시작...")
                    self.log("   💡 문장 분할 차이를 고려하여 번역문 유사도로 최적 매칭")
                    matcher = self.smart_match_sentences_by_target_only

                all_gt_sentences = list(gt_grouped.items())
                all_pred_sentences = list(pred_grouped.items())
                final_matches = matcher(all_gt_sentences, all_pred_sentences)
                row_shift_meta.update({
                    'row_shift_applied': -999,
                    'row_shift_overlap': len(final_matches),
                    'row_shift_improved': len(final_matches),
                })
                self.log(f"   ✅ {match_label} 기준 매칭 완료: {len(final_matches)}/{len(all_gt_sentences)} 쌍 매칭")
            else:
                self.log("\n• 행 번호 1:1 매칭 시작...")
                self.log("   💡 매칭 방식: 동일 행 인덱스(i=i)로 직접 매칭, 교차 매칭 없음")
                gt_ids = sorted(gt_grouped.keys())
                pred_ids = sorted(pred_grouped.keys())
                common_ids = sorted(set(gt_ids).intersection(set(pred_ids)))
                final_matches = [(i, i, 1.0) for i in common_ids]
                row_shift_meta.update({
                    'row_shift_applied': 0,
                    'row_shift_overlap': len(common_ids),
                })
        else:
            self.log("\n• {0} 기준 매칭 시작...".format(match_label))
            self.log(f"   💡 매칭 방식: {self.group_label} 식별자는 통계용, 순수 {match_label} 유사도로 매칭")
            all_gt_sentences = list(gt_grouped.items())
            all_pred_sentences = list(pred_grouped.items())
            if match_on == 'source':
                final_matches = self.smart_match_sentences_by_source_only(all_gt_sentences, all_pred_sentences)
            else:
                final_matches = self.smart_match_sentences_by_target_only(all_gt_sentences, all_pred_sentences)

        return final_matches, row_shift_meta

    def _compute_metrics_for_matches(
        self,
        gt_grouped: Dict[int, List[Dict[str, str]]],
        pred_grouped: Dict[int, List[Dict[str, str]]],
        base_matches: List[Tuple[int, int, float]],
        unit: str,
        match_basis: str,
        row_shift_meta: Dict[str, any],
        *,
        mismatch_container: List[Dict[str, any]],
        global_integrity: Dict[str, any] | None = None,
        log_integrity: bool = True,
        log_prefix: str = '',
    ) -> Dict[str, any]:
        """공통 지표 계산을 수행하고 결과를 반환한다."""
        unit_label = '행' if unit == 'row' else f"{self.group_label}"

        matched_gt_ids = {gt_id for gt_id, _, _ in base_matches}
        unmatched_gt_ids = [gt_id for gt_id in gt_grouped.keys() if gt_id not in matched_gt_ids]
        final_matches = list(base_matches)
        for gt_id in unmatched_gt_ids:
            final_matches.append((gt_id, None, 0.0))

        self.log(f"\n{log_prefix}📊 최종 매칭 결과 ({'원문' if match_basis=='source' else '번역문'} 기준):")
        self.log(f"{log_prefix}  • 총 매칭된 {unit_label} 쌍: {len([m for m in final_matches if m[1] is not None])}개")
        self.log(f"{log_prefix}  • 매칭되지 않은 정답 {unit_label}: {len(unmatched_gt_ids)}개 (평가에 포함)")
        self.log(f"{log_prefix}  • 매칭되지 않은 예측 {unit_label}: {len(pred_grouped) - len([m for m in final_matches if m[1] is not None])}개")

        sentence_results = []
        overall_metrics = defaultdict(list)
        source_mismatch_count = 0

        for gt_id, pred_id, match_similarity in final_matches:
            gt_segments = gt_grouped.get(gt_id, [])
            pred_segments = pred_grouped.get(pred_id, []) if pred_id is not None else []

            accuracy = self.calculate_sentence_accuracy(gt_segments, pred_segments, gt_id, mismatch_container=mismatch_container)
            accuracy['sentence_id'] = gt_id
            accuracy['group_id'] = gt_id
            accuracy['group_label'] = self.group_label
            accuracy['matched_pred_id'] = pred_id
            accuracy['match_similarity'] = match_similarity
            sentence_results.append(accuracy)

            if not accuracy['source_text_match']:
                source_mismatch_count += 1

            for metric, value in accuracy.items():
                if metric in ['sentence_id', 'matched_pred_id', 'match_similarity', 'group_label', 'group_id']:
                    continue
                if isinstance(value, (int, float)) or isinstance(value, bool):
                    overall_metrics[metric].append(float(value))

        if source_mismatch_count > 0:
            self.log(f"{log_prefix}\n🔍 원문 불일치 요약 (평가 대상에 포함됨):")
            self.log(f"{log_prefix}   총 {source_mismatch_count}개 문장에서 원문 불일치 발생")
            denom = len(final_matches) if len(final_matches) > 0 else 1
            self.log(f"{log_prefix}   전체 대비 비율: {source_mismatch_count/denom:.1%}")
            self.log(f"{log_prefix}   원문 불일치는 평가에 포함되어 전체 정확도에 반영됩니다.")
        else:
            self.log(f"{log_prefix}\n✅ 모든 문장의 원문이 일치합니다!")

        overall_accuracy = {}
        matched_results = [r for r in sentence_results if r.get('match_similarity', 0) > 0]
        matched_metrics = defaultdict(list)

        for result in matched_results:
            for metric, value in result.items():
                if metric in ['sentence_id', 'matched_pred_id', 'match_similarity', 'group_label', 'group_id']:
                    continue
                if isinstance(value, (int, float)) or isinstance(value, bool):
                    matched_metrics[metric].append(float(value))

        for metric, values in overall_metrics.items():
            if metric in ['gt_segments', 'pred_segments', 'matched_pairs', 'correct_translation_pairs', 'source_based_matches',
                          'source_text_len_gt', 'source_text_len_pred', 'source_text_len_delta',
                          'target_text_len_gt', 'target_text_len_pred', 'target_text_len_delta']:
                overall_accuracy[f'total_{metric}'] = sum(values)
            overall_accuracy[f'avg_{metric}'] = (sum(values) / len(values)) if values else 0

        for metric, values in matched_metrics.items():
            overall_accuracy[f'avg_{metric}_matched_only'] = (sum(values) / len(values)) if values else 0

        matched_count = len(matched_results)
        total_count = len(sentence_results)
        matching_rate = (matched_count / total_count) if total_count > 0 else 0.0
        overall_accuracy['matching_rate'] = matching_rate
        overall_accuracy['matched_count'] = matched_count
        overall_accuracy['total_count'] = total_count

        for metric, values in overall_metrics.items():
            matched_avg = (sum(matched_metrics.get(metric, [0])) / len(matched_metrics.get(metric, [1]))) if matched_metrics.get(metric) else 0.0
            weighted_avg = matched_avg * matching_rate
            overall_accuracy[f'avg_{metric}_weighted'] = weighted_avg

        if global_integrity:
            if log_integrity:
                self.log("\n🧩 전역 텍스트 무결성 체크:")
                self.log(f"  • 원문 길이: GT {global_integrity['global_source_len_gt']} / Pred {global_integrity['global_source_len_pred']} (Δ={global_integrity['global_source_delta']})")
                self.log(f"  • 번역 길이: GT {global_integrity['global_target_len_gt']} / Pred {global_integrity['global_target_len_pred']} (Δ={global_integrity['global_target_delta']})")
            overall_accuracy.update(global_integrity)

        row_meta = {
            'row_shift_applied': int(row_shift_meta.get('row_shift_applied', 0)),
            'row_shift_overlap': int(row_shift_meta.get('row_shift_overlap', 0)),
            'row_shift_improved': int(row_shift_meta.get('row_shift_improved', 0)),
            'row_shift_zero_eq': int(row_shift_meta.get('row_shift_zero_eq', 0)),
            'row_shift_best_eq': int(row_shift_meta.get('row_shift_best_eq', 0)),
            'row_shift_best_avg_sim': float(row_shift_meta.get('row_shift_best_avg_sim', 0.0)),
        }
        overall_accuracy.update(row_meta)

        summary = {
            'total_sentences': len(final_matches),
            'total_gt_sentences': len(gt_grouped),
            'total_pred_sentences': len(pred_grouped),
            'total_groups': len(final_matches),
            'total_gt_groups': len(gt_grouped),
            'total_pred_groups': len(pred_grouped),
            'total_gt_rows': int(len(self.gt_data)),
            'total_pred_rows': int(len(self.pred_data)),
            'unit': unit,
            'group_label': self.group_label,
            'source_based_matches': len(final_matches),
            'unmatched_gt': len(gt_grouped) - len(final_matches),
            'unmatched_pred': len(pred_grouped) - len(final_matches),
            'source_mismatch_count': source_mismatch_count,
            'match_basis': match_basis,
        }
        summary.update(row_meta)

        return {
            'sentence_results': sentence_results,
            'overall_accuracy': overall_accuracy,
            'summary': summary,
            'source_mismatch_details': mismatch_container,
        }

    def evaluate_accuracy(self, unit: str = 'row', include_translation_matching: bool = True) -> Dict[str, any]:
        """전체 정확도 평가.

        - 원문 기준(원문 유사도로 매칭) 1회 + (옵션) 번역문 기준(번역문 유사도로 매칭) 1회.
        - 프로젝트(PA/SA)는 임계값(등급) 적용에만 영향을 주며,
          매칭/평가 축(원문기준/번역기준) 자체는 항상 동일하게 계산된다.
        """
        if self.gt_data is None or self.pred_data is None:
            print("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            return {}

        if unit == 'row':
            self.log("평가 단위: 행(로우) 단위")
            self.group_label = '행'
            if getattr(self, '_row_align_by_keys', False):
                try:
                    self.align_prediction_rows_by_keys()
                except Exception as _e_align:
                    self.log(f"행 정렬(키 기반) 실패: {_e_align}")
            gt_grouped = self.group_by_row(self.gt_data)
            pred_grouped = self.group_by_row(self.pred_data)
        else:
            gt_grouped = self.group_by_sentence_id(self.gt_data)
            pred_grouped = self.group_by_sentence_id(self.pred_data)
            self.log(f"평가 단위: {self.group_label}(식별자 그룹) 단위")

        unit_label = '행' if unit == 'row' else f"{self.group_label}"
        self.log(f"\n정답 데이터 {unit_label} 수: {len(gt_grouped)}")
        self.log(f"예측 데이터 {unit_label} 수: {len(pred_grouped)}")
        if unit == 'row':
            try:
                gt_rows = int(len(self.gt_data))
                pd_rows = int(len(self.pred_data))
                if len(gt_grouped) != gt_rows:
                    self.log(f"⚠️ 정답 원본 행 수({gt_rows})와 그룹 수({len(gt_grouped)})가 다릅니다. 빈 셀/머지/시트 확인 필요")
                if len(pred_grouped) != pd_rows:
                    self.log(f"⚠️ 예측 원본 행 수({pd_rows})와 그룹 수({len(pred_grouped)})가 다릅니다. 올바른 파일/시트인지 확인")
            except Exception:
                pass

        global_integrity = self.compute_global_text_integrity()

        source_matches, source_row_shift = self._build_matches(gt_grouped, pred_grouped, unit, match_on='source')
        source_results = self._compute_metrics_for_matches(
            gt_grouped,
            pred_grouped,
            source_matches,
            unit,
            match_basis='source',
            row_shift_meta=source_row_shift,
            mismatch_container=self.source_mismatch_details,
            global_integrity=global_integrity,
            log_integrity=True,
            log_prefix='',
        )

        target_results = None
        if include_translation_matching:
            self.log("\n================ 번역문 기준 추가 평가 실행 ================")
            translation_mismatch_details: List[Dict[str, any]] = []
            target_matches, target_row_shift = self._build_matches(gt_grouped, pred_grouped, unit, match_on='target')
            target_results = self._compute_metrics_for_matches(
                gt_grouped,
                pred_grouped,
                target_matches,
                unit,
                match_basis='target',
                row_shift_meta=target_row_shift,
                mismatch_container=translation_mismatch_details,
                global_integrity=global_integrity,
                log_integrity=False,
                log_prefix='[번역기준] ',
            )
            target_results['source_mismatch_details'] = translation_mismatch_details

        # 프로젝트별 primary 기준 선택
        requested_primary = self._primary_match_basis_for_project()
        primary_basis = requested_primary
        primary_results = source_results
        secondary_results = target_results
        if requested_primary == 'target' and target_results is not None:
            primary_results = target_results
            secondary_results = source_results
        elif requested_primary == 'target' and target_results is None:
            # SA인데 번역기준 평가를 꺼둔 경우: 안전하게 원문기준으로 폴백
            primary_basis = 'source'
            primary_results = source_results

        wrapper: Dict[str, any] = {
            'project': self.project,
            'primary_basis': primary_basis,
            'primary': primary_results,
            'secondary_basis': ('target' if primary_basis == 'source' else 'source') if secondary_results is not None else None,
            'secondary': secondary_results,
            # 호환/가독성용 별칭
            'source_based': source_results,
            'target_based': target_results,
        }
        return wrapper

    def analyze_unmatched_gt_rows(self, output_file: str = None) -> Dict[str, any]:
        """키 기반 정렬 후 매칭되지 않은 GT 행들만 분석
        - align_prediction_rows_by_keys()를 먼저 실행해야 함
        - 매칭되지 않은 GT 행(예측 데이터에 빈 값으로 채워진 행) 추출
        - 통계 및 상세 정보 반환
        """
        if self.gt_data is None or self.pred_data is None:
            self.log("데이터가 로드되지 않았습니다.")
            return {}
        
        # 원문 컬럼 찾기
        source_col = None
        for col in self.gt_data.columns:
            if '원문' in str(col) or 'source' in str(col).lower():
                source_col = col
                break
        
        if source_col is None:
            self.log("원문 컬럼을 찾을 수 없습니다.")
            return {}
        
        # 예측 데이터에서 원문이 비어있는 행 추출
        unmatched_mask = (self.pred_data[source_col].isna()) | (self.pred_data[source_col] == '')
        unmatched_gt = self.gt_data[unmatched_mask].copy()
        unmatched_pred = self.pred_data[unmatched_mask].copy()
        
        # 통계
        total_unmatched = len(unmatched_gt)
        total_chars = unmatched_gt[source_col].astype(str).str.len().sum()
        
        self.log(f"\n📌 매칭되지 않은 GT 행 분석:")
        self.log(f"  • 총 unmatched 행: {total_unmatched}행")
        self.log(f"  • 총 원문 문자수: {total_chars}자")
        self.log(f"  • 비율: {total_unmatched/len(self.gt_data)*100:.1f}% of total")
        
        # 상세 정보 저장
        result = {
            'total_unmatched': total_unmatched,
            'total_chars': total_chars,
            'total_gt_rows': len(self.gt_data),
            'unmatched_percentage': total_unmatched / len(self.gt_data) * 100 if len(self.gt_data) > 0 else 0,
            'unmatched_data': unmatched_gt
        }
        
        # 파일로 저장
        if output_file:
            try:
                parent = os.path.dirname(output_file)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                unmatched_gt.to_excel(output_file, index=False)
                self.log(f"  ✅ 저장: {output_file}")
            except Exception as e:
                self.log(f"  ❌ 저장 실패: {e}")
        
        return result

    def align_prediction_rows_by_keys(self):
        """GT의 (문장식별자, 구식별자) 순서에 맞춰 예측 DF를 재정렬한다.
        - 키가 양쪽 모두 있을 때만 적용
        - 매칭 실패 행은 빈 원문/번역문으로 채운다(길이 일치 보장)
        """
        gt = self.gt_data.copy()
        pd_df = self.pred_data.copy()
        key_sid = None
        key_gid = None
        for col in gt.columns:
            if str(col) == '문장식별자':
                key_sid = col
            if str(col) == '구식별자':
                key_gid = col
        if key_sid is None or key_gid is None:
            self.log("GT에 문장식별자/구식별자 키가 없어 정렬을 건너뜁니다.")
            return
        if key_sid not in pd_df.columns or key_gid not in pd_df.columns:
            self.log("예측 데이터에 문장식별자/구식별자가 없어 정렬을 건너뜁니다.")
            return
        # 예측 인덱스 맵 구성(첫 매칭 우선)
        pred_map = {}
        for idx, row in pd_df.iterrows():
            try:
                key = (int(row[key_sid]), int(row[key_gid]))
            except Exception:
                continue
            if key not in pred_map:
                pred_map[key] = idx
        # 새 행 목록 생성
        rows = []
        matched = 0
        for _, r in gt.iterrows():
            try:
                key = (int(r[key_sid]), int(r[key_gid]))
            except Exception:
                key = None
            if key and key in pred_map:
                rows.append(pd_df.loc[pred_map[key]].to_dict())
                matched += 1
            else:
                # 빈 행 생성(키는 GT와 동일, 원문/번역문은 공백)
                new_row = {c: '' for c in pd_df.columns}
                if key_sid in new_row:
                    new_row[key_sid] = int(r[key_sid]) if not pd.isna(r[key_sid]) else ''
                if key_gid in new_row:
                    new_row[key_gid] = int(r[key_gid]) if not pd.isna(r[key_gid]) else ''
                rows.append(new_row)
        new_pred = pd.DataFrame(rows, columns=pd_df.columns)
        self.pred_data = new_pred
        self.log(f"🔧 키 기반 정렬 적용: {matched}/{len(gt)}행 매칭, 미매칭 {len(gt)-matched}행은 빈 값으로 채움")

    def detect_best_row_shift(self, gt_grouped: Dict[int, List[Dict[str, str]]], pred_grouped: Dict[int, List[Dict[str, str]]], max_shift: int = 50) -> Dict[str, any]:
        """행 단위에서 시스템적 인덱스 오프셋을 탐지해 최적 shift를 찾는다.
        기준: 정규화된 원문이 완전히 동일한 행 수(우선), 동률이면 평균 유사도.
        반환: best_shift, best_count_eq, best_avg_sim, best_overlap, zero_* 통계 포함
        """
        # 정렬된 ID와 원문 리스트 구성
        gt_ids = sorted(gt_grouped.keys())
        pred_ids = sorted(pred_grouped.keys())
        gt_src = [self.normalize_for_matching(gt_grouped[i][0]['source'] if gt_grouped[i] else '') for i in gt_ids]
        pred_src = [self.normalize_for_matching(pred_grouped[i][0]['source'] if pred_grouped[i] else '') for i in pred_ids]

        def stats_for_shift(s: int) -> Tuple[int, float, int]:
            # 겹치는 범위 계산 (리스트 인덱스 기준)
            if s >= 0:
                start = 0
                end = min(len(gt_src), len(pred_src) - s)
                g_slice = gt_src[start:end]
                p_slice = pred_src[start + s:start + s + (end - start)]
            else:
                start = -s
                end = min(len(gt_src), len(pred_src) - 0)
                g_slice = gt_src[start:end]
                p_slice = pred_src[0:(end - start)]
            overlap = len(g_slice)
            if overlap <= 0:
                return 0, 0.0, 0
            eq = 0
            sim_sum = 0.0
            for a, b in zip(g_slice, p_slice):
                if a == b and a != '':
                    eq += 1
                # 유사도는 원문 원본 문자열 기반으로 계산하는 것이 더 낫지만, 여기서는 정규화 문자열로 근사
                sim_sum += (1.0 if a == b else self.calculate_text_similarity(a, b))
            return eq, (sim_sum / overlap), overlap

        # s=0 기준
        zero_eq, zero_avg_sim, zero_overlap = stats_for_shift(0)
        best = {
            'best_shift': 0,
            'best_count_eq': zero_eq,
            'best_avg_sim': zero_avg_sim,
            'best_overlap': zero_overlap,
            'zero_count_eq': zero_eq,
            'zero_avg_sim': zero_avg_sim,
            'zero_overlap': zero_overlap,
        }
        # 검색 범위 [-max_shift, +max_shift]
        for s in range(-max_shift, max_shift + 1):
            if s == 0:
                continue
            eq, avg_sim, overlap = stats_for_shift(s)
            # 선택 기준: eq 최댓값, 동률이면 avg_sim 큰 것
            if eq > best['best_count_eq'] or (eq == best['best_count_eq'] and avg_sim > best['best_avg_sim']):
                best.update({'best_shift': s, 'best_count_eq': eq, 'best_avg_sim': avg_sim, 'best_overlap': overlap})
        return best
    
    def print_detailed_results(self, results: Dict[str, any]):
        """상세 결과 출력"""
        normalized = self._normalize_results_object(results)
        primary = normalized.get('primary') or {}
        secondary = normalized.get('secondary')
        primary_basis = normalized.get('primary_basis', (primary.get('summary') or {}).get('match_basis', 'source'))
        secondary_basis = normalized.get('secondary_basis')

        self.log("\n" + "="*80)
        self.log(f"정확도 평가 결과 (Primary: {self._basis_label(primary_basis)})")
        self.log("="*80)

        # 전체 요약
        summary = primary['summary']
        overall = primary['overall_accuracy']

        self.log(f"\n📊 전체 요약:")
        self.log(f"  • 📋 매칭 요약:")
        self.log(f"    - Primary 매칭 기준: {self._basis_label(primary_basis)}")
        self.log(f"    - 총 평가 {summary.get('group_label','문장')} 쌍: {summary['total_sentences']}개")
        # secondary가 있을 때만 표시
        if secondary is not None:
            sec_summary = (secondary.get('summary') or {})
            sec_basis = secondary_basis or sec_summary.get('match_basis', 'target')
            self.log(f"    - Secondary 매칭 기준: {self._basis_label(sec_basis)}")
        self.log(f"    - 총 평가 {summary.get('group_label','문장')} 쌍: {summary['total_sentences']}개")
        self.log(f"    - 평가 단위: {'행' if summary.get('unit')=='row' else summary.get('group_label','문장')}")
        self.log(f"  • 🗂️ 데이터 현황:")
        self.log(f"    - 정답 행(로우) 수: {summary.get('total_gt_rows', 0)}행")
        self.log(f"    - 예측 행(로우) 수: {summary.get('total_pred_rows', 0)}행")
        self.log(f"    - 정답 {summary.get('group_label','문장')} 그룹 수: {summary.get('total_gt_groups', summary.get('total_gt_sentences', 0))}개")
        self.log(f"    - 예측 {summary.get('group_label','문장')} 그룹 수: {summary.get('total_pred_groups', summary.get('total_pred_sentences', 0))}개")
        self.log(f"    - 매칭되지 않은 정답: {summary.get('unmatched_gt', 0)}개")
        self.log(f"    - 매칭되지 않은 예측: {summary.get('unmatched_pred', 0)}개")
        if summary['total_sentences'] > 0:
            self.log(f"  • ⚠️ 원문 불일치: {summary['source_mismatch_count']}개 ({summary['source_mismatch_count']/summary['total_sentences']:.1%}) - 평가에 포함됨")
        else:
            self.log(f"  • ⚠️ 원문 불일치: {summary['source_mismatch_count']}개 (대상 없음)")
        self.log(f"  • 평균 정답 세그먼트 수: {overall.get('avg_gt_segments', 0):.1f}")
        self.log(f"  • 평균 예측 세그먼트 수: {overall.get('avg_pred_segments', 0):.1f}")

    # 전역 무결성 간단 요약 (전역 유사도 수치 제외)
        if 'global_source_len_gt' in overall:
            self.log("\n🧩 전역 텍스트 무결성:")
            self.log(f"  • 원문 길이 Δ: {overall.get('global_source_delta', 0)} (GT {overall.get('global_source_len_gt', 0)} / Pred {overall.get('global_source_len_pred', 0)})")
            self.log(f"  • 번역 길이 Δ: {overall.get('global_target_delta', 0)} (GT {overall.get('global_target_len_gt', 0)} / Pred {overall.get('global_target_len_pred', 0)})")
            # 문자 단위 diff 요약 로그 (간결 표시)
            if 'global_source_ops_insert' in overall:
                self.log("  • 원문 diff: 삽입 {0}, 삭제 {1}, 치환 {2}".format(
                    int(overall.get('global_source_ops_insert', 0)),
                    int(overall.get('global_source_ops_delete', 0)),
                    int(overall.get('global_source_ops_replace', 0))
                ))
            if 'global_target_ops_insert' in overall:
                self.log("  • 번역 diff: 삽입 {0}, 삭제 {1}, 치환 {2}".format(
                    int(overall.get('global_target_ops_insert', 0)),
                    int(overall.get('global_target_ops_delete', 0)),
                    int(overall.get('global_target_ops_replace', 0))
                ))
            # 첫 차이 스니펫 (있을 때만 간략 표시)
            s_ctx_gt = overall.get('global_source_first_diff_context_gt')
            s_ctx_pd = overall.get('global_source_first_diff_context_pred')
            if s_ctx_gt or s_ctx_pd:
                self.log("  • 원문 첫 차이 주변(GT | Pred):")
                self.log(f"    GT: {s_ctx_gt}")
                self.log(f"    PD: {s_ctx_pd}")
            t_ctx_gt = overall.get('global_target_first_diff_context_gt')
            t_ctx_pd = overall.get('global_target_first_diff_context_pred')
            if t_ctx_gt or t_ctx_pd:
                self.log("  • 번역 첫 차이 주변(GT | Pred):")
                self.log(f"    GT: {t_ctx_gt}")
                self.log(f"    PD: {t_ctx_pd}")

        # 임계값 등급 요약 (primary 기준)
        if self.thresholds:
            grading = self.grade_with_thresholds(overall)
            self.log(f"\n🏷️ 임계값 등급 요약 (Primary: {self._basis_label(primary_basis)}):")
            unit_ko = '행' if grading.get('unit') == 'row' else (self.group_label or '문장')
            self.log(f"• 프로젝트: {grading['project'].upper()} (단위: {unit_ko})")
            # 메트릭 한글 라벨 매핑
            metric_label_ko = {
                'partial_match': '부분 일치율',
                'target_avg_similarity': '번역문 평균 유사도',
                'target_text_similarity': '번역문 전체 유사도',
                'source_text_similarity': '원문 전체 유사도',
                'source_avg_similarity': '원문 평균 유사도',
                'source_jaccard': '원문 Jaccard 유사도',
                'target_accuracy': '번역문 정확도',
                'f1_score': 'F1 점수',
                'source_f1_score': '원문 F1',
                'target_f1_score': '번역문 F1',
                'precision': '정밀도',
                'recall': '재현율',
            }
            for m, info in grading['per_metric'].items():
                label = metric_label_ko.get(m, m)
                self.log(f"- {label}: {info['value']:.3f} → {info['label']} [min {info['min']}, rec {info['recommended']}, top {info['top']}]")
            self.log(f"• 전체 등급: {grading['overall_label']}")

        # 간결 모드: 핵심 지표만 출력하고 나머지 상세는 생략
        if self.brief:
            def log_brief_metrics(*, basis: str, overall_metrics: Dict[str, any], role: str):
                """brief 모드 핵심 지표 출력.

                - 원문기준(source) 매칭 결과: 번역 품질(번역문 유사도)을 핵심으로 표시
                - 번역기준(target) 매칭 결과: 원문 품질(원문 유사도)을 핵심으로 표시

                매칭 기준과 무관하게 계산되는 지표들도 있지만, brief에서는 혼동을 줄이기 위해
                '매칭 기준과 반대쪽 텍스트'의 유사도를 핵심으로 보여준다.
                """
                self.log(f"\n🎯 핵심 지표 (간결 모드, {role}: {self._basis_label(basis)})")
                self.log(f"• 부분 일치율: {overall_metrics.get('avg_partial_match', 0):.3f}")
                if basis == 'target':
                    # 번역기준 매칭에서는 '원문 유사도'를 핵심으로 표시
                    self.log(f"• 원문 일치율: {overall_metrics.get('avg_source_text_match', 0):.3f}")
                    self.log(f"• 원문 F1: {overall_metrics.get('avg_source_f1_score', 0):.3f}")
                    self.log(f"• 원문 평균 유사도: {overall_metrics.get('avg_source_avg_similarity', 0):.3f}")
                    self.log(f"• 원문 전체 유사도: {overall_metrics.get('avg_source_text_similarity', 0):.3f}")
                else:
                    # 원문기준 매칭에서는 '번역문 유사도'를 핵심으로 표시
                    self.log(f"• 번역문 일치율: {overall_metrics.get('avg_target_text_match', 0):.3f}")
                    self.log(f"• 번역문 F1: {overall_metrics.get('avg_target_f1_score', 0):.3f}")
                    self.log(f"• 번역문 평균 유사도: {overall_metrics.get('avg_target_avg_similarity', 0):.3f}")
                    self.log(f"• 번역문 전체 유사도: {overall_metrics.get('avg_target_text_similarity', 0):.3f}")

            log_brief_metrics(basis=primary_basis, overall_metrics=overall, role='Primary')
            if secondary is not None:
                sec_overall = (secondary.get('overall_accuracy') or {})
                sec_basis = secondary_basis or (secondary.get('summary') or {}).get('match_basis', 'target')
                log_brief_metrics(basis=sec_basis, overall_metrics=sec_overall, role='Secondary')
            return

        self.log(f"\n🎯 주요 정확도 지표 (현재 출력: Primary={self._basis_label(primary_basis)}):")
        self.log(f"  📌 평가 방식 안내:")
        unit_mode = summary.get('unit')
        group_label = summary.get('group_label', self.group_label or '문장')
        if unit_mode == 'row':
            self.log("    - 평가 단위: 행(로우)")
            self.log("    - 매칭: 기본은 1:1(행 번호) 또는(설정에 따라) 유사도 기반 스마트 매칭")
        else:
            self.log(f"    - 평가 단위: {group_label} 식별자 그룹")
            self.log("    - 주의: 식별자는 그룹(묶기) 용도이며, 매칭은 유사도 기반으로 수행(식별자 값 자체로 직접 매칭하지 않음)")
        self.log("    - 원문/번역문은 각 단위의 (원문, 번역문) 쌍으로 함께 평가")
        self.log("")
        self.log(f"  • 완전 일치율: {overall.get('avg_exact_match', 0):.1%}")
        self.log(f"  • 전체 텍스트 일치율: {overall.get('avg_text_match', 0):.1%}")
        self.log(f"    - 원문 일치율: {overall.get('avg_source_text_match', 0):.1%}")
        self.log(f"    - 번역문 일치율: {overall.get('avg_target_text_match', 0):.1%}")
        self.log(f"  • 세그먼트 수 일치율: {overall.get('avg_segment_count_match', 0):.1%}")
        self.log(f"  • 📊 원문 기준 순서대로 매칭 분석:")
        self.log(f"    - 매칭된 원문-번역문 쌍: {overall.get('avg_matched_pairs', 0):.1f}개")
        self.log(f"    - 원문 기준 매칭 수: {overall.get('avg_source_based_matches', 0):.1f}개")
        self.log(f"  • 📊 번역문 정확도 평가 (원문이 매칭된 쌍에서만):")
        self.log(f"    - 번역문 정확한 쌍: {overall.get('avg_correct_translation_pairs', 0):.1f}개")
        self.log(f"    - 번역문 정확도: {overall.get('avg_target_accuracy', 0):.1%}")
        self.log(f"    - 번역문 평균 유사도: {overall.get('avg_target_avg_similarity', 0):.1%}")
        self.log(f"  • 부분 일치율: {overall.get('avg_partial_match', 0):.1%}")
        self.log(f"    - 원문 부분 일치율: {overall.get('avg_source_partial_match', 0):.1%}")
        self.log(f"      • Jaccard 유사도: {overall.get('avg_source_jaccard', 0):.1%}")
        self.log(f"      • 전체 텍스트 유사도: {overall.get('avg_source_text_similarity', 0):.1%}")
        self.log(f"      • 세그먼트별 평균 유사도: {overall.get('avg_source_avg_similarity', 0):.1%}")
        self.log(f"    - 번역문 부분 일치율: {overall.get('avg_target_partial_match', 0):.1%}")
        self.log(f"      • 전체 텍스트 유사도: {overall.get('avg_target_text_similarity', 0):.1%}")
        self.log(f"      • 매칭된 쌍 평균 유사도: {overall.get('avg_target_avg_similarity', 0):.1%}")
        self.log(f"  • F1 점수: {overall.get('avg_f1_score', 0):.1%}")
        self.log(f"    - 원문 F1: {overall.get('avg_source_f1_score', 0):.1%}")
        self.log(f"    - 번역문 F1: {overall.get('avg_target_f1_score', 0):.1%}")
        self.log(f"  • 정밀도: {overall.get('avg_precision', 0):.1%}")
        self.log(f"  • 재현율: {overall.get('avg_recall', 0):.1%}")

        if secondary is not None:
            t_overall = secondary.get('overall_accuracy', {})
            sec_basis = secondary_basis or (secondary.get('summary') or {}).get('match_basis', 'target')
            self.log(f"\n🌐 Secondary 기준 추가 지표 ({self._basis_label(sec_basis)}):")
            self.log(f"  • 완전 일치율: {t_overall.get('avg_exact_match', 0):.1%}")
            self.log(f"  • 번역문 정확도: {t_overall.get('avg_target_accuracy', 0):.1%}")
            self.log(f"  • 부분 일치율: {t_overall.get('avg_partial_match', 0):.1%}")
            self.log(f"  • 매칭률: {t_overall.get('matching_rate', 0):.1%}")

            # 사용자가 보는 '기준(원문/번역)'에 따라, 반대쪽 텍스트 품질 지표를 함께 노출
            # - 번역기준(target): 원문 정확도/유사도(핵심)
            # - 원문기준(source): 번역문 유사도(핵심)
            if sec_basis == 'target':
                self.log(f"  • 원문 일치율: {t_overall.get('avg_source_text_match', 0):.1%}")
                self.log(f"  • 원문 부분 일치율: {t_overall.get('avg_source_partial_match', 0):.1%}")
                self.log(f"    - Jaccard 유사도: {t_overall.get('avg_source_jaccard', 0):.1%}")
                self.log(f"  • 원문 F1: {t_overall.get('avg_source_f1_score', 0):.1%}")
                self.log(f"    - 원문 정밀도: {t_overall.get('avg_source_precision', 0):.1%}")
                self.log(f"    - 원문 재현율: {t_overall.get('avg_source_recall', 0):.1%}")
                self.log(f"  • 원문 평균 유사도: {t_overall.get('avg_source_avg_similarity', 0):.1%}")
                self.log(f"  • 원문 전체 유사도: {t_overall.get('avg_source_text_similarity', 0):.1%}")
            else:
                self.log(f"  • 번역문 일치율: {t_overall.get('avg_target_text_match', 0):.1%}")
                self.log(f"  • 번역문 부분 일치율: {t_overall.get('avg_target_partial_match', 0):.1%}")
                self.log(f"  • 번역문 F1: {t_overall.get('avg_target_f1_score', 0):.1%}")
                self.log(f"    - 번역문 정밀도: {t_overall.get('avg_target_precision', 0):.1%}")
                self.log(f"    - 번역문 재현율: {t_overall.get('avg_target_recall', 0):.1%}")
                self.log(f"  • 번역문 평균 유사도: {t_overall.get('avg_target_avg_similarity', 0):.1%}")
                self.log(f"  • 번역문 전체 유사도: {t_overall.get('avg_target_text_similarity', 0):.1%}")

            # 임계값 등급(secondary도 함께 표시)
            if self.thresholds:
                grading_t = self.grade_with_thresholds(t_overall)
                if grading_t:
                    self.log(f"\n🏷️ 임계값 등급 요약 (Secondary: {self._basis_label(sec_basis)}):")
                    unit_ko_t = '행' if grading_t.get('unit') == 'row' else (self.group_label or '문장')
                    self.log(f"• 프로젝트: {grading_t['project'].upper()} (단위: {unit_ko_t})")
                    metric_label_ko = {
                        'partial_match': '부분 일치율',
                        'target_avg_similarity': '번역문 평균 유사도',
                        'target_text_similarity': '번역문 전체 유사도',
                        'source_text_similarity': '원문 전체 유사도',
                        'source_avg_similarity': '원문 평균 유사도',
                        'source_jaccard': '원문 Jaccard 유사도',
                        'target_accuracy': '번역문 정확도',
                        'f1_score': 'F1 점수',
                        'source_f1_score': '원문 F1',
                        'target_f1_score': '번역문 F1',
                        'precision': '정밀도',
                        'recall': '재현율',
                    }
                    for m, info in grading_t.get('per_metric', {}).items():
                        label = metric_label_ko.get(m, m)
                        self.log(f"- {label}: {info['value']:.3f} → {info['label']} [min {info['min']}, rec {info['recommended']}, top {info['top']}]")
                    self.log(f"• 전체 등급: {grading_t.get('overall_label')}")

        # 문장별 상세 결과 (상위 10개 + 하위 10개)
        sentence_results = primary['sentence_results']
        sentence_results.sort(key=lambda x: x['f1_score'], reverse=True)

        self.log(f"\n📈 성능 상위 10개 {summary.get('group_label','문장') }:")
        self.log("ID\tF1\t완전일치\t세그먼트수(정답/예측)\t원문F1\t번역문F1\t원문매칭\t번역정확")
        for result in sentence_results[:10]:
            self.log(f"{result['sentence_id']}\t{result['f1_score']:.2f}\t{result['exact_match']:.0f}\t\t{result['gt_segments']}/{result['pred_segments']}\t\t{result['source_f1_score']:.2f}\t{result['target_f1_score']:.2f}\t{result['matched_pairs']}\t\t{result['correct_translation_pairs']}")

        self.log(f"\n📉 성능 하위 10개 {summary.get('group_label','문장') }:")
        self.log("ID\tF1\t완전일치\t세그먼트수(정답/예측)\t원문F1\t번역문F1\t원문매칭\t번역정확")
        for result in sentence_results[-10:]:
            self.log(f"{result['sentence_id']}\t{result['f1_score']:.2f}\t{result['exact_match']:.0f}\t\t{result['gt_segments']}/{result['pred_segments']}\t\t{result['source_f1_score']:.2f}\t{result['target_f1_score']:.2f}\t{result['matched_pairs']}\t\t{result['correct_translation_pairs']}")
    
    def save_results(self, results: Dict[str, any], output_file: str, csv_dir: str | None = None):
        """결과를 엑셀 파일로 저장"""
        try:
            normalized = self._normalize_results_object(results)
            primary = normalized.get('primary') or {}
            secondary = normalized.get('secondary')
            primary_basis = normalized.get('primary_basis', (primary.get('summary') or {}).get('match_basis', 'source'))
            secondary_basis = normalized.get('secondary_basis')

            project_label = (self.project or '').upper() or ''

            # 출력 디렉터리 자동 생성
            parent = os.path.dirname(output_file)
            if parent:
                os.makedirs(parent, exist_ok=True)

            def build_summary_rows(overall: Dict[str, any], *, match_basis: str, role: str, res_summary: Dict[str, any] | None = None) -> List[Dict[str, any]]:
                summary_rows: List[Dict[str, any]] = []
                res_summary = res_summary or {}
                unit = str(res_summary.get('unit', '') or '')
                group_label = str(res_summary.get('group_label', '') or '')
                # 어떤 결과(Primary/Secondary)이며 어떤 단위/프로젝트인지 먼저 명시
                if project_label:
                    summary_rows.append({'지표(한글)': '프로젝트', '지표(영문)': 'project', '값': project_label})
                if unit:
                    summary_rows.append({'지표(한글)': '평가 단위(unit)', '지표(영문)': 'unit', '값': unit})
                if group_label:
                    summary_rows.append({'지표(한글)': '평가 단위(라벨)', '지표(영문)': 'group_label', '값': group_label})
                summary_rows.append({'지표(한글)': '결과 구분', '지표(영문)': 'result_role', '값': role})
                # 어떤 매칭 기준으로 계산된 지표인지 명시
                summary_rows.append({'지표(한글)': '매칭 기준', '지표(영문)': 'match_basis', '값': self._basis_label(match_basis)})
                if self.minimal_summary:
                    # minimal-summary는 "핵심 지표"가 match_basis에 따라 달라져야 혼동이 줄어듦.
                    # - match_basis=target(번역기준): 원문(소스) 일치율/F1/유사도가 핵심
                    # - match_basis=source(원문기준): 번역문(타겟) 정확도/F1/유사도가 핵심
                    if match_basis == 'target':
                        core_keys = [
                            # 원문(소스) 핵심 지표
                            'avg_source_text_match',
                            'avg_source_partial_match',
                            'avg_source_jaccard',
                            'avg_source_f1_score',
                            'avg_source_precision',
                            'avg_source_recall',
                            'avg_source_avg_similarity',
                            'avg_source_text_similarity',
                            # 번역(타겟)도 함께 제공 (컨텍스트)
                            'avg_target_accuracy',
                            'avg_target_partial_match',
                            'avg_target_f1_score',
                            'avg_target_avg_similarity',
                            'avg_target_text_similarity',
                        ]
                    else:
                        core_keys = [
                            # 번역(타겟) 핵심 지표
                            'avg_target_accuracy',
                            'avg_target_partial_match',
                            'avg_target_f1_score',
                            'avg_target_avg_similarity',
                            'avg_target_text_similarity',
                            # 원문(소스)도 함께 제공 (컨텍스트)
                            'avg_source_text_match',
                            'avg_source_partial_match',
                            'avg_source_jaccard',
                            'avg_source_f1_score',
                            'avg_source_precision',
                            'avg_source_recall',
                            'avg_source_avg_similarity',
                            'avg_source_text_similarity',
                        ]

                    keys = [
                        # 기존 호환(있으면 포함)
                        'avg_partial_match',
                        # 전역 일치율(있으면 포함)
                        'avg_exact_match',
                        'avg_text_match',
                        *core_keys,
                        # 규모/카운트
                        'total_gt_segments', 'total_pred_segments',
                        'total_matched_pairs', 'total_correct_translation_pairs',
                        # 편집거리/쉬프트(디버그용)
                        'global_source_delta', 'global_target_delta',
                        'global_source_ops_insert', 'global_source_ops_delete', 'global_source_ops_replace',
                        'global_target_ops_insert', 'global_target_ops_delete', 'global_target_ops_replace',
                        'row_shift_applied', 'row_shift_overlap', 'row_shift_improved',
                        'row_shift_zero_eq', 'row_shift_best_eq', 'row_shift_best_avg_sim',
                    ]
                    for key in keys:
                        if key in overall:
                            ko_label = self.metric_label_ko.get(key, key)
                            summary_rows.append({'지표(한글)': ko_label, '지표(영문)': key, '값': overall[key]})
                else:
                    exclude_keys = {
                        'global_source_text_similarity', 'global_target_text_similarity',
                        'global_source_text_match', 'global_target_text_match',
                        'global_source_text_match_strict', 'global_source_text_match_lenient',
                        'global_target_text_match_strict', 'global_target_text_match_lenient',
                        'avg_source_text_match_strict', 'avg_source_text_match_lenient',
                        'avg_target_text_match_strict', 'avg_target_text_match_lenient',
                    }
                    for key, value in overall.items():
                        if key in exclude_keys:
                            continue
                        ko_label = self.metric_label_ko.get(key, key)
                        summary_rows.append({'지표(한글)': ko_label, '지표(영문)': key, '값': value})
                # 임계값 등급 병합
                if self.thresholds:
                    grading = self.grade_with_thresholds(overall)
                    summary_rows.append({'지표(한글)': '프로젝트', '지표(영문)': 'grading_project', '값': grading['project']})
                    summary_rows.append({'지표(한글)': '평가 단위', '지표(영문)': 'grading_unit', '값': grading['unit']})
                    summary_rows.append({'지표(한글)': '전체 등급', '지표(영문)': 'grading_overall_label', '값': grading['overall_label']})
                    for m, info in grading['per_metric'].items():
                        ko_m = self.metric_label_ko.get(f'avg_{m}', m)
                        summary_rows.append({'지표(한글)': f'{ko_m} (등급)', '지표(영문)': f'grade_{m}', '값': info['label']})
                        summary_rows.append({'지표(한글)': f'{ko_m} (임계값)', '지표(영문)': f'grade_{m}_thresholds', '값': f"min={info['min']}, rec={info['recommended']}, top={info['top']}"})
                return summary_rows

            def build_meta_df() -> pd.DataFrame:
                rows: List[Dict[str, any]] = []
                rows.append({'항목': 'project', '값': project_label})
                rows.append({'항목': 'unit', '값': (primary.get('summary') or {}).get('unit', '')})
                rows.append({'항목': 'group_label', '값': (primary.get('summary') or {}).get('group_label', '')})
                rows.append({'항목': 'primary_basis', '값': self._basis_label(primary_basis)})
                if secondary is not None:
                    sec_basis = secondary_basis or (secondary.get('summary') or {}).get('match_basis', 'target')
                    rows.append({'항목': 'secondary_basis', '값': self._basis_label(sec_basis)})
                else:
                    rows.append({'항목': 'secondary_basis', '값': ''})
                rows.append({'항목': 'ground_truth_file', '값': self.ground_truth_file})
                rows.append({'항목': 'prediction_file', '값': self.prediction_file})
                rows.append({'항목': 'output_file', '값': output_file})
                if csv_dir:
                    rows.append({'항목': 'csv_dir', '값': csv_dir})
                return pd.DataFrame(rows)

            def write_single_result(res_obj: Dict[str, any], *, match_basis: str, role: str, sheet_suffix: str = '', mismatch_details_override: List[Dict[str, any]] | None = None):
                res_summary = (res_obj.get('summary') or {})
                sentence_df = pd.DataFrame(res_obj['sentence_results'])
                # 프로젝트/단위/구분(Primary/Secondary)/기준을 컬럼으로 포함(필터링/재가공 용이)
                if project_label:
                    sentence_df.insert(0, 'project', project_label)
                sentence_df.insert(0 if not project_label else 1, 'result_role', role)
                sentence_df.insert(0 if not project_label else 2, 'unit', str(res_summary.get('unit', '') or ''))
                # 어떤 매칭 기준의 결과인지 컬럼으로도 포함(필터링/재가공 용이)
                sentence_df.insert(0 if not project_label else 3, 'match_basis', self._basis_label(match_basis))
                sentence_df.to_excel(writer, sheet_name=f'문장별_상세결과{sheet_suffix}', index=False)

                summary_rows = build_summary_rows(res_obj['overall_accuracy'], match_basis=match_basis, role=role, res_summary=res_summary)
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name=f'전체_요약{sheet_suffix}', index=False)

                mismatch_details = mismatch_details_override if mismatch_details_override is not None else res_obj.get('source_mismatch_details')
                if mismatch_details:
                    mismatch_df = pd.DataFrame(mismatch_details)
                    cols = ['sentence_id', 'length_diff', 'similarity', 'gt_source', 'pred_source']
                    for c in cols:
                        if c not in mismatch_df.columns:
                            mismatch_df[c] = None
                    mismatch_df = mismatch_df[cols]
                    mismatch_df.to_excel(writer, sheet_name=f'원문불일치_상세{sheet_suffix}', index=False)
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 메타(프로젝트/단위/Primary-Secondary 기준) 시트 먼저 기록
                try:
                    build_meta_df().to_excel(writer, sheet_name='메타', index=False)
                except Exception:
                    pass

                write_single_result(primary, match_basis=primary_basis, role='Primary', sheet_suffix='')

                try:
                    gi = self._global_integrity or {}
                    if gi:
                        mismatch_rows = [
                            {
                                '구분': '원문',
                                'len_gt': gi.get('global_source_len_gt', 0),
                                'len_pred': gi.get('global_source_len_pred', 0),
                                'delta': gi.get('global_source_delta', 0),
                                'ops_insert': gi.get('global_source_ops_insert', 0),
                                'ops_delete': gi.get('global_source_ops_delete', 0),
                                'ops_replace': gi.get('global_source_ops_replace', 0),
                                'first_diff_index': gi.get('global_source_first_diff_index', -1),
                                'first_diff_context_gt': gi.get('global_source_first_diff_context_gt', ''),
                                'first_diff_context_pred': gi.get('global_source_first_diff_context_pred', ''),
                            },
                            {
                                '구분': '번역',
                                'len_gt': gi.get('global_target_len_gt', 0),
                                'len_pred': gi.get('global_target_len_pred', 0),
                                'delta': gi.get('global_target_delta', 0),
                                'ops_insert': gi.get('global_target_ops_insert', 0),
                                'ops_delete': gi.get('global_target_ops_delete', 0),
                                'ops_replace': gi.get('global_target_ops_replace', 0),
                                'first_diff_index': gi.get('global_target_first_diff_index', -1),
                                'first_diff_context_gt': gi.get('global_target_first_diff_context_gt', ''),
                                'first_diff_context_pred': gi.get('global_target_first_diff_context_pred', ''),
                            },
                        ]
                        mismatch_df2 = pd.DataFrame(mismatch_rows)
                        mismatch_df2.to_excel(writer, sheet_name='전역_불일치', index=False)
                except Exception:
                    pass

                if secondary is not None:
                    sec_basis = secondary_basis or (secondary.get('summary') or {}).get('match_basis', 'target')
                    write_single_result(secondary, match_basis=sec_basis, role='Secondary', sheet_suffix=self._basis_suffix(sec_basis), mismatch_details_override=secondary.get('source_mismatch_details'))

                import datetime
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_data = []
                log_data.append(f"[{current_time}] 정확도 평가 실행 로그")
                log_data.append(f"정답 파일: {self.ground_truth_file}")
                log_data.append(f"예측 파일: {self.prediction_file}")
                log_data.append("")
                log_data.extend(self.execution_log)
                
                log_df = pd.DataFrame({'실행_로그': log_data})
                log_df.to_excel(writer, sheet_name='실행_로그', index=False)
                
            # 선택: CSV 디렉터리로 개별 시트 내보내기
            if csv_dir:
                try:
                    os.makedirs(csv_dir, exist_ok=True)
                    # 메타 CSV
                    try:
                        build_meta_df().to_csv(os.path.join(csv_dir, '메타.csv'), index=False, encoding='utf-8-sig')
                    except Exception:
                        pass
                    # 재사용 가능한 DataFrame들 구성
                    sentence_df = pd.DataFrame(primary['sentence_results'])
                    res_summary_p = (primary.get('summary') or {})
                    if project_label:
                        sentence_df.insert(0, 'project', project_label)
                    sentence_df.insert(0 if not project_label else 1, 'result_role', 'Primary')
                    sentence_df.insert(0 if not project_label else 2, 'unit', str(res_summary_p.get('unit', '') or ''))
                    sentence_df.insert(0 if not project_label else 3, 'match_basis', self._basis_label(primary_basis))
                    summary_pairs = build_summary_rows(primary['overall_accuracy'], match_basis=primary_basis, role='Primary', res_summary=res_summary_p)
                    mismatch_df = pd.DataFrame(primary.get('source_mismatch_details', self.source_mismatch_details)) if primary.get('source_mismatch_details', self.source_mismatch_details) else pd.DataFrame()
                    log_df = pd.DataFrame({'실행_로그': self.execution_log})

                    sentence_df.to_csv(os.path.join(csv_dir, '문장별_상세결과.csv'), index=False, encoding='utf-8-sig')
                    pd.DataFrame(summary_pairs).to_csv(os.path.join(csv_dir, '전체_요약.csv'), index=False, encoding='utf-8-sig')
                    if not mismatch_df.empty:
                        cols = ['sentence_id', 'length_diff', 'similarity', 'gt_source', 'pred_source']
                        for c in cols:
                            if c not in mismatch_df.columns:
                                mismatch_df[c] = None
                        mismatch_df = mismatch_df[cols]
                        mismatch_df.to_csv(os.path.join(csv_dir, '원문불일치_상세.csv'), index=False, encoding='utf-8-sig')
                    if secondary is not None:
                        sec_basis = secondary_basis or (secondary.get('summary') or {}).get('match_basis', 'target')
                        sec_suffix = self._basis_suffix(sec_basis)
                        sec_sentence_df = pd.DataFrame(secondary['sentence_results'])
                        res_summary_s = (secondary.get('summary') or {})
                        if project_label:
                            sec_sentence_df.insert(0, 'project', project_label)
                        sec_sentence_df.insert(0 if not project_label else 1, 'result_role', 'Secondary')
                        sec_sentence_df.insert(0 if not project_label else 2, 'unit', str(res_summary_s.get('unit', '') or ''))
                        sec_sentence_df.insert(0 if not project_label else 3, 'match_basis', self._basis_label(sec_basis))
                        sec_sentence_df.to_csv(os.path.join(csv_dir, f'문장별_상세결과{sec_suffix}.csv'), index=False, encoding='utf-8-sig')
                        pd.DataFrame(build_summary_rows(secondary['overall_accuracy'], match_basis=sec_basis, role='Secondary', res_summary=res_summary_s)).to_csv(os.path.join(csv_dir, f'전체_요약{sec_suffix}.csv'), index=False, encoding='utf-8-sig')
                        t_md = pd.DataFrame(secondary.get('source_mismatch_details', []))
                        if not t_md.empty:
                            cols = ['sentence_id', 'length_diff', 'similarity', 'gt_source', 'pred_source']
                            for c in cols:
                                if c not in t_md.columns:
                                    t_md[c] = None
                            t_md = t_md[cols]
                            t_md.to_csv(os.path.join(csv_dir, f'원문불일치_상세{sec_suffix}.csv'), index=False, encoding='utf-8-sig')
                    # 전역 불일치 CSV 저장 (가능한 경우)
                    try:
                        gi = self._global_integrity or {}
                        if gi:
                            mismatch_rows = [
                                {
                                    '구분': '원문',
                                    'len_gt': gi.get('global_source_len_gt', 0),
                                    'len_pred': gi.get('global_source_len_pred', 0),
                                    'delta': gi.get('global_source_delta', 0),
                                    'ops_insert': gi.get('global_source_ops_insert', 0),
                                    'ops_delete': gi.get('global_source_ops_delete', 0),
                                    'ops_replace': gi.get('global_source_ops_replace', 0),
                                    'first_diff_index': gi.get('global_source_first_diff_index', -1),
                                    'first_diff_context_gt': gi.get('global_source_first_diff_context_gt', ''),
                                    'first_diff_context_pred': gi.get('global_source_first_diff_context_pred', ''),
                                },
                                {
                                    '구분': '번역',
                                    'len_gt': gi.get('global_target_len_gt', 0),
                                    'len_pred': gi.get('global_target_len_pred', 0),
                                    'delta': gi.get('global_target_delta', 0),
                                    'ops_insert': gi.get('global_target_ops_insert', 0),
                                    'ops_delete': gi.get('global_target_ops_delete', 0),
                                    'ops_replace': gi.get('global_target_ops_replace', 0),
                                    'first_diff_index': gi.get('global_target_first_diff_index', -1),
                                    'first_diff_context_gt': gi.get('global_target_first_diff_context_gt', ''),
                                    'first_diff_context_pred': gi.get('global_target_first_diff_context_pred', ''),
                                },
                            ]
                            mismatch_df2 = pd.DataFrame(mismatch_rows)
                            mismatch_df2.to_csv(os.path.join(csv_dir, '전역_불일치.csv'), index=False, encoding='utf-8-sig')
                            # 🆕 전역 문자 빈도 차이 CSV도 저장 (원인 규명 보조)
                            try:
                                from collections import Counter
                                # 원문/번역 전역 문자열 재구성
                                gt_src_col, gt_tgt_col = self._detect_source_target_cols(self.gt_data)
                                pd_src_col, pd_tgt_col = self._detect_source_target_cols(self.pred_data)
                                def combine(df, col):
                                    try:
                                        return ''.join(df[col].astype(str).fillna(''))
                                    except Exception:
                                        return ''
                                gt_src_all = combine(self.gt_data, gt_src_col) if gt_src_col else ''
                                pd_src_all = combine(self.pred_data, pd_src_col) if pd_src_col else ''
                                gt_tgt_all = combine(self.gt_data, gt_tgt_col) if gt_tgt_col else ''
                                pd_tgt_all = combine(self.pred_data, pd_tgt_col) if pd_tgt_col else ''

                                def char_delta_df(a: str, b: str) -> pd.DataFrame:
                                    ca, cb = Counter(a), Counter(b)
                                    all_chars = set(ca.keys()) | set(cb.keys())
                                    rows = []
                                    for ch in all_chars:
                                        ga = ca.get(ch, 0)
                                        gb = cb.get(ch, 0)
                                        rows.append({
                                            'char': ch,
                                            'codepoint': f"U+{ord(ch):04X}",
                                            'count_gt': ga,
                                            'count_pred': gb,
                                            'delta': ga - gb,
                                        })
                                    dfc = pd.DataFrame(rows)
                                    # 보기 좋게 공백류 이름 보조 컬럼 추가
                                    def nice_name(ch: str) -> str:
                                        names = {
                                            ' ': 'SPACE',
                                            '\u3000': 'IDEOGRAPHIC SPACE',
                                            '\u00A0': 'NO-BREAK SPACE',
                                            '\t': 'TAB',
                                            '\n': 'LF',
                                            '\r': 'CR',
                                        }
                                        if ch in [' ', '\t', '\n', '\r', '\u3000', '\u00A0']:
                                            return names.get(ch, '')
                                        return ''
                                    dfc['char_name'] = dfc['char'].map(nice_name)
                                    return dfc.sort_values(by='delta', ascending=False)

                                # 저장: 원문/번역 각각
                                src_delta_df = char_delta_df(gt_src_all, pd_src_all)
                                tgt_delta_df = char_delta_df(gt_tgt_all, pd_tgt_all)
                                src_delta_df.to_csv(os.path.join(csv_dir, '전역_문자_빈도_차이_원문.csv'), index=False, encoding='utf-8-sig')
                                tgt_delta_df.to_csv(os.path.join(csv_dir, '전역_문자_빈도_차이_번역.csv'), index=False, encoding='utf-8-sig')
                                # 로그에 상위 차이 간단 요약
                                top_removed = tgt_delta_df.sort_values('delta').head(10)
                                top_added = tgt_delta_df.sort_values('delta', ascending=False).head(10)
                                self.log('🧪 번역 전역 문자 빈도 차이(상위, 제거된 쪽):')
                                for _, r in top_removed.iterrows():
                                    ch = r['char']
                                    disp = r['char_name'] or (ch if ch.strip() != '' else 'SPACE')
                                    self.log(f"  - {disp} ({r['codepoint']}): Δ={int(r['delta'])} (GT {int(r['count_gt'])} vs Pred {int(r['count_pred'])})")
                                self.log('🧪 번역 전역 문자 빈도 차이(상위, 추가된 쪽):')
                                for _, r in top_added.iterrows():
                                    ch = r['char']
                                    disp = r['char_name'] or (ch if ch.strip() != '' else 'SPACE')
                                    self.log(f"  - {disp} ({r['codepoint']}): Δ={int(r['delta'])} (GT {int(r['count_gt'])} vs Pred {int(r['count_pred'])})")
                            except Exception as _ce:
                                # 빈도 분석 실패는 치명적이지 않으므로 무시하고 로그만 남김
                                self.log(f"전역 문자 빈도 차이 분석 실패: {_ce}")
                    except Exception:
                        pass
                    log_df.to_csv(os.path.join(csv_dir, '실행_로그.csv'), index=False, encoding='utf-8-sig')
                    self.log(f"📄 CSV로도 저장됨: {csv_dir}")
                except Exception as ce:
                    self.log(f"CSV 저장 중 오류: {ce}")

            self.log(f"\n💾 결과가 저장되었습니다: {output_file}")
            
        except Exception as e:
            self.log(f"결과 저장 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description='관자 원문 분할 정확도 평가')
    parser.add_argument('ground_truth', help='정답 파일 경로 (구병렬 기준)')
    parser.add_argument('prediction', help='예측 파일 경로 (output01 등)')
    parser.add_argument('--output', '-o', help='결과 저장 파일 경로', default='test_results/sa/row_eval_combined_similarity.xlsx')
    parser.add_argument('--csv-dir', help='각 시트를 CSV로도 저장할 디렉터리 경로(미지정 시 자동 생성)', default=None)
    parser.add_argument('--unit', choices=['sentence', 'row'], default='row', help='평가 단위: row(행 단위) 또는 sentence(식별자 그룹)')
    parser.add_argument('--project', choices=['pa', 'sa'], default='sa', help='프로젝트 유형에 따른 임계값 적용')
    parser.add_argument('--brief', action='store_true', help='간결 모드: 핵심 지표만 콘솔 출력')
    parser.add_argument('--minimal-summary', action='store_true', help='전체_요약(엑셀/CSV)에 핵심 지표만 저장')
    # 행 모드 자동 오프셋 감지 옵션
    parser.add_argument('--row-auto-shift', action='store_true', default=True, help='행 단위에서 시스템적 인덱스 오프셋 자동 감지/보정 시도(기본: 활성)')
    parser.add_argument('--row-auto-shift-range', type=int, default=50, help='행 오프셋 자동 감지 시 검사 범위(±N)')
    # 공백/구두점 무시 일치 판정 옵션
    parser.add_argument('--ignore-space-punct', action='store_true', help='원문/번역문 일치 여부를 판단할 때 공백/구두점을 무시하여 관대하게 계산')
    parser.add_argument('--ignore-space-only', action='store_true', default=True, help='공백(스페이스/개행/탭)만 무시하여 관대하게 계산(구두점은 유지) (기본: 활성)')
    parser.add_argument('--ignore-brackets', action='store_true', default=True, help='[-텍스트] 패턴을 비교 시 무시 (예: [-曰] 제거) (기본: 활성)')
    parser.add_argument('--warn-lenient-mismatch', action='store_true', help='관대 일치로는 동일하지만 엄격 기준으로는 불일치인 경우 경고 로그를 남김')
    parser.add_argument('--no-translation-match', action='store_true', help='번역문 기준 추가 평가를 비활성화')
    # 번역문 조사(토씨) 힌트 매칭 옵션
    parser.add_argument('--use-ko-particle-hint', action='store_true', help='번역문 내 한국어 조사 겹침을 힌트로 사용해 원문-번역 경계 매칭을 보조')
    parser.add_argument('--particle-weight', type=float, default=0.15, help='조사 힌트 가중치(0.0~1.0), 기본 0.15')
    # 중복 매칭 및 단조 정렬 옵션(문장 단위 매칭용)
    parser.add_argument('--max-dup-per-gt', type=int, default=1, help='각 GT 단위당 허용할 예측 중복 매칭 수(k). 예: 2')
    parser.add_argument('--no-monotonic', action='store_true', help='단조 제약을 해제(기본은 단조 유지)')
    # 행 단위 정렬(키 기반)
    parser.add_argument('--row-align-by-keys', action='store_true', help='행 단위에서 (문장식별자,구식별자) 순서대로 예측을 재정렬 후 1:1 매칭')
    # Unmatched GT 행 분석 옵션
    parser.add_argument('--analyze-unmatched', action='store_true', help='키 기반 정렬 후 매칭되지 않은 GT 행만 별도 분석 및 저장')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.ground_truth):
        print(f"정답 파일을 찾을 수 없습니다: {args.ground_truth}")
        sys.exit(1)
        
    if not os.path.exists(args.prediction):
        print(f"예측 파일을 찾을 수 없습니다: {args.prediction}")
        sys.exit(1)
    
    # 정확도 평가 실행
    evaluator = AccuracyEvaluator(
        args.ground_truth,
        args.prediction,
        project=args.project,
        brief=args.brief,
        minimal_summary=args.minimal_summary,
        ignore_space_punct=args.ignore_space_punct,
        ignore_space_only=args.ignore_space_only,
        ignore_brackets=args.ignore_brackets,
        use_ko_particle_hint=args.use_ko_particle_hint,
        particle_weight=args.particle_weight,
        max_dup_per_gt=args.max_dup_per_gt,
        monotonic_alignment=(not args.no_monotonic),
    )
    evaluator._warn_lenient_mismatch = bool(args.warn_lenient_mismatch)
    # 옵션 전달
    evaluator._detect_row_shift = bool(args.row_auto_shift)
    evaluator._row_shift_range = int(args.row_auto_shift_range)
    evaluator._row_align_by_keys = bool(args.row_align_by_keys)
    evaluator.load_data()
    results = evaluator.evaluate_accuracy(unit=args.unit, include_translation_matching=not args.no_translation_match)
    
    # unmatched GT 행 분석 (행 모드이고 키 기반 정렬 적용 시)
    if args.unit == 'row' and args.row_align_by_keys and args.analyze_unmatched:
        unmatched_output = None
        if args.output:
            base = os.path.splitext(os.path.basename(args.output))[0]
            parent = os.path.dirname(args.output) or '.'
            unmatched_output = os.path.join(parent, f"{base}_unmatched.xlsx")
        unmatched_results = evaluator.analyze_unmatched_gt_rows(output_file=unmatched_output)
    
    # 결과 출력 및 저장
    evaluator.print_detailed_results(results)
    # CSV 디렉터리 기본값 자동 설정: output 파일명 기반
    csv_dir = args.csv_dir
    if csv_dir is None and args.output:
        try:
            base = os.path.splitext(os.path.basename(args.output))[0]
            parent = os.path.dirname(args.output) or '.'
            csv_dir = os.path.join(parent, f"{base}_csv")
        except Exception:
            csv_dir = None
    evaluator.save_results(results, args.output, csv_dir=csv_dir)

if __name__ == "__main__":
    main()
