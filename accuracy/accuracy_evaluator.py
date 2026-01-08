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

class AccuracyEvaluator:
    def __init__(self, ground_truth_file: str, prediction_file: str):
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
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """문자열 유사도 계산 (SequenceMatcher 사용)"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        return difflib.SequenceMatcher(None, text1, text2).ratio()
        
    def log(self, message: str):
        """실행 로그 기록"""
        print(message)
        self.execution_log.append(message)
        
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
        """텍스트 정규화 (공백, 특수문자 제거)"""
        if pd.isna(text):
            return ""
        return str(text).strip().replace(" ", "").replace("\n", "")
    
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
            elif '원문' in str(col) or 'source' in str(col).lower() or 'original' in str(col).lower():
                source_col = col
            elif '번역문' in str(col) or 'target' in str(col).lower() or 'translation' in str(col).lower():
                target_col = col
        
        if sentence_col is None or source_col is None or target_col is None:
            self.log(f"필요한 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(data.columns)}")
            self.log(f"필요 컬럼: 문장식별자 또는 문단식별자, 원문, 번역문")
            sys.exit(1)
            
        self.log(f"사용 컬럼: 문장식별자={sentence_col}, 원문={source_col}, 번역문={target_col}")
        
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
    
    def normalize_for_matching(self, text: str) -> str:
        """매칭용 텍스트 정규화 (더 관대한 매칭을 위해)"""
        if pd.isna(text):
            return ""
        # 공백, 탭, 개행, 구두점 제거하고 소문자 변환
        import re
        normalized = str(text).strip()
        normalized = re.sub(r'[\s\t\n\r]+', '', normalized)  # 모든 공백류 제거
        normalized = re.sub(r'[。，、；：！？""''「」『』（）〈〉《》【】〔〕]+', '', normalized)  # 구두점 제거
        return normalized.lower()
    
    def calculate_sentence_accuracy(self, gt_segments: List[Dict[str, str]], pred_segments: List[Dict[str, str]], sentence_id: int) -> Dict[str, float]:
        """단일 문장의 분할 정확도 계산 (원문 기준 순서대로 매칭 + 번역문 평가)"""
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
        
        # 🚨 원문 일치 여부 확인 및 불일치 로깅
        source_text_match = gt_source_full == pred_source_full
        if not source_text_match:
            self.log(f"⚠️  문장 {sentence_id}: 원문 불일치 감지!")
            self.log(f"   정답 원문: '{gt_source_full}'")
            self.log(f"   예측 원문: '{pred_source_full}'")
            # 원문이 다르면 길이 차이도 로깅
            len_diff = len(pred_source_full) - len(gt_source_full)
            if len_diff != 0:
                self.log(f"   길이 차이: {len_diff:+d} 글자")
            
            # 상세 정보 저장
            self.source_mismatch_details.append({
                'sentence_id': sentence_id,
                'gt_source': gt_source_full,
                'pred_source': pred_source_full,
                'length_diff': len_diff,
                'similarity': self.calculate_text_similarity(gt_source_full, pred_source_full)
            })
        
        # 번역문 일치 여부
        target_text_match = gt_target_full == pred_target_full
        # 전체 텍스트 일치 (원문 + 번역문 모두)
        text_match = source_text_match and target_text_match
        
        # 세그먼트 수 일치 여부
        segment_count_match = len(gt_segments) == len(pred_segments)
        
        # 완전 일치 (순서와 내용 모두 일치)
        exact_match = gt_segments == pred_segments
        
        # 🎯 핵심: 원문 기준 순서대로 매칭 후 번역문 평가 (PA 방식)
        self.log(f"   🔄 문장 {sentence_id}: 원문 기준 순서대로 매칭 시작...")
        
        # 원문 기준 순서대로 정렬 찾기
        alignments = self.find_best_alignment_by_source(gt_sources, pred_sources)
        
        # 매칭된 원문-번역문 쌍에서 번역문 정확도 평가
        correct_translation_pairs = 0
        translation_similarities = []
        matched_pairs = []
        
        for gt_idx, pred_idx, source_similarity in alignments:
            if pred_idx >= 0:  # 매칭 성공
                gt_target = gt_targets[gt_idx]
                pred_target = pred_targets[pred_idx]
                
                # 번역문 비교 평가
                gt_target_norm = self.normalize_for_matching(gt_target)
                pred_target_norm = self.normalize_for_matching(pred_target)
                
                # 번역문 정확도 판정
                if gt_target == pred_target or gt_target_norm == pred_target_norm:
                    correct_translation_pairs += 1
                    translation_similarities.append(1.0)
                else:
                    # 부분 일치 유사도 계산
                    similarity = self.calculate_text_similarity(gt_target, pred_target)
                    translation_similarities.append(similarity)
                    # 90% 이상 유사하면 정확한 번역으로 간주
                    if similarity >= 0.9:
                        correct_translation_pairs += 1
                
                # 매칭 정보 저장
                matched_pairs.append({
                    'gt_idx': gt_idx, 'pred_idx': pred_idx,
                    'gt_seg': gt_segments[gt_idx], 'pred_seg': pred_segments[pred_idx],
                    'source_match_type': 'source_based',
                    'source_similarity': source_similarity
                })
            else:  # 매칭 실패
                translation_similarities.append(0.0)
        
        # 3) 정확도 지표 계산
        total_gt_segments = len(gt_segments)
        total_pred_segments = len(pred_segments)
        matched_segments = len(matched_pairs)
        
        # 원문 매칭 기반 정확도
        source_precision = matched_segments / total_pred_segments if total_pred_segments > 0 else 0.0
        source_recall = matched_segments / total_gt_segments if total_gt_segments > 0 else 0.0
        source_f1_score = (2 * source_precision * source_recall / (source_precision + source_recall) 
                          if source_precision + source_recall > 0 else 0.0)
        
        # 번역문 정확도 (매칭된 원문 쌍에서만 평가)
        if matched_segments > 0:
            target_accuracy = correct_translation_pairs / matched_segments
            target_precision = target_accuracy  # 매칭된 쌍에서의 번역문 정확도
            target_recall = correct_translation_pairs / total_gt_segments  # 전체 대비 올바른 번역 비율
            target_f1_score = (2 * target_precision * target_recall / (target_precision + target_recall)
                              if target_precision + target_recall > 0 else 0.0)
            # 번역문 평균 유사도 계산
            target_avg_similarity = sum(translation_similarities) / len(translation_similarities)
        else:
            target_accuracy = 0.0
            target_precision = 0.0
            target_recall = 0.0
            target_f1_score = 0.0
            target_avg_similarity = 0.0
        
        # 전체 F1 점수 (원문 매칭과 번역문 정확도의 조화평균)
        f1_score = (source_f1_score + target_f1_score) / 2
        
        # 🆕 부분 일치 계산 (원문 기준)
        # 1) 원문 세그먼트별 최대 유사도 평균
        source_segment_similarities = []
        for gt_src in gt_sources:
            if pred_sources:
                max_sim = max([self.calculate_text_similarity(gt_src, pred_src) for pred_src in pred_sources])
                source_segment_similarities.append(max_sim)
            else:
                source_segment_similarities.append(0.0)
        source_avg_similarity = sum(source_segment_similarities) / len(source_segment_similarities) if source_segment_similarities else 0.0
        
        # 2) 전체 텍스트 유사도
        source_text_similarity = self.calculate_text_similarity(gt_source_full, pred_source_full)
        target_text_similarity = self.calculate_text_similarity(gt_target_full, pred_target_full)
        
        # 3) 세트 기반 Jaccard 유사도 (원문)
        gt_source_set = set(gt_sources)
        pred_source_set = set(pred_sources)
        source_jaccard = (len(gt_source_set.intersection(pred_source_set)) / 
                         len(gt_source_set.union(pred_source_set)) 
                         if len(gt_source_set.union(pred_source_set)) > 0 else 0.0)
        
        # 4) 최종 부분 일치 점수
        source_partial_match = (source_jaccard + source_text_similarity + source_avg_similarity) / 3
        target_partial_match = target_avg_similarity  # 매칭된 쌍에서의 번역문 유사도
        partial_match = (source_partial_match + target_partial_match) / 2
        
        # 🆕 매칭 상세 정보 로깅
        if matched_segments != total_gt_segments:
            self.log(f"   문장 {sentence_id} 매칭 상세: 원문기준매칭 {matched_segments}개, 미매칭 {total_gt_segments - matched_segments}개")
        
        return {
            'text_match': float(text_match),
            'source_text_match': float(source_text_match),
            'target_text_match': float(target_text_match),
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
            # 🆕 원문 기준 매칭 지표들
            'matched_pairs': matched_segments,
            'correct_translation_pairs': correct_translation_pairs,
            'source_based_matches': matched_segments,  # 원문 기준 매칭 수
            # 🆕 번역문 평가 지표들  
            'target_accuracy': target_accuracy,
            'target_avg_similarity': target_avg_similarity,
            # 🆕 부분 일치 세부 지표들
            'source_jaccard': source_jaccard,
            'source_text_similarity': source_text_similarity,
            'target_text_similarity': target_text_similarity,
            'source_avg_similarity': source_avg_similarity
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
    
    def evaluate_accuracy(self) -> Dict[str, any]:
        """전체 정확도 평가 (원문 기준으로만 매칭)"""
        if self.gt_data is None or self.pred_data is None:
            print("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            return {}
        
        # 문장식별자별로 그룹화
        gt_grouped = self.group_by_sentence_id(self.gt_data)
        pred_grouped = self.group_by_sentence_id(self.pred_data)
        
        self.log(f"\n정답 데이터 문장 수: {len(gt_grouped)}")
        self.log(f"예측 데이터 문장 수: {len(pred_grouped)}")
        
        # 🆕 원문 기준으로만 매칭 (식별자 매칭 제거)
        self.log(f"\n� 원문 기준 매칭 시작...")
        self.log(f"   💡 매칭 방식: 문장식별자 무시하고 순수 원문 유사도로만 매칭")
        
        # 모든 정답과 예측 문장들을 원문 기준으로 매칭
        all_gt_sentences = [(gt_id, gt_segments) for gt_id, gt_segments in gt_grouped.items()]
        all_pred_sentences = [(pred_id, pred_segments) for pred_id, pred_segments in pred_grouped.items()]
        
        # 원문 기준 스마트 매칭
        final_matches = self.smart_match_sentences_by_source_only(all_gt_sentences, all_pred_sentences)
        
        self.log(f"\n📊 최종 매칭 결과 (원문 기준):")
        self.log(f"  • 총 매칭된 문장 쌍: {len(final_matches)}개")
        self.log(f"  • 매칭되지 않은 정답: {len(gt_grouped) - len(final_matches)}개")
        self.log(f"  • 매칭되지 않은 예측: {len(pred_grouped) - len(final_matches)}개")
        
        # 각 문장별 정확도 계산
        sentence_results = []
        overall_metrics = defaultdict(list)
        source_mismatch_count = 0  # 원문 불일치 카운트
        
        for gt_id, pred_id, match_similarity in final_matches:
            gt_segments = gt_grouped[gt_id]
            pred_segments = pred_grouped[pred_id]
            
            accuracy = self.calculate_sentence_accuracy(gt_segments, pred_segments, gt_id)
            accuracy['sentence_id'] = gt_id
            accuracy['matched_pred_id'] = pred_id
            accuracy['match_similarity'] = match_similarity
            sentence_results.append(accuracy)
            
            # 원문 불일치 카운트
            if not accuracy['source_text_match']:
                source_mismatch_count += 1
            
            # 전체 메트릭 누적
            for metric, value in accuracy.items():
                if metric not in ['sentence_id', 'matched_pred_id', 'match_similarity']:
                    overall_metrics[metric].append(value)
        
        # 원문 불일치 요약 로깅 (평가 대상에 포함)
        if source_mismatch_count > 0:
            self.log(f"\n🔍 원문 불일치 요약 (평가 대상에 포함됨):")
            self.log(f"   총 {source_mismatch_count}개 문장에서 원문 불일치 발생")
            self.log(f"   전체 대비 비율: {source_mismatch_count/len(final_matches):.1%}")
            self.log(f"   원문 불일치는 평가에 포함되어 전체 정확도에 반영됩니다.")
        else:
            self.log(f"\n✅ 모든 문장의 원문이 일치합니다!")
        
        # 전체 평균 계산
        overall_accuracy = {}
        for metric, values in overall_metrics.items():
            if metric in ['gt_segments', 'pred_segments']:
                overall_accuracy[f'total_{metric}'] = sum(values)
                overall_accuracy[f'avg_{metric}'] = sum(values) / len(values) if values else 0
            else:
                overall_accuracy[f'avg_{metric}'] = sum(values) / len(values) if values else 0
        
        return {
            'sentence_results': sentence_results,
            'overall_accuracy': overall_accuracy,
            'summary': {
                'total_sentences': len(final_matches),
                'total_gt_sentences': len(gt_grouped),
                'total_pred_sentences': len(pred_grouped),
                'source_based_matches': len(final_matches),  # 원문 기준 매칭만
                'unmatched_gt': len(gt_grouped) - len(final_matches),
                'unmatched_pred': len(pred_grouped) - len(final_matches),
                'source_mismatch_count': source_mismatch_count  # 원문 불일치 개수 추가
            }
        }
    
    def print_detailed_results(self, results: Dict[str, any]):
        """상세 결과 출력"""
        self.log("\n" + "="*80)
        self.log("정확도 평가 결과")
        self.log("="*80)
        
        # 전체 요약
        summary = results['summary']
        overall = results['overall_accuracy']
        
        self.log(f"\n📊 전체 요약:")
        self.log(f"  • 📋 매칭 요약:")
        self.log(f"    - 원문 기준 매칭: {summary.get('source_based_matches', 0)}개")
        self.log(f"    - 총 평가 문장 쌍: {summary['total_sentences']}개")
        self.log(f"  • 🗂️ 데이터 현황:")
        self.log(f"    - 정답 문장 총 개수: {summary.get('total_gt_sentences', 0)}개")
        self.log(f"    - 예측 문장 총 개수: {summary.get('total_pred_sentences', 0)}개")
        self.log(f"    - 매칭되지 않은 정답: {summary.get('unmatched_gt', 0)}개")
        self.log(f"    - 매칭되지 않은 예측: {summary.get('unmatched_pred', 0)}개")
        self.log(f"  • ⚠️ 원문 불일치: {summary['source_mismatch_count']}개 ({summary['source_mismatch_count']/summary['total_sentences']:.1%}) - 평가에 포함됨")
        self.log(f"  • 평균 정답 세그먼트 수: {overall['avg_gt_segments']:.1f}")
        self.log(f"  • 평균 예측 세그먼트 수: {overall['avg_pred_segments']:.1f}")
        
        self.log(f"\n🎯 주요 정확도 지표 (원문 기준 순수 매칭 + 번역문 평가):")
        self.log(f"  📌 평가 방식 안내:")
        self.log(f"    - 문장식별자 무시: 식별자와 관계없이 순수 원문 유사도로만 매칭")
        self.log(f"    - 원문-번역문 쌍 단위 평가: 각 행은 [원문,번역문] 한 쌍")
        self.log(f"    - 순서대로 매칭: 정답 원문을 순서대로 처리하여 가장 유사한 예측 원문 찾기")
        self.log(f"    - 번역문 평가: 매칭된 쌍의 번역문 정확도 측정")
        self.log(f"    - 중복 방지: 한 예측 쌍은 하나의 정답 쌍과만 매칭")
        self.log(f"")
        self.log(f"  • 완전 일치율: {overall['avg_exact_match']:.1%}")
        self.log(f"  • 전체 텍스트 일치율: {overall['avg_text_match']:.1%}")
        self.log(f"    - 원문 일치율: {overall['avg_source_text_match']:.1%}")
        self.log(f"    - 번역문 일치율: {overall['avg_target_text_match']:.1%}")
        self.log(f"  • 세그먼트 수 일치율: {overall['avg_segment_count_match']:.1%}")
        self.log(f"  • 📊 원문 기준 순서대로 매칭 분석:")
        self.log(f"    - 매칭된 원문-번역문 쌍: {overall.get('avg_matched_pairs', 0):.1f}개")
        self.log(f"    - 원문 기준 매칭 수: {overall.get('avg_source_based_matches', 0):.1f}개")
        self.log(f"  • 📊 번역문 정확도 평가 (원문이 매칭된 쌍에서만):")
        self.log(f"    - 번역문 정확한 쌍: {overall.get('avg_correct_translation_pairs', 0):.1f}개") 
        self.log(f"    - 번역문 정확도: {overall.get('avg_target_accuracy', 0):.1%}")
        self.log(f"    - 번역문 평균 유사도: {overall.get('avg_target_avg_similarity', 0):.1%}")
        self.log(f"  • 부분 일치율: {overall['avg_partial_match']:.1%}")
        self.log(f"    - 원문 부분 일치율: {overall['avg_source_partial_match']:.1%}")
        self.log(f"      • Jaccard 유사도: {overall.get('avg_source_jaccard', 0):.1%}")
        self.log(f"      • 전체 텍스트 유사도: {overall.get('avg_source_text_similarity', 0):.1%}")
        self.log(f"      • 세그먼트별 평균 유사도: {overall.get('avg_source_avg_similarity', 0):.1%}")
        self.log(f"    - 번역문 부분 일치율: {overall['avg_target_partial_match']:.1%}")
        self.log(f"      • 전체 텍스트 유사도: {overall.get('avg_target_text_similarity', 0):.1%}")
        self.log(f"      • 매칭된 쌍 평균 유사도: {overall.get('avg_target_avg_similarity', 0):.1%}")
        self.log(f"  • F1 점수: {overall['avg_f1_score']:.1%}")
        self.log(f"    - 원문 F1: {overall['avg_source_f1_score']:.1%}")
        self.log(f"    - 번역문 F1: {overall['avg_target_f1_score']:.1%}")
        self.log(f"  • 정밀도: {overall['avg_precision']:.1%}")
        self.log(f"  • 재현율: {overall['avg_recall']:.1%}")
        
        # 문장별 상세 결과 (상위 10개 + 하위 10개)
        sentence_results = results['sentence_results']
        sentence_results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        self.log(f"\n📈 성능 상위 10개 문장:")
        self.log("ID\tF1\t완전일치\t세그먼트수(정답/예측)\t원문F1\t번역문F1\t원문매칭\t번역정확")
        for result in sentence_results[:10]:
            self.log(f"{result['sentence_id']}\t{result['f1_score']:.2f}\t{result['exact_match']:.0f}\t\t{result['gt_segments']}/{result['pred_segments']}\t\t{result['source_f1_score']:.2f}\t{result['target_f1_score']:.2f}\t{result['matched_pairs']}\t\t{result['correct_translation_pairs']}")
        
        self.log(f"\n📉 성능 하위 10개 문장:")
        self.log("ID\tF1\t완전일치\t세그먼트수(정답/예측)\t원문F1\t번역문F1\t원문매칭\t번역정확")
        for result in sentence_results[-10:]:
            self.log(f"{result['sentence_id']}\t{result['f1_score']:.2f}\t{result['exact_match']:.0f}\t\t{result['gt_segments']}/{result['pred_segments']}\t\t{result['source_f1_score']:.2f}\t{result['target_f1_score']:.2f}\t{result['matched_pairs']}\t\t{result['correct_translation_pairs']}")
    
    def save_results(self, results: Dict[str, any], output_file: str):
        """결과를 엑셀 파일로 저장"""
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 문장별 상세 결과
                sentence_df = pd.DataFrame(results['sentence_results'])
                sentence_df.to_excel(writer, sheet_name='문장별_상세결과', index=False)
                
                # 전체 요약
                summary_data = []
                for key, value in results['overall_accuracy'].items():
                    summary_data.append({'지표': key, '값': value})
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='전체_요약', index=False)
                
                # 🆕 실행 로그 추가
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
                
            self.log(f"\n💾 결과가 저장되었습니다: {output_file}")
            
        except Exception as e:
            self.log(f"결과 저장 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description='관자 원문 분할 정확도 평가')
    parser.add_argument('ground_truth', help='정답 파일 경로 (구병렬 기준)')
    parser.add_argument('prediction', help='예측 파일 경로 (output01 등)')
    parser.add_argument('--output', '-o', help='결과 저장 파일 경로', default='accuracy_results.xlsx')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.ground_truth):
        print(f"정답 파일을 찾을 수 없습니다: {args.ground_truth}")
        sys.exit(1)
        
    if not os.path.exists(args.prediction):
        print(f"예측 파일을 찾을 수 없습니다: {args.prediction}")
        sys.exit(1)
    
    # 정확도 평가 실행
    evaluator = AccuracyEvaluator(args.ground_truth, args.prediction)
    evaluator.load_data()
    results = evaluator.evaluate_accuracy()
    
    # 결과 출력 및 저장
    evaluator.print_detailed_results(results)
    evaluator.save_results(results, args.output)

if __name__ == "__main__":
    main()
