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
            elif '원문' in str(col) or 'source' in str(col).lower() or 'original' in str(col).lower():
                source_col = col
            elif '번역문' in str(col) or 'target' in str(col).lower() or 'translation' in str(col).lower():
                target_col = col
        
        if sentence_col is None or source_col is None or target_col is None:
            self.log(f"필요한 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(data.columns)}")
            self.log(f"필요 컬럼: 문장식별자, 원문, 번역문")
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
    
    def calculate_sentence_accuracy(self, gt_segments: List[Dict[str, str]], pred_segments: List[Dict[str, str]], sentence_id: int) -> Dict[str, float]:
        """단일 문장의 분할 정확도 계산 (원문 + 번역문)"""
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
        
        # 🔄 새로운 방식: 원문 기준 페어 매칭 후 번역문 비교
        # 1) 원문이 정확히 일치하는 세그먼트 쌍 찾기
        matched_pairs = []
        unmatched_gt = gt_segments.copy()
        unmatched_pred = pred_segments.copy()
        
        for gt_seg in gt_segments:
            for pred_seg in pred_segments:
                if (gt_seg['source'] == pred_seg['source'] and 
                    gt_seg in unmatched_gt and pred_seg in unmatched_pred):
                    matched_pairs.append((gt_seg, pred_seg))
                    unmatched_gt.remove(gt_seg)
                    unmatched_pred.remove(pred_seg)
                    break
        
        # 2) 매칭된 쌍에서 번역문 일치 여부 확인
        correct_translation_pairs = 0
        for gt_seg, pred_seg in matched_pairs:
            if gt_seg['target'] == pred_seg['target']:
                correct_translation_pairs += 1
        
        # 3) 정확도 계산
        total_gt_segments = len(gt_segments)
        total_pred_segments = len(pred_segments)
        matched_segments = len(matched_pairs)
        
        # 원문 기준 정확도
        source_precision = matched_segments / total_pred_segments if total_pred_segments > 0 else 0.0
        source_recall = matched_segments / total_gt_segments if total_gt_segments > 0 else 0.0
        source_f1_score = (2 * source_precision * source_recall / (source_precision + source_recall) 
                          if source_precision + source_recall > 0 else 0.0)
        
        # 번역문 기준 정확도 (원문이 일치하는 쌍에서만)
        if matched_segments > 0:
            target_accuracy = correct_translation_pairs / matched_segments
            target_precision = target_accuracy  # 매칭된 쌍에서의 번역문 정확도
            target_recall = correct_translation_pairs / total_gt_segments  # 전체 대비 올바른 번역 비율
            target_f1_score = (2 * target_precision * target_recall / (target_precision + target_recall)
                              if target_precision + target_recall > 0 else 0.0)
        else:
            target_accuracy = 0.0
            target_precision = 0.0
            target_recall = 0.0
            target_f1_score = 0.0
        
        # 전체 정확도
        f1_score = (source_f1_score + target_f1_score) / 2
        
        # 🆕 개선된 부분 일치 계산 (세트 기반 + 문자열 유사도)
        
        # 1) 기존 세트 기반 Jaccard 유사도
        gt_source_set = set(gt_sources)
        pred_source_set = set(pred_sources)
        source_jaccard = (len(gt_source_set.intersection(pred_source_set)) / 
                         len(gt_source_set.union(pred_source_set)) 
                         if len(gt_source_set.union(pred_source_set)) > 0 else 0.0)
        
        # 2) 문자열 유사도 기반 부분 일치 (전체 텍스트)
        source_text_similarity = self.calculate_text_similarity(gt_source_full, pred_source_full)
        target_text_similarity = self.calculate_text_similarity(gt_target_full, pred_target_full)
        
        # 3) 세그먼트별 평균 유사도
        if len(gt_segments) > 0 and len(pred_segments) > 0:
            # 원문 세그먼트별 최대 유사도 평균
            source_segment_similarities = []
            for gt_src in gt_sources:
                max_sim = max([self.calculate_text_similarity(gt_src, pred_src) for pred_src in pred_sources])
                source_segment_similarities.append(max_sim)
            source_avg_similarity = sum(source_segment_similarities) / len(source_segment_similarities)
            
            # 번역문 세그먼트별 최대 유사도 평균 (매칭된 쌍에서만)
            if matched_pairs:
                target_segment_similarities = []
                for gt_seg, pred_seg in matched_pairs:
                    sim = self.calculate_text_similarity(gt_seg['target'], pred_seg['target'])
                    target_segment_similarities.append(sim)
                target_avg_similarity = sum(target_segment_similarities) / len(target_segment_similarities)
            else:
                target_avg_similarity = 0.0
        else:
            source_avg_similarity = 0.0
            target_avg_similarity = 0.0
        
        # 4) 최종 부분 일치 점수 (여러 방식의 평균)
        source_partial_match = (source_jaccard + source_text_similarity + source_avg_similarity) / 3
        target_partial_match = target_avg_similarity  # 매칭된 쌍에서의 번역문 유사도
        partial_match = (source_partial_match + target_partial_match) / 2
        
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
            # 🆕 새로운 페어 매칭 기반 지표들
            'matched_pairs': matched_segments,
            'correct_translation_pairs': correct_translation_pairs,
            # 🆕 부분 일치 세부 지표들
            'source_jaccard': source_jaccard,
            'source_text_similarity': source_text_similarity,
            'target_text_similarity': target_text_similarity,
            'source_avg_similarity': source_avg_similarity,
            'target_avg_similarity': target_avg_similarity,
            'target_accuracy': target_accuracy if matched_segments > 0 else 0.0
        }
    
    def evaluate_accuracy(self) -> Dict[str, any]:
        """전체 정확도 평가"""
        if self.gt_data is None or self.pred_data is None:
            print("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")
            return {}
        
        # 문장식별자별로 그룹화
        gt_grouped = self.group_by_sentence_id(self.gt_data)
        pred_grouped = self.group_by_sentence_id(self.pred_data)
        
        self.log(f"\n정답 데이터 문장 수: {len(gt_grouped)}")
        self.log(f"예측 데이터 문장 수: {len(pred_grouped)}")
        
        # 공통 문장 ID 추출
        common_ids = set(gt_grouped.keys()).intersection(set(pred_grouped.keys()))
        missing_in_pred = set(gt_grouped.keys()) - set(pred_grouped.keys())
        extra_in_pred = set(pred_grouped.keys()) - set(gt_grouped.keys())
        
        self.log(f"공통 문장 수: {len(common_ids)}")
        if missing_in_pred:
            self.log(f"예측에서 누락된 문장 ID: {sorted(missing_in_pred)}")
        if extra_in_pred:
            self.log(f"예측에서 추가된 문장 ID: {sorted(extra_in_pred)}")
        
        # 각 문장별 정확도 계산
        sentence_results = []
        overall_metrics = defaultdict(list)
        source_mismatch_count = 0  # 원문 불일치 카운트
        
        for sentence_id in sorted(common_ids):
            gt_segments = gt_grouped[sentence_id]
            pred_segments = pred_grouped[sentence_id]
            
            accuracy = self.calculate_sentence_accuracy(gt_segments, pred_segments, sentence_id)
            accuracy['sentence_id'] = sentence_id
            sentence_results.append(accuracy)
            
            # 원문 불일치 카운트
            if not accuracy['source_text_match']:
                source_mismatch_count += 1
            
            # 전체 메트릭 누적
            for metric, value in accuracy.items():
                if metric != 'sentence_id':
                    overall_metrics[metric].append(value)
        
        # 원문 불일치 요약 로깅 (평가 대상에 포함)
        if source_mismatch_count > 0:
            self.log(f"\n� 원문 불일치 요약 (평가 대상에 포함됨):")
            self.log(f"   총 {source_mismatch_count}개 문장에서 원문 불일치 발생")
            self.log(f"   전체 대비 비율: {source_mismatch_count/len(common_ids):.1%}")
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
                'total_sentences': len(common_ids),
                'missing_in_prediction': len(missing_in_pred),
                'extra_in_prediction': len(extra_in_pred),
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
        self.log(f"  • 평가된 문장 수: {summary['total_sentences']} (원문 불일치 포함)")
        self.log(f"  • 예측에서 누락: {summary['missing_in_prediction']}")
        self.log(f"  • 예측에서 추가: {summary['extra_in_prediction']}")
        self.log(f"  • � 원문 불일치: {summary['source_mismatch_count']}개 ({summary['source_mismatch_count']/summary['total_sentences']:.1%}) - 평가에 포함됨")
        self.log(f"  • 평균 정답 세그먼트 수: {overall['avg_gt_segments']:.1f}")
        self.log(f"  • 평균 예측 세그먼트 수: {overall['avg_pred_segments']:.1f}")
        
        self.log(f"\n🎯 주요 정확도 지표 (개선된 페어 매칭 방식, 원문 불일치 포함):")
        self.log(f"  📌 평가 방식 안내:")
        self.log(f"    - 완전일치: 원문, 번역문, 세그먼트 수가 모두 정확히 일치")
        self.log(f"    - 부분일치: 50% 이상 유사성 (Jaccard + 텍스트 유사도 + 세그먼트 유사도)")
        self.log(f"    - 원문불일치: 평가 대상에 포함하여 전체 성능에 반영")
        self.log(f"")
        self.log(f"  • 완전 일치율: {overall['avg_exact_match']:.1%}")
        self.log(f"  • 전체 텍스트 일치율: {overall['avg_text_match']:.1%}")
        self.log(f"    - 원문 일치율: {overall['avg_source_text_match']:.1%}")
        self.log(f"    - 번역문 일치율: {overall['avg_target_text_match']:.1%}")
        self.log(f"  • 세그먼트 수 일치율: {overall['avg_segment_count_match']:.1%}")
        self.log(f"  • 📊 페어 매칭 분석:")
        self.log(f"    - 평균 매칭된 쌍: {overall.get('avg_matched_pairs', 0):.1f}개")
        self.log(f"    - 평균 번역 정확 쌍: {overall.get('avg_correct_translation_pairs', 0):.1f}개") 
        self.log(f"    - 번역 정확도 (매칭된 쌍 기준): {overall.get('avg_target_accuracy', 0):.1%}")
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
        self.log("ID\tF1\t완전일치\t세그먼트수(정답/예측)\t원문F1\t번역문F1\t부분일치")
        for result in sentence_results[:10]:
            self.log(f"{result['sentence_id']}\t{result['f1_score']:.2f}\t{result['exact_match']:.0f}\t\t{result['gt_segments']}/{result['pred_segments']}\t\t{result['source_f1_score']:.2f}\t{result['target_f1_score']:.2f}\t{result['partial_match']:.2f}")
        
        self.log(f"\n📉 성능 하위 10개 문장:")
        self.log("ID\tF1\t완전일치\t세그먼트수(정답/예측)\t원문F1\t번역문F1\t부분일치")
        for result in sentence_results[-10:]:
            self.log(f"{result['sentence_id']}\t{result['f1_score']:.2f}\t{result['exact_match']:.0f}\t\t{result['gt_segments']}/{result['pred_segments']}\t\t{result['source_f1_score']:.2f}\t{result['target_f1_score']:.2f}\t{result['partial_match']:.2f}")
    
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
