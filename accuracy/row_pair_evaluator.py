#!/usr/bin/env python3
"""
행 단위 원문-번역문 쌍 정확도 평가 도구
각 행의 원문+번역문을 결합하여 텍스트 유사도로 정확도 측정
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
import argparse
import os
from datetime import datetime
import difflib
from collections import defaultdict

class RowPairAccuracyEvaluator:
    """행 단위 원문-번역문 쌍 정확도 평가 클래스"""
    
    def __init__(self, similarity_threshold: float = 0.5, skip_threshold: float = 0.1):
        """
        초기화
        
        Args:
            similarity_threshold: 부분 일치 판정 임계값 (기본값: 0.5)
            skip_threshold: 건너뛰기 판정 임계값 (기본값: 0.1)
        """
        self.similarity_threshold = similarity_threshold
        self.skip_threshold = skip_threshold
        self.execution_log = []
        
    def log(self, message: str):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.execution_log.append(log_message)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Excel 파일 로드"""
        try:
            df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
            self.log(f"✅ 파일 로드 성공: {file_path} ({len(df)}행)")
            return df
        except Exception as e:
            self.log(f"❌ 파일 로드 실패: {file_path} - {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).strip()
        # 공백 정규화
        text = ' '.join(text.split())
        return text
    
    def combine_source_target(self, source: str, target: str) -> str:
        """원문과 번역문을 결합 (번역문 우선)"""
        source_clean = self.preprocess_text(source)
        target_clean = self.preprocess_text(target)
        
        # 번역문을 기준으로 하되, 원문도 참고용으로 포함
        return f"{target_clean} | {source_clean}"
    
    def extract_target_only(self, source: str, target: str) -> str:
        """번역문만 추출 (순수 번역문 기준 매칭용)"""
        return self.preprocess_text(target)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간 유사도 계산 (difflib 사용)"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def find_best_alignment(self, gt_combined: List[str], pred_combined: List[str]) -> List[Tuple[int, int, float]]:
        """최적 정렬 찾기 (동적 프로그래밍 기반)"""
        self.log("🔄 최적 정렬 계산 중...")
        
        gt_len = len(gt_combined)
        pred_len = len(pred_combined)
        
        # 간단한 휴리스틱: 각 정답 행에 대해 가장 유사한 예측 행 찾기
        alignments = []
        used_pred = set()
        
        for gt_idx, gt_text in enumerate(gt_combined):
            best_pred_idx = -1
            best_similarity = 0.0
            
            # 현재 위치 근처에서 우선 검색 (±3 범위)
            search_start = max(0, gt_idx - 3)
            search_end = min(pred_len, gt_idx + 4)
            
            for pred_idx in range(search_start, search_end):
                if pred_idx in used_pred:
                    continue
                    
                pred_text = pred_combined[pred_idx]
                similarity = self.calculate_text_similarity(gt_text, pred_text)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pred_idx = pred_idx
            
            # 근처에서 못 찾으면 전체 검색
            if best_similarity < self.skip_threshold:
                for pred_idx in range(pred_len):
                    if pred_idx in used_pred:
                        continue
                        
                    pred_text = pred_combined[pred_idx]
                    similarity = self.calculate_text_similarity(gt_text, pred_text)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pred_idx = pred_idx
            
            if best_pred_idx >= 0:
                alignments.append((gt_idx, best_pred_idx, best_similarity))
                used_pred.add(best_pred_idx)
                self.log(f"  정답 {gt_idx+1} → 예측 {best_pred_idx+1} (유사도: {best_similarity:.2f})")
            else:
                alignments.append((gt_idx, -1, 0.0))
                self.log(f"  정답 {gt_idx+1} → 매칭 없음")
        
        self.log(f"✅ 정렬 완료: {len(alignments)}개 쌍")
        return alignments
    
    def calculate_row_accuracy(self, gt_text: str, pred_text: str, gt_idx: int, pred_idx: int) -> Dict[str, Any]:
        """행별 정확도 계산"""
        
        # 완전 일치 체크
        exact_match = gt_text == pred_text
        
        # 텍스트 유사도 계산
        text_similarity = self.calculate_text_similarity(gt_text, pred_text)
        
        # 길이 기반 유사도
        if len(gt_text) > 0 and len(pred_text) > 0:
            length_similarity = min(len(gt_text), len(pred_text)) / max(len(gt_text), len(pred_text))
        else:
            length_similarity = 0.0
        
        # 종합 유사도
        final_similarity = (text_similarity * 0.8 + length_similarity * 0.2)
        
        # 부분 일치 판정
        has_content = final_similarity >= self.similarity_threshold
        
        return {
            'gt_row_index': gt_idx + 1,
            'pred_row_index': pred_idx + 1 if pred_idx >= 0 else -1,
            'exact_match': 1 if exact_match else 0,
            'text_similarity': text_similarity,
            'length_similarity': length_similarity,
            'final_similarity': final_similarity,
            'has_content': 1 if has_content else 0,
            'gt_length': len(gt_text),
            'pred_length': len(pred_text) if pred_idx >= 0 else 0,
            'gt_text': gt_text,
            'pred_text': pred_text if pred_idx >= 0 else ""
        }
    
    def evaluate_accuracy(self, ground_truth_file: str, prediction_file: str) -> Dict[str, Any]:
        """행 단위 정확도 평가 실행"""
        
        self.log("🚀 행 단위 원문-번역문 쌍 정확도 평가 시작")
        self.log(f"   정답 파일: {ground_truth_file}")
        self.log(f"   예측 파일: {prediction_file}")
        
        # 데이터 로드
        gt_df = self.load_data(ground_truth_file)
        pred_df = self.load_data(prediction_file)
        
        # 각 행의 원문+번역문 결합
        self.log("🔄 원문-번역문 결합 중...")
        
        gt_combined = []
        for _, row in gt_df.iterrows():
            combined = self.combine_source_target(row.get('원문', ''), row.get('번역문', ''))
            gt_combined.append(combined)
        
        pred_combined = []
        for _, row in pred_df.iterrows():
            combined = self.combine_source_target(row.get('원문', ''), row.get('번역문', ''))
            pred_combined.append(combined)
        
        self.log(f"📋 정답 결합 텍스트: {len(gt_combined)}개")
        self.log(f"📋 예측 결합 텍스트: {len(pred_combined)}개")
        
        # 최적 정렬 찾기
        alignments = self.find_best_alignment(gt_combined, pred_combined)
        
        # 행별 정확도 계산
        self.log(f"\n🔄 행별 정확도 계산 중...")
        
        row_results = []
        overall_metrics = defaultdict(list)
        matched_count = 0
        skipped_count = 0
        
        for gt_idx, pred_idx, similarity in alignments:
            gt_text = gt_combined[gt_idx]
            pred_text = pred_combined[pred_idx] if pred_idx >= 0 else ""
            
            # 정확도 계산
            accuracy = self.calculate_row_accuracy(gt_text, pred_text, gt_idx, pred_idx)
            row_results.append(accuracy)
            
            if pred_idx >= 0:
                matched_count += 1
            else:
                skipped_count += 1
            
            # 전체 메트릭 수집
            for key, value in accuracy.items():
                if isinstance(value, (int, float)) and key not in ['gt_row_index', 'pred_row_index']:
                    overall_metrics[key].append(value)
        
        self.log(f"✅ {len(row_results)}개 행 평가 완료")
        self.log(f"   매칭된 행: {matched_count}개")
        self.log(f"   건너뛴 행: {skipped_count}개")
        
        # 전체 평균 계산
        overall_accuracy = {}
        for metric, values in overall_metrics.items():
            overall_accuracy[f'avg_{metric}'] = sum(values) / len(values) if values else 0
        
        return {
            'row_results': row_results,
            'overall_accuracy': overall_accuracy,
            'summary': {
                'total_gt_rows': len(gt_combined),
                'total_pred_rows': len(pred_combined),
                'matched_rows': matched_count,
                'skipped_rows': skipped_count,
                'alignment_rate': matched_count / len(gt_combined) if gt_combined else 0
            }
        }
    
    def print_detailed_results(self, results: Dict[str, any]):
        """상세 결과 출력"""
        self.log("\n" + "="*80)
        self.log("행 단위 원문-번역문 쌍 정확도 평가 결과")
        self.log("="*80)
        
        # 전체 요약
        summary = results['summary']
        overall = results['overall_accuracy']
        
        self.log(f"\n📊 전체 요약:")
        self.log(f"  • 정답 총 행 수: {summary['total_gt_rows']}")
        self.log(f"  • 예측 총 행 수: {summary['total_pred_rows']}")
        self.log(f"  • 매칭된 행 수: {summary['matched_rows']}")
        self.log(f"  • 건너뛴 행 수: {summary['skipped_rows']}")
        self.log(f"  • 정렬 성공률: {summary['alignment_rate']:.1%}")
        self.log(f"  • 평균 정답 텍스트 길이: {overall['avg_gt_length']:.1f}자")
        self.log(f"  • 평균 예측 텍스트 길이: {overall['avg_pred_length']:.1f}자")
        
        self.log(f"\n🎯 주요 정확도 지표:")
        self.log(f"  📌 평가 방식 안내:")
        self.log(f"    - 행단위 매칭: 각 행의 원문+번역문 쌍을 직접 비교")
        self.log(f"    - 동적 정렬: 문장 수 차이로 인한 밀림 현상 보정")
        self.log(f"    - 완전일치: 결합 텍스트가 정확히 일치")
        self.log(f"    - 유사도 기반: 텍스트 유사도로 정확도 측정")
        self.log(f"")
        self.log(f"  • 완전 일치율: {overall['avg_exact_match']:.1%}")
        self.log(f"  • 텍스트 유사도: {overall['avg_text_similarity']:.1%}")
        self.log(f"  • 길이 유사도: {overall['avg_length_similarity']:.1%}")
        self.log(f"  • 최종 유사도 (가중평균): {overall['avg_final_similarity']:.1%}")
        self.log(f"  • 유효 행 비율: {overall['avg_has_content']:.1%}")
        
        # 행별 상세 결과 (상위 10개 + 하위 10개)
        row_results = results['row_results']
        row_results.sort(key=lambda x: x['final_similarity'], reverse=True)
        
        self.log(f"\n📈 최종 유사도 상위 10개 행:")
        self.log("정답행\t예측행\t최종유사도\t텍스트유사도\t완전일치")
        for result in row_results[:10]:
            pred_row = result['pred_row_index'] if result['pred_row_index'] > 0 else "X"
            self.log(f"{result['gt_row_index']}\t{pred_row}\t{result['final_similarity']:.2f}\t\t{result['text_similarity']:.2f}\t\t{result['exact_match']:.0f}")
        
        self.log(f"\n📉 최종 유사도 하위 10개 행:")
        self.log("정답행\t예측행\t최종유사도\t텍스트유사도\t완전일치")
        for result in row_results[-10:]:
            pred_row = result['pred_row_index'] if result['pred_row_index'] > 0 else "X"
            self.log(f"{result['gt_row_index']}\t{pred_row}\t{result['final_similarity']:.2f}\t\t{result['text_similarity']:.2f}\t\t{result['exact_match']:.0f}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """결과를 Excel 파일로 저장"""
        self.log(f"\n💾 결과 저장 중: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 요약 시트
            summary_data = {
                '지표': ['정답 총 행 수', '예측 총 행 수', '매칭된 행 수', '건너뛴 행 수', '정렬 성공률',
                        '완전 일치율', '텍스트 유사도', '길이 유사도', '최종 유사도', '유효 행 비율'],
                '값': [
                    results['summary']['total_gt_rows'],
                    results['summary']['total_pred_rows'],
                    results['summary']['matched_rows'],
                    results['summary']['skipped_rows'],
                    f"{results['summary']['alignment_rate']:.1%}",
                    f"{results['overall_accuracy']['avg_exact_match']:.1%}",
                    f"{results['overall_accuracy']['avg_text_similarity']:.1%}",
                    f"{results['overall_accuracy']['avg_length_similarity']:.1%}",
                    f"{results['overall_accuracy']['avg_final_similarity']:.1%}",
                    f"{results['overall_accuracy']['avg_has_content']:.1%}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 상세 결과 시트
            results_df = pd.DataFrame(results['row_results'])
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # 실행 로그 시트
            log_df = pd.DataFrame({'Execution_Log': self.execution_log})
            log_df.to_excel(writer, sheet_name='Execution_Log', index=False)
        
        self.log(f"✅ 결과 저장 완료: {output_file}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='행 단위 원문-번역문 쌍 정확도 평가 도구')
    parser.add_argument('ground_truth', help='정답 파일 경로 (Excel)')
    parser.add_argument('prediction', help='예측 파일 경로 (Excel)')
    parser.add_argument('--output', '-o', default='row_pair_accuracy_results.xlsx', help='결과 출력 파일명')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='부분 일치 임계값')
    parser.add_argument('--skip-threshold', '-s', type=float, default=0.1, help='건너뛰기 임계값')
    
    args = parser.parse_args()
    
    # 평가 실행
    evaluator = RowPairAccuracyEvaluator(
        similarity_threshold=args.threshold, 
        skip_threshold=args.skip_threshold
    )
    
    try:
        results = evaluator.evaluate_accuracy(args.ground_truth, args.prediction)
        evaluator.print_detailed_results(results)
        evaluator.save_results(results, args.output)
        
        print(f"\n🎉 행 단위 원문-번역문 쌍 정확도 평가 완료!")
        print(f"   결과 파일: {args.output}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
