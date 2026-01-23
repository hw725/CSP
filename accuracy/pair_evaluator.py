#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair-based Evaluator for P2S/S2P

평가 기준: (원문, 번역문) 쌍 단위 비교
- F1: 정확히 일치하는 쌍 비율
- 유사도: 원문이 일치하는 쌍에서 번역문 유사도 측정
"""

import pandas as pd
import argparse
import os
import sys
from difflib import SequenceMatcher
from typing import Dict, Set, Tuple


def normalize_text(text) -> str:
    """텍스트 정규화: 공백/개행/탭 제거"""
    if pd.isna(text):
        return ''
    return str(text).replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '').strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트의 문자열 유사도 계산"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def load_data(file_path: str) -> pd.DataFrame:
    """CSV/Excel 파일 로드"""
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, engine='openpyxl')
    return pd.read_csv(file_path)


def evaluate_pairs(
    gold_df: pd.DataFrame, 
    pred_df: pd.DataFrame,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Pair-based 평가: (원문, 번역문) 쌍 단위 비교
    """
    # 정규화
    gold_df = gold_df.copy()
    pred_df = pred_df.copy()
    
    # 전역 무결성 체크
    pred_src_global = normalize_text(''.join(pred_df['원문'].astype(str)))
    gold_src_global = normalize_text(''.join(gold_df['원문'].astype(str)))
    is_global_integrity_ok = (pred_src_global == gold_src_global)
    
    integrity_details = ""
    if not is_global_integrity_ok:
        integrity_details = f"Gold 길이: {len(gold_src_global)}, Pred 길이: {len(pred_src_global)}\n"
        min_len = min(len(gold_src_global), len(pred_src_global))
        for i in range(min_len):
            if gold_src_global[i] != pred_src_global[i]:
                integrity_details += f"첫 불일치 위치: {i}\n"
                integrity_details += f"Gold (snippet): ...{gold_src_global[max(0, i-10):i+10]}...\n"
                integrity_details += f"Pred (snippet): ...{pred_src_global[max(0, i-10):i+10]}...\n"
                break
    
    # (원문, 번역문) 쌍 생성
    gold_pairs: Set[Tuple[str, str]] = set()
    pred_pairs: Set[Tuple[str, str]] = set()
    
    # Gold 쌍: 단순 집합 (중복 무시)
    for _, row in gold_df.iterrows():
        src_norm = normalize_text(row['원문'])
        tgt_norm = normalize_text(row['번역문'])
        gold_pairs.add((src_norm, tgt_norm))
    
    # Pred 쌍
    for _, row in pred_df.iterrows():
        src_norm = normalize_text(row['원문'])
        tgt_norm = normalize_text(row['번역문'])
        pred_pairs.add((src_norm, tgt_norm))
    
    if verbose:
        print(f"Gold 쌍 수: {len(gold_pairs)}")
        print(f"Pred 쌍 수: {len(pred_pairs)}")
    
    # F1 계산 (쌍 Exact Match)
    exact_matches = gold_pairs & pred_pairs
    tp = len(exact_matches)
    precision = tp / len(pred_pairs) if pred_pairs else 0
    recall = tp / len(gold_pairs) if gold_pairs else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # 유사도 계산: 불일치 쌍만 비교 (일치 쌍은 F1에서 이미 처리)
    # 일치하는 쌍은 유사도 1.0으로 처리
    non_match_pred = pred_pairs - exact_matches
    non_match_gold = gold_pairs - exact_matches
    
    similarities = [1.0] * tp  # 일치 쌍은 유사도 1.0
    
    # 불일치 쌍만 비교 (속도 최적화)
    if non_match_pred and non_match_gold:
        gold_combined = [(src + tgt, src, tgt) for src, tgt in non_match_gold]
        
        for pred_src, pred_tgt in non_match_pred:
            pred_comb = pred_src + pred_tgt
            best_sim = 0.0
            for gold_comb, gold_src, gold_tgt in gold_combined:
                sim = calculate_similarity(pred_comb, gold_comb)
                if sim > best_sim:
                    best_sim = sim
                if sim >= 0.99:  # 거의 일치하면 조기 종료
                    break
            similarities.append(best_sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    return {
        'global_integrity': is_global_integrity_ok,
        'integrity_details': integrity_details,
        'gold_pair_count': len(gold_pairs),
        'pred_pair_count': len(pred_pairs),
        'exact_match_count': tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_similarity': avg_similarity,
        'sim_ge_90': sum(1 for s in similarities if s >= 0.9) / len(similarities) if similarities else 0,
        'sim_ge_80': sum(1 for s in similarities if s >= 0.8) / len(similarities) if similarities else 0,
    }


def print_results(results: Dict[str, float]):
    """결과 출력"""
    print("\n" + "=" * 60)
    print("📊 Pair-based 평가 결과 (원문-번역문 쌍 단위)")
    print("=" * 60)
    
    if results.get('global_integrity'):
        print("전역 무결성 (원문 보존): PASS")
    else:
        print("전역 무결성 (원문 보존): FAIL")
        print("-" * 60)
        print(results.get('integrity_details', '').strip())
        print("-" * 60)
    
    print(f"Gold 쌍 수: {results['gold_pair_count']:,}")
    print(f"Pred 쌍 수: {results['pred_pair_count']:,}")
    print("-" * 60)
    print(f"[Exact Match] 쌍 완전일치: {results['exact_match_count']:,}개")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1: {results['f1']:.4f} ({results['f1']*100:.2f}%)")
    print("-" * 60)
    print(f"[유사도] 가장 유사한 Gold 쌍 기준")
    print(f"  평균 유사도: {results['avg_similarity']:.4f} ({results['avg_similarity']*100:.2f}%)")
    print(f"  유사도 >= 0.9: {results['sim_ge_90']*100:.1f}%")
    print(f"  유사도 >= 0.8: {results['sim_ge_80']*100:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Pair-based 평가 (원문-번역문 쌍 단위)')
    parser.add_argument('ground_truth', help='정답 파일 경로')
    parser.add_argument('prediction', help='예측 파일 경로')
    parser.add_argument('--output', '-o', help='결과 저장 경로', default='test_results/pair_eval_result.csv')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ground_truth):
        print(f"❌ 정답 파일을 찾을 수 없습니다: {args.ground_truth}")
        sys.exit(1)
    if not os.path.exists(args.prediction):
        print(f"❌ 예측 파일을 찾을 수 없습니다: {args.prediction}")
        sys.exit(1)
    
    print(f"📂 정답 파일: {args.ground_truth}")
    gold_df = load_data(args.ground_truth)
    print(f"📂 예측 파일: {args.prediction}")
    pred_df = load_data(args.prediction)
    
    results = evaluate_pairs(gold_df, pred_df, verbose=args.verbose)
    print_results(results)
    
    # CSV 저장
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    pd.DataFrame([results]).to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
