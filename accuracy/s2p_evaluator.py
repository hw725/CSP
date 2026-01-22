#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2P 정확도 평가 스크립트 (간소화 버전)

핵심 지표: src exact subset 내에서의 번역문 F1과 유사도
"""

import pandas as pd
import argparse
import os
import sys
from difflib import SequenceMatcher
from typing import Dict


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


def evaluate_src_exact_subset(
    gold_df: pd.DataFrame, 
    pred_df: pd.DataFrame,
    verbose: bool = False
) -> Dict[str, float]:
    """
    src exact subset 평가: 원문 exact match 행에서 번역문 F1/유사도 계산
    """
    # 정규화된 원문 추가
    gold_df = gold_df.copy()
    pred_df = pred_df.copy()
    gold_df['_src_norm'] = gold_df['원문'].apply(normalize_text)
    pred_df['_src_norm'] = pred_df['원문'].apply(normalize_text)
    
    # Gold/Pred 모두 문장식별자 사용
    gold_df['_key'] = list(zip(
        gold_df['book_name'].fillna(''), 
        gold_df['문장식별자'], 
        gold_df['_src_norm']
    ))
    pred_df['_key'] = list(zip(
        pred_df['book_name'].fillna(''), 
        pred_df['문장식별자'], 
        pred_df['_src_norm']
    ))
    
    # 공통 키 (원문 exact match)
    common_keys = set(gold_df['_key']) & set(pred_df['_key'])
    
    if verbose:
        print(f"Gold 행 수: {len(gold_df)}")
        print(f"Pred 행 수: {len(pred_df)}")
        print(f"원문 Exact Match 키 수: {len(common_keys)}")
    
    if len(common_keys) == 0:
        return {
            'src_exact_match_count': 0,
            'target_exact_match_count': 0,
            'target_f1': 0.0,
            'target_avg_similarity': 0.0,
        }
    
    # 그룹 딕셔너리 생성 (동일 키의 번역문 연결)
    gold_dict = {}
    for _, row in gold_df.iterrows():
        k = row['_key']
        if k not in gold_dict:
            gold_dict[k] = []
        gold_dict[k].append(normalize_text(row['번역문']))
    
    pred_dict = {}
    for _, row in pred_df.iterrows():
        k = row['_key']
        if k not in pred_dict:
            pred_dict[k] = []
        pred_dict[k].append(normalize_text(row['번역문']))
    
    # 번역문 비교
    exact_matches = 0
    similarities = []
    
    for key in common_keys:
        gold_tgt = ''.join(gold_dict[key])
        pred_tgt = ''.join(pred_dict[key])
        
        if gold_tgt == pred_tgt:
            exact_matches += 1
            similarities.append(1.0)
        else:
            similarities.append(calculate_similarity(gold_tgt, pred_tgt))
    
    total = len(common_keys)
    f1 = exact_matches / total if total > 0 else 0
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    return {
        'src_exact_match_count': total,
        'target_exact_match_count': exact_matches,
        'target_f1': f1,
        'target_avg_similarity': avg_similarity,
        'target_sim_ge_90': sum(1 for s in similarities if s >= 0.9) / len(similarities) if similarities else 0,
        'target_sim_ge_80': sum(1 for s in similarities if s >= 0.8) / len(similarities) if similarities else 0,
    }


def print_results(results: Dict[str, float]):
    """결과 출력"""
    print("\n" + "=" * 50)
    print("📊 S2P 평가 결과 (src exact subset)")
    print("=" * 50)
    print(f"원문 Exact Match: {results['src_exact_match_count']:,}개")
    print(f"번역문 Exact Match: {results['target_exact_match_count']:,}개")
    print("-" * 50)
    print(f"번역문 F1: {results['target_f1']:.4f} ({results['target_f1']*100:.2f}%)")
    print(f"번역문 평균 유사도: {results['target_avg_similarity']:.4f} ({results['target_avg_similarity']*100:.2f}%)")
    print("-" * 50)
    print(f"유사도 >= 0.9: {results['target_sim_ge_90']*100:.1f}%")
    print(f"유사도 >= 0.8: {results['target_sim_ge_80']*100:.1f}%")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='S2P 정확도 평가 (src exact subset)')
    parser.add_argument('ground_truth', help='정답 파일 경로')
    parser.add_argument('prediction', help='예측 파일 경로')
    parser.add_argument('--output', '-o', help='결과 저장 경로', default='test_results/s2p_eval_result.csv')
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
    
    results = evaluate_src_exact_subset(gold_df, pred_df, verbose=args.verbose)
    print_results(results)
    
    # CSV 저장
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    pd.DataFrame([results]).to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
