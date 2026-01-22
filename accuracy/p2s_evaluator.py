#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2S 정확도 평가 스크립트 (간소화 버전)

핵심 지표: 
- 원문 경계 F1 (문장 경계 위치 기반)
- 번역문 완전일치 시 원문 유사도
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path
import difflib
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
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def load_data(file_path: str) -> pd.DataFrame:
    """CSV/Excel 파일 로드"""
    if str(file_path).endswith('.xlsx'):
        return pd.read_excel(file_path, engine='openpyxl')
    return pd.read_csv(file_path)


def boundary_positions(segments: list) -> set:
    """문장 경계 위치 집합 계산 (정규화 기준)"""
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        seg_norm = normalize_text(seg)
        cursor += len(seg_norm)
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def calculate_prf1(tp: int, fp: int, fn: int) -> tuple:
    """Precision, Recall, F1 계산"""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_p2s(
    pred_df: pd.DataFrame, 
    gold_df: pd.DataFrame,
    verbose: bool = False
) -> Dict[str, float]:
    """
    P2S 평가: 원문 경계 F1 + 번역문 완전일치 시 원문 유사도
    """
    # 필수 컬럼 확인
    pred_has_book = 'book_name' in pred_df.columns
    
    # 데이터 정리
    pred_df = pred_df.copy()
    gold_df = gold_df.copy()
    for col in ('원문', '번역문'):
        pred_df[col] = pred_df[col].fillna('')
        gold_df[col] = gold_df[col].fillna('')
    if 'book_name' in gold_df.columns:
        gold_df['book_name'] = gold_df['book_name'].fillna('')
    if pred_has_book:
        pred_df['book_name'] = pred_df['book_name'].fillna('')
    
    # 그룹화
    if pred_has_book:
        pred_groups = pred_df.groupby(['book_name', '문단식별자'], sort=False)
        gold_groups = gold_df.groupby(['book_name', '문단식별자'], sort=False)
    else:
        pred_groups = pred_df.groupby('문단식별자', sort=False)
        gold_groups = gold_df.groupby('문단식별자', sort=False)
    
    pred_keys = set(pred_groups.groups.keys())
    gold_keys = set(gold_groups.groups.keys())
    common_keys = pred_keys & gold_keys
    
    if verbose:
        print(f"Pred 문단 수: {len(pred_keys)}")
        print(f"Gold 문단 수: {len(gold_keys)}")
        print(f"공통 문단 수: {len(common_keys)}")
    
    if len(common_keys) == 0:
        return {
            'total_paragraphs': 0,
            'boundary_f1': 0.0,
            'target_exact_match': 0,
            'target_exact_rate': 0.0,
            'source_similarity_on_tgt_match': 0.0,
        }
    
    # 평가
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
    
    # 평가
    tp = fp = fn = 0
    tgt_exact_ok = 0        # 번역문 리스트 전체 일치 문단 수
    
    # 문장 단위 통계
    sent_tgt_exact_count = 0 
    sent_tgt_total_match_subset = 0
    src_sims_on_sent_match = [] # 번역문 일치하는 문장들의 원문 유사도
    
    for key in common_keys:
        pred_g = pred_groups.get_group(key)
        gold_g = gold_groups.get_group(key)
        
        pred_src = [str(x) for x in pred_g['원문'].tolist()]
        pred_tgt = [str(x) for x in pred_g['번역문'].tolist()]
        gold_src = [str(x) for x in gold_g['원문'].tolist()]
        gold_tgt = [str(x) for x in gold_g['번역문'].tolist()]
        
        # 번역문 완전일치 확인 (문단 단위)
        pred_tgt_norm = [normalize_text(s) for s in pred_tgt]
        gold_tgt_norm = [normalize_text(s) for s in gold_tgt]
        
        if pred_tgt_norm == gold_tgt_norm:
            tgt_exact_ok += 1
            
        # 문장 단위 비교(번역문이 일치하는 경우 원문 유사도 측정)
        # 길이가 다르면 앞에서부터 가능한 만큼 비교
        min_len = min(len(pred_tgt_norm), len(gold_tgt_norm))
        for i in range(min_len):
            if pred_tgt_norm[i] == gold_tgt_norm[i]:
                sent_tgt_exact_count += 1
                # 대응되는 원문의 유사도 계산
                sim = calculate_similarity(normalize_text(pred_src[i]), normalize_text(gold_src[i]))
                src_sims_on_sent_match.append(sim)
        
        # 원문 경계 F1 계산
        pred_b = boundary_positions(pred_src)
        gold_b = boundary_positions(gold_src)
        inter = pred_b & gold_b
        
        tp += len(inter)
        fp += len(pred_b - gold_b)
        fn += len(gold_b - pred_b)
    
    p, r, f1 = calculate_prf1(tp, fp, fn)
    avg_src_sim = sum(src_sims_on_sent_match) / len(src_sims_on_sent_match) if src_sims_on_sent_match else 0
    
    return {
        'total_paragraphs': len(common_keys),
        'global_integrity': is_global_integrity_ok,
        'integrity_details': integrity_details,
        'boundary_precision': p,
        'boundary_recall': r,
        'boundary_f1': f1,
        'target_exact_match_para': tgt_exact_ok,
        'target_exact_rate_para': tgt_exact_ok / len(common_keys) if common_keys else 0,
        'target_exact_match_sent_count': sent_tgt_exact_count,
        'source_similarity_on_tgt_match_sent': avg_src_sim,
    }


def print_results(results: Dict[str, float]):
    """결과 출력"""
    print("\n" + "=" * 50)
    print("📊 P2S 평가 결과")
    print("=" * 50)
    print(f"총 문단 수: {results['total_paragraphs']:,}")
    if results.get('global_integrity'):
        print("전역 무결성 (원문 보존): PASS")
    else:
        print("전역 무결성 (원문 보존): FAIL")
        print("-" * 50)
        print(results.get('integrity_details', '').strip())
        print("-" * 50)
    print("-" * 50)
    print(f"원문 경계 F1: {results['boundary_f1']:.4f} ({results['boundary_f1']*100:.2f}%)")
    print(f"  - Precision: {results['boundary_precision']:.4f}")
    print(f"  - Recall: {results['boundary_recall']:.4f}")
    print("-" * 50)
    print(f"번역문 완전일치 (문단): {results['target_exact_match_para']}/{results['total_paragraphs']} ({results['target_exact_rate_para']*100:.2f}%)")
    print(f"원문 유사도 (번역문 일치 문장 기준): {results['source_similarity_on_tgt_match_sent']:.4f} ({results['source_similarity_on_tgt_match_sent']*100:.2f}%)")
    print(f"  - (근거: 번역문 일치 문장 수 {results['target_exact_match_sent_count']}개)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='P2S 정확도 평가')
    parser.add_argument('prediction', help='예측 파일 경로 (P2S 출력)')
    parser.add_argument('ground_truth', help='정답 파일 경로')
    parser.add_argument('--output', '-o', help='결과 저장 경로', default='test_results/p2s_eval_result.csv')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prediction):
        print(f"❌ 예측 파일을 찾을 수 없습니다: {args.prediction}")
        sys.exit(1)
    if not os.path.exists(args.ground_truth):
        print(f"❌ 정답 파일을 찾을 수 없습니다: {args.ground_truth}")
        sys.exit(1)
    
    print(f"📂 예측 파일: {args.prediction}")
    pred_df = load_data(args.prediction)
    print(f"📂 정답 파일: {args.ground_truth}")
    gold_df = load_data(args.ground_truth)
    
    results = evaluate_p2s(pred_df, gold_df, verbose=args.verbose)
    print_results(results)
    
    # CSV 저장
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    pd.DataFrame([results]).to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
