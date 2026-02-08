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
        return ""
    return (
        str(text)
        .replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("\r", "")
        .strip()
    )

def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트의 문자열 유사도 계산"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def load_data(file_path: str) -> pd.DataFrame:
    """CSV/Excel 파일 로드"""
    if str(file_path).endswith(".xlsx"):
        return pd.read_excel(file_path, engine="openpyxl")
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
    pred_df: pd.DataFrame, gold_df: pd.DataFrame, verbose: bool = False
) -> Dict[str, float]:
    """
    P2S 평가: 원문 경계 F1 + 번역문 완전일치 시 원문 유사도
    """
    # 필수 컬럼 확인: book_name 또는 책명 지원
    pred_df = pred_df.copy()
    gold_df = gold_df.copy()

    # 책명 → book_name 통합
    for df in (pred_df, gold_df):
        if "책명" in df.columns and "book_name" not in df.columns:
            df["book_name"] = df["책명"]

    pred_has_book = "book_name" in pred_df.columns
    gold_has_book = "book_name" in gold_df.columns
    use_book = pred_has_book and gold_has_book

    # 데이터 정리
    for col in ("원문", "번역문"):
        pred_df[col] = pred_df[col].fillna("")
        gold_df[col] = gold_df[col].fillna("")
    if use_book:
        pred_df["book_name"] = pred_df["book_name"].fillna("")
        gold_df["book_name"] = gold_df["book_name"].fillna("")

    # 그룹화 키 결정
    if use_book:
        group_cols = ["book_name", "문단식별자"]
    else:
        group_cols = ["문단식별자"]

    single_col = len(group_cols) == 1
    pred_groups = pred_df.groupby(group_cols, sort=False)
    gold_groups = gold_df.groupby(group_cols, sort=False)

    pred_keys = set(pred_groups.groups.keys())
    gold_keys = set(gold_groups.groups.keys())
    common_keys = pred_keys & gold_keys

    if verbose:
        print(f"Pred 문단 수: {len(pred_keys)}")
        print(f"Gold 문단 수: {len(gold_keys)}")
        print(f"공통 문단 수: {len(common_keys)}")

    if len(common_keys) == 0:
        return {
            "total_paragraphs": 0,
            "boundary_f1": 0.0,
            "target_exact_match": 0,
            "target_exact_rate": 0.0,
            "source_similarity_on_tgt_match": 0.0,
        }

    # 전역 무결성 체크: 문단식별자 기준 정렬 후 비교 (파일 순서 무관)
    sorted_keys = sorted(common_keys)
    pred_src_parts, gold_src_parts = [], []
    integrity_fail_details = []
    for key in sorted_keys:
        gkey = (key,) if single_col else key
        p_src = normalize_text("".join(pred_groups.get_group(gkey)["원문"].astype(str)))
        g_src = normalize_text("".join(gold_groups.get_group(gkey)["원문"].astype(str)))
        pred_src_parts.append(p_src)
        gold_src_parts.append(g_src)
        if p_src != g_src:
            integrity_fail_details.append(
                f"문단 {key}: Gold 길이={len(g_src)}, Pred 길이={len(p_src)}"
            )

    pred_src_global = "".join(pred_src_parts)
    gold_src_global = "".join(gold_src_parts)
    is_global_integrity_ok = pred_src_global == gold_src_global

    integrity_details = ""
    if not is_global_integrity_ok:
        integrity_details = (
            f"Gold 총 길이: {len(gold_src_global)}, Pred 총 길이: {len(pred_src_global)}\n"
        )
        if integrity_fail_details:
            integrity_details += f"불일치 문단 수: {len(integrity_fail_details)}\n"
            for d in integrity_fail_details[:5]:
                integrity_details += f"  - {d}\n"

    # 평가
    tp = fp = fn = 0
    tgt_exact_ok = 0  # 번역문 리스트 전체 일치 문단 수

    # 문장 단위 통계
    sent_tgt_exact_count = 0
    src_sims_on_sent_match = []  # 번역문 일치하는 문장들의 원문 유사도

    for key in common_keys:
        gkey = (key,) if single_col else key
        pred_g = pred_groups.get_group(gkey)
        gold_g = gold_groups.get_group(gkey)

        pred_src = [str(x) for x in pred_g["원문"].tolist()]
        pred_tgt = [str(x) for x in pred_g["번역문"].tolist()]
        gold_src = [str(x) for x in gold_g["원문"].tolist()]
        gold_tgt = [str(x) for x in gold_g["번역문"].tolist()]

        # 번역문 완전일치 확인 (문단 단위)
        pred_tgt_norm = [normalize_text(s) for s in pred_tgt]
        gold_tgt_norm = [normalize_text(s) for s in gold_tgt]

        if pred_tgt_norm == gold_tgt_norm:
            tgt_exact_ok += 1

        # 문장 단위 비교: 번역문이 일치하는 문장에서만 원문 F1 계산
        min_len = min(len(pred_tgt_norm), len(gold_tgt_norm))
        for i in range(min_len):
            if pred_tgt_norm[i] == gold_tgt_norm[i]:
                sent_tgt_exact_count += 1
                # 대응되는 원문의 유사도 계산
                pred_src_norm = normalize_text(pred_src[i])
                gold_src_norm = normalize_text(gold_src[i])
                sim = calculate_similarity(pred_src_norm, gold_src_norm)
                src_sims_on_sent_match.append(sim)

                # 원문 F1: 원문이 정확히 일치하면 TP, 아니면 FN
                if pred_src_norm == gold_src_norm:
                    tp += 1
                else:
                    fn += 1

    p, r, f1 = calculate_prf1(tp, fp, fn)
    avg_src_sim = (
        sum(src_sims_on_sent_match) / len(src_sims_on_sent_match)
        if src_sims_on_sent_match
        else 0
    )

    return {
        "total_paragraphs": len(common_keys),
        "global_integrity": is_global_integrity_ok,
        "integrity_details": integrity_details,
        "boundary_precision": p,
        "boundary_recall": r,
        "boundary_f1": f1,
        "target_exact_match_para": tgt_exact_ok,
        "target_exact_rate_para": tgt_exact_ok / len(common_keys) if common_keys else 0,
        "target_exact_match_sent_count": sent_tgt_exact_count,
        "source_similarity_on_tgt_match_sent": avg_src_sim,
    }

def print_results(results: Dict[str, float]):
    """결과 출력"""
    print("\n" + "=" * 50)
    print("📊 P2S 평가 결과")
    print("=" * 50)
    print(f"총 문단 수: {results['total_paragraphs']:,}")
    if results.get("global_integrity"):
        print("전역 무결성 (원문 보존): PASS")
    else:
        print("전역 무결성 (원문 보존): FAIL")
        print("-" * 50)
        print(results.get("integrity_details", "").strip())
        print("-" * 50)
    print("-" * 50)
    print(
        f"원문 경계 F1: {results['boundary_f1']:.4f} ({results['boundary_f1']*100:.2f}%)"
    )
    print(f"  - Precision: {results['boundary_precision']:.4f}")
    print(f"  - Recall: {results['boundary_recall']:.4f}")
    print("-" * 50)
    print(
        f"번역문 완전일치 (문단): {results['target_exact_match_para']}/{results['total_paragraphs']} ({results['target_exact_rate_para']*100:.2f}%)"
    )
    print(
        f"원문 유사도 (번역문 일치 문장 기준): {results['source_similarity_on_tgt_match_sent']:.4f} ({results['source_similarity_on_tgt_match_sent']*100:.2f}%)"
    )
    print(
        f"  - (근거: 번역문 일치 문장 수 {results['target_exact_match_sent_count']}개)"
    )
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="P2S 정확도 평가")
    parser.add_argument("prediction", help="예측 파일 경로 (P2S 출력)")
    parser.add_argument("ground_truth", help="정답 파일 경로")
    parser.add_argument(
        "--output",
        "-o",
        help="결과 저장 경로",
        default="test_results/p2s_eval_result.csv",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame([results]).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n💾 결과 저장: {args.output}")

if __name__ == "__main__":
    main()
