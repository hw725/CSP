#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클러스터 라벨 변화 분석: 1:1(균등) vs 3:1(가중)

균등 가중치와 Canon3 가중치 클러스터링 결과를 비교하여
클러스터 라벨이 얼마나 달라지는지 정량 분석합니다.

지표:
- Adjusted Rand Index (ARI): 클러스터 일치도
- Normalized Mutual Information (NMI): 정보 일치도
- Label Transition Matrix: 라벨 전환 패턴
- Canon/Other별 라벨 변화율
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import json

# Canon 정의 (실제 CSV의 book 컬럼 값)
SASEO_BOOKS = {"논어집주", "맹자집주", "대학장구", "중용장구"}
SEKYUNG_COMPLETE = {
    "서경집전(상)", "서경집전(하)",
    "시경집전(상)", "시경집전(하)",
    "주역전의(상)", "주역전의(하)"
}
CANON_BOOKS = SASEO_BOOKS | SEKYUNG_COMPLETE

def load_comparison_data(csv_1_1: Path, csv_3_1: Path, book_col: str = "book"):
    """두 클러스터링 결과 로드 및 정렬"""
    print(f"📂 균등(1:1) 로드: {csv_1_1}")
    df_1_1 = pd.read_csv(csv_1_1)

    print(f"📂 가중(3:1) 로드: {csv_3_1}")
    df_3_1 = pd.read_csv(csv_3_1)

    if len(df_1_1) != len(df_3_1):
        raise ValueError(f"데이터 크기 불일치: {len(df_1_1)} vs {len(df_3_1)}")

    # 정렬 (안전성 확보)
    if '문장식별자' in df_1_1.columns and '문장식별자' in df_3_1.columns:
        sort_cols = ['book', '문장식별자']
    elif 'sentence_id' in df_1_1.columns:
        sort_cols = ['book', 'sentence_id']
    else:
        sort_cols = ['book']

    df_1_1 = df_1_1.sort_values(sort_cols).reset_index(drop=True)
    df_3_1 = df_3_1.sort_values(sort_cols).reset_index(drop=True)

    # Canon 마스크
    canon_mask = df_1_1[book_col].isin(CANON_BOOKS)

    return df_1_1, df_3_1, canon_mask

def compute_cluster_agreement_metrics(labels_1_1, labels_3_1):
    """클러스터 일치도 메트릭 계산"""
    ari = adjusted_rand_score(labels_1_1, labels_3_1)
    nmi = normalized_mutual_info_score(labels_1_1, labels_3_1)

    # 완전 일치율
    exact_match_rate = (labels_1_1 == labels_3_1).mean()

    return {
        "ari": ari,
        "nmi": nmi,
        "exact_match_rate": exact_match_rate
    }

def analyze_label_transitions(labels_1_1, labels_3_1, canon_mask):
    """라벨 전환 패턴 분석"""
    # 전체 전환 매트릭스
    transition_matrix = confusion_matrix(labels_1_1, labels_3_1)

    # Canon vs Other 전환율
    canon_labels_1_1 = labels_1_1[canon_mask]
    canon_labels_3_1 = labels_3_1[canon_mask]

    other_labels_1_1 = labels_1_1[~canon_mask]
    other_labels_3_1 = labels_3_1[~canon_mask]

    canon_change_rate = (canon_labels_1_1 != canon_labels_3_1).mean()
    other_change_rate = (other_labels_1_1 != other_labels_3_1).mean()

    return {
        "transition_matrix": transition_matrix,
        "canon_change_rate": canon_change_rate,
        "other_change_rate": other_change_rate
    }

def generate_analysis_report(
    metrics: dict,
    transitions: dict,
    output_path: Path,
    tag: str = "sentence"
):
    """분석 리포트 생성"""
    lines = [
        f"# 클러스터 라벨 변화 분석: 1:1 vs 3:1 ({tag.upper()})",
        "",
        "**분석 목적**: 균등 가중치와 Canon3 가중치 클러스터링 결과 비교",
        "",
        "---",
        "",
        "## 1. 전체 일치도 지표",
        "",
        f"- **Adjusted Rand Index (ARI)**: {metrics['ari']:.4f}",
        "  - 범위: [-1, 1], 1에 가까울수록 클러스터 구조 일치",
        "  - 0.8 이상: 매우 유사, 0.5~0.8: 중간, 0.5 미만: 상당한 차이",
        "",
        f"- **Normalized Mutual Information (NMI)**: {metrics['nmi']:.4f}",
        "  - 범위: [0, 1], 1에 가까울수록 정보 일치",
        "",
        f"- **완전 일치율**: {metrics['exact_match_rate']*100:.2f}%",
        "  - 동일한 클러스터 라벨을 받은 샘플 비율",
        "",
        "---",
        "",
        "## 2. 라벨 전환 패턴",
        "",
        "### 전환 매트릭스 (1:1 → 3:1)",
        "",
        "| 1:1 \\ 3:1 | Cluster 0 | Cluster 1 | Cluster 2 |",
        "|:---:|---:|---:|---:|"
    ]

    tm = transitions['transition_matrix']
    for i in range(tm.shape[0]):
        row = " | ".join([str(tm[i, j]) for j in range(tm.shape[1])])
        lines.append(f"| Cluster {i} | {row} |")

    lines.extend([
        "",
        "### Canon vs Other 라벨 변화율",
        "",
        f"- **Canon 샘플**: {transitions['canon_change_rate']*100:.2f}% 변경",
        f"- **Other 샘플**: {transitions['other_change_rate']*100:.2f}% 변경",
        "",
        "**해석**:",
        "- Canon 변화율이 Other보다 **높으면**: 가중치가 Canon 클러스터 경계를 재편",
        "- Canon 변화율이 Other보다 **낮으면**: Canon은 안정적, Other가 재배치됨",
        "",
        "---",
        "",
        "## 3. 결론",
        "",
        f"- ARI {metrics['ari']:.4f}는 ",
    ])

    if metrics['ari'] >= 0.8:
        lines.append("  → **클러스터 구조가 매우 유사**함. 가중치 효과가 제한적.")
    elif metrics['ari'] >= 0.5:
        lines.append("  → **중간 정도의 구조 변화**. 가중치가 일부 클러스터 경계를 이동시킴.")
    else:
        lines.append("  → **상당한 구조 변화**. 가중치가 클러스터 경계를 크게 재편함.")

    lines.extend([
        "",
        f"- 완전 일치율 {metrics['exact_match_rate']*100:.1f}%는 ",
    ])

    if metrics['exact_match_rate'] >= 0.8:
        lines.append("  → 대부분의 샘플이 **동일한 클러스터 유지**. 가중치 영향 미미.")
    elif metrics['exact_match_rate'] >= 0.5:
        lines.append("  → 약 절반의 샘플이 **클러스터 이동**. 가중치가 재분류 유도.")
    else:
        lines.append("  → 다수의 샘플이 **클러스터 변경**. 가중치가 분류 체계 재구성.")

    lines.extend([
        "",
        "---",
        "",
        "**권장 사항**:",
        "- ARI < 0.7: 가중 클러스터링이 균등과 **본질적으로 다른 패턴** 발견",
        "- ARI ≥ 0.7: 가중치는 **평가 관점 변경**에 적합, 클러스터 재생성은 불필요",
        ""
    ])

    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ 리포트 저장: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="클러스터 라벨 변화 분석")
    parser.add_argument("--csv-1-1", type=Path, required=True, help="균등(1:1) CSV")
    parser.add_argument("--csv-3-1", type=Path, required=True, help="가중(3:1) CSV")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉토리")
    parser.add_argument("--tag", type=str, default="comparison", help="출력 태그")
    args = parser.parse_args()

    print("="*70)
    print("📊 클러스터 라벨 변화 분석: 1:1 vs 3:1")
    print("="*70)

    # 1. 데이터 로드
    df_1_1, df_3_1, canon_mask = load_comparison_data(args.csv_1_1, args.csv_3_1)

    labels_1_1 = df_1_1['cluster_id'].values
    labels_3_1 = df_3_1['cluster_id'].values

    # 2. 일치도 메트릭
    print("\n🔬 일치도 메트릭 계산...")
    metrics = compute_cluster_agreement_metrics(labels_1_1, labels_3_1)
    print(f"   ARI: {metrics['ari']:.4f}")
    print(f"   NMI: {metrics['nmi']:.4f}")
    print(f"   완전 일치율: {metrics['exact_match_rate']*100:.2f}%")

    # 3. 전환 패턴
    print("\n🔄 라벨 전환 분석...")
    transitions = analyze_label_transitions(labels_1_1, labels_3_1, canon_mask)
    print(f"   Canon 변화율: {transitions['canon_change_rate']*100:.2f}%")
    print(f"   Other 변화율: {transitions['other_change_rate']*100:.2f}%")

    # 4. 리포트 생성
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f"{args.tag}_LABEL_CHANGE_ANALYSIS.md"
    generate_analysis_report(metrics, transitions, report_path, tag=args.tag)

    # 5. JSON 저장
    results = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "transitions": {
            "transition_matrix": transitions['transition_matrix'].tolist(),
            "canon_change_rate": float(transitions['canon_change_rate']),
            "other_change_rate": float(transitions['other_change_rate'])
        }
    }

    json_path = args.out_dir / f"{args.tag}_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장: {json_path}")

    print("\n" + "="*70)
    print("✅ 분석 완료!")
    print("="*70)

if __name__ == "__main__":
    main()
