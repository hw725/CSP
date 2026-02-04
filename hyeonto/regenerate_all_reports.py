#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 리포트 재생성 파이프라인 (1:1 + 3:1)

균등(1:1)과 Canon3(3:1) 가중치 리포트를 모두 재생성합니다:
- 클러스터링 (가중치 적용)
- 메트릭 계산
- 시각화 (Overlay, Sankey)
- 심층 분석 (K=3 클러스터 분석)
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list, description: str, allow_fail: bool = False):
    """명령 실행 및 출력"""
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)
    print(f"명령: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        if allow_fail:
            print(f"⚠️  경고: {description} (계속 진행)")
        else:
            print(f"❌ 실패: {description}")
            sys.exit(1)
    else:
        print(f"\n✅ 완료: {description}")

def main():
    print("\n" + "=" * 70)
    print("📊 전체 리포트 재생성 파이프라인 (1:1 + 3:1)")
    print("=" * 70)

    # ===================================================================
    # Phase 1: 균등(1:1) 리포트 재생성
    # ===================================================================
    print("\n" + "=" * 70)
    print("📌 Phase 1: 균등(1:1) 리포트")
    print("=" * 70)

    report_1_1 = "hyeonto/report_1-1"
    sentence_1_1_csv = f"{report_1_1}/sentence_k3_normalized/sentence_clusters.csv"
    phrase_1_1_csv = f"{report_1_1}/phrase_k3_normalized/phrase_clusters.csv"

    # 1-1. 가중 메트릭 (균등이므로 weight=1.0)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/weight_sensitivity/recompute_weighted_metrics.py",
        "--csv", sentence_1_1_csv,
        "--out-dir", f"{report_1_1}/sentence_k3_normalized",
        "--tag", "sentence",
        "--canon-weight", "1.0"
    ], "1:1 Sentence 메트릭")

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/weight_sensitivity/recompute_weighted_metrics.py",
        "--csv", phrase_1_1_csv,
        "--out-dir", f"{report_1_1}/phrase_k3_normalized",
        "--tag", "phrase",
        "--canon-weight", "1.0"
    ], "1:1 Phrase 메트릭")

    # 1-2. Embedding Overlay (1:1) - 흑백 인포그래픽
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/analyze_embedding_overlay.py",
        "--sentence-cache-npy", "hyeonto/results/sentence_embeddings_cache.npy",
        "--phrase-cache-npy", "hyeonto/results/phrase_embeddings_cache.npy",
        "--weight-mode", "1-1",
        "--sample", "10000",
        "--grayscale"
    ], "1:1 Embedding Overlay (흑백)")

    # 1-3. Sankey (1:1)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/visualize_p2s_s2p_sankey.py",
        "--p2s-csv", phrase_1_1_csv,
        "--s2p-csv", sentence_1_1_csv,
        "--p2s-k", "3",
        "--s2p-k", "3",
        "--out-dir", f"{report_1_1}/visualizations_k3"
    ], "1:1 Sankey")

    # 1-4. K=3 클러스터 심층 분석 (1:1)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/analyze_k3_advanced.py"
    ], "1:1 K=3 심층 분석", allow_fail=True)

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/analyze_k3_clusters.py"
    ], "1:1 K=3 클러스터 분석", allow_fail=True)

    # ===================================================================
    # Phase 2: Canon3(3:1) 리포트 재생성
    # ===================================================================
    print("\n" + "=" * 70)
    print("📌 Phase 2: Canon3(3:1) 리포트")
    print("=" * 70)

    report_3_1 = "hyeonto/report_3-1"
    sentence_3_1_csv = f"{report_3_1}/sentence_k3_normalized/sentence_clusters.csv"
    phrase_3_1_csv = f"{report_3_1}/phrase_k3_normalized/phrase_clusters.csv"

    # 2-1. 가중 메트릭 (Canon3 weight=3.0)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/weight_sensitivity/recompute_weighted_metrics.py",
        "--csv", sentence_3_1_csv,
        "--out-dir", f"{report_3_1}/sentence_k3_normalized",
        "--tag", "sentence",
        "--canon-weight", "3.0"
    ], "3:1 Sentence 메트릭")

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/weight_sensitivity/recompute_weighted_metrics.py",
        "--csv", phrase_3_1_csv,
        "--out-dir", f"{report_3_1}/phrase_k3_normalized",
        "--tag", "phrase",
        "--canon-weight", "3.0"
    ], "3:1 Phrase 메트릭")

    # 2-2. Embedding Overlay (3:1) - 흑백 인포그래픽
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/analyze_embedding_overlay.py",
        "--sentence-cache-npy", "hyeonto/results/sentence_embeddings_cache.npy",
        "--phrase-cache-npy", "hyeonto/results/phrase_embeddings_cache.npy",
        "--weight-mode", "3-1",
        "--sample", "10000",
        "--grayscale"
    ], "3:1 Embedding Overlay (흑백)")

    # 2-3. Sankey (3:1)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/visualize_p2s_s2p_sankey.py",
        "--p2s-csv", phrase_3_1_csv,
        "--s2p-csv", sentence_3_1_csv,
        "--p2s-k", "3",
        "--s2p-k", "3",
        "--out-dir", f"{report_3_1}/visualizations_k3"
    ], "3:1 Sankey")

    # 2-4. K=3 클러스터 심층 분석 (3:1) - 스크립트가 고정 경로 사용
    print("\n" + "=" * 70)
    print("⚠️  K=3 심층 분석 스크립트는 고정 경로를 사용합니다")
    print("   report_3-1 분석을 위해서는 스크립트 내부 경로 수정 필요")
    print("=" * 70)

    # ===================================================================
    # Phase 3: 라벨 변화 분석 (1:1 vs 3:1)
    # ===================================================================
    print("\n" + "=" * 70)
    print("📌 Phase 3: 라벨 변화 분석 (1:1 vs 3:1)")
    print("=" * 70)

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/analyze_cluster_label_changes.py",
        "--csv-1-1", sentence_1_1_csv,
        "--csv-3-1", sentence_3_1_csv,
        "--out-dir", "hyeonto/cluster_label_analysis",
        "--tag", "sentence"
    ], "Sentence 라벨 변화 분석")

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/analyze_cluster_label_changes.py",
        "--csv-1-1", phrase_1_1_csv,
        "--csv-3-1", phrase_3_1_csv,
        "--out-dir", "hyeonto/cluster_label_analysis",
        "--tag", "phrase"
    ], "Phrase 라벨 변화 분석")

    # ===================================================================
    # Phase 4: 검증 분석 (공기어 + 이상치 탐지)
    # ===================================================================
    print("\n" + "=" * 70)
    print("📌 Phase 4: 검증 분석 (공기어 + 이상치)")
    print("=" * 70)

    # 4-1. 한자-현토 공기어 정규화 분석
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/analyze_cooccurrence_normalized.py"
    ], "한자-현토 공기어 정규화 분석", allow_fail=True)

    # 4-2. 클러스터 이상치 탐지
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/detect_outliers_boundary.py"
    ], "클러스터 이상치(Outlier) 탐지", allow_fail=True)

    # ===================================================================
    # 완료 메시지
    # ===================================================================
    print("\n" + "=" * 70)
    print("✅ 전체 파이프라인 완료!")
    print("=" * 70)
    print("📂 생성된 리포트:")
    print(f"   - {report_1_1}/ (균등 1:1, 흑백 인포그래픽)")
    print(f"   - {report_3_1}/ (Canon3 3:1, 흑백 인포그래픽)")
    print(f"   - hyeonto/cluster_label_analysis/ (라벨 변화)")
    print(f"   - hyeonto/results/cooccurrence_normalized.* (공기어 분석)")
    print(f"   - hyeonto/results/outliers_*.csv (이상치 탐지)")
    print("\n📌 모든 시각화는 흑백 인포그래픽(라이트모드)으로 생성됩니다.")
    print("=" * 70)

if __name__ == "__main__":
    main()
