#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report 3-1 완전 재생성 파이프라인

Canon3 가중 클러스터링 결과로부터 모든 시각화와 메트릭을 생성합니다:
1. 가중 메트릭 계산 (Canonicity, Entropy)
2. Embedding Overlay 시각화 (2D/3D)
3. P2S-S2P Sankey 다이어그램
4. 클러스터 프로파일 문서
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: list, description: str):
    """명령 실행 및 출력"""
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)
    print(f"명령: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"❌ 실패: {description}")
        sys.exit(1)

    print(f"\n✅ 완료: {description}")

def main():
    print("\n" + "=" * 70)
    print("📊 Report 3-1 완전 재생성 파이프라인")
    print("=" * 70)

    # 경로 설정
    sentence_csv = "hyeonto/report_3-1/sentence_k3_normalized/sentence_clusters.csv"
    phrase_csv = "hyeonto/report_3-1/phrase_k3_normalized/phrase_clusters.csv"
    sentence_emb = "hyeonto/results/sentence_embeddings_cache.npy"
    phrase_emb = "hyeonto/results/phrase_embeddings_cache.npy"
    out_dir = "hyeonto/report_3-1"

    # 1. Sentence 가중 메트릭 계산
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/weight_sensitivity/recompute_weighted_metrics.py",
        "--csv", sentence_csv,
        "--out-dir", out_dir + "/sentence_k3_normalized",
        "--tag", "sentence",
        "--canon-weight", "3.0"
    ], "Sentence 가중 메트릭 계산")

    # 2. Phrase 가중 메트릭 계산
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/weight_sensitivity/recompute_weighted_metrics.py",
        "--csv", phrase_csv,
        "--out-dir", out_dir + "/phrase_k3_normalized",
        "--tag", "phrase",
        "--canon-weight", "3.0"
    ], "Phrase 가중 메트릭 계산")

    # 3. Embedding Overlay 시각화 (2D + 3D)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "hyeonto/analyze_embedding_overlay.py",
        "--sentence-cache-npy", sentence_emb,
        "--phrase-cache-npy", phrase_emb,
        "--weight-mode", "3-1",
        "--sample", "10000"
    ], "Embedding Overlay 시각화 (2D + 3D)")

    # 4. P2S-S2P Sankey 다이어그램
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/visualize_p2s_s2p_sankey.py",
        "--p2s-csv", phrase_csv,
        "--s2p-csv", sentence_csv,
        "--p2s-k", "3",
        "--s2p-k", "3",
        "--out-dir", out_dir + "/visualizations_k3"
    ], "P2S-S2P Sankey 다이어그램")

    print("\n" + "=" * 70)
    print("✅ 전체 재생성 완료!")
    print("=" * 70)
    print(f"📂 결과 위치: {out_dir}")
    print(f"   - sentence_k3_normalized/: Sentence 클러스터 + 메트릭")
    print(f"   - phrase_k3_normalized/: Phrase 클러스터 + 메트릭")
    print(f"   - visualizations_k3/: 시각화 (Overlay, Sankey)")
    print("\n" + "=" * 70)
    print("📌 추가 분석 스크립트 (별도 실행 필요)")
    print("=" * 70)
    print("\n1️⃣  K=3 클러스터 심층 분석:")
    print("   docker compose exec csp python hyeonto/analyze_k3_advanced.py")
    print("   docker compose exec csp python hyeonto/analyze_k3_clusters.py")
    print("\n2️⃣  가중치 민감도 분석:")
    print("   docker compose exec csp python hyeonto/scripts/analyze_weight_sensitivity_v6.py \\")
    print(f"     --pa-csv {sentence_csv} --out-dir hyeonto/weight_sensitivity")
    print("\n3️⃣  흑백 인쇄용 시각화:")
    print("   docker compose exec csp python hyeonto/analyze_embedding_overlay.py \\")
    print("     --weight-mode 3-1 --grayscale")
    print("\n4️⃣  마커 통사 기능 분석:")
    print("   docker compose exec csp python hyeonto/scripts/analyze_marker_syntactic_function_v6.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
