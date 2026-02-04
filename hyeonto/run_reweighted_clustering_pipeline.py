#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
균등(K=3) 기준 통합 파이프라인

1. 균등(1:1) 기준 마커 통사 기능 분석
2. 균등(1:1) vs 가중(3:1) Sankey 시각화

※ 가중 클러스터링 자체는 별도 수행된 결과를 사용
"""

import subprocess
from pathlib import Path

def run_command(cmd, desc):
    """명령 실행"""
    print(f"\n{'='*70}")
    print(f"🚀 {desc}")
    print(f"{'='*70}")
    print(f"명령: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True)
    return result.returncode == 0

def main():
    # Docker 내부 경로 사용
    sentence_1_1_csv = "hyeonto/report_1-1/sentence_k3_normalized/sentence_clusters.csv"
    phrase_1_1_csv = "hyeonto/report_1-1/phrase_k3_normalized/phrase_clusters.csv"

    out_dir_reweighted = "hyeonto/report_3-1"
    out_dir_analysis = "hyeonto/report_3-1"
    out_dir_syntax = "hyeonto/report_1-1/syntactic_function"
    mapping_json = "configs/syntactic_function_mapping.json"

    # 1. 균등/가중 Sankey (흑백 기본)
    # ⚠️ 주의: --k-uniform과 --k-weighted를 지정하여 서로 다른 K값 비교 가능
    # 예: --k-uniform 3 --k-weighted 2 (또는 16)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/visualize_uniform_weighted_sankey.py",
        "--metrics-json", out_dir_analysis + "/sentence_metrics.json",
        "--out-dir", out_dir_analysis,
        "--tag", "sentence",
        "--k-uniform", "3"
    ], "Sentence 균등/가중 Sankey (Uniform K=3)")

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/visualize_uniform_weighted_sankey.py",
        "--metrics-json", out_dir_analysis + "/phrase_metrics.json",
        "--out-dir", out_dir_analysis,
        "--tag", "phrase",
        "--k-uniform", "3"
    ], "Phrase 균등/가중 Sankey (Uniform K=3)")

    # 2. 마커 통사 기능 분석 (균등만)
    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/classify_syntactic_function.py",
        "--csv", sentence_1_1_csv,
        "--mapping", mapping_json,
        "--out-csv", out_dir_syntax + "/sentence_uniform_syntax.csv"
    ], "Sentence 마커 통사 기능 분석 (균등)")

    run_command([
        "docker", "compose", "exec", "csp", "python",
        "scripts/classify_syntactic_function.py",
        "--csv", phrase_1_1_csv,
        "--mapping", mapping_json,
        "--out-csv", out_dir_syntax + "/phrase_uniform_syntax.csv"
    ], "Phrase 마커 통사 기능 분석 (균등)")

    print(f"\n{'='*70}")
    print("✅ 전체 파이프라인 완료!")
    print(f"{'='*70}")
    print(f"📂 가중 클러스터링 결과(기존): {out_dir_reweighted}")
    print(f"📂 균등/가중 Sankey: {out_dir_analysis}")
    print(f"📂 마커 통사 기능 분석(균등): {out_dir_syntax}")

if __name__ == "__main__":
    main()
