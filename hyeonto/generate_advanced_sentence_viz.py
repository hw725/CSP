#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentence 고급 산점도 생성 (K=3 임베딩 2D)

입력: report_1-1/visualizations_k3/k3_embedding_overlay_2d.csv
출력: report_1-1/exploratory/viz_advanced_sentence/advanced_cluster_viz.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

def main() -> int:
    p = argparse.ArgumentParser(description="Sentence 고급 산점도 생성")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    if not args.input.exists():
        print(f"❌ 파일 없음: {args.input}")
        return 1

    df = pd.read_csv(args.input)
    df_sentence = df[df["boundary_type"] == "Sentence"].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "advanced_cluster_viz.html"

    fig = px.scatter(
        df_sentence,
        x="x",
        y="y",
        color="cluster_id",
        title="Sentence 클러스터 고급 산점도 (K=3)",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#000")))
    fig.update_layout(height=700, width=900)

    fig.write_html(str(out_path))
    print(f"✅ 저장: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
