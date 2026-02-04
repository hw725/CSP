#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
균등(1:1) vs 가중(3:1) 클러스터 Sankey 다이어그램

동일 데이터의 클러스터 라벨 전환을 Sankey로 시각화합니다.
기본 출력은 흑백(그레이스케일)입니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from hyeonto.generate_sankey_diagrams import compute_flow_matrix, generate_sankey_html

def _pick_first(columns: Iterable[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None

def _build_key(df: pd.DataFrame, book_col: str | None, id_col: str | None) -> pd.Series:
    if book_col and id_col:
        return df[book_col].astype(str) + "_" + df[id_col].astype(str)
    if id_col:
        return df[id_col].astype(str)
    return df.index.astype(str)

def build_uniform_weighted_flow(
    df_uniform: pd.DataFrame, df_weighted: pd.DataFrame
) -> pd.DataFrame:
    """균등/가중 데이터 간 클러스터 흐름 계산"""
    book_candidates = ["book_name", "book", "책명"]
    id_candidates = [
        "left_sentence_id",
        "sentence_id",
        "문장식별자",
        "left_phrase_id",
        "phrase_id",
        "구식별자",
        "paragraph_id",
        "문단식별자",
    ]

    book_col_u = _pick_first(df_uniform.columns, book_candidates)
    book_col_w = _pick_first(df_weighted.columns, book_candidates)
    id_col_u = _pick_first(df_uniform.columns, id_candidates)
    id_col_w = _pick_first(df_weighted.columns, id_candidates)

    df_u = df_uniform.copy()
    df_w = df_weighted.copy()

    df_u["_key"] = _build_key(df_u, book_col_u, id_col_u)
    df_w["_key"] = _build_key(df_w, book_col_w, id_col_w)

    merged = df_u[["_key", "cluster_id"]].merge(
        df_w[["_key", "cluster_id"]], on="_key", suffixes=("_u", "_w")
    )
    merged.columns = ["key", "cluster1", "cluster2"]

    flow = (
        merged.groupby(["cluster1", "cluster2"]).size().reset_index(name="count")
    )
    return flow

def main() -> int:
    p = argparse.ArgumentParser(description="균등/가중 클러스터 Sankey 생성")
    p.add_argument("--csv-1-1", type=Path, help="균등(1:1) CSV")
    p.add_argument("--csv-3-1", type=Path, help="가중(3:1) CSV")
    p.add_argument(
        "--metrics-json",
        type=Path,
        help="라벨 전환 metrics JSON (transition_matrix 기반)",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="출력 디렉토리")
    p.add_argument("--tag", type=str, default="dataset", help="출력 태그")
    p.add_argument(
        "--k-uniform",
        type=int,
        help="균등(Uniform) K 값 명시 (생략 시 자동 계산)",
    )
    p.add_argument(
        "--k-weighted",
        type=int,
        help="가중(Weighted) K 값 명시 (생략 시 자동 계산)",
    )
    p.add_argument(
        "--color",
        action="store_true",
        help="컬러 출력 (기본은 흑백)",
    )
    args = p.parse_args()

    flow_df = None
    left_k = right_k = None

    if args.metrics_json:
        if not args.metrics_json.exists():
            print(f"❌ 파일 없음: {args.metrics_json}")
            return 1
        print(f"📂 metrics 로드: {args.metrics_json}")
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        tm = metrics.get("transitions", {}).get("transition_matrix", [])
        if not tm:
            print("⚠️ transition_matrix가 비어 있습니다. Sankey 생성을 건너뜁니다.")
            return 0
        flow_list = []
        for i, row in enumerate(tm):
            for j, cnt in enumerate(row):
                if cnt > 0:
                    flow_list.append({"cluster1": i, "cluster2": j, "count": int(cnt)})
        flow_df = pd.DataFrame(flow_list)
        # 명시적 k 값이 없으면 transition_matrix 크기에서 자동 계산
        left_k = args.k_uniform if args.k_uniform is not None else len(tm)
        right_k = args.k_weighted if args.k_weighted is not None else (len(tm[0]) if tm else 0)
    else:
        if not args.csv_1_1 or not args.csv_3_1:
            print("❌ csv-1-1/csv-3-1 또는 metrics-json 중 하나는 필요합니다.")
            return 1
        if not args.csv_1_1.exists():
            print(f"❌ 파일 없음: {args.csv_1_1}")
            return 1
        if not args.csv_3_1.exists():
            print(f"❌ 파일 없음: {args.csv_3_1}")
            return 1

        print(f"📂 균등(1:1) 로드: {args.csv_1_1}")
        df_uniform = pd.read_csv(args.csv_1_1)
        print(f"📂 가중(3:1) 로드: {args.csv_3_1}")
        df_weighted = pd.read_csv(args.csv_3_1)

        print("🔄 흐름 계산...")
        flow_df = build_uniform_weighted_flow(df_uniform, df_weighted)

        if flow_df.empty:
            print("⚠️ 매칭된 샘플이 없습니다. Sankey 생성을 건너뜁니다.")
            return 0

        # 명시적 k 값이 없으면 데이터에서 자동 계산
        left_k = args.k_uniform if args.k_uniform is not None else int(df_uniform["cluster_id"].max()) + 1
        right_k = args.k_weighted if args.k_weighted is not None else int(df_weighted["cluster_id"].max()) + 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    title = f"{args.tag.upper()} 균등(1:1) ↔ 가중(3:1) 라벨 전환"
    out_path = args.out_dir / f"{args.tag}_uniform_vs_weighted_sankey.html"

    generate_sankey_html(
        flow_df,
        "Uniform",
        "Weighted",
        left_k,
        right_k,
        out_path,
        title,
        grayscale=not args.color,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
