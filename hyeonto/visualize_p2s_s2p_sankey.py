#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentence-Phrase 간 클러스터 Sankey 다이어그램

도서(book_name)를 공통 기준으로 Sentence 클러스터와 Phrase 클러스터 간의 매핑을 시각화합니다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

def load_cluster_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def _pick_first(columns: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"필수 컬럼 없음. 후보: {candidates}, 사용 가능: {columns}")

def build_cross_dataset_sankey(
    sentence_df: pd.DataFrame, phrase_df: pd.DataFrame, sentence_k: int, phrase_k: int
):
    """book_name + sentence_id를 기준으로 Sentence-Phrase 클러스터 간 Sankey 데이터 생성 (최적화)"""

    # Sentence 문장 키 생성
    sentence_book_col = _pick_first(list(sentence_df.columns), ["book_name", "book", "책명"])
    sentence_sent_col = _pick_first(
        list(sentence_df.columns),
        [
            "left_sentence_id",
            "sentence_id",
            "문장식별자",
            "paragraph_id",
            "문단식별자",
        ],
    )
    sentence_df = sentence_df[[sentence_book_col, sentence_sent_col, "cluster_id"]].copy()
    sentence_df.columns = ["book_name", "sent_id", "sentence_cluster"]
    sentence_df["sent_key"] = (
        sentence_df["book_name"].astype(str) + "_" + sentence_df["sent_id"].astype(str)
    )

    # Phrase 문장 키 생성
    phrase_book_col = _pick_first(list(phrase_df.columns), ["book_name", "book", "책명"])
    phrase_sent_col = _pick_first(
        list(phrase_df.columns),
        ["sentence_id", "문장식별자", "left_sentence_id", "paragraph_id", "문단식별자"],
    )
    phrase_df = phrase_df[[phrase_book_col, phrase_sent_col, "cluster_id"]].copy()
    phrase_df.columns = ["book_name", "sent_id", "phrase_cluster"]
    phrase_df["sent_key"] = (
        phrase_df["book_name"].astype(str) + "_" + phrase_df["sent_id"].astype(str)
    )

    # 공통 키 기반 병합 (vectorized)
    merged = pd.merge(
        sentence_df[["sent_key", "sentence_cluster"]],
        phrase_df[["sent_key", "phrase_cluster"]],
        on="sent_key",
        how="inner",
    )

    print(f"  매핑된 쌍: {len(merged):,}개")

    # 클러스터 쌍별 카운트
    flow_df = (
        merged.groupby(["sentence_cluster", "phrase_cluster"]).size().reset_index(name="count")
    )

    # 노드 정의
    sentence_nodes = [f"Sentence_p{i}" for i in range(sentence_k)]
    phrase_nodes = [f"Phrase_p{i}" for i in range(phrase_k)]
    all_nodes = sentence_nodes + phrase_nodes
    node_indices = {n: i for i, n in enumerate(all_nodes)}

    sources = []
    targets = []
    values = []

    for _, row in flow_df.iterrows():
        sentence_cid = int(row["sentence_cluster"])
        phrase_cid = int(row["phrase_cluster"])
        cnt = int(row["count"])
        if sentence_cid < sentence_k and phrase_cid < phrase_k:
            sources.append(node_indices[f"Sentence_p{sentence_cid}"])
            targets.append(node_indices[f"Phrase_p{phrase_cid}"])
            values.append(cnt)

    return all_nodes, sources, targets, values

def generate_sankey_html(
    all_nodes: list,
    sources: list,
    targets: list,
    values: list,
    sentence_k: int,
    phrase_k: int,
    out_path: Path,
):
    # 흑백 색상 팔레트 (색약자 접근성 높음)
    # Sentence: 진하색 (검정~진회색)
    # Phrase: 밝은색 (밝은회색~흰색)
    
    # 더 정확한 흑백 그라데이션
    sentence_colors = [f"#{int(255 - (i * 200 / max(sentence_k-1, 1))):02x}{int(255 - (i * 200 / max(sentence_k-1, 1))):02x}{int(255 - (i * 200 / max(sentence_k-1, 1))):02x}" for i in range(sentence_k)]
    phrase_colors = [f"#{int(100 + (i * 155 / max(phrase_k-1, 1))):02x}{int(100 + (i * 155 / max(phrase_k-1, 1))):02x}{int(100 + (i * 155 / max(phrase_k-1, 1))):02x}" for i in range(phrase_k)]
    
    node_colors = sentence_colors + phrase_colors

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="#000000", width=1),
                    label=all_nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=["rgba(100,100,100,0.4)" for _ in sources],
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Sentence(K={sentence_k}) → Phrase(K={phrase_k}) 클러스터 연결 (도서 기반, 흑백 인포그래픽)",
        font_size=12,
        height=700,
        width=1000,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))

def generate_report(
    sentence_df: pd.DataFrame, phrase_df: pd.DataFrame, sentence_k: int, phrase_k: int, out_path: Path
):
    sentence_book_col = (
        "book_name"
        if "book_name" in sentence_df.columns
        else ("book" if "book" in sentence_df.columns else None)
    )
    phrase_book_col = (
        "book_name"
        if "book_name" in phrase_df.columns
        else ("book" if "book" in phrase_df.columns else None)
    )
    if sentence_book_col is None or phrase_book_col is None:
        raise KeyError("book_name 또는 book 컬럼이 필요합니다.")

    common_books = set(sentence_df[sentence_book_col].unique()) & set(
        phrase_df[phrase_book_col].unique()
    )
    lines = [
        f"# Sentence(K={sentence_k}) ↔ Phrase(K={phrase_k}) 클러스터 연결 분석",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 개요",
        "",
        f"- Sentence 데이터: {len(sentence_df):,}건, {sentence_k}개 클러스터",
        f"- Phrase 데이터: {len(phrase_df):,}건, {phrase_k}개 클러스터",
        f"- 공통 도서 수: {len(common_books)}",
        "",
        "## 해석",
        "",
        "Sankey 다이어그램은 **도서(book_name)**를 공통 기준으로 사용하여,",
        "Sentence의 각 클러스터에 속한 데이터가 Phrase에서는 어떤 클러스터에 분포하는지를 보여줍니다.",
        "",
        "- 굵은 연결선: 두 클러스터가 유사한 도서 구성을 공유함",
        "- 가는 연결선: 도서 구성이 다르지만 일부 겹침이 있음",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Sentence-Phrase 간 클러스터 Sankey")
    parser.add_argument("--sentence-csv", type=Path, required=True)
    parser.add_argument("--phrase-csv", type=Path, required=True)
    parser.add_argument("--sentence-k", type=int, required=True)
    parser.add_argument("--phrase-k", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Sentence 데이터 로드: {args.sentence_csv}")
    sentence_df = load_cluster_data(args.sentence_csv)

    print(f"[2/4] Phrase 데이터 로드: {args.phrase_csv}")
    phrase_df = load_cluster_data(args.phrase_csv)

    print(f"[3/4] Sankey 데이터 생성...")
    all_nodes, sources, targets, values = build_cross_dataset_sankey(
        sentence_df, phrase_df, args.sentence_k, args.phrase_k
    )

    print(f"[4/4] 시각화 생성...")
    generate_sankey_html(
        all_nodes,
        sources,
        targets,
        values,
        args.sentence_k,
        args.phrase_k,
        args.out_dir / f"sentence_k{args.sentence_k}_phrase_k{args.phrase_k}_sankey.html",
    )

    generate_report(
        sentence_df,
        phrase_df,
        args.sentence_k,
        args.phrase_k,
        args.out_dir / f"sentence_k{args.sentence_k}_phrase_k{args.phrase_k}_sankey.md",
    )

    print(f"완료: {args.out_dir}")

if __name__ == "__main__":
    main()
