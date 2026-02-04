#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현토 마커 음운 패턴 분석

입력:
- sentence_clusters.csv
- phrase_clusters.csv

출력:
- phonetic_analysis_sentence+phrase.md
- phonetic_heatmap.png
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# 초성/종성 테이블
CHO = [
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]
JONG = [
    "∅",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]

def _decompose_char(ch: str) -> tuple[str | None, str | None]:
    base = ord(ch) - 0xAC00
    if base < 0 or base >= 11172:
        return None, None
    cho = base // 588
    jung = (base % 588) // 28
    jong = base % 28
    _ = jung  # 중성은 현재 분석에서 제외
    return CHO[cho], JONG[jong]

def _pick_marker_series(df: pd.DataFrame) -> pd.Series:
    if "marker_normalized" in df.columns:
        return df["marker_normalized"].fillna("")
    if "marker" in df.columns:
        return df["marker"].fillna("")
    return df.get("원문", "").fillna("")

def _collect_phonetics(markers: pd.Series) -> Counter:
    counts = Counter()
    for raw in markers.astype(str):
        if not raw:
            continue
        first = raw.split(",")[0].strip()
        if not first:
            continue
        ch = first[0]
        cho, jong = _decompose_char(ch)
        if cho is None:
            continue
        counts[(cho, jong)] += 1
    return counts

def _render_heatmap(counts: Counter, output_path: Path) -> None:
    matrix = np.zeros((len(CHO), len(JONG)), dtype=int)
    for (cho, jong), cnt in counts.items():
        matrix[CHO.index(cho), JONG.index(jong)] = cnt

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="Greys")

    ax.set_xticks(range(len(JONG)))
    ax.set_xticklabels(JONG, fontsize=8)
    ax.set_yticks(range(len(CHO)))
    ax.set_yticklabels(CHO, fontsize=8)
    ax.set_xlabel("종성")
    ax.set_ylabel("초성")
    ax.set_title("현토 마커 음운 분포 (초성×종성)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> int:
    p = argparse.ArgumentParser(description="현토 마커 음운 패턴 분석")
    p.add_argument("--sentence-csv", type=Path, required=True)
    p.add_argument("--phrase-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    if not args.sentence_csv.exists():
        print(f"❌ 파일 없음: {args.sentence_csv}")
        return 1
    if not args.phrase_csv.exists():
        print(f"❌ 파일 없음: {args.phrase_csv}")
        return 1

    df_sentence = pd.read_csv(args.sentence_csv)
    df_phrase = pd.read_csv(args.phrase_csv)

    sentence_counts = _collect_phonetics(_pick_marker_series(df_sentence))
    phrase_counts = _collect_phonetics(_pick_marker_series(df_phrase))

    combined = sentence_counts + phrase_counts

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = out_dir / "phonetic_heatmap.png"
    _render_heatmap(combined, heatmap_path)

    report_path = out_dir / "phonetic_analysis_sentence+phrase.md"
    top_items = combined.most_common(20)

    lines = [
        "# 음운 패턴 분석 (Sentence + Phrase)",
        "",
        f"- Sentence 행 수: {len(df_sentence):,}",
        f"- Phrase 행 수: {len(df_phrase):,}",
        f"- 고유 초성×종성 조합: {len(combined):,}",
        "",
        "## 상위 20 초성×종성 조합",
        "",
        "| 순위 | 초성 | 종성 | 빈도 |",
        "|:---:|:---:|:---:|---:|",
    ]

    for idx, ((cho, jong), cnt) in enumerate(top_items, 1):
        lines.append(f"| {idx} | {cho} | {jong} | {cnt:,} |")

    lines.extend([
        "",
        f"## 히트맵",
        "",
        f"- 파일: {heatmap_path.name}",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 리포트 저장: {report_path}")
    print(f"✅ 히트맵 저장: {heatmap_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
