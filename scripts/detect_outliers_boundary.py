#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""클러스터 이상치(Outlier) 탐지 스크립트

기존 클러스터링 결과를 재현하고, 각 데이터 포인트와 클러스터 중심 간의
거리를 계산하여 이상치를 추출합니다.

출력:
- outliers_pa.csv / outliers_sa.csv: 이상치 목록 (거리 상위 N개)
- outlier_analysis.md: 이상치 분석 리포트
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sys

import numpy as np
import time
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

@dataclass
class BoundaryInstance:
    book_name: str
    paragraph_id: int
    left_sentence_id: int
    right_sentence_id: int
    src_left: str
    src_right: str
    tgt_left: str
    tgt_right: str

    def to_embed_text(self, use_src: bool = True, use_tgt: bool = True) -> str:
        parts = []
        if use_src:
            parts.append(f"[SRC_L]{self.src_left}[SRC_R]{self.src_right}")
        if use_tgt:
            parts.append(f"[TGT_L]{self.tgt_left}[TGT_R]{self.tgt_right}")
        return " ".join(parts)

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

def compute_embeddings_batched(
    texts: List[str], batch_size: int, device_id: Optional[int]
) -> np.ndarray:
    """배치로 임베딩 계산 (모델 한 번만 로드)"""
    from FlagEmbedding import BGEM3FlagModel

    print(f"  -> BGE-M3 모델 로딩 중...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        device=f"cuda:{device_id}" if device_id is not None else "cpu",
    )
    print(f"  -> 모델 로드 완료. 임베딩 시작...")

    all_vecs = []
    total = len(texts)
    start = time.time()
    
    for idx, i in enumerate(range(0, total, batch_size), start=1):
        batch = texts[i : i + batch_size]
        out = model.encode(batch, batch_size=64, max_length=512)
        vecs = np.array(out["dense_vecs"], dtype=np.float32)
        all_vecs.append(vecs)
        
        done = min(i + batch_size, total)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else 0.0
        print(
            f"  임베딩 진행: {done}/{total} | 경과 {elapsed/60:.1f}분 | 남은 {remaining/60:.1f}분 (속도: {rate:.1f}개/초)"
        )
    
    return np.vstack(all_vecs)

def load_pa_sentence_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

def _safe_int(x: object, default: int = -1) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return default

def iter_boundary_instances(df: pd.DataFrame):
    book_col = "book_name" if "book_name" in df.columns else "book"
    df = df.sort_values(by=[book_col, "문단식별자", "문장식별자"])
    for (book, para), grp in df.groupby([book_col, "문단식별자"], sort=False):
        rows = grp.to_dict("records")
        for i in range(len(rows) - 1):
            left_row = rows[i]
            right_row = rows[i + 1]
            yield BoundaryInstance(
                book_name=str(book),
                paragraph_id=int(para),
                left_sentence_id=_safe_int(left_row.get("문장식별자", -1)),
                right_sentence_id=_safe_int(right_row.get("문장식별자", -1)),
                src_left=str(left_row.get("원문", "")),
                src_right=str(right_row.get("원문", "")),
                tgt_left=str(left_row.get("번역문", "")),
                tgt_right=str(right_row.get("번역문", "")),
            )

def compute_outliers(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    instances: List[BoundaryInstance],
    top_n: int = 100,
) -> pd.DataFrame:
    """각 데이터 포인트와 해당 클러스터 중심 간의 거리를 계산하여 이상치 추출"""
    from sklearn.cluster import MiniBatchKMeans

    # 클러스터 중심 계산
    unique_clusters = np.unique(cluster_ids)
    centroids = {}
    for c in unique_clusters:
        mask = cluster_ids == c
        centroids[c] = X[mask].mean(axis=0)

    # 각 포인트와 해당 클러스터 중심 간의 거리 계산
    distances = []
    for i, (x, c) in enumerate(zip(X, cluster_ids)):
        dist = np.linalg.norm(x - centroids[c])
        distances.append(dist)

    distances = np.array(distances)

    # 거리 기준 상위 N개 추출
    top_indices = np.argsort(distances)[-top_n:][::-1]

    outlier_data = []
    for idx in top_indices:
        inst = instances[idx]
        outlier_data.append(
            {
                "rank": len(outlier_data) + 1,
                "distance": float(distances[idx]),
                "cluster_id": int(cluster_ids[idx]),
                "book_name": inst.book_name,
                "paragraph_id": inst.paragraph_id,
                "left_sentence_id": inst.left_sentence_id,
                "right_sentence_id": inst.right_sentence_id,
                "src_left": inst.src_left[:100],
                "src_right": inst.src_right[:100],
                "tgt_left": inst.tgt_left[:100],
                "tgt_right": inst.tgt_right[:100],
            }
        )

    return pd.DataFrame(outlier_data)

def write_outlier_report(
    out_path: Path, df_outliers: pd.DataFrame, analysis_type: str
) -> None:
    """이상치 분석 마크다운 리포트 작성"""
    lines = [
        f"# {analysis_type} 이상치 분석 리포트",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**이상치 수**: {len(df_outliers)}건",
        "",
        "---",
        "",
        "## 도서별 이상치 분포",
        "",
    ]

    book_dist = df_outliers["book_name"].value_counts().head(10)
    lines.append("| 도서명 | 이상치 수 |")
    lines.append("|:---|---:|")
    for book, cnt in book_dist.items():
        lines.append(f"| {book} | {cnt} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 클러스터별 이상치 분포",
            "",
        ]
    )

    cluster_dist = df_outliers["cluster_id"].value_counts().sort_index()
    lines.append("| 클러스터 ID | 이상치 수 |")
    lines.append("|:---:|---:|")
    for c, cnt in cluster_dist.items():
        lines.append(f"| p{c} | {cnt} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 상위 20개 이상치 샘플",
            "",
        ]
    )

    for _, row in df_outliers.head(20).iterrows():
        lines.append(
            f"### Rank {row['rank']} (거리: {row['distance']:.4f}, 클러스터: p{row['cluster_id']})"
        )
        lines.append(f"- **도서**: {row['book_name']}")
        lines.append(f"- **좌측 원문**: {row['src_left']}")
        lines.append(f"- **우측 원문**: {row['src_right']}")
        lines.append(f"- **좌측 번역**: {row['tgt_left']}")
        lines.append(f"- **우측 번역**: {row['tgt_right']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        raise SystemExit(f"scikit-learn이 필요합니다: {e}")

    p = argparse.ArgumentParser(description="클러스터 이상치 탐지")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 CSV (sentence/phrase clusters 또는 경계 CSV)",
    )
    p.add_argument("--out-dir", type=str, required=True, help="출력 디렉토리")
    p.add_argument("--analysis-type", type=str, default="Sentence", help="분석 태그")
    p.add_argument(
        "--max-boundaries", type=int, default=500000, help="최대 경계 샘플 수"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=16, help="클러스터 수")
    p.add_argument("--top-n", type=int, default=200, help="추출할 이상치 수")
    p.add_argument("--device-id", type=int, default=None, help="GPU ID")
    p.add_argument("--batch", type=int, default=128, help="임베딩 배치 크기")
    p.add_argument("--embedding-cache", type=str, default=None, help="임베딩 캐시 .npy 경로")
    p.add_argument(
        "--save-embedding-cache",
        action="store_true",
        help="임베딩 계산 결과를 캐시에 저장",
    )

    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 데이터 로딩: {in_path}")
    df = load_pa_sentence_pairs(in_path)

    instances: List[BoundaryInstance] = []
    for inst in iter_boundary_instances(df):
        instances.append(inst)
        if len(instances) >= args.max_boundaries:
            break

    print(f"  -> {len(instances)}개 경계 인스턴스 로드")

    if not instances:
        raise SystemExit("경계 인스턴스가 0개입니다.")

    print(f"[2/4] 임베딩 준비 중...")
    texts = [x.to_embed_text(use_src=True, use_tgt=True) for x in instances]
    X = None
    if args.embedding_cache:
        cache_path = Path(args.embedding_cache)
        if cache_path.exists():
            cached = np.load(cache_path)
            if len(cached) == len(texts):
                print(f"  -> 임베딩 캐시 사용: {cache_path}")
                X = cached
            else:
                print(
                    f"  ⚠️ 캐시 크기 불일치: {len(cached)} vs {len(texts)} (재계산)"
                )
        else:
            print(f"  ⚠️ 캐시 없음: {cache_path} (재계산)")

    if X is None:
        X = compute_embeddings_batched(
            texts, batch_size=args.batch, device_id=args.device_id
        )
        if args.embedding_cache and args.save_embedding_cache:
            cache_path = Path(args.embedding_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, X)
            print(f"  -> 임베딩 캐시 저장: {cache_path}")
    X = _l2_normalize(X)

    print(f"[3/4] 클러스터링 (K={args.k})...")
    km = MiniBatchKMeans(
        n_clusters=args.k, random_state=args.seed, batch_size=1024, n_init="auto"
    )
    cluster_ids = km.fit_predict(X)

    print(f"[4/4] 이상치 탐지 (상위 {args.top_n}개)...")
    df_outliers = compute_outliers(X, cluster_ids, instances, top_n=args.top_n)

    # 결과 저장
    out_csv = out_dir / f"outliers_{args.analysis_type.lower()}.csv"
    out_md = out_dir / f"outlier_analysis_{args.analysis_type.lower()}.md"

    df_outliers.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_outlier_report(out_md, df_outliers, args.analysis_type)

    print(f"완료: {out_csv}")
    print(f"리포트: {out_md}")

    # 요약 통계 출력
    print("\n=== 요약 ===")
    print(f"총 경계 수: {len(instances)}")
    print(f"이상치 수: {len(df_outliers)}")
    print(f"이상치 평균 거리: {df_outliers['distance'].mean():.4f}")
    print(f"이상치 최대 거리: {df_outliers['distance'].max():.4f}")
    print(
        f"이상치 최다 도서: {df_outliers['book_name'].value_counts().head(3).to_dict()}"
    )

if __name__ == "__main__":
    main()
