#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canon3 가중치 적용 클러스터링 (Reweighted Clustering)

MiniBatchKMeans의 sample_weight 파라미터를 사용하여
사서+삼경(Canon) 샘플에 3배 가중치를 적용한 클러스터링을 수행합니다.

결과:
- 클러스터 중심이 Canon 쪽으로 이동
- 클러스터 라벨 자체가 균등(1:1)과 달라짐
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import json

# Canon 정의 (실제 CSV의 book 컬럼 값)
SASEO_BOOKS = {"논어집주", "맹자집주", "대학장구", "중용장구"}
SEKYUNG_COMPLETE = {
    "서경집전(상)", "서경집전(하)",
    "시경집전(상)", "시경집전(하)",
    "주역전의(상)", "주역전의(하)"
}
CANON_BOOKS = SASEO_BOOKS | SEKYUNG_COMPLETE

def load_embeddings_with_weights(
    csv_path: Path,
    embedding_cache: Path,
    book_col: str = "book",
    canon_weight: float = 3.0
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    임베딩과 가중치 로드

    Returns:
        embeddings: (N, D) array
        sample_weights: (N,) array (canon=3.0, other=1.0)
        df: 원본 데이터프레임
    """
    print(f"📂 데이터 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   {len(df):,}건")

    if not embedding_cache.exists():
        raise FileNotFoundError(f"임베딩 캐시 필요: {embedding_cache}")

    print(f"📂 임베딩 캐시 로드: {embedding_cache}")
    embeddings = np.load(embedding_cache)

    if len(embeddings) != len(df):
        raise ValueError(f"임베딩({len(embeddings)})과 데이터({len(df)}) 크기 불일치")

    # 가중치 생성
    print(f"⚖️ 가중치 생성: Canon {canon_weight}x, Other 1.0x")
    sample_weights = np.ones(len(df))
    canon_mask = df[book_col].isin(CANON_BOOKS)
    sample_weights[canon_mask] = canon_weight

    canon_count = canon_mask.sum()
    other_count = (~canon_mask).sum()
    print(f"   Canon: {canon_count:,}건 (가중 {canon_count * canon_weight:,.0f})")
    print(f"   Other: {other_count:,}건 (가중 {other_count * 1.0:,.0f})")
    print(f"   총 가중치: {sample_weights.sum():,.0f}")

    return embeddings, sample_weights, df

def weighted_clustering(
    embeddings: np.ndarray,
    sample_weights: np.ndarray,
    k: int = 3,
    seed: int = 42
) -> np.ndarray:
    """가중치 적용 MiniBatchKMeans 클러스터링"""
    print(f"\n🔬 가중 클러스터링: K={k}, seed={seed}")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=1024,
        n_init="auto"
    )

    # sample_weight 적용
    kmeans.fit(embeddings, sample_weight=sample_weights)
    labels = kmeans.labels_

    print(f"   클러스터 분포:")
    unique, counts = np.unique(labels, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f"      Cluster {c}: {cnt:,}건 ({cnt/len(labels)*100:.1f}%)")

    return labels

def save_reweighted_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
    tag: str = "sentence"
):
    """가중 클러스터링 결과 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    df_out['cluster_id'] = labels

    # 클러스터 크기 추가
    cluster_sizes = df_out['cluster_id'].value_counts().to_dict()
    df_out['cluster_size'] = df_out['cluster_id'].map(cluster_sizes)

    csv_path = output_dir / f"{tag}_clusters.csv"
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장: {csv_path}")

    # 메타데이터
    meta = {
        "total_samples": len(df_out),
        "num_clusters": len(cluster_sizes),
        "cluster_sizes": {int(k): int(v) for k, v in cluster_sizes.items()},
        "reweighted": True,
        "canon_weight": 3.0
    }

    meta_path = output_dir / f"{tag}_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ 메타데이터 저장: {meta_path}")

def main():
    parser = argparse.ArgumentParser(description="Canon3 가중 클러스터링")
    parser.add_argument("--csv", type=Path, required=True, help="입력 CSV (sentence/phrase_clusters.csv)")
    parser.add_argument("--embedding-cache", type=Path, required=True, help="임베딩 캐시 .npy")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉토리")
    parser.add_argument("--k", type=int, default=3, help="클러스터 수 (default: 3)")
    parser.add_argument("--canon-weight", type=float, default=3.0, help="Canon 가중치 (default: 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tag", type=str, default="reweighted", help="출력 파일 태그")
    args = parser.parse_args()

    print("="*70)
    print("🎯 Canon3 가중 클러스터링")
    print("="*70)

    # 1. 데이터 로드
    embeddings, sample_weights, df = load_embeddings_with_weights(
        args.csv,
        args.embedding_cache,
        canon_weight=args.canon_weight
    )

    # 2. 가중 클러스터링
    labels = weighted_clustering(embeddings, sample_weights, k=args.k, seed=args.seed)

    # 3. 결과 저장
    save_reweighted_clusters(df, labels, args.out_dir, tag=args.tag)

    print("\n" + "="*70)
    print("✅ 가중 클러스터링 완료!")
    print("="*70)

if __name__ == "__main__":
    main()
