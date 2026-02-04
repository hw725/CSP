#!/usr/bin/env python3
"""
K값 최적화 스크립트 (Elbow Method + Silhouette Analysis)

phrase_full.csv의 임베딩에 대해 K=2~20 범위에서 최적 K값을 찾습니다.
시각화 결과를 reports/ 디렉터리에 저장합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Docker

# 한글 폰트 설정
import matplotlib.font_manager as fm

font_paths = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/nanum/NanumGothic.ttf",
]
for fp in font_paths:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = "NanumGothic"
        break
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

import regex

# Canon 정의 (실제 CSV의 book 컬럼 값)
SASEO_BOOKS = {"논어집주", "맹자집주", "대학장구", "중용장구"}
SEKYUNG_COMPLETE = {
    "서경집전(상)",
    "서경집전(하)",
    "시경집전(상)",
    "시경집전(하)",
    "주역전의(상)",
    "주역전의(하)",
}
CANON_BOOKS = SASEO_BOOKS | SEKYUNG_COMPLETE

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def extract_markers_from_text(text: str) -> list:
    """텍스트에서 현토 마커 추출"""
    if pd.isna(text):
        return []
    return regex.findall(r"\p{Hangul}+", str(text))

def main():
    import argparse

    parser = argparse.ArgumentParser(description="K값 최적화 분석")
    parser.add_argument(
        "--boundary",
        choices=["phrase", "sentence"],
        default="phrase",
        help="데이터 유형",
    )
    parser.add_argument(
        "--use-weights",
        action="store_true",
        help="Canon 가중치 적용 (MiniBatchKMeans sample_weight)",
    )
    parser.add_argument(
        "--canon-weight",
        type=float,
        default=3.0,
        help="Canon 가중치 배수 (default: 3.0)",
    )
    args = parser.parse_args()

    boundary_type = args.boundary
    print("=" * 70)
    print(f"K값 최적화 분석 ({boundary_type.capitalize()} 데이터)")
    print("=" * 70)

    # 데이터 로드 (정규화된 데이터 사용)
    data_path = DATASETS_DIR / f"{boundary_type}_normalized.csv"
    if not data_path.exists():
        data_path = DATASETS_DIR / f"{boundary_type}_full.csv"
    print(f"\n[1/4] 데이터 로드: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  행수: {len(df):,}")

    # 텍스트 준비
    texts = df["원문"].fillna("").tolist()

    # 가중치 준비 (선택)
    sample_weights = None
    if args.use_weights:
        book_col = None
        for c in ["book_name", "book", "책명"]:
            if c in df.columns:
                book_col = c
                break
        if book_col is None:
            print("⚠️ book 컬럼이 없어 가중치를 적용할 수 없습니다. (균등으로 진행)")
        else:
            sample_weights = np.ones(len(df))
            canon_mask = df[book_col].isin(CANON_BOOKS)
            sample_weights[canon_mask] = args.canon_weight
            print(
                f"⚖️ 가중치 적용: Canon {args.canon_weight}x, Other 1.0x (Canon {canon_mask.sum():,}건)"
            )

    # 임베딩 캐시 확인
    embedding_cache_path = RESULTS_DIR / f"{boundary_type}_embeddings_cache.npy"

    if embedding_cache_path.exists():
        print(f"\n[2/4] 캐시된 임베딩 로드: {embedding_cache_path}")
        embeddings = np.load(embedding_cache_path)
        print(f"  임베딩 로드 완료: shape={embeddings.shape}")
    else:
        print(f"\n[2/4] BGE-M3 임베딩 생성 중...")
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        # 배치 처리
        batch_size = 256
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="임베딩"):
            batch = texts[i : i + batch_size]
            emb = model.encode(batch, max_length=512)["dense_vecs"]
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings)
        print(f"  임베딩 완료: shape={embeddings.shape}")

        # 캐시 저장
        np.save(embedding_cache_path, embeddings)
        print(f"  캐시 저장: {embedding_cache_path}")

    # K값 평가
    print(f"\n[3/4] K값 평가 중... (K=2~20)")
    print("-" * 60)

    results = []
    k_range = range(2, 21)

    for k in k_range:
        print(f"K={k:2d}...", end=" ", flush=True)

        # 클러스터링
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=1024, n_init=3
        )
        if sample_weights is not None:
            kmeans.fit(embeddings, sample_weight=sample_weights)
            labels = kmeans.labels_
        else:
            labels = kmeans.fit_predict(embeddings)

        # Silhouette Score (높을수록 좋음, -1~1)
        silhouette = silhouette_score(embeddings, labels, sample_size=5000)

        # Davies-Bouldin Index (낮을수록 좋음)
        davies_bouldin = davies_bouldin_score(embeddings, labels)

        # Inertia (낮을수록 좋음)
        inertia = kmeans.inertia_

        results.append(
            {
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "inertia": inertia,
            }
        )

        print(f"Silhouette={silhouette:.3f}, DB={davies_bouldin:.3f}")

    # 결과 분석
    df_results = pd.DataFrame(results)

    print(f"\n{'=' * 70}")
    print("최적 K값 분석 결과")
    print(f"{'=' * 70}\n")

    # Silhouette 기준 최적 K
    best_silhouette_k = df_results.loc[df_results["silhouette"].idxmax(), "k"]
    best_silhouette_val = df_results["silhouette"].max()
    print(
        f"  Silhouette Score 최고: K={int(best_silhouette_k)}, Score={best_silhouette_val:.3f}"
    )

    # Davies-Bouldin 기준 최적 K (낮을수록 좋음)
    best_db_k = df_results.loc[df_results["davies_bouldin"].idxmin(), "k"]
    best_db_val = df_results["davies_bouldin"].min()
    print(f"  Davies-Bouldin Index 최저: K={int(best_db_k)}, Index={best_db_val:.3f}")

    # [4/4] 시각화 생성
    print(f"\n[4/4] 시각화 생성 중...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Elbow Method (Inertia)
    ax1 = axes[0]
    ax1.plot(df_results["k"], df_results["inertia"], "bo-", linewidth=2, markersize=8)
    ax1.axvline(x=4, color="r", linestyle="--", label="K=4 (선택값)")
    ax1.set_xlabel("K (클러스터 수)", fontsize=12)
    ax1.set_ylabel("Inertia", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Silhouette Score
    ax2 = axes[1]
    colors = ["red" if k == best_silhouette_k else "steelblue" for k in df_results["k"]]
    ax2.bar(df_results["k"], df_results["silhouette"], color=colors, edgecolor="black")
    ax2.axhline(
        y=best_silhouette_val,
        color="r",
        linestyle="--",
        alpha=0.7,
    )
    ax2.set_xlabel("K (클러스터 수)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score (높을수록 좋음)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Davies-Bouldin Index
    ax3 = axes[2]
    colors = ["red" if k == best_db_k else "steelblue" for k in df_results["k"]]
    ax3.bar(
        df_results["k"], df_results["davies_bouldin"], color=colors, edgecolor="black"
    )
    ax3.axhline(
        y=best_db_val,
        color="r",
        linestyle="--",
        alpha=0.7,
    )
    ax3.set_xlabel("K (클러스터 수)", fontsize=12)
    ax3.set_ylabel("Davies-Bouldin Index", fontsize=12)
    ax3.set_title(
        "Davies-Bouldin Index (낮을수록 좋음)", fontsize=14, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # 저장
    viz_path = REPORTS_DIR / f"{boundary_type}_k_optimization_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"  시각화 저장: {viz_path}")

    # 추천 K값
    print(f"\n{'=' * 70}")
    print("추천 K값")
    print(f"{'=' * 70}")
    # 최적 K: Silhouette과 DB 모두 고려하여 결정
    optimal_k = (
        best_silhouette_k
        if best_silhouette_k == best_db_k
        else min(best_silhouette_k, best_db_k)
    )
    print(f"  {boundary_type.upper()} 최적 K: {int(optimal_k)}")
    print(
        f"  (Silhouette 기준: K={int(best_silhouette_k)}, DB 기준: K={int(best_db_k)})"
    )

    # 결과 JSON 저장
    results_json = {
        "analysis_type": "k_optimization",
        "data_type": boundary_type,
        "weighted": bool(sample_weights is not None),
        "canon_weight": float(args.canon_weight) if sample_weights is not None else 1.0,
        "sample_size": len(df),
        "embedding_dim": int(embeddings.shape[1]),
        "k_range": [2, 20],
        "results": results,
        "best_silhouette_k": int(best_silhouette_k),
        "best_silhouette_val": float(best_silhouette_val),
        "best_db_k": int(best_db_k),
        "best_db_val": float(best_db_val),
        "recommended_k": int(optimal_k),
    }

    result_path = RESULTS_DIR / f"{boundary_type}_k_optimization_analysis.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\n  결과 저장: {result_path}")
    print("\n" + "=" * 70)
    print("K값 최적화 분석 완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()
