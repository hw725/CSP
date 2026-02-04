"""
Canon3 가중치로 클러스터 메트릭 재계산

기존 클러스터링 결과(CSV)에 canon3(3:1) 가중치를 적용하여
Canonicity, Entropy, Distribution 메트릭을 재계산합니다.

클러스터 경계는 변경하지 않고, 평가 관점만 변경합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from collections import Counter
from scipy.stats import entropy

# 사서삼경(Canon) 정의 (실제 CSV의 book 컬럼 값)
SASEO_BOOKS = {"논어집주", "맹자집주", "대학장구", "중용장구"}
SEKYUNG_COMPLETE = {
    "서경집전(상)", "서경집전(하)",
    "시경집전(상)", "시경집전(하)",
    "주역전의(상)", "주역전의(하)"
}
CANON_BOOKS = SASEO_BOOKS | SEKYUNG_COMPLETE

def compute_weighted_canonicity(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    book_col: str = "book",
    canon_weight: float = 3.0,
) -> dict:
    """Canon3 가중치로 Canonicity 재계산"""
    results = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id]

        # Canon/Other 분류
        canon_mask = cluster_df[book_col].isin(CANON_BOOKS)
        n_canon = canon_mask.sum()
        n_other = (~canon_mask).sum()

        # 가중 카운트
        weighted_canon = n_canon * canon_weight
        weighted_other = n_other * 1.0
        weighted_total = weighted_canon + weighted_other

        # Canonicity (%)
        canonicity = (
            (weighted_canon / weighted_total * 100) if weighted_total > 0 else 0.0
        )

        results[cluster_id] = {
            "raw_canon_count": int(n_canon),
            "raw_other_count": int(n_other),
            "raw_total": int(n_canon + n_other),
            "raw_canonicity": float(
                (n_canon / (n_canon + n_other) * 100)
                if (n_canon + n_other) > 0
                else 0.0
            ),
            "weighted_canon": float(weighted_canon),
            "weighted_other": float(weighted_other),
            "weighted_total": float(weighted_total),
            "weighted_canonicity": float(canonicity),
        }

    return results

def compute_weighted_entropy(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    book_col: str = "book",
    canon_weight: float = 3.0,
) -> dict:
    """Canon3 가중치로 서종 엔트로피 재계산"""
    results = {}

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id]

        # 가중치 적용 카운트
        weighted_counts = []
        for book in cluster_df[book_col]:
            weight = canon_weight if book in CANON_BOOKS else 1.0
            weighted_counts.append(weight)

        # 서종별 가중 합산
        book_weighted = {}
        for book, weight in zip(cluster_df[book_col], weighted_counts):
            book_weighted[book] = book_weighted.get(book, 0.0) + weight

        # 엔트로피 계산
        total_weight = sum(book_weighted.values())
        probs = [w / total_weight for w in book_weighted.values()]
        ent = entropy(probs, base=2)

        results[cluster_id] = {
            "num_unique_books": int(len(book_weighted)),
            "weighted_entropy": float(ent),
            "raw_entropy": float(
                entropy(list(Counter(cluster_df[book_col]).values()), base=2)
            ),
        }

    return results

def save_weighted_profile(canonicity: dict, entropy_data: dict, out_path: Path):
    """가중 메트릭 프로파일 저장"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# K=3 Cluster Profile (Canon3 Weighted)\n\n")
        f.write("**가중치**: Canon(사서+삼경) 3.0x, Other 1.0x\n")
        f.write("**비고**: 클러스터 경계는 균등(1:1) 결과와 동일, 평가 관점만 변경\n\n")
        f.write("---\n\n")

        for cluster_id in sorted(canonicity.keys()):
            can = canonicity[cluster_id]
            ent = entropy_data[cluster_id]

            f.write(f"## Cluster {cluster_id}\n\n")
            f.write(f"### Canonicity\n")
            f.write(
                f"- **균등(1:1)**: {can['raw_canonicity']:.2f}% ({can['raw_canon_count']}/{can['raw_total']})\n"
            )
            f.write(
                f"- **Canon3(3:1)**: {can['weighted_canonicity']:.2f}% ({can['weighted_canon']:.1f}/{can['weighted_total']:.1f})\n\n"
            )

            f.write(f"### Entropy\n")
            f.write(f"- **균등(1:1)**: {ent['raw_entropy']:.4f}\n")
            f.write(f"- **Canon3(3:1)**: {ent['weighted_entropy']:.4f}\n")
            f.write(f"- **Unique Books**: {ent['num_unique_books']}\n\n")
            f.write("---\n\n")

        # Summary
        avg_raw_can = np.mean([c["raw_canonicity"] for c in canonicity.values()])
        avg_weighted_can = np.mean(
            [c["weighted_canonicity"] for c in canonicity.values()]
        )
        avg_raw_ent = np.mean([e["raw_entropy"] for e in entropy_data.values()])
        avg_weighted_ent = np.mean(
            [e["weighted_entropy"] for e in entropy_data.values()]
        )

        f.write("## Summary\n\n")
        f.write(f"| Metric | 균등(1:1) | Canon3(3:1) | 변화 |\n")
        f.write(f"|--------|-----------|-------------|------|\n")
        f.write(
            f"| 평균 Canonicity | {avg_raw_can:.2f}% | {avg_weighted_can:.2f}% | {avg_weighted_can - avg_raw_can:+.2f}%p |\n"
        )
        f.write(
            f"| 평균 Entropy | {avg_raw_ent:.4f} | {avg_weighted_ent:.4f} | {avg_weighted_ent - avg_raw_ent:+.4f} |\n"
        )

def main():
    parser = argparse.ArgumentParser(
        description="Canon3 가중치로 클러스터 메트릭 재계산"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="클러스터 CSV (sentence_clusters.csv or phrase_clusters.csv)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉토리")
    parser.add_argument(
        "--canon-weight", type=float, default=3.0, help="Canon 가중치 (default: 3.0)"
    )
    parser.add_argument(
        "--tag", type=str, default="", help="파일명 태그 (sentence/phrase)"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Canon3 가중치 메트릭 재계산")
    print(f"   입력: {args.csv}")
    print(f"   가중치: Canon {args.canon_weight}x, Other 1.0x")

    # 데이터 로드
    df = pd.read_csv(args.csv)
    print(f"   데이터: {len(df):,}건, {df['cluster_id'].nunique()}개 클러스터")

    # Canonicity 재계산
    canonicity = compute_weighted_canonicity(df, canon_weight=args.canon_weight)

    # Entropy 재계산
    entropy_data = compute_weighted_entropy(df, canon_weight=args.canon_weight)

    # Profile 저장
    tag_prefix = f"{args.tag}_" if args.tag else ""
    profile_path = args.out_dir / f"{tag_prefix}cluster_profile_weighted.md"
    save_weighted_profile(canonicity, entropy_data, profile_path)

    # JSON 저장
    json_path = args.out_dir / f"{tag_prefix}weighted_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "canonicity": {str(k): v for k, v in canonicity.items()},
                "entropy": {str(k): v for k, v in entropy_data.items()},
                "canon_weight": args.canon_weight,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✅ 완료:")
    print(f"   - {profile_path}")
    print(f"   - {json_path}")

if __name__ == "__main__":
    main()
