"""
K=4 클러스터 집중 분석 스크립트
- 클러스터별 서종(書種) 분포 상세 분석
- 사서(四書), 삼경(三經), 사서(史書), 집부(集部) 분류
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# 서종 분류 정의
BOOK_CATEGORIES = {
    # 사서 (四書) - Saseo
    "논어집주": "사서",
    "맹자집주": "사서",
    "대학장구": "사서",
    "중용장구": "사서",
    
    # 삼경 (三經) - Samgyeong  
    "주역전의(상)": "삼경",
    "주역전의(하)": "삼경",
    "서경집전(상)": "삼경",
    "서경집전(하)": "삼경",
    "시경집전(상)": "삼경",
    "시경집전(하)": "삼경",
    
    # 사서 (史書) - History
    "춘추좌씨전1": "사서(史)",
    "춘추좌씨전2": "사서(史)",
    "춘추좌씨전3": "사서(史)",
    "춘추좌씨전4": "사서(史)",
    "춘추좌씨전5": "사서(史)",
    "춘추좌씨전6": "사서(史)",
    "자치통감강목1": "사서(史)",
    "자치통감강목2": "사서(史)",
    "자치통감강목3": "사서(史)",
    "자치통감강목4": "사서(史)",
    "자치통감강목5": "사서(史)",
    "자치통감강목6": "사서(史)",
    "자치통감강목7": "사서(史)",
    
    # 예기 (禮記)
    "예기집설대전1": "예기",
    "예기집설대전2": "예기",
    "예기집설대전3": "예기",
    "예기집설대전4": "예기",
}

# 집부(集部) - 당송팔대가문초
TANGSUNG_AUTHORS = ["한유", "유종원", "구양수", "소순", "소식", "소철", "왕안석", "증공"]


def categorize_book(book_name: str) -> str:
    """책 이름을 서종으로 분류"""
    if book_name in BOOK_CATEGORIES:
        return BOOK_CATEGORIES[book_name]
    
    # 당송팔대가문초 처리
    if "당송팔대가문초" in book_name:
        for author in TANGSUNG_AUTHORS:
            if author in book_name:
                return f"집부({author})"
        return "집부(기타)"
    
    return "기타"


def analyze_cluster_book_distribution(df: pd.DataFrame, cluster_col: str = "cluster_id"):
    """클러스터별 서종 분포 분석"""
    
    # 서종 분류 추가
    df["book_category"] = df["book_name"].apply(categorize_book)
    
    results = []
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster_id]
        total = len(cluster_df)
        
        # 서종별 분포
        category_dist = cluster_df["book_category"].value_counts()
        category_pct = (category_dist / total * 100).round(2)
        
        # 개별 서적 분포 (상위 15개)
        book_dist = cluster_df["book_name"].value_counts().head(15)
        book_pct = (book_dist / total * 100).round(2)
        
        # 사서 비율 계산
        saseo_count = category_dist.get("사서", 0)
        saseo_ratio = saseo_count / total * 100
        
        # 경서(삼경) 비율
        samgyeong_count = category_dist.get("삼경", 0)
        samgyeong_ratio = samgyeong_count / total * 100
        
        # 사서(史) 비율
        history_count = category_dist.get("사서(史)", 0)
        history_ratio = history_count / total * 100
        
        # 집부 합계
        jipbu_count = sum(v for k, v in category_dist.items() if k.startswith("집부"))
        jipbu_ratio = jipbu_count / total * 100
        
        results.append({
            "cluster_id": cluster_id,
            "total": total,
            "saseo_count": saseo_count,
            "saseo_ratio": saseo_ratio,
            "samgyeong_count": samgyeong_count,
            "samgyeong_ratio": samgyeong_ratio,
            "history_count": history_count,
            "history_ratio": history_ratio,
            "jipbu_count": jipbu_count,
            "jipbu_ratio": jipbu_ratio,
            "category_dist": category_pct.to_dict(),
            "top_books": book_pct.to_dict()
        })
    
    return results, df


def print_detailed_analysis(results: list, layer_name: str):
    """상세 분석 결과 출력"""
    
    print(f"\n{'='*80}")
    print(f"📊 {layer_name} K=4 클러스터별 서종 분포 상세 분석")
    print(f"{'='*80}")
    
    for r in results:
        print(f"\n{'─'*80}")
        print(f"🔹 Cluster {r['cluster_id']} (n={r['total']:,})")
        print(f"{'─'*80}")
        
        print(f"\n  ▶ 서종 대분류 비율:")
        print(f"    • 사서(四書): {r['saseo_count']:,}건 ({r['saseo_ratio']:.2f}%)")
        print(f"    • 삼경(三經): {r['samgyeong_count']:,}건 ({r['samgyeong_ratio']:.2f}%)")
        print(f"    • 사서(史書): {r['history_count']:,}건 ({r['history_ratio']:.2f}%)")
        print(f"    • 집부(集部): {r['jipbu_count']:,}건 ({r['jipbu_ratio']:.2f}%)")
        
        print(f"\n  ▶ 서종 세부 분포:")
        for cat, pct in sorted(r["category_dist"].items(), key=lambda x: -x[1]):
            print(f"    • {cat}: {pct:.2f}%")
        
        print(f"\n  ▶ 상위 15개 개별 서적:")
        for i, (book, pct) in enumerate(r["top_books"].items(), 1):
            category = categorize_book(book)
            print(f"    {i:2d}. {book} ({category}): {pct:.2f}%")


def generate_markdown_report(results: list, layer_name: str, output_path: Path):
    """마크다운 보고서 생성"""
    
    lines = [
        f"# {layer_name} K=4 클러스터별 서종(書種) 분포 상세 분석",
        "",
        f"- **분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"- **총 클러스터 수**: {len(results)}",
        f"- **총 데이터 수**: {sum(r['total'] for r in results):,}",
        "",
        "## 📋 요약 테이블",
        "",
        "| Cluster | Size | 사서(四書) | 삼경(三經) | 사서(史書) | 집부(集部) | 기타 |",
        "|:-------:|-----:|-----------:|-----------:|-----------:|-----------:|-----:|",
    ]
    
    for r in results:
        other = 100 - r['saseo_ratio'] - r['samgyeong_ratio'] - r['history_ratio'] - r['jipbu_ratio']
        # 예기 등 기타 분류 포함
        lines.append(
            f"| p{r['cluster_id']} | {r['total']:,} | "
            f"{r['saseo_ratio']:.1f}% | {r['samgyeong_ratio']:.1f}% | "
            f"{r['history_ratio']:.1f}% | {r['jipbu_ratio']:.1f}% | {other:.1f}% |"
        )
    
    lines.extend([
        "",
        "---",
        "",
    ])
    
    for r in results:
        lines.extend([
            f"## Cluster p{r['cluster_id']} (n={r['total']:,})",
            "",
            "### 서종 분포",
            "",
            "| 서종 | 비율 |",
            "|:-----|-----:|",
        ])
        
        for cat, pct in sorted(r["category_dist"].items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {pct:.2f}% |")
        
        lines.extend([
            "",
            "### 상위 15개 서적",
            "",
            "| 순위 | 서적명 | 서종 | 비율 |",
            "|:----:|:-------|:-----|-----:|",
        ])
        
        for i, (book, pct) in enumerate(r["top_books"].items(), 1):
            category = categorize_book(book)
            marker = "⭐" if category == "사서" else ""
            lines.append(f"| {i} | {marker}{book} | {category} | {pct:.2f}% |")
        
        lines.extend(["", "---", ""])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 보고서 저장: {output_path}")


def main():
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    
    # 1. Sentence Boundary K=4 분석
    sentence_path = reports_dir / "sentence_k4_normalized" / "sentence_clusters.csv"
    if sentence_path.exists():
        print(f"\n📂 Loading: {sentence_path}")
        sentence_df = pd.read_csv(sentence_path)
        print(f"   → Loaded {len(sentence_df):,} rows")

        sentence_results, sentence_df = analyze_cluster_book_distribution(sentence_df)
        print_detailed_analysis(sentence_results, "문장경계(Sentence)")

        output_sentence = reports_dir / "sentence_k4_normalized" / "k4_book_distribution_analysis.md"
        generate_markdown_report(sentence_results, "문장경계(Sentence)", output_sentence)
    else:
        print(f"⚠️ File not found: {sentence_path}")

    # 2. Phrase Boundary K=4 분석
    phrase_path = reports_dir / "phrase_k4_normalized" / "phrase_clusters.csv"
    if phrase_path.exists():
        print(f"\n📂 Loading: {phrase_path}")
        phrase_df = pd.read_csv(phrase_path)
        print(f"   → Loaded {len(phrase_df):,} rows")

        phrase_results, phrase_df = analyze_cluster_book_distribution(phrase_df)
        print_detailed_analysis(phrase_results, "구경계(Phrase)")

        output_phrase = reports_dir / "phrase_k4_normalized" / "k4_book_distribution_analysis.md"
        generate_markdown_report(phrase_results, "구경계(Phrase)", output_phrase)
    else:
        print(f"⚠️ File not found: {phrase_path}")


if __name__ == "__main__":
    main()
