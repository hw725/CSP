#!/usr/bin/env python3
"""
장르별 고유 현토 예문 추출 스크립트

K=4 클러스터 분석 결과를 바탕으로 각 장르(사서/삼경/역사서/문집)의 
고유한 현토 패턴을 보이는 문장을 추출합니다.
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import json

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "genre_examples"
OUTPUT_DIR.mkdir(exist_ok=True)

# 데이터 로드
print("데이터 로딩 중...")
df = pd.read_csv(DATASETS_DIR / "sentence_normalized.csv")
print(f"총 문장 수: {len(df):,}")

# 서명(book_name)을 장르로 매핑 - 전체 53개 서명 분류
GENRE_MAPPING = {
    # 사서(四書) - Saseo (4종)
    "논어집주": "사서(四書)",
    "맹자집주": "사서(四書)",
    "대학장구": "사서(四書)",
    "중용장구": "사서(四書)",
    # 삼경(三經) - Samgyeong (6종: 상/하 분권)
    "주역전의(상)": "삼경(三經)",
    "주역전의(하)": "삼경(三經)",
    "서경집전(상)": "삼경(三經)",
    "서경집전(하)": "삼경(三經)",
    "시경집전(상)": "삼경(三經)",
    "시경집전(하)": "삼경(三經)",
    # 역사서(史書) - History (15종: 자치통감강목 7 + 춘추좌씨전 8)
    "자치통감강목1": "역사서(史書)",
    "자치통감강목2": "역사서(史書)",
    "자치통감강목3": "역사서(史書)",
    "자치통감강목4": "역사서(史書)",
    "자치통감강목5": "역사서(史書)",
    "자치통감강목6": "역사서(史書)",
    "자치통감강목7": "역사서(史書)",
    "춘추좌씨전1": "역사서(史書)",
    "춘추좌씨전2": "역사서(史書)",
    "춘추좌씨전3": "역사서(史書)",
    "춘추좌씨전4": "역사서(史書)",
    "춘추좌씨전5": "역사서(史書)",
    "춘추좌씨전6": "역사서(史書)",
    "춘추좌씨전7": "역사서(史書)",
    "춘추좌씨전8": "역사서(史書)",
    # 문집(集部) - Collections (24종: 당송팔대가문초 + 당시삼백수)
    "당송팔대가문초구양수1": "문집(集部)",
    "당송팔대가문초구양수2": "문집(集部)",
    "당송팔대가문초구양수3": "문집(集部)",
    "당송팔대가문초구양수4": "문집(集部)",
    "당송팔대가문초구양수5": "문집(集部)",
    "당송팔대가문초구양수6": "문집(集部)",
    "당송팔대가문초소순1": "문집(集部)",
    "당송팔대가문초소식1": "문집(集部)",
    "당송팔대가문초소식2": "문집(集部)",
    "당송팔대가문초소식3": "문집(集部)",
    "당송팔대가문초소식4": "문집(集部)",
    "당송팔대가문초소식5": "문집(集部)",
    "당송팔대가문초소철1": "문집(集部)",
    "당송팔대가문초소철2": "문집(集部)",
    "당송팔대가문초소철3": "문집(集部)",
    "당송팔대가문초왕안석1": "문집(集部)",
    "당송팔대가문초왕안석2": "문집(集部)",
    "당송팔대가문초유종원1": "문집(集部)",
    "당송팔대가문초유종원2": "문집(集部)",
    "당송팔대가문초증공1": "문집(集部)",
    "당송팔대가문초한유1": "문집(集部)",
    "당송팔대가문초한유2": "문집(集部)",
    "당송팔대가문초한유3": "문집(集部)",
    "당시삼백수1": "문집(集部)",
    "당시삼백수2": "문집(集部)",
    "당시삼백수3": "문집(集部)",
    # 예학(禮學) - Ritual Studies (2종)
    "예기집설대전1": "예학(禮學)",
    "예기집설대전2": "예학(禮學)",
}

df["genre"] = df["book"].map(GENRE_MAPPING).fillna("기타")

print("\n=== 장르별 분포 ===")
genre_dist = df["genre"].value_counts()
for genre, count in genre_dist.items():
    print(f"  {genre}: {count:,} ({count/len(df)*100:.1f}%)")

# 마커 빈도 분석 함수
def analyze_markers(df_subset, top_n=20):
    """마커 빈도 분석"""
    markers = Counter()
    for marker_str in df_subset["marker_normalized"].dropna():
        markers[marker_str] += 1
    return markers.most_common(top_n)

# 장르별 마커 분포 분석
print("\n=== 장르별 고빈도 마커 ===")
genre_markers = {}
for genre in df["genre"].unique():
    df_genre = df[df["genre"] == genre]
    genre_markers[genre] = analyze_markers(df_genre, top_n=30)
    print(f"\n[{genre}] (n={len(df_genre):,})")
    for marker, count in genre_markers[genre][:10]:
        print(f"  {marker}: {count:,}")

# 장르 특이적 마커 찾기 (특정 장르에서만 높은 비율로 나타나는 마커)
print("\n=== 장르 특이적 마커 분석 ===")

# 전체 마커 빈도
all_markers = Counter()
for marker_str in df["marker_normalized"].dropna():
    all_markers[marker_str] += 1

# 장르별 마커 비율 계산
genre_marker_ratios = {}
for genre in df["genre"].unique():
    df_genre = df[df["genre"] == genre]
    genre_total = len(df_genre)
    genre_markers_count = Counter()
    for marker_str in df_genre["marker_normalized"].dropna():
        genre_markers_count[marker_str] += 1
    
    # 비율 계산 (장르 내 비율 / 전체 비율)
    ratios = {}
    for marker, count in genre_markers_count.items():
        genre_ratio = count / genre_total
        overall_ratio = all_markers[marker] / len(df)
        if overall_ratio > 0:
            specificity = genre_ratio / overall_ratio
            ratios[marker] = {
                "count": count,
                "genre_ratio": genre_ratio,
                "overall_ratio": overall_ratio,
                "specificity": specificity
            }
    
    # 특이성 순으로 정렬 (빈도가 일정 이상인 것만)
    significant_ratios = {k: v for k, v in ratios.items() if v["count"] >= 50}
    sorted_ratios = sorted(significant_ratios.items(), key=lambda x: x[1]["specificity"], reverse=True)
    genre_marker_ratios[genre] = sorted_ratios
    
    print(f"\n[{genre}] 특이적 마커 (특이성 > 1.2)")
    for marker, stats in sorted_ratios[:15]:
        if stats["specificity"] > 1.2:
            print(f"  {marker}: 특이성={stats['specificity']:.2f}, 빈도={stats['count']:,}")

# 각 장르별 대표 예문 추출
print("\n=== 장르별 대표 예문 추출 ===")

def extract_representative_examples(df, genre, specific_markers, n=5):
    """장르 특이적 마커를 포함하는 대표 예문 추출"""
    df_genre = df[df["genre"] == genre]
    examples = []
    
    for marker in specific_markers:
        df_with_marker = df_genre[df_genre["marker_normalized"] == marker]
        if len(df_with_marker) > 0:
            sample = df_with_marker.head(n)
            for _, row in sample.iterrows():
                examples.append({
                    "genre": genre,
                    "book": row["book"],
                    "marker": marker,
                    "src_left": row.get("src_left", ""),
                    "src_right": row.get("src_right", ""),
                    "tgt_left": row.get("tgt_left", ""),
                    "tgt_right": row.get("tgt_right", ""),
                    "marker_left": row.get("marker_left", ""),
                    "marker_right": row.get("marker_right", ""),
                })
    return examples

# 장르별 특이 마커 추출 (상위 5개)
genre_specific_markers = {}
for genre, ratios in genre_marker_ratios.items():
    # 특이성 > 1.2이고 빈도 >= 50인 마커
    specific = [m for m, s in ratios if s["specificity"] > 1.2 and s["count"] >= 50][:5]
    genre_specific_markers[genre] = specific
    print(f"\n[{genre}] 특이 마커: {specific}")

# 예문 추출 및 저장
all_examples = []
for genre, markers in genre_specific_markers.items():
    examples = extract_representative_examples(df, genre, markers, n=3)
    all_examples.extend(examples)
    print(f"  {genre}: {len(examples)}개 예문 추출")

# JSON으로 저장
output_json = OUTPUT_DIR / "genre_specific_examples.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_examples, f, ensure_ascii=False, indent=2)
print(f"\n예문 저장 완료: {output_json}")

# Markdown 보고서 생성
report_md = OUTPUT_DIR / "GENRE_SPECIFIC_EXAMPLES.md"
with open(report_md, "w", encoding="utf-8") as f:
    f.write("# 장르별 고유 현토 예문 보고서\n\n")
    f.write("이 보고서는 K=4 클러스터 분석 결과를 바탕으로 각 장르(사서/삼경/기타)의 고유한 현토 패턴을 보이는 문장을 추출한 것입니다.\n\n")
    
    f.write("## 1. 장르별 분포\n\n")
    f.write("| 장르 | 문장 수 | 비율 |\n")
    f.write("|:-----|-------:|-----:|\n")
    for genre, count in genre_dist.items():
        f.write(f"| {genre} | {count:,} | {count/len(df)*100:.1f}% |\n")
    
    f.write("\n## 2. 장르별 특이적 마커\n\n")
    f.write("'특이성'은 해당 마커가 특정 장르에서 전체 평균 대비 얼마나 높은 비율로 출현하는지를 나타냅니다.\n\n")
    
    for genre, ratios in genre_marker_ratios.items():
        f.write(f"### {genre}\n\n")
        f.write("| 마커 | 특이성 | 빈도 |\n")
        f.write("|:-----|-------:|-----:|\n")
        for marker, stats in ratios[:10]:
            if stats["specificity"] > 1.0:
                f.write(f"| {marker} | {stats['specificity']:.2f} | {stats['count']:,} |\n")
        f.write("\n")
    
    f.write("\n## 3. 장르별 대표 예문\n\n")
    
    current_genre = None
    for ex in all_examples:
        if ex["genre"] != current_genre:
            current_genre = ex["genre"]
            f.write(f"### {current_genre}\n\n")
        
        f.write(f"**서명**: {ex['book']} | **마커**: `{ex['marker']}`\n\n")
        f.write(f"- **원문(左)**: {ex['src_left']}\n")
        f.write(f"- **원문(右)**: {ex['src_right']}\n")
        f.write(f"- **번역(左)**: {ex['tgt_left']}\n")
        f.write(f"- **번역(右)**: {ex['tgt_right']}\n")
        f.write(f"- **현토(左)**: {ex['marker_left']}\n")
        f.write(f"- **현토(右)**: {ex['marker_right']}\n\n")
        f.write("---\n\n")

print(f"보고서 저장 완료: {report_md}")
print("\n완료!")
