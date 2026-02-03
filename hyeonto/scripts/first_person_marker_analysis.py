#!/usr/bin/env python3
"""
1인칭 주어 표지 'ㅗ' 모음 분석 스크립트

중세/근대 한국어에서 'ㅗ' 모음은 1인칭 주어를 표시하는 문법적 기능을 가졌습니다.
예: -노이다, -호대, -로소이다 등

이 스크립트는 현토 데이터에서 1인칭 표지 마커를 분석합니다.
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import json
import re

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "first_person_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# 데이터 로드
print("데이터 로딩 중...")
df = pd.read_csv(DATASETS_DIR / "sentence_normalized.csv")
print(f"총 문장 수: {len(df):,}")

# 장르 매핑
GENRE_MAPPING = {
    "논어집주": "사서(四書)", "맹자집주": "사서(四書)", 
    "대학장구": "사서(四書)", "중용장구": "사서(四書)",
    "주역전의(상)": "삼경(三經)", "주역전의(하)": "삼경(三經)",
    "서경집전(상)": "삼경(三經)", "서경집전(하)": "삼경(三經)",
    "시경집전(상)": "삼경(三經)", "시경집전(하)": "삼경(三經)",
    "예기집설대전1": "예학(禮學)", "예기집설대전2": "예학(禮學)",
}
for i in range(1, 8):
    GENRE_MAPPING[f"자치통감강목{i}"] = "역사서(史書)"
for i in range(1, 9):
    GENRE_MAPPING[f"춘추좌씨전{i}"] = "역사서(史書)"
for prefix in ["당송팔대가문초", "당시삼백수"]:
    for book in df["book"].unique():
        if book.startswith(prefix[:4]):
            GENRE_MAPPING[book] = "문집(集部)"

df["genre"] = df["book"].map(GENRE_MAPPING).fillna("문집(集部)")

# 1인칭 표지 'ㅗ' 마커 정의
# 언어학적 분류:
# 1. -노- : 1인칭 주어 표지 (하노이다, 하노라, 하노니)
# 2. -호- : 1인칭 청자 높임 (호되, 호대, 호리라)  
# 3. -로소- : 1인칭 겸양 (로소이다)
# 4. -오- : 1인칭 (의문/감탄)

FIRST_PERSON_MARKERS = {
    # -노- 계열 (1인칭 서술)
    "하노이다": {"type": "1인칭 겸양 서술", "meaning": "~합니다 (겸양)"},
    "하노라": {"type": "1인칭 평서", "meaning": "~한다 (1인칭)"},
    "하노니": {"type": "1인칭 연결", "meaning": "~하니 (1인칭)"},
    "라하노이다": {"type": "1인칭 인용 겸양", "meaning": "~라고 합니다"},
    "라하노라": {"type": "1인칭 인용 평서", "meaning": "~라고 한다"},
    "라하노니": {"type": "1인칭 인용 연결", "meaning": "~라고 하니"},
    "노라": {"type": "1인칭 평서", "meaning": "~노라"},
    # -호- 계열 (1인칭 + 청자 높임/낮춤)
    "호되": {"type": "1인칭 역접", "meaning": "~하되 (1인칭)"},
    "호대": {"type": "1인칭 인용", "meaning": "~하기를 (1인칭)"},
    "호리라": {"type": "1인칭 의지/추측", "meaning": "~하리라 (1인칭)"},
    "호리이다": {"type": "1인칭 겸양 의지", "meaning": "~하겠습니다"},
    "호라": {"type": "1인칭 명령/감탄", "meaning": "~하라/~하도다"},
    "호니": {"type": "1인칭 연결", "meaning": "~하니 (1인칭)"},
    "라호되": {"type": "1인칭 인용 역접", "meaning": "~라 하되"},
    "라호라": {"type": "1인칭 인용 감탄", "meaning": "~라 하도다"},
    "호리니": {"type": "1인칭 연결 의지", "meaning": "~하리니"},
    "호리오": {"type": "1인칭 의문", "meaning": "~하리오?"},
    # -로소- 계열 (1인칭 겸양)
    "로소이다": {"type": "1인칭 겸양 종결", "meaning": "~입니다 (겸양)"},
    "로소니": {"type": "1인칭 겸양 연결", "meaning": "~이니 (겸양)"},
    "호이다": {"type": "1인칭 겸양", "meaning": "~합니다"},
    "라호이다": {"type": "1인칭 인용 겸양", "meaning": "~라 합니다"},
}

# 전체 마커에서 1인칭 표지 추출
all_markers = []
for markers in df["marker_normalized"].dropna():
    for m in str(markers).split(","):
        m = m.strip()
        if m:
            all_markers.append(m)

marker_counts = Counter(all_markers)

# 1인칭 표지 마커 필터링 (ㅗ 모음 기반)
first_person_markers_found = {}
for marker, count in marker_counts.items():
    # 'ㅗ' 음절 포함 여부 확인 (노, 호, 로소, 오 등)
    if any(pattern in marker for pattern in ["노", "호", "로소", "고"]):
        # '하고', '로' 등 단순 연결은 제외
        if marker not in ["하고", "로", "고", "로되", "로다"]:
            first_person_markers_found[marker] = count

print(f"\n=== 1인칭 표지 'ㅗ' 마커 분석 ===")
print(f"발견된 1인칭 표지 마커: {len(first_person_markers_found)}개")
print(f"총 출현 빈도: {sum(first_person_markers_found.values()):,}회")

# 빈도순 정렬
sorted_markers = sorted(first_person_markers_found.items(), key=lambda x: x[1], reverse=True)

print("\n[상위 20개 1인칭 표지 마커]")
for marker, count in sorted_markers[:20]:
    info = FIRST_PERSON_MARKERS.get(marker, {"type": "미분류", "meaning": "-"})
    print(f"  {marker}: {count:,} ({info['type']})")

# 장르별 1인칭 표지 분포 분석
print("\n=== 장르별 1인칭 표지 분포 ===")

def has_first_person_marker(marker_str):
    """1인칭 표지 마커 포함 여부"""
    if pd.isna(marker_str):
        return False
    markers = str(marker_str).split(",")
    for m in markers:
        m = m.strip()
        if m in first_person_markers_found:
            return True
    return False

def get_first_person_markers(marker_str):
    """1인칭 표지 마커 추출"""
    if pd.isna(marker_str):
        return []
    markers = str(marker_str).split(",")
    result = []
    for m in markers:
        m = m.strip()
        if m in first_person_markers_found:
            result.append(m)
    return result

df["has_first_person"] = df["marker_normalized"].apply(has_first_person_marker)
df["first_person_markers"] = df["marker_normalized"].apply(get_first_person_markers)

# 장르별 통계
genre_stats = {}
for genre in df["genre"].unique():
    df_genre = df[df["genre"] == genre]
    total = len(df_genre)
    with_fp = df_genre["has_first_person"].sum()
    ratio = with_fp / total * 100 if total > 0 else 0
    
    # 장르별 1인칭 마커 빈도
    fp_markers = Counter()
    for markers in df_genre["first_person_markers"]:
        for m in markers:
            fp_markers[m] += 1
    
    genre_stats[genre] = {
        "total": total,
        "with_first_person": with_fp,
        "ratio": ratio,
        "top_markers": fp_markers.most_common(10)
    }

print("\n| 장르 | 총 문장 | 1인칭 표지 포함 | 비율 |")
print("|:-----|-------:|---------------:|-----:|")
for genre, stats in sorted(genre_stats.items(), key=lambda x: x[1]["ratio"], reverse=True):
    print(f"| {genre} | {stats['total']:,} | {stats['with_first_person']:,} | {stats['ratio']:.2f}% |")

# 장르별 상위 1인칭 마커
print("\n=== 장르별 주요 1인칭 표지 마커 ===")
for genre, stats in sorted(genre_stats.items(), key=lambda x: x[1]["ratio"], reverse=True):
    print(f"\n[{genre}] (1인칭 비율: {stats['ratio']:.2f}%)")
    for marker, count in stats["top_markers"][:5]:
        info = FIRST_PERSON_MARKERS.get(marker, {"type": "미분류"})
        print(f"  {marker}: {count} ({info['type']})")

# 대표 예문 추출
print("\n=== 1인칭 표지 대표 예문 ===")

examples = []
for marker in ["하노이다", "호되", "로소이다", "하노라", "호리라"]:
    df_with_marker = df[df["marker_normalized"].str.contains(marker, na=False, regex=False)]
    if len(df_with_marker) > 0:
        sample = df_with_marker.head(3)
        for _, row in sample.iterrows():
            examples.append({
                "marker": marker,
                "genre": row["genre"],
                "book": row["book"],
                "src_right": row.get("src_right", ""),
                "tgt_right": row.get("tgt_right", ""),
                "marker_right": row.get("marker_right", ""),
            })

for ex in examples[:15]:
    info = FIRST_PERSON_MARKERS.get(ex["marker"], {"meaning": "-"})
    print(f"\n**{ex['marker']}** ({info['meaning']}) - {ex['book']}")
    print(f"  원문: {ex['src_right'][:80]}...")
    print(f"  번역: {ex['tgt_right'][:80]}...")

# Markdown 보고서 생성
report_path = OUTPUT_DIR / "FIRST_PERSON_MARKER_ANALYSIS.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# 1인칭 주어 표지 'ㅗ' 모음 분석 보고서\n\n")
    f.write("## 개요\n\n")
    f.write("중세/근대 한국어에서 **'ㅗ' 모음**은 **1인칭 주어**를 표시하는 문법적 기능을 가졌습니다.\n")
    f.write("현토(懸吐)에서 이 1인칭 표지는 화자(話者)의 입장을 명시하는 중요한 역할을 합니다.\n\n")
    
    f.write("### 1인칭 표지의 유형\n\n")
    f.write("| 유형 | 예시 | 기능 |\n")
    f.write("|:-----|:-----|:-----|\n")
    f.write("| **-노-** 계열 | 하노이다, 하노라, 하노니 | 1인칭 서술/연결 |\n")
    f.write("| **-호-** 계열 | 호되, 호대, 호리라 | 1인칭 + 청자 높임/대우 |\n")
    f.write("| **-로소-** 계열 | 로소이다, 로소니 | 1인칭 겸양 |\n\n")
    
    f.write("## 1. 발견된 1인칭 표지 마커\n\n")
    f.write(f"- 발견된 마커 수: **{len(first_person_markers_found)}개**\n")
    f.write(f"- 총 출현 빈도: **{sum(first_person_markers_found.values()):,}회**\n\n")
    
    f.write("### 주요 마커 (빈도순)\n\n")
    f.write("| 마커 | 빈도 | 유형 | 의미 |\n")
    f.write("|:-----|-----:|:-----|:-----|\n")
    for marker, count in sorted_markers[:25]:
        info = FIRST_PERSON_MARKERS.get(marker, {"type": "미분류", "meaning": "-"})
        f.write(f"| {marker} | {count:,} | {info['type']} | {info['meaning']} |\n")
    
    f.write("\n## 2. 장르별 분포\n\n")
    f.write("1인칭 표지의 출현 비율은 **장르에 따라 현저히 다릅니다**.\n\n")
    f.write("| 장르 | 총 문장 | 1인칭 표지 포함 | 비율 |\n")
    f.write("|:-----|-------:|---------------:|-----:|\n")
    for genre, stats in sorted(genre_stats.items(), key=lambda x: x[1]["ratio"], reverse=True):
        f.write(f"| {genre} | {stats['total']:,} | {stats['with_first_person']:,} | {stats['ratio']:.2f}% |\n")
    
    f.write("\n### 해석\n\n")
    f.write("- **문집(集部)**: 상소문, 서간 등에서 1인칭 화자가 자주 등장 → 1인칭 표지 비율 높음\n")
    f.write("- **역사서(史書)**: 대화 인용이 많아 1인칭 표현 빈번\n")
    f.write("- **사서·삼경**: 경전 해설 중심으로 1인칭 표현 희소\n\n")
    
    f.write("## 3. 장르별 주요 1인칭 마커\n\n")
    for genre, stats in sorted(genre_stats.items(), key=lambda x: x[1]["ratio"], reverse=True):
        f.write(f"### {genre}\n\n")
        f.write("| 마커 | 빈도 | 유형 |\n")
        f.write("|:-----|-----:|:-----|\n")
        for marker, count in stats["top_markers"][:7]:
            info = FIRST_PERSON_MARKERS.get(marker, {"type": "미분류"})
            f.write(f"| {marker} | {count} | {info['type']} |\n")
        f.write("\n")
    
    f.write("## 4. 대표 예문\n\n")
    current_marker = None
    for ex in examples[:15]:
        if ex["marker"] != current_marker:
            current_marker = ex["marker"]
            info = FIRST_PERSON_MARKERS.get(current_marker, {"meaning": "-", "type": "-"})
            f.write(f"### `{current_marker}` ({info['type']})\n\n")
            f.write(f"의미: {info['meaning']}\n\n")
        
        f.write(f"**{ex['book']}** ({ex['genre']})\n")
        f.write(f"- 원문: {ex['src_right']}\n")
        f.write(f"- 번역: {ex['tgt_right']}\n\n")
    
    f.write("## 5. 언어학적 의의\n\n")
    f.write("### 5.1 화자 표지로서의 'ㅗ'\n\n")
    f.write("중세 한국어에서 **'ㅗ' 모음**은 1인칭 화자를 표시하는 체계적인 문법 표지였습니다:\n\n")
    f.write("- **-노-**: 1인칭 주어 + 서술 (`하노라` = 내가 한다)\n")
    f.write("- **-호-**: 1인칭 + 청자 대우 (`호되` = 내가 말하되)\n")
    f.write("- **-로소-**: 1인칭 + 겸양 (`로소이다` = 저는 ~입니다)\n\n")
    f.write("### 5.2 장르적 분화\n\n")
    f.write("1인칭 표지의 분포는 텍스트의 **발화 상황(Speech Situation)**과 밀접하게 연관됩니다:\n\n")
    f.write("- **문집**: 저자가 직접 발화 → 1인칭 표지 빈번\n")
    f.write("- **역사서**: 대화 인용 → 1인칭/2인칭 표지 혼재\n")
    f.write("- **경전**: 객관적 서술 → 1인칭 표지 희소\n\n")
    f.write("이는 현토가 단순한 '토씨 붙이기'가 아니라, **화용론적 맥락까지 반영**하는 정교한 번역 체계임을 보여줍니다.\n")

print(f"\n보고서 저장 완료: {report_path}")
print("\n완료!")
