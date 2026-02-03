#!/usr/bin/env python3
"""
현토 통합 리서치 스크립트

1. '하다' 분석: sentence_full.csv 기반 (문장 단위)
2. Classified Markers 분석: phrase_full.csv 기반 (구절 단위)

사용법: python run_full_research.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
import json
import re
import regex
import sys

# 경로 설정
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# hyeonto_normalizer import
sys.path.insert(0, str(BASE_DIR))
from hyeonto_normalizer import normalize_hyeonto_marker

# ============================================================================
# TAXONOMY 정의 (phase4_premodern_classify.py에서 가져옴)
# ============================================================================
PREMODERN_TAXONOMY = {
    "제외_구두점": {
        "description": "구두점 - 현토 아님",
        "source": "오류",
        "patterns": [r"^ㆍ$", r"^,$", r"^\."]
    },
    "단사_미절": {
        "description": "약하게 끊음 (微絶)",
        "source": "임규직 《구두해법》",
        "patterns": [r"^(이)?라$"]
    },
    "단사_쾌절": {
        "description": "단호하게 결정하여 끊음 (夬絶)",
        "source": "임규직 《구두해법》",
        "patterns": [r".*니라$", r".*시니라$"]
    },
    "단사_기사지단": {
        "description": "정령, 조서 등 공적 기록",
        "source": "임규직 《구두해법》",
        "patterns": [r"^하다$", r".*하시다$"]
    },
    "단사_서술지단": {
        "description": "사건의 전말이나 행적을 서술함",
        "source": "임규직 《구두해법》",
        "patterns": [r".*러라$", r".*더라$"]
    },
    "단사_유사이단": {
        "description": "감탄이나 여운을 남김 (游辭以斷)",
        "source": "임규직 《구두해법》 / 박문호 《이두해》 30번",
        "patterns": [r".*로다$", r".*놋다$", r".*도다$"]
    },
    "주체_한대": {
        "description": "주체의 행위, 동일 주어 유지",
        "source": "이삼환 《구두지남》",
        "patterns": [r"^한대$", r"^하신대$", r".*한대$", r"^로대$", r"^이로대$"]
    },
    "객체_어늘": {
        "description": "객체의 행위, 주어 전환",
        "source": "이삼환 《구두지남》",
        "patterns": [r".*어늘$", r".*거늘$", r".*커늘$", r".*커든$"]
    },
    "과거_러니": {
        "description": "과거/회상 (往昔事)",
        "source": "이삼환 《구두지남》 / 박문호 13번",
        "patterns": [r".*러니$", r".*더니$", r".*러시니$", r".*더시니$"]
    },
    "미래_리니": {
        "description": "미래/필연 (未來事)",
        "source": "이삼환 《구두지남》 / 박문호 15번",
        "patterns": [r".*리니$"]
    },
    "미래_리라": {
        "description": "미래 추측/의지",
        "source": "이삼환 《구두지남》",
        "patterns": [r".*리라$", r"^하리다$", r"^리다$"]
    },
    "진행_할새": {
        "description": "한창 진행 중임 (方將)",
        "source": "임규직 11번 / 박문호 12번",
        "patterns": [r".*할새$", r".*ㄹ새$"]
    },
    "의문_설명": {
        "description": "설명의문 (何, 豈 등 아래)",
        "source": "이삼환 《구두지남》 / 박문호 31번",
        "patterns": [r"^리오$", r"^고$", r"^오$", r".*리오$", r".*잇고$"]
    },
    "의문_판정": {
        "description": "판정의문/반어",
        "source": "이삼환 《구두지남》 / 박문호 31번",
        "patterns": [r"^잇가$", r".*잇가$", r"^인저$", r".*ㄴ저$"]
    },
    "일의상승_하야": {
        "description": "하나의 뜻이 이어짐 / 인과관계",
        "source": "임규직 1번 / 박문호 1번",
        "patterns": [r"^하야$", r"^하여$", r"^하샤$", r"^하사$"]
    },
    "대우_하고하며": {
        "description": "짝을 이루는 말 / 나열",
        "source": "임규직 3번 / 박문호 2번",
        "patterns": [r"^하고$", r"^하며$", r".*하시고$", r".*하시며$", r"^코$"]
    },
    "대우_이오이며": {
        "description": "대우 (이오/이며 계열)",
        "source": "임규직 3번",
        "patterns": [r"^이오$", r"^요$", r"^이며$", r"^며$"]
    },
    "승상_하니": {
        "description": "위 구절을 잇는 말 (하니 계열)",
        "source": "임규직 6번",
        "patterns": [r"^하니$", r".*하시니$"]
    },
    "승상_이니": {
        "description": "위 구절을 잇는 말 (이니 계열)",
        "source": "임규직 6번",
        "patterns": [r"^이니$", r"^니$", r"^로니$"]
    },
    "상반_호되": {
        "description": "서로 반대되는 내용",
        "source": "임규직 9번 / 박문호 21번",
        "patterns": [r"^호되$", r".*로되$", r".*하되$"]
    },
    "양보_이나이라도": {
        "description": "양보/반어 (雖 아래)",
        "source": "이삼환 / 박문호 17번",
        "patterns": [r"^이나$", r"^나$", r".*라도$", r".*어니와$"]
    },
    "직하_조사": {
        "description": "곧장 내려오는 조사",
        "source": "임규직 5번",
        "patterns": [r"^이$", r"^은$", r"^는$", r"^의$", r"^를$", r"^을$"]
    },
    "처소_에": {
        "description": "처소/대상 (於, 于, 乎 아래)",
        "source": "이삼환 / 박문호 19번",
        "patterns": [r"^에$", r"^애$"]
    },
    "나열_와과": {
        "description": "낱낱이 셈 / 일일이 거론",
        "source": "임규직 12번 / 박문호 11번",
        "patterns": [r"^와$", r"^과$"]
    },
    "가정": {
        "description": "가정/조건 (若, 如 아래)",
        "source": "이삼환 / 박문호 14번",
        "patterns": [r"^면$", r".*이면$", r".*어든$", r".*하면$"]
    },
    "수단_으로": {
        "description": "수단/기점 (以, 使 아래)",
        "source": "이삼환 / 박문호 23번",
        "patterns": [r"^로$", r"^으로$"]
    },
    "존칭": {
        "description": "존칭/겸양 표현",
        "source": "임규직 전체 / 박문호 32번",
        "patterns": [r".*이다$", r".*니이다$", r".*소서$", r".*노이다$"]
    },
    "감탄": {
        "description": "감탄/호격 표현",
        "source": "구두 전통",
        "patterns": [r"^여$", r"^저$", r"^ㄴ저$", r"^야$"]
    },
}


def classify_genre(book_name: str) -> str:
    """장르 분류"""
    if pd.isna(book_name):
        return '기타'
    book_name = str(book_name)
    if '자치통감' in book_name or '춘추좌씨전' in book_name:
        return '역사서'
    elif '당송팔대가' in book_name:
        return '문집'
    elif '예기' in book_name:
        return '경전'
    elif '당시삼백수' in book_name:
        return '시'
    else:
        return '기타'


def classify_marker(marker):
    """마커 분류"""
    if pd.isna(marker) or marker == '':
        return ("미분류", [])
    marker = str(marker)
    for category, info in PREMODERN_TAXONOMY.items():
        for pattern in info["patterns"]:
            if re.match(pattern, marker):
                return (category, [])
    return ("미분류", [])


def extract_marker(text):
    """번역문에서 마커 추출: 마지막 어절"""
    if pd.isna(text):
        return ''
    text = str(text).strip()
    text = regex.sub(r'\([^)]*\)', '', text)
    text = text.replace('ㆍ', '')
    words = regex.split(r'[^\p{Hangul}]+', text)
    words = [w.strip() for w in words if w.strip()]
    if words:
        return normalize_hyeonto_marker(words[-1])
    return ''


# ============================================================================
# 1. '하다' 분석 (sentence_full.csv 기반)
# ============================================================================
def analyze_hada_sentence():
    """'하다' 종결어 장르별 분석 (문장 단위)"""
    print("=" * 70)
    print("📊 '하다' 종결어 분석 (sentence_full.csv)")
    print("=" * 70)
    
    df = pd.read_csv(DATASETS_DIR / 'sentence_full.csv')
    print(f"총 문장 데이터: {len(df):,}건")
    
    # 장르 분류
    df['genre'] = df['book'].apply(classify_genre)
    
    # '하다' 종결어 탐지 (번역문 끝에서 '~하다.' 패턴)
    hada_pattern = re.compile(r'하다[.。]?$')
    df['is_hada'] = df['번역문'].apply(
        lambda x: bool(hada_pattern.search(str(x))) if pd.notna(x) else False
    )
    
    total_hada = df['is_hada'].sum()
    print(f"\n'하다' 종결어 전체: {total_hada:,}건")
    
    # 장르별 집계
    print("\n" + "-" * 60)
    print("장르별 '하다' 분포")
    print("-" * 60)
    
    results = []
    for genre in ['역사서', '문집', '경전', '시', '기타']:
        genre_df = df[df['genre'] == genre]
        total = len(genre_df)
        hada_count = genre_df['is_hada'].sum()
        ratio = (hada_count / total * 100) if total > 0 else 0
        results.append({
            'genre': genre,
            'total': int(total),
            'hada_count': int(hada_count),
            'ratio': float(ratio)
        })
        print(f"{genre:8s}: {total:>8,}건 중 {hada_count:>6,}건 ({ratio:>5.2f}%)")
    
    # Chi-squared 검정 (역사서 vs 비역사서)
    history = df[df['genre'] == '역사서']
    non_history = df[df['genre'] != '역사서']
    
    table = np.array([
        [history['is_hada'].sum(), len(history) - history['is_hada'].sum()],
        [non_history['is_hada'].sum(), len(non_history) - non_history['is_hada'].sum()]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    print("\n" + "-" * 60)
    print("통계 검정: 역사서 vs 비역사서")
    print("-" * 60)
    print(f"χ² = {chi2:.2f}, p = {p_value:.2e}")
    print(f"결론: {'H₀ 기각 ✅ (역사서에서 유의미하게 높음)' if p_value < 0.05 else 'H₀ 기각 실패'}")
    
    # 결과 저장
    output = {
        'analysis': 'Level 3: 기사지단 (하다 장르별 분석) - sentence_full.csv 기반',
        'total_records': len(df),
        'total_hada': int(total_hada),
        'by_genre': results,
        'chi2_test': {
            'history_vs_non_history': {
                'chi2': float(chi2),
                'p_value': float(p_value),
                'reject_h0': bool(p_value < 0.05)
            }
        }
    }
    
    with open(RESULTS_DIR / 'hada_sentence_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과 저장: {RESULTS_DIR / 'hada_sentence_analysis.json'}")
    return output


# ============================================================================
# 2. Classified Markers 분석 (phrase_full.csv 기반)
# ============================================================================
def analyze_classified_markers():
    """전근대 원전 기준 현토 분류 (구절 단위)"""
    print("\n" + "=" * 70)
    print("📊 Classified Markers 분석 (phrase_full.csv)")
    print("=" * 70)
    
    df = pd.read_csv(DATASETS_DIR / 'phrase_full.csv')
    print(f"총 구절 데이터: {len(df):,}건")
    
    # 번역문에서 현토 마커 추출
    print("마커 추출 및 정규화 중...")
    df['marker_final'] = df['번역문'].apply(extract_marker)
    
    # 마커별 빈도
    marker_counts = df['marker_final'].value_counts().to_dict()
    print(f"고유 마커: {len(marker_counts):,}개")
    
    # 분류 실행
    classified = defaultdict(list)
    
    for marker, count in marker_counts.items():
        if marker == '':
            continue
        category, base_cats = classify_marker(marker)
        classified[category].append((marker, count))
    
    # 정렬 (빈도순)
    for cat in classified:
        classified[cat].sort(key=lambda x: -x[1])
    
    # 결과 출력
    print("\n" + "-" * 60)
    print("분류별 요약")
    print("-" * 60)
    
    summary = []
    for cat in sorted(classified.keys()):
        markers = classified[cat]
        total = sum(c for _, c in markers)
        summary.append((cat, len(markers), total))
        print(f"{cat:20s}: {len(markers):>5}개, {total:>8,}건")
    
    # 결과 저장 (JSON)
    json_data = {}
    for cat, markers in classified.items():
        marker_list = [{"marker": m, "count": c} for m, c in markers]
        json_data[cat] = {
            "description": PREMODERN_TAXONOMY.get(cat, {}).get("description", "미분류"),
            "source": PREMODERN_TAXONOMY.get(cat, {}).get("source", ""),
            "markers": marker_list,
            "total_count": sum(c for _, c in markers)
        }
    
    with open(RESULTS_DIR / 'classified_markers_phrase.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # 결과 저장 (Markdown)
    lines = ["# 전근대 원전 기준 현토 분류 결과 (phrase_full.csv 기반)\n\n"]
    lines.append("## 분류별 요약\n\n")
    lines.append("| 분류 | 고유 마커 | 총 빈도 |\n")
    lines.append("|------|----------|--------|\n")
    for cat, unique, total in summary:
        lines.append(f"| {cat} | {unique} | {total:,} |\n")
    
    with open(RESULTS_DIR / 'CLASSIFIED_MARKERS_phrase.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"\n✅ 결과 저장: {RESULTS_DIR / 'classified_markers_phrase.json'}")
    print(f"✅ 결과 저장: {RESULTS_DIR / 'CLASSIFIED_MARKERS_phrase.md'}")
    
    return classified, summary


def main():
    print("🔬 현토 리서치 시작\n")
    
    # 1. '하다' 분석 (sentence_full.csv)
    analyze_hada_sentence()
    
    # 2. Classified Markers 분석 (phrase_full.csv)
    analyze_classified_markers()
    
    print("\n" + "=" * 70)
    print("✅ 모든 분석 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
