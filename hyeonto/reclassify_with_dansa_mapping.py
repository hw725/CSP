#!/usr/bin/env python3
"""
정확한 한문 구두학 기반 분류로 CSV 재분류 및 마크다운 보고서 생성
근거: classified_markers.json (임규직, 박문호, 이삼환)
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

def load_mapping():
    """분류 매핑 로드"""
    with open("configs/marker_classification_dansa.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["mappings"], data["sources"]

def classify_marker(marker, mapping):
    """마커를 분류로 매핑"""
    if pd.isna(marker) or marker == "":
        return "미분류"
    
    marker_str = str(marker).strip()
    
    # 정확한 매칭
    if marker_str in mapping:
        return mapping[marker_str]
    
    # 복합 마커에서 마지막 부분으로 재시도
    if "," in marker_str:
        parts = marker_str.split(",")
        for part in reversed(parts):
            if part.strip() in mapping:
                return mapping[part.strip()]
    
    return "미분류"

def process_csv(csv_path, mapping, tag):
    """CSV 파일 재분류"""
    print(f"\n{'='*60}")
    print(f"📋 {tag} CSV 처리: {csv_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    print(f"총 행 수: {len(df):,}")
    
    # 마커 열 찾기
    marker_col = None
    for col in ["marker", "marker_normalized", "current_marker"]:
        if col in df.columns:
            marker_col = col
            break
    
    if not marker_col:
        print(f"❌ 마커 열을 찾을 수 없음")
        return None
    
    # 분류 적용
    df["분류"] = df[marker_col].apply(lambda x: classify_marker(x, mapping))
    
    # 출력 경로 설정
    output_csv = csv_path.replace(".csv", "_재분류.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 재분류 CSV 저장: {output_csv}")
    
    # 분류 분포
    dist = df["분류"].value_counts()
    print(f"\n분류 분포:")
    for cat, count in dist.head(20).items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count:,}개 ({pct:.1f}%)")
    
    return df, output_csv

def generate_markdown_report(df, output_csv, tag):
    """마크다운 보고서 생성"""
    report_path = output_csv.replace(".csv", "_보고서.md")
    
    # 분류별 마커 분석
    classification_data = defaultdict(lambda: {"count": 0, "markers": Counter()})
    
    # 마커 열 찾기
    marker_col = None
    for col in ["marker_normalized", "marker", "current_marker"]:
        if col in df.columns:
            marker_col = col
            break
    
    for _, row in df.iterrows():
        classification = row["분류"]
        count = 1
        marker = str(row.get(marker_col, "") if marker_col else "").strip()
        
        classification_data[classification]["count"] += count
        if marker:
            classification_data[classification]["markers"][marker] += count
    
    # 마크다운 생성
    total_count = sum(d["count"] for d in classification_data.values())
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {tag} 마커 분류 분석 보고서\n\n")
        f.write(f"**근거**: 임규직, 박문호, 이삼환 등 한문 구두학 전문가 분류\n")
        f.write(f"**데이터**: {output_csv.replace('_재분류.csv', '.csv')}\n")
        f.write(f"**총 마커 수**: {total_count:,}개\n\n")
        
        f.write("## 분류별 통계\n\n")
        f.write("| 분류 | 개수 | 비율 |\n")
        f.write("|------|------|------|\n")
        
        sorted_data = sorted(classification_data.items(), 
                            key=lambda x: x[1]["count"], 
                            reverse=True)
        
        for category, data in sorted_data:
            count = data["count"]
            pct = count / total_count * 100
            f.write(f"| {category} | {count:,} | {pct:.1f}% |\n")
        
        f.write("\n## 상위 분류별 주요 마커\n\n")
        
        for category, data in sorted_data[:10]:
            f.write(f"### {category}\n\n")
            f.write(f"총 {data['count']:,}개 마커\n\n")
            
            top_markers = data["markers"].most_common(10)
            for marker, count in top_markers:
                pct = count / data["count"] * 100
                f.write(f"- **{marker}**: {count:,}개 ({pct:.1f}%)\n")
            f.write("\n")
    
    print(f"✅ 마크다운 보고서 저장: {report_path}")
    return report_path

def main():
    mapping, sources = load_mapping()
    
    # 처리할 CSV 파일들
    csv_files = [
        ("sentence", "hyeonto/report_1-1/sentence_k3_normalized/sentence_clusters.csv"),
        ("phrase", "hyeonto/report_1-1/phrase_k3_normalized/phrase_clusters.csv"),
    ]
    
    for tag, csv_path in csv_files:
        if not Path(csv_path).exists():
            print(f"⚠️ {csv_path} 파일 없음")
            continue
        
        result = process_csv(csv_path, mapping, tag)
        if result:
            df, output_csv = result
            generate_markdown_report(df, output_csv, tag)
    
    print(f"\n{'='*60}")
    print("✅ 모든 재분류 및 보고서 생성 완료!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
