#!/usr/bin/env python3
"""
classified_markers.json과 compound_tags.json에서 마커 분류 매핑을 생성
근거: 임규직, 박문호, 이삼환 등 한문 구두학 전문가 분류
"""

import json
from pathlib import Path
from collections import defaultdict

def build_marker_mapping():
    """classified_markers.json과 compound_tags.json에서 마커 매핑 생성"""
    
    # 파일 읽기
    classified_path = Path("hyeonto/results/classified_markers.json")
    compound_path = Path("hyeonto/results/compound_tags.json")
    
    if not classified_path.exists():
        print(f"❌ {classified_path} 파일 없음")
        return None
    
    with open(classified_path, "r", encoding="utf-8") as f:
        classified = json.load(f)
    
    with open(compound_path, "r", encoding="utf-8") as f:
        compound = json.load(f)
    
    # 마커 → 분류 매핑 생성
    marker_to_category = {}
    marker_to_source = {}
    
    # 1. classified_markers.json에서 단일 마커 추출
    for category, data in classified.items():
        if not isinstance(data, dict) or "markers" not in data:
            continue
        
        source = data.get("source", "")
        markers = data.get("markers", [])
        
        for marker_obj in markers:
            if isinstance(marker_obj, dict) and "marker" in marker_obj:
                marker = marker_obj["marker"]
                if marker and marker.strip():
                    marker_to_category[marker] = category
                    marker_to_source[marker] = source
    
    # 2. compound_tags.json에서 복합 마커 추출
    for compound_marker, categories in compound.items():
        if isinstance(categories, list) and len(categories) > 0:
            category = categories[0]  # 첫 번째 카테고리 사용
            marker_to_category[compound_marker] = category
            marker_to_source[compound_marker] = "임규직/박문호 복합 마커"
    
    print(f"✅ 총 마커 수: {len(marker_to_category):,}개")
    print(f"   - 단일 마커: {sum(1 for m in marker_to_category if len(m) <= 5):,}개")
    print(f"   - 복합 마커: {sum(1 for m in marker_to_category if len(m) > 5):,}개")
    
    # 카테고리 분포
    category_counts = defaultdict(int)
    for cat in marker_to_category.values():
        category_counts[cat] += 1
    
    print("\n📊 카테고리 분포 (top 15):")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {cat}: {count:,}개")
    
    # 최종 매핑 저장
    output = {
        "metadata": {
            "source": "classified_markers.json + compound_tags.json",
            "reference": "임규직, 박문호, 이삼환 등 한문 구두학 저작",
            "total_markers": len(marker_to_category)
        },
        "mappings": marker_to_category,
        "sources": marker_to_source
    }
    
    output_path = Path("configs/marker_classification_dansa.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 마커 분류 매핑 저장: {output_path}")
    
    return output

if __name__ == "__main__":
    mapping = build_marker_mapping()
