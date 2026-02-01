"""
한자-현토 공기어 분석 (현토 정규화 반영)

기존 공기어 분석을 현토 마커 정규화를 반영하여 재수행합니다.
을/를, 은/는, ㄴ댄/인댄 등 변이형을 대표형으로 통일합니다.

출력:
- cooccurrence_normalized.csv: 정규화된 공기 매트릭스
- associations_normalized.csv: 정규화된 PMI 연관 분석
- cooccurrence_network_normalized.html: 인터랙티브 네트워크 그래프
- cooccurrence_analysis_normalized.md: 분석 리포트
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import json
import regex
from datetime import datetime

# 현토 정규화 모듈 임포트
from hyeonto_normalizer import (
    normalize_hyeonto_marker, 
    get_normalization_stats,
    NORMALIZATION_TABLE
)


def extract_hanja(text: str) -> list:
    """텍스트에서 한자 추출"""
    if pd.isna(text):
        return []
    return regex.findall(r'\p{Han}', str(text))


def extract_hyeonto(text: str) -> list:
    """텍스트에서 한글(현토) 추출"""
    if pd.isna(text):
        return []
    return regex.findall(r'\p{Hangul}+', str(text))


def calculate_pmi(cooccur_count: int, hanja_count: int, hyeonto_count: int, 
                  total: int, epsilon: float = 1e-10) -> float:
    """PMI(Pointwise Mutual Information) 계산"""
    p_xy = cooccur_count / total
    p_x = hanja_count / total
    p_y = hyeonto_count / total
    
    if p_x == 0 or p_y == 0:
        return 0
    
    return np.log2((p_xy + epsilon) / (p_x * p_y + epsilon))


def analyze_cooccurrence(df: pd.DataFrame, 
                         src_col: str = 'left_sentence',
                         normalize_markers: bool = True,
                         top_hanja: int = 100,
                         top_hyeonto: int = 50) -> dict:
    """
    한자-현토 공기 분석 (정규화 옵션 포함)
    
    Args:
        df: 입력 데이터프레임
        src_col: 원문 컬럼명
        normalize_markers: 현토 마커 정규화 적용 여부
        top_hanja: 분석할 상위 한자 수
        top_hyeonto: 분석할 상위 현토 수
    
    Returns:
        분석 결과 딕셔너리
    """
    
    print(f"\n{'='*60}")
    print(f"한자-현토 공기 분석 (정규화: {'ON' if normalize_markers else 'OFF'})")
    print(f"{'='*60}")
    
    # 1. 한자/현토 추출
    print("\n? Step 1: 한자/현토 추출...")
    hanja_counter = Counter()
    hyeonto_counter = Counter()
    cooccur_counter = Counter()
    
    all_hyeonto_raw = []  # 정규화 전 현토 목록
    all_hyeonto_normalized = []  # 정규화 후 현토 목록
    
    for idx, row in df.iterrows():
        text = row.get(src_col, '')
        if pd.isna(text):
            continue
        text = str(text)
        
        # 한자 추출
        hanjas = extract_hanja(text)
        hanja_counter.update(hanjas)
        
        # 현토 추출 및 정규화
        hyeontos_raw = extract_hyeonto(text)
        all_hyeonto_raw.extend(hyeontos_raw)
        
        if normalize_markers:
            hyeontos = [normalize_hyeonto_marker(h) for h in hyeontos_raw]
        else:
            hyeontos = hyeontos_raw
        
        all_hyeonto_normalized.extend(hyeontos)
        hyeonto_counter.update(hyeontos)
        
        # 공기 관계 추출 (한자-현토 쌍)
        for hanja in set(hanjas):
            for hyeonto in set(hyeontos):
                cooccur_counter[(hanja, hyeonto)] += 1
    
    # 정규화 통계
    if normalize_markers:
        norm_stats = get_normalization_stats(all_hyeonto_raw)
        print(f"\n? 정규화 통계:")
        print(f"   ? 정규화 전 고유 마커: {norm_stats['before_unique']:,}개")
        print(f"   ? 정규화 후 고유 마커: {norm_stats['after_unique']:,}개")
        print(f"   ? 감소율: {norm_stats['reduction_rate']:.1f}%")
        print(f"\n   상위 정규화 쌍 (원본 → 대표형):")
        for (orig, norm), count in norm_stats['normalization_pairs'][:10]:
            print(f"      '{orig}' → '{norm}': {count:,}회")
    
    print(f"\n? 고유 한자: {len(hanja_counter):,}개")
    print(f"? 고유 현토: {len(hyeonto_counter):,}개")
    print(f"? 공기 쌍: {len(cooccur_counter):,}개")
    
    # 2. 상위 한자/현토 선택
    top_hanja_list = [h for h, _ in hanja_counter.most_common(top_hanja)]
    top_hyeonto_list = [h for h, _ in hyeonto_counter.most_common(top_hyeonto)]
    
    print(f"\n? 분석 대상: 상위 {len(top_hanja_list)}개 한자 × 상위 {len(top_hyeonto_list)}개 현토")
    
    # 3. 공기 매트릭스 생성
    print("\n? Step 2: 공기 매트릭스 생성...")
    total = len(df)
    
    cooccur_matrix = pd.DataFrame(
        0, index=top_hanja_list, columns=top_hyeonto_list
    )
    
    for (hanja, hyeonto), count in cooccur_counter.items():
        if hanja in cooccur_matrix.index and hyeonto in cooccur_matrix.columns:
            cooccur_matrix.loc[hanja, hyeonto] = count
    
    # 4. PMI 계산
    print("\n? Step 3: PMI 연관 분석...")
    associations = []
    
    for hanja in top_hanja_list:
        hanja_count = hanja_counter[hanja]
        for hyeonto in top_hyeonto_list:
            hyeonto_count = hyeonto_counter[hyeonto]
            cooccur_count = cooccur_matrix.loc[hanja, hyeonto]
            
            if cooccur_count > 0:
                pmi = calculate_pmi(cooccur_count, hanja_count, hyeonto_count, total)
                associations.append({
                    'hanja': hanja,
                    'hyeonto': hyeonto,
                    'cooccur_count': cooccur_count,
                    'hanja_count': hanja_count,
                    'hyeonto_count': hyeonto_count,
                    'pmi': round(pmi, 3)
                })
    
    associations_df = pd.DataFrame(associations)
    associations_df = associations_df.sort_values('pmi', ascending=False)
    
    # 5. 핵심 한자별 현토 선호 분석
    print("\n? Step 4: 핵심 한자별 현토 선호 분석...")
    core_hanja = ['之', '而', '不', '以', '也', '其', '者', '人', '子', '曰', '道', '德', '禮']
    hanja_preferences = {}
    
    for hanja in core_hanja:
        if hanja in cooccur_matrix.index:
            row = cooccur_matrix.loc[hanja].sort_values(ascending=False)
            top5 = [(h, int(c)) for h, c in row.head(5).items() if c > 0]
            hanja_preferences[hanja] = top5
    
    print(f"? {len(hanja_preferences)}개 핵심 한자 분석 완료")
    
    return {
        'cooccur_matrix': cooccur_matrix,
        'associations': associations_df,
        'hanja_counter': hanja_counter,
        'hyeonto_counter': hyeonto_counter,
        'hanja_preferences': hanja_preferences,
        'normalization_stats': norm_stats if normalize_markers else None,
        'top_hanja': top_hanja_list,
        'top_hyeonto': top_hyeonto_list
    }


def generate_network_html(associations_df: pd.DataFrame, 
                          output_sentenceth: Path,
                          title: str = "한자-현토 공기 네트워크",
                          pmi_threshold: float = 1.5,
                          max_edges: int = 200):
    """인터랙티브 네트워크 그래프 생성"""
    
    # 상위 연관만 선택
    filtered = associations_df[associations_df['pmi'] >= pmi_threshold].head(max_edges)
    
    # 노드와 엣지 준비
    nodes = set()
    edges = []
    
    for _, row in filtered.iterrows():
        hanja = row['hanja']
        hyeonto = row['hyeonto']
        nodes.add(('hanja', hanja))
        nodes.add(('hyeonto', hyeonto))
        edges.append({
            'source': hanja,
            'target': hyeonto,
            'value': row['cooccur_count'],
            'pmi': row['pmi']
        })
    
    nodes_json = json.dumps([
        {'id': n, 'group': 1 if t == 'hanja' else 2, 'type': t}
        for t, n in nodes
    ], ensure_ascii=False)
    
    edges_json = json.dumps(edges, ensure_ascii=False)
    
    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Pretendard', sans-serif;
            margin: 0;
            background: #1a1a2e;
            color: #fff;
        }}
        #network {{
            width: 100%;
            height: 100vh;
        }}
        .node-hanja {{ fill: #e94560; }}
        .node-hyeonto {{ fill: #4ecca3; }}
        .link {{ stroke: #555; stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
            pointer-events: none;
        }}
        h1 {{
            position: absolute;
            top: 10px;
            left: 20px;
            margin: 0;
            font-size: 1.5rem;
        }}
        .legend {{
            position: absolute;
            top: 50px;
            left: 20px;
            background: rgba(0,0,0,0.5);
            padding: 10px 15px;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .info {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.5);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <h1>? {title} (정규화 반영)</h1>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: #e94560;"></div>
            <span>한자</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #4ecca3;"></div>
            <span>현토 (정규화)</span>
        </div>
    </div>
    <div class="info">
        PMI ≥ {pmi_threshold} | 상위 {len(edges)}개 연관
    </div>
    <div id="network"></div>
    <div class="tooltip" style="display: none;"></div>
    
    <script>
        const nodes = {nodes_json};
        const links = {edges_json};
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#network")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.value / 100));
        
        const node = svg.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", d => d.type === 'hanja' ? 12 : 8)
            .attr("class", d => d.type === 'hanja' ? 'node-hanja' : 'node-hyeonto')
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        const label = svg.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .text(d => d.id)
            .attr("font-size", d => d.type === 'hanja' ? 14 : 11)
            .attr("fill", "#fff")
            .attr("text-anchor", "middle")
            .attr("dy", d => d.type === 'hanja' ? -18 : -12);
        
        const tooltip = d3.select(".tooltip");
        
        node.on("mouseover", function(event, d) {{
            const connections = links.filter(l => l.source.id === d.id || l.target.id === d.id);
            tooltip
                .style("display", "block")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px")
                .html(`<strong>${{d.id}}</strong> (${{d.type}})<br>연결: ${{connections.length}}개`);
        }})
        .on("mouseout", function() {{
            tooltip.style("display", "none");
        }});
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''
    
    output_sentenceth.write_text(html_content, encoding='utf-8')
    print(f"? 네트워크 그래프 저장: {output_sentenceth}")


def generate_report(results: dict, output_sentenceth: Path, normalize: bool = True):
    """마크다운 리포트 생성"""
    
    associations = results['associations']
    hanja_prefs = results['hanja_preferences']
    norm_stats = results.get('normalization_stats')
    
    lines = [
        f"# 한자-현토 공기 네트워크 분석 리포트 {'(정규화 반영)' if normalize else ''}",
        "",
        f"**분석 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 개요",
        "",
        f"- 분석 한자 수: {len(results['top_hanja'])}개",
        f"- 분석 현토 수: {len(results['top_hyeonto'])}개",
        f"- 유의미한 연관 쌍: {len(associations)}개",
    ]
    
    # 정규화 정보 추가
    if norm_stats:
        lines.extend([
            "",
            "### 현토 마커 정규화 적용",
            "",
            f"- 정규화 전 고유 마커: {norm_stats['before_unique']:,}개",
            f"- 정규화 후 고유 마커: {norm_stats['after_unique']:,}개",
            f"- **감소율**: {norm_stats['reduction_rate']:.1f}%",
            "",
            "주요 정규화 규칙:",
            "- `을` → `를` (목적격 조사 통일)",
            "- `은` → `는` (주제 조사 통일)",
            "- `인댄`/`은댄` → `ㄴ댄` (인용 어미 통일)",
            "- `으니` → `니`, `으면` → `면` 등",
        ])
    
    lines.extend([
        "",
        "---",
        "",
        "## 2. 상위 PMI 연관 쌍 (한자 → 현토)",
        "",
        "| 순위 | 한자 | 현토 | 공기 횟수 | PMI |",
        "|:---:|:---:|:---:|---:|---:|",
    ])
    
    for i, row in associations.head(50).iterrows():
        rank = list(associations.index).index(i) + 1
        lines.append(
            f"| {rank} | {row['hanja']} | {row['hyeonto']} | "
            f"{row['cooccur_count']:,} | {row['pmi']:.3f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 3. 핵심 한자별 현토 선호",
        "",
    ])
    
    for hanja, prefs in hanja_prefs.items():
        pref_str = ", ".join([f"{h}({c:,})" for h, c in prefs])
        lines.append(f"- **{hanja}**: {pref_str}")
    
    lines.extend([
        "",
        "---",
        "",
        "**분석 완료**"
    ])
    
    output_sentenceth.write_text("\n".join(lines), encoding='utf-8')
    print(f"? 리포트 저장: {output_sentenceth}")


def main():
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports" / "exploratory" / "cooccurrence_normalized"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드 (Phrase 데이터 사용)
    data_path = base_dir / "reports" / "phrase_k4_normalized" / "phrase_clusters.csv"
    
    if not data_path.exists():
        # 대안 경로
        data_path = base_dir / "datasets" / "s2p_merged_v2.csv"
    
    print(f"? 데이터 로드: {data_path}")
    df = pd.read_csv(data_path)
    print(f"? {len(df):,}행 로드")
    
    # 컬럼 확인
    print(f"컬럼: {df.columns.tolist()[:10]}...")
    
    # 원문, left_sentence 또는 src_left 컬럼 찾기
    if '원문' in df.columns:
        src_col = '원문'
    elif 'left_sentence' in df.columns:
        src_col = 'left_sentence'
    elif 'src_left' in df.columns:
        src_col = 'src_left'
    elif 'src_L' in df.columns:
        src_col = 'src_L'
    else:
        # 첫 번째 문자열 컬럼 사용
        src_col = df.select_dtypes(include=['object']).columns[0]
    
    print(f"분석 컬럼: {src_col}")
    
    # 정규화 적용 분석
    results = analyze_cooccurrence(
        df, 
        src_col=src_col,
        normalize_markers=True,
        top_hanja=100,
        top_hyeonto=50
    )
    
    # 결과 저장
    print("\n? 결과 저장 중...")
    
    # 1. 공기 매트릭스
    matrix_path = reports_dir / "cooccurrence_matrix_normalized.csv"
    results['cooccur_matrix'].to_csv(matrix_path, encoding='utf-8-sig')
    print(f"? 매트릭스 저장: {matrix_path}")
    
    # 2. PMI 연관 분석
    assoc_path = reports_dir / "associations_normalized.csv"
    results['associations'].to_csv(assoc_path, index=False, encoding='utf-8-sig')
    print(f"? 연관 분석 저장: {assoc_path}")
    
    # 3. 네트워크 그래프
    network_path = reports_dir / "cooccurrence_network_normalized.html"
    generate_network_html(results['associations'], network_path)
    
    # 4. 리포트
    report_path = reports_dir / "cooccurrence_analysis_normalized.md"
    generate_report(results, report_path, normalize=True)
    
    print("\n" + "="*60)
    print("? 공기어 분석 완료 (현토 정규화 반영)")
    print("="*60)


if __name__ == "__main__":
    main()
