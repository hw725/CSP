"""
기존 CSV 데이터를 사용해 흑백 인쇄용 2D 오버레이 HTML 재생성

기존에 생성된 k4_embedding_overlay_2d.csv를 사용하여
마커 심볼이 포함된 흑백 버전 HTML을 생성합니다.
"""
import pandas as pd
import json
from pathlib import Path

# 흑백 인쇄용 색상 및 심볼
GRAYSCALE_COLORS = {
    'sentence': ['#000000', '#666666', '#BBBBBB', '#EEEEEE'],  # 검정계열 4단계
    'phrase': ['#000000', '#666666', '#BBBBBB', '#EEEEEE']  # 동일한 그레이스케일
}

GRAYSCALE_SYMBOLS = {
    'sentence': ['circle', 'square', 'diamond', 'cross'],
    'phrase': ['triangle-up', 'triangle-down', 'hexagon', 'star']
}

def generate_bw_2d_viz(result_df: pd.DataFrame, output_path: Path):
    """흑백 인쇄용 2D 시각화 HTML 생성 (마커 심볼 포함)"""
    
    sentence_colors = GRAYSCALE_COLORS['sentence']
    phrase_colors = GRAYSCALE_COLORS['phrase']
    
    traces_data = []
    for boundary_type in ['Sentence', 'Phrase']:
        colors = sentence_colors if boundary_type == 'Sentence' else phrase_colors
        symbols = GRAYSCALE_SYMBOLS['sentence' if boundary_type == 'Sentence' else 'phrase']
        type_df = result_df[result_df['boundary_type'] == boundary_type]
        
        for i, cluster_id in enumerate(sorted(type_df['cluster_id'].unique())):
            cluster_df = type_df[type_df['cluster_id'] == cluster_id]
            trace = {
                'x': cluster_df['x'].tolist(),
                'y': cluster_df['y'].tolist(),
                'name': f'{boundary_type}-p{int(cluster_id)}',
                'color': colors[int(cluster_id) % len(colors)],
                'type': boundary_type,
                'symbol': symbols[i % len(symbols)]  # 항상 심볼 포함
            }
            traces_data.append(trace)

    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Sentence/Phrase K=4 임베딩 오버레이 2D (흑백)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
            background: #ffffff;
            min-height: 100vh;
            color: #333;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 10px;
            color: #333;
        }}
        .subtitle {{ text-align: center; color: #666; margin-bottom: 20px; }}
        #scatter-plot {{ width: 100%; height: 700px; }}
        .controls {{
            display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap;
        }}
        .control-group {{
            background: #f0f0f0; padding: 12px 20px; border-radius: 10px;
        }}
        .control-group label {{ color: #333; margin-right: 10px; }}
        .control-group select, .control-group input {{
            background: #fff; border: 1px solid #ccc;
            color: #333; padding: 6px 12px; border-radius: 6px;
        }}
        .legend-guide {{
            text-align: center; margin: 10px 0; padding: 10px;
            background: #f9f9f9; border-radius: 8px;
        }}
        .legend-guide span {{ margin: 0 15px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentence/Phrase K=4 임베딩 오버레이 (2D UMAP) - 흑백 인쇄용</h1>
        <p class="subtitle">정규화 반영 | 심볼로 구분: Sentence(○□◇+), Phrase(△▽?★)</p>
        <div class="legend-guide">
            <span><strong>Sentence:</strong> ○circle, □square, ◇diamond, +cross</span>
            <span><strong>Phrase:</strong> △triangle-up, ▽triangle-down, ?hexagon, ★star</span>
        </div>
        <div class="controls">
            <div class="control-group">
                <label>포인트 크기:</label>
                <input type="range" id="point-size" min="2" max="10" value="6" onchange="updatePlot()">
            </div>
            <div class="control-group">
                <label>투명도:</label>
                <input type="range" id="opacity" min="10" max="100" value="70" onchange="updatePlot()">
            </div>
            <div class="control-group">
                <label>표시:</label>
                <select id="show-mode" onchange="updatePlot()">
                    <option value="both">Sentence + Phrase</option>
                    <option value="sentence">Sentence만</option>
                    <option value="phrase">Phrase만</option>
                </select>
            </div>
        </div>
        <div id="scatter-plot"></div>
    </div>
    <script>
        const tracesData = {json.dumps(traces_data)};
        function updatePlot() {{
            const pointSize = parseInt(document.getElementById('point-size').value);
            const opacity = parseInt(document.getElementById('opacity').value) / 100;
            const showMode = document.getElementById('show-mode').value;
            const traces = [];
            for (const t of tracesData) {{
                if (showMode === 'both' || (showMode === 'sentence' && t.type === 'Sentence') || (showMode === 'phrase' && t.type === 'Phrase')) {{
                    const markerConfig = {{ size: pointSize, color: t.color, opacity: opacity }};
                    if (t.symbol) {{ markerConfig.symbol = t.symbol; }}
                    traces.push({{
                        x: t.x, y: t.y, mode: 'markers', type: 'scatter', name: t.name,
                        marker: markerConfig,
                        hovertemplate: '<b>' + t.name + '</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'
                    }});
                }}
            }}
            Plotly.react('scatter-plot', traces, {{
                paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                xaxis: {{ title: 'UMAP-1', color: '#333', gridcolor: '#ddd' }},
                yaxis: {{ title: 'UMAP-2', color: '#333', gridcolor: '#ddd' }},
                legend: {{ font: {{ color: '#333' }}, bgcolor: '#f9f9f9' }},
                margin: {{ t: 30, b: 60, l: 60, r: 30 }}
            }}, {{ responsive: true }});
        }}
        updatePlot();
    </script>
</body>
</html>'''

    output_path.write_text(html_content, encoding='utf-8')
    print(f"? 흑백 2D 시각화 저장: {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    
    # 기존 CSV 로드
    csv_path = reports_dir / "k4_embedding_overlay_2d.csv"
    if not csv_path.exists():
        print(f"? CSV 파일이 없습니다: {csv_path}")
        exit(1)
    
    result_df = pd.read_csv(csv_path)
    print(f"? 로드: {len(result_df)} rows from {csv_path}")
    
    # 흑백 버전 HTML 생성
    output_path = reports_dir / "k4_embedding_overlay_2d_bw.html"
    generate_bw_2d_viz(result_df, output_path)
    
    print("\n? 흑백 인쇄용 시각화 생성 완료!")
