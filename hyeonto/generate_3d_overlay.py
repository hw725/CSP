"""
Sentence/Phrase K=4 임베딩 3D 오버레이 시각화 생성

기존 2D 좌표 CSV를 기반으로 3D UMAP을 실행하거나,
2D 좌표에 z축을 추가하여 3D 시각화를 생성합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def generate_3d_overlay_html(csv_path: Path, output_path: Path):
    """기존 2D 좌표에서 3D 시각화 생성"""
    
    print(f"? 좌표 데이터 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # z 좌표가 없으면 클러스터 ID + 노이즈로 z 생성
    if 'z' not in df.columns:
        # 클러스터별로 z값 할당 + 노이즈 추가
        np.random.seed(42)
        cluster_z_base = {0: 0, 1: 3, 2: 6, 3: 9}
        df['z'] = df['cluster_id'].map(cluster_z_base) + np.random.normal(0, 0.5, len(df))
    
    # 클러스터별 색상
    sentence_colors = ['#e94560', '#ff6b6b', '#ffc93c', '#ff8c42']
    phrase_colors = ['#4ecca3', '#45b7aa', '#38a3a5', '#22577a']
    
    # JSON 데이터 준비
    traces_data = []
    
    for boundary_type in ['Sentence', 'Phrase']:
        colors = sentence_colors if boundary_type == 'Sentence' else phrase_colors
        type_df = df[df['boundary_type'] == boundary_type]
        
        for cluster_id in sorted(type_df['cluster_id'].unique()):
            cluster_df = type_df[type_df['cluster_id'] == cluster_id]
            traces_data.append({
                'x': cluster_df['x'].tolist(),
                'y': cluster_df['y'].tolist(),
                'z': cluster_df['z'].tolist(),
                'name': f'{boundary_type}-p{cluster_id}',
                'color': colors[int(cluster_id) % len(colors)],
                'type': boundary_type
            })
    
    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentence/Phrase K=4 임베딩 3D 오버레이</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #e94560, #4ecca3);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        #scatter-3d {{ width: 100%; height: 800px; }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .control-group {{
            background: rgba(255,255,255,0.05);
            padding: 12px 20px;
            border-radius: 10px;
        }}
        .control-group label {{
            color: #4ecca3;
            margin-right: 10px;
        }}
        .control-group select, .control-group input {{
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 6px 12px;
            border-radius: 6px;
        }}
        .legend-box {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            font-size: 0.9rem;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .insight-box {{
            background: rgba(78, 204, 163, 0.1);
            border: 1px solid rgba(78, 204, 163, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
        }}
        .insight-box h3 {{ color: #4ecca3; margin-bottom: 15px; }}
        .insight-item {{
            padding: 10px 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin: 10px 0;
            border-left: 3px solid #e94560;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>? Sentence/Phrase K=4 임베딩 3D 오버레이</h1>
        <p class="subtitle">UMAP 차원 축소 + 클러스터 분리 축 (정규화 반영) | 마우스 드래그로 회전</p>

        <div class="controls">
            <div class="control-group">
                <label>포인트 크기:</label>
                <input type="range" id="point-size" min="1" max="8" value="3" onchange="updatePlot()">
            </div>
            <div class="control-group">
                <label>투명도:</label>
                <input type="range" id="opacity" min="20" max="100" value="60" onchange="updatePlot()">
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

        <div class="legend-box">
            <div class="legend-item"><div class="legend-dot" style="background:#e94560"></div>Sentence-p0</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div>Sentence-p1</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ffc93c"></div>Sentence-p2</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ff8c42"></div>Sentence-p3</div>
            <div class="legend-item"><div class="legend-dot" style="background:#4ecca3"></div>Phrase-p0</div>
            <div class="legend-item"><div class="legend-dot" style="background:#45b7aa"></div>Phrase-p1</div>
            <div class="legend-item"><div class="legend-dot" style="background:#38a3a5"></div>Phrase-p2</div>
            <div class="legend-item"><div class="legend-dot" style="background:#22577a"></div>Phrase-p3</div>
        </div>

        <div id="scatter-3d"></div>

        <div class="insight-box">
            <h3>? 3D 시각화 해석</h3>
            <div class="insight-item">
                <strong>클러스터 분리</strong>: Z축은 클러스터 ID를 기반으로 분리하여 각 클러스터의 공간적 분포를 명확히 표시
            </div>
            <div class="insight-item">
                <strong>Sentence-Phrase 대응</strong>: 유사한 색상 계열(빨강=Sentence, 초록=Phrase)이 비슷한 X-Y 위치에 분포하면 동일 의미역 공유
            </div>
            <div class="insight-item">
                <strong>인터랙션</strong>: 마우스로 드래그하여 회전, 스크롤로 확대/축소
            </div>
        </div>
    </div>

    <script>
        const tracesData = {json.dumps(traces_data)};

        function updatePlot() {{
            const pointSize = parseInt(document.getElementById('point-size').value);
            const opacity = parseInt(document.getElementById('opacity').value) / 100;
            const showMode = document.getElementById('show-mode').value;

            const traces = [];

            for (const t of tracesData) {{
                if (showMode === 'both' ||
                    (showMode === 'pa' && t.type === 'Sentence') ||
                    (showMode === 'sa' && t.type === 'Phrase')) {{

                    traces.push({{
                        x: t.x,
                        y: t.y,
                        z: t.z,
                        mode: 'markers',
                        type: 'scatter3d',
                        name: t.name,
                        marker: {{
                            size: pointSize,
                            color: t.color,
                            opacity: opacity
                        }},
                        hovertemplate: '<b>' + t.name + '</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>'
                    }});
                }}
            }}

            Plotly.react('scatter-3d', traces, {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                scene: {{
                    xaxis: {{
                        title: 'UMAP-1',
                        color: '#fff',
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.2)'
                    }},
                    yaxis: {{
                        title: 'UMAP-2',
                        color: '#fff',
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.2)'
                    }},
                    zaxis: {{
                        title: 'Cluster Layer',
                        color: '#fff',
                        gridcolor: 'rgba(255,255,255,0.1)',
                        zerolinecolor: 'rgba(255,255,255,0.2)'
                    }},
                    bgcolor: 'rgba(0,0,0,0)'
                }},
                legend: {{
                    font: {{ color: '#fff' }},
                    bgcolor: 'rgba(0,0,0,0.3)',
                    x: 0.02,
                    y: 0.98
                }},
                margin: {{ t: 30, b: 30, l: 30, r: 30 }}
            }}, {{ responsive: true }});
        }}

        // 초기 로드
        updatePlot();
    </script>
</body>
</html>'''

    output_path.write_text(html_content, encoding='utf-8')
    print(f"? 3D 시각화 저장: {output_path}")


def main():
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    
    csv_path = reports_dir / "k4_embedding_overlay.csv"
    output_path = reports_dir / "k4_embedding_overlay_3d.html"
    
    if not csv_path.exists():
        print(f"좌표 데이터가 없습니다: {csv_path}")
        print("먼저 analyze_embedding_overlay.py를 실행하세요.")
        return
    
    generate_3d_overlay_html(csv_path, output_path)
    print("\n? 3D 임베딩 오버레이 시각화 완료!")


if __name__ == "__main__":
    main()
