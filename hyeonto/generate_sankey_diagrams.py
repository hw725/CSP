"""
Sentence/Phrase 클러스터 Sankey 다이어그램 생성

4가지 비교:
1. Sentence K=4 ↔ Phrase K=4
2. Sentence K=4 → Sentence K=14
3. Phrase K=4 → Phrase K=24
4. Sentence K=14 ↔ Phrase K=24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_cluster_data(path: Path) -> pd.DataFrame:
    """클러스터 데이터 로드"""
    if not path.exists():
        print(f"파일 없음: {path}")
        return None
    return pd.read_csv(path)


def compute_flow_matrix(df1: pd.DataFrame, df2: pd.DataFrame, 
                        id_col: str = 'left_sentence_id') -> dict:
    """두 클러스터링 결과 간 흐름 매트릭스 계산"""
    
    # 공통 ID로 매칭
    if id_col not in df1.columns or id_col not in df2.columns:
        # ID 컬럼이 없으면 인덱스 기반 매칭
        merged = pd.DataFrame({
            'cluster1': df1['cluster_id'].values[:min(len(df1), len(df2))],
            'cluster2': df2['cluster_id'].values[:min(len(df1), len(df2))]
        })
    else:
        merged = df1[[id_col, 'cluster_id']].merge(
            df2[[id_col, 'cluster_id']], 
            on=id_col, 
            suffixes=('_1', '_2')
        )
        merged.columns = [id_col, 'cluster1', 'cluster2']
    
    # 흐름 계산
    flow = merged.groupby(['cluster1', 'cluster2']).size().reset_index(name='count')
    
    return flow


def generate_sankey_html(flow_df: pd.DataFrame, 
                         left_label: str, right_label: str,
                         left_k: int, right_k: int,
                         output_sentenceth: Path,
                         title: str):
    """Sankey 다이어그램 HTML 생성"""
    
    # 노드 생성
    left_nodes = [f"{left_label}-p{i}" for i in range(left_k)]
    right_nodes = [f"{right_label}-p{i}" for i in range(right_k)]
    all_nodes = left_nodes + right_nodes
    
    # 링크 생성
    links = []
    for _, row in flow_df.iterrows():
        source_idx = int(row['cluster1'])
        target_idx = left_k + int(row['cluster2'])
        links.append({
            'source': source_idx,
            'target': target_idx,
            'value': int(row['count'])
        })
    
    # 색상
    left_colors = ['rgba(233, 69, 96, 0.8)', 'rgba(255, 107, 107, 0.8)', 
                   'rgba(255, 201, 60, 0.8)', 'rgba(255, 140, 66, 0.8)'] * 4
    right_colors = ['rgba(78, 204, 163, 0.8)', 'rgba(69, 183, 170, 0.8)',
                    'rgba(56, 163, 165, 0.8)', 'rgba(34, 87, 122, 0.8)'] * 7
    
    node_colors = left_colors[:left_k] + right_colors[:right_k]
    
    html_content = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
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
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #e94560, #4ecca3);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
        #sankey {{ width: 100%; height: 600px; }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px 25px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }}
        .stat-value {{ font-size: 1.5rem; color: #4ecca3; }}
        .stat-label {{ font-size: 0.9rem; color: #888; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">클러스터 간 데이터 흐름 시각화</p>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{left_label} K={left_k}</div>
                <div class="stat-label">왼쪽</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{right_label} K={right_k}</div>
                <div class="stat-label">오른쪽</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{flow_df['count'].sum():,}</div>
                <div class="stat-label">총 매칭</div>
            </div>
        </div>
        <div id="sankey"></div>
    </div>
    <script>
        const data = {{
            type: "sankey",
            orientation: "h",
            node: {{
                pad: 15,
                thickness: 20,
                line: {{ color: "rgba(255,255,255,0.3)", width: 0.5 }},
                label: {json.dumps(all_nodes)},
                color: {json.dumps(node_colors)}
            }},
            link: {{
                source: {json.dumps([l['source'] for l in links])},
                target: {json.dumps([l['target'] for l in links])},
                value: {json.dumps([l['value'] for l in links])},
                color: "rgba(255,255,255,0.2)"
            }}
        }};
        
        Plotly.newPlot('sankey', [data], {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#fff', size: 12 }},
            margin: {{ t: 30, b: 30, l: 30, r: 30 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''

    output_sentenceth.write_text(html_content, encoding='utf-8')
    print(f"✅ Sankey 저장: {output_sentenceth}")


def main():
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    output_dir = reports_dir / "sankey_diagrams"
    output_dir.mkdir(exist_ok=True)
    
    # 클러스터 데이터 경로 (K=4만 사용)
    paths = {
        'sentence_k4': reports_dir / "sentence_k4_normalized" / "sentence_clusters.csv",
        'phrase_k4': reports_dir / "phrase_k4_normalized" / "phrase_clusters.csv",
    }
    
    # 데이터 로드
    data = {}
    for key, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            data[key] = df
            print(f"✅ {key}: {len(df):,}건")
        else:
            print(f"❌ 파일 없음: {path}")
    
    print()
    
    # Sentence K=4 ↔ Phrase K=4 (서종 기준)
    if 'sentence_k4' in data and 'phrase_k4' in data:
        print("📊 Sentence K=4 ↔ Phrase K=4 (서종 기준)")
        
        # 서종별 클러스터 분포 계산
        sentence_book_cluster = data['sentence_k4'].groupby(['book_name', 'cluster_id']).size().reset_index(name='sentence_count')
        phrase_book_cluster = data['phrase_k4'].groupby(['book_name', 'cluster_id']).size().reset_index(name='phrase_count')
        
        sentence_totals = data['sentence_k4'].groupby('book_name').size().reset_index(name='sentence_total')
        phrase_totals = data['phrase_k4'].groupby('book_name').size().reset_index(name='phrase_total')
        
        sentence_book_cluster = sentence_book_cluster.merge(sentence_totals, on='book_name')
        phrase_book_cluster = phrase_book_cluster.merge(phrase_totals, on='book_name')
        
        sentence_book_cluster['sentence_ratio'] = sentence_book_cluster['sentence_count'] / sentence_book_cluster['sentence_total']
        phrase_book_cluster['phrase_ratio'] = phrase_book_cluster['phrase_count'] / phrase_book_cluster['phrase_total']
        
        # 서종 기준 Sentence-Phrase 클러스터 쌍별 흐름 계산
        flow_list = []
        common_books = set(sentence_book_cluster['book_name'].unique()) & set(phrase_book_cluster['book_name'].unique())
        
        for book in common_books:
            sentence_book = sentence_book_cluster[sentence_book_cluster['book_name'] == book]
            phrase_book = phrase_book_cluster[phrase_book_cluster['book_name'] == book]
            book_weight = min(sentence_book['sentence_total'].iloc[0], phrase_book['phrase_total'].iloc[0])
            
            for _, sentence_row in sentence_book.iterrows():
                for _, phrase_row in phrase_book.iterrows():
                    weight = sentence_row['sentence_ratio'] * phrase_row['phrase_ratio'] * book_weight
                    if weight > 0:
                        flow_list.append({
                            'cluster1': sentence_row['cluster_id'],
                            'cluster2': phrase_row['cluster_id'],
                            'count': weight
                        })
        
        flow = pd.DataFrame(flow_list).groupby(['cluster1', 'cluster2'])['count'].sum().reset_index()
        flow['count'] = flow['count'].round().astype(int)
        flow = flow[flow['count'] > 0]
        
        generate_sankey_html(flow, 'Sentence', 'Phrase', 4, 4, 
                            output_dir / "sankey_sentence4_phrase4.html",
                            "Sentence K=4 ↔ Phrase K=4 (서종 분포 기반)")
    
    print(f"\n✅ Sankey 다이어그램 생성 완료!")
    print(f"   저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
