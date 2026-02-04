"""

K=3 클러스터 심층 분석 스크립트 (확장판)

- 클러스터별 현토마커 분석

- 언어학적 특성 비교

- 히트맵 시각화

"""

import pandas as pd

import numpy as np

from pathlib import Path

from collections import Counter

import json

# 서종 분류 정의 (간략화)

def categorize_book_simple(book_name: str) -> str:
    """간략화된 서종 분류"""

    saseo_books = ["논어집주", "맹자집주", "대학장구", "중용장구"]

    if book_name in saseo_books:

        return "사서(四書)"

    samgyeong = ["주역전의", "서경집전", "시경집전"]

    if any(s in book_name for s in samgyeong):

        return "삼경(三經)"

    if "춘추좌씨전" in book_name or "자치통감" in book_name:

        return "사서(史書)"

    if "당송팔대가문초" in book_name:

        return "집부(集部)"

    if "예기" in book_name:

        return "예기(禮記)"

    if "당시삼백수" in book_name:

        return "당시(唐詩)"

    return "기타"

def analyze_marker_patterns(df: pd.DataFrame, cluster_col: str = "cluster_id"):
    """클러스터별 현토마커 패턴 분석"""

    results = []

    # 마커 컬럼 확인 (hyeonto_markers 또는 markers 컬럼 찾기)

    marker_cols = [
        c for c in df.columns if "marker" in c.lower() or "hyeonto" in c.lower()
    ]

    print(f"Found marker columns: {marker_cols}")

    for cluster_id in sorted(df[cluster_col].unique()):

        cluster_df = df[df[cluster_col] == cluster_id]

        # 서종별 분포 계산

        cluster_df["category"] = cluster_df["book"].apply(categorize_book_simple)

        cat_dist = cluster_df["category"].value_counts(normalize=True) * 100

        result = {
            "cluster_id": cluster_id,
            "total": len(cluster_df),
            "category_distribution": cat_dist.to_dict(),
        }

        results.append(result)

    return results

def generate_heatmap_data(results: list, output_path: Path):
    """히트맵용 데이터 생성"""

    categories = [
        "사서(四書)",
        "삼경(三經)",
        "사서(史書)",
        "집부(集部)",
        "예기(禮記)",
        "당시(唐詩)",
        "기타",
    ]

    # 매트릭스 생성

    matrix = []

    for r in results:

        row = []

        for cat in categories:

            row.append(r["category_distribution"].get(cat, 0))

        matrix.append(row)

    heatmap_data = {
        "categories": categories,
        "clusters": [f"p{r['cluster_id']}" for r in results],
        "matrix": matrix,
        "cluster_sizes": [r["total"] for r in results],
    }

    output_path.write_text(
        json.dumps(heatmap_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n? 히트맵 데이터 저장: {output_path}")

def generate_html_heatmap(results: list, layer_name: str, output_path: Path):
    """인터랙티브 HTML 히트맵 생성"""

    categories = [
        "사서(四書)",
        "삼경(三經)",
        "사서(史書)",
        "집부(集部)",
        "예기(禮記)",
        "당시(唐詩)",
        "기타",
    ]

    clusters = [f"p{r['cluster_id']}" for r in results]

    # 매트릭스 생성

    z_data = []

    for r in results:

        row = []

        for cat in categories:

            row.append(round(r["category_distribution"].get(cat, 0), 2))

        z_data.append(row)

    # Summary cards 생성

    summary_cards = ""

    for r in results:

        summary_cards += f"""

            <div class="summary-card">

                <h3>Cluster {r['cluster_id']}</h3>

                <div class="value">{r['total']:,}</div>

                <div class="label">데이터 수</div>

            </div>

        """

    # Insights 생성 - 미리 계산

    p0_top = max(results[0]["category_distribution"].items(), key=lambda x: x[1])

    p0_top_cat = p0_top[0]

    p0_top_val = p0_top[1]

    p1_saseo = results[1]["category_distribution"].get("사서(四書)", 0)

    p2_samgyeong = results[2]["category_distribution"].get("삼경(三經)", 0)

    p3_history = results[3]["category_distribution"].get("사서(史書)", 0)

    p3_jipbu = results[3]["category_distribution"].get("집부(集部)", 0)

    z_data_json = json.dumps(z_data)

    categories_json = json.dumps(categories)

    clusters_json = json.dumps(clusters)

    html_content = f"""<!DOCTYPE html>

<html lang="ko">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>{layer_name} K=3 서종 분포 히트맵</title>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>

        body {{

            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;

            margin: 0;

            padding: 20px;

            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);

            min-height: 100vh;

            color: #fff;

        }}

        .container {{

            max-width: 1400px;

            margin: 0 auto;

        }}

        h1 {{

            text-align: center;

            color: #e94560;

            font-size: 2.2rem;

            margin-bottom: 10px;

            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);

        }}

        .subtitle {{

            text-align: center;

            color: #a0a0a0;

            margin-bottom: 30px;

            font-size: 1.1rem;

        }}

        #heatmap {{

            width: 100%;

            height: 500px;

            background: rgba(255,255,255,0.05);

            border-radius: 15px;

            padding: 20px;

            box-shadow: 0 8px 32px rgba(0,0,0,0.3);

        }}

        .summary {{

            display: grid;

            grid-template-columns: repeat(4, 1fr);

            gap: 20px;

            margin-top: 30px;

        }}

        .summary-card {{

            background: rgba(255,255,255,0.08);

            border-radius: 12px;

            padding: 20px;

            text-align: center;

            border: 1px solid rgba(255,255,255,0.1);

            transition: transform 0.3s ease;

        }}

        .summary-card:hover {{

            transform: translateY(-5px);

            background: rgba(255,255,255,0.12);

        }}

        .summary-card h3 {{

            color: #e94560;

            margin-bottom: 10px;

            font-size: 1.3rem;

        }}

        .summary-card .value {{

            font-size: 2rem;

            font-weight: bold;

            color: #4ecca3;

        }}

        .summary-card .label {{

            color: #888;

            font-size: 0.9rem;

            margin-top: 5px;

        }}

        .insights {{

            margin-top: 40px;

            background: rgba(255,255,255,0.05);

            border-radius: 15px;

            padding: 30px;

            border: 1px solid rgba(255,255,255,0.1);

        }}

        .insights h2 {{

            color: #4ecca3;

            margin-bottom: 20px;

        }}

        .insight-item {{

            background: rgba(0,0,0,0.2);

            border-radius: 8px;

            padding: 15px 20px;

            margin-bottom: 15px;

            border-left: 4px solid #e94560;

        }}

        .insight-item strong {{

            color: #e94560;

        }}

    </style>

</head>

<body>

    <div class="container">

        <h1>? {layer_name} K=3 클러스터별 서종 분포</h1>

        <p class="subtitle">현토(懸吐) 코퍼스 클러스터링 분석 - 서종(書種) 분포 히트맵</p>

        <div id="heatmap"></div>

        <div class="summary">

            {summary_cards}

        </div>

        <div class="insights">

            <h2>? 핵심 발견사항</h2>

            <div class="insight-item">

                <strong>Cluster p0</strong>: {p0_top_cat}가 {p0_top_val:.1f}%로 지배적 - <em>서사/역사 중심 클러스터</em>

            </div>

            <div class="insight-item">

                <strong>Cluster p1</strong>: 사서(四書)가 {p1_saseo:.1f}%로 최고 - <em>사서 핵심 클러스터</em>

            </div>

            <div class="insight-item">

                <strong>Cluster p2</strong>: 삼경(三經)이 {p2_samgyeong:.1f}%로 압도적 - <em>경서(시경/서경/주역) 집중 클러스터</em>

            </div>

            <div class="insight-item">

                <strong>Cluster p3</strong>: 사서(史書) {p3_history:.1f}% + 집부 {p3_jipbu:.1f}% - <em>서사/문집 혼합 클러스터</em>

            </div>

        </div>

    </div>

    <script>

        var data = [{{

            z: {z_data_json},

            x: {categories_json},

            y: {clusters_json},

            type: 'heatmap',

            colorscale: [

                [0, '#0a0a23'],

                [0.2, '#1a1a4e'],

                [0.4, '#e94560'],

                [0.6, '#ff6b6b'],

                [0.8, '#ffc93c'],

                [1, '#4ecca3']

            ],

            hoverongaps: false,

            hovertemplate: '<b>%{{y}}</b><br>%{{x}}: %{{z:.1f}}%<extra></extra>',

            showscale: true,

            colorbar: {{

                title: '비율 (%)',

                titlefont: {{color: '#fff'}},

                tickfont: {{color: '#fff'}}

            }}

        }}];

        var layout = {{

            title: {{

                text: '',

                font: {{color: '#fff', size: 18}}

            }},

            paper_bgcolor: 'rgba(0,0,0,0)',

            plot_bgcolor: 'rgba(0,0,0,0)',

            xaxis: {{

                title: '서종(書種)',

                tickfont: {{color: '#fff', size: 12}},

                titlefont: {{color: '#fff'}},

                tickangle: -30

            }},

            yaxis: {{

                title: '클러스터',

                tickfont: {{color: '#fff', size: 14}},

                titlefont: {{color: '#fff'}}

            }},

            margin: {{t: 30, l: 80, r: 50, b: 100}},

            annotations: []

        }};

        // 각 셀에 값 표시

        var zData = {z_data_json};

        var xLabels = {categories_json};

        var yLabels = {clusters_json};

        for (var i = 0; i < yLabels.length; i++) {{

            for (var j = 0; j < xLabels.length; j++) {{

                var val = zData[i][j];

                if (val > 5) {{

                    layout.annotations.push({{

                        x: xLabels[j],

                        y: yLabels[i],

                        text: val.toFixed(1) + '%',

                        font: {{color: val > 30 ? '#000' : '#fff', size: 11}},

                        showarrow: false

                    }});

                }}

            }}

        }}

        Plotly.newPlot('heatmap', data, layout, {{responsive: true}});

    </script>

</body>

</html>"""

    output_path.write_text(html_content, encoding="utf-8")

    print(f"? HTML 히트맵 저장: {output_path}")

def analyze_linguistic_characteristics(
    df: pd.DataFrame, cluster_col: str = "cluster_id"
):
    """클러스터별 언어학적 특성 분석"""

    print("\n" + "=" * 80)

    print("? 클러스터별 언어학적 특성 비교")

    print("=" * 80)

    # 서종 분류 추가

    df["category"] = df["book"].apply(categorize_book_simple)

    for cluster_id in sorted(df[cluster_col].unique()):

        cluster_df = df[df[cluster_col] == cluster_id]

        print(f"\n{'─'*60}")

        print(f"? Cluster {cluster_id} (n={len(cluster_df):,})")

        print(f"{'─'*60}")

        # 서종 분포

        cat_dist = cluster_df["category"].value_counts()

        cat_pct = cat_dist / len(cluster_df) * 100

        # 상위 3개 서종

        top_categories = cat_pct.head(3)

        print(f"\n  ▶ 지배적 서종 (Top 3):")

        for cat, pct in top_categories.items():

            print(f"    ? {cat}: {pct:.1f}%")

        # 사서 비율로 정전성(Canonicity) 계산

        saseo_ratio = cat_pct.get("사서(四書)", 0)

        samgyeong_ratio = cat_pct.get("삼경(三經)", 0)

        canonical_ratio = saseo_ratio + samgyeong_ratio

        print(f"\n  ▶ 정전성 지표:")

        print(f"    ? 사서(四書) 비율: {saseo_ratio:.2f}%")

        print(f"    ? 삼경(三經) 비율: {samgyeong_ratio:.2f}%")

        print(f"    ? 총 정전성 (사서+삼경): {canonical_ratio:.2f}%")

        # 클러스터 특성 해석

        print(f"\n  ▶ 클러스터 해석:")

        if saseo_ratio > 15:

            print(f"    → '사서 핵심' 클러스터 (유교 경전 중심)")

        elif samgyeong_ratio > 40:

            print(f"    → '경서 정의' 클러스터 (시경/서경/주역 집중)")

        elif cat_pct.get("사서(史書)", 0) > 35:

            print(f"    → '역사 서사' 클러스터 (춘추좌전/강목 중심)")

        elif cat_pct.get("집부(集部)", 0) > 35:

            print(f"    → '문집' 클러스터 (당송팔대가 문초)")

        else:

            print(f"    → '혼합' 클러스터")

def main():

    base_dir = Path(__file__).parent

    reports_dir = base_dir / "reports"

    print("=" * 80)

    print("? K=3 클러스터 심층 분석 (확장판)")

    print("=" * 80)

    # 1. Sentence Boundary K=3 분석

    sentence_path = reports_dir / "sentence_boundary_k3_full" / "boundary_clusters.csv"

    if sentence_path.exists():

        print(f"\n? Loading Sentence: {sentence_path}")

        sentence_df = pd.read_csv(sentence_path)

        print(f"   → Loaded {len(sentence_df):,} rows")

        sentence_results = analyze_marker_patterns(sentence_df)

        generate_heatmap_data(
            sentence_results,
            reports_dir / "sentence_boundary_k3_full" / "heatmap_data.json",
        )

        generate_html_heatmap(
            sentence_results,
            "문장경계(Sentence)",
            reports_dir / "sentence_boundary_k3_full" / "k3_heatmap.html",
        )

        analyze_linguistic_characteristics(sentence_df)

    # 2. Phrase Boundary K=3 분석

    phrase_path = (
        reports_dir / "phrase_boundary_k3_full" / "phrase_boundary_clusters.csv"
    )

    if phrase_path.exists():

        print(f"\n? Loading Phrase: {phrase_path}")

        phrase_df = pd.read_csv(phrase_path)

        print(f"   → Loaded {len(phrase_df):,} rows")

        phrase_results = analyze_marker_patterns(phrase_df)

        generate_heatmap_data(
            phrase_results,
            reports_dir / "phrase_boundary_k3_full" / "heatmap_data.json",
        )

        generate_html_heatmap(
            phrase_results,
            "구경계(Phrase)",
            reports_dir / "phrase_boundary_k3_full" / "k3_heatmap.html",
        )

        analyze_linguistic_characteristics(phrase_df)

    print("\n" + "=" * 80)

    print("? 분석 완료!")

    print("=" * 80)

if __name__ == "__main__":

    main()
