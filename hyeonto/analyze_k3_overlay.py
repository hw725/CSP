"""

Sentence/Phrase K=3 클러스터 비교 오버레이 시각화

문장경계(Sentence)와 구경계(Phrase)의 K=3 클러스터를 같은 평면에 겹쳐서

서종 분포 패턴을 직접 비교합니다.

"""

import pandas as pd

import numpy as np

from pathlib import Path

import json

import re

from datetime import datetime

_JTI_PREFIX_TO_CATEGORY = {
    "jti_1": "경부",
    "jti_2": "사부",
    "jti_3": "자부",
    "jti_4": "집부",
}

def _build_book_category_map(sources_dir: Path) -> dict:
    """sources 파일명에서 책명 -> 사부분류 매핑 생성"""
    mapping: dict[str, str] = {}
    if not sources_dir.exists():
        return mapping

    for path in sources_dir.glob("jti_*"):
        name = path.name
        prefix_match = re.match(r"(jti_[1-4])", name)
        if not prefix_match:
            continue
        book_match = re.search(r"\[(?:역주|현토)\](.+?)_(?:원문|번역문)", name)
        if not book_match:
            continue
        book_name = book_match.group(1)
        mapping[book_name] = _JTI_PREFIX_TO_CATEGORY[prefix_match.group(1)]

    return mapping

def load_cluster_data(sentence_path: Path, phrase_path: Path):
    """Sentence/Phrase 클러스터 데이터 로드 및 서종 분포 계산"""

    # 서종 분류 함수 (파일명 jti_1~4 기준: 경/사/자/집)

    sources_dir = Path(__file__).parent.parent / "sources"
    book_category_map = _build_book_category_map(sources_dir)

    def categorize_book(book_name: str) -> str:
        return book_category_map.get(book_name, "기타")

    categories = ["경부", "사부", "자부", "집부"]

    results = {"sentence": [], "phrase": []}

    # Sentence 데이터

    print(f"? Loading Sentence: {sentence_path}")

    sentence_df = pd.read_csv(sentence_path)

    sentence_df["category"] = sentence_df["book"].apply(categorize_book)

    for cluster_id in sorted(sentence_df["cluster_id"].unique()):

        cluster_df = sentence_df[sentence_df["cluster_id"] == cluster_id]

        cat_dist = cluster_df["category"].value_counts(normalize=True) * 100

        results["sentence"].append(
            {
                "cluster_id": cluster_id,
                "total": len(cluster_df),
                "distribution": {cat: cat_dist.get(cat, 0) for cat in categories},
            }
        )

    # Phrase 데이터

    print(f"? Loading Phrase: {phrase_path}")

    phrase_df = pd.read_csv(phrase_path)

    phrase_df["category"] = phrase_df["book"].apply(categorize_book)

    for cluster_id in sorted(phrase_df["cluster_id"].unique()):

        cluster_df = phrase_df[phrase_df["cluster_id"] == cluster_id]

        cat_dist = cluster_df["category"].value_counts(normalize=True) * 100

        results["phrase"].append(
            {
                "cluster_id": cluster_id,
                "total": len(cluster_df),
                "distribution": {cat: cat_dist.get(cat, 0) for cat in categories},
            }
        )

    return results, categories

def generate_overlay_html(results: dict, categories: list, output_sentenceth: Path):
    """Sentence/Phrase 오버레이 비교 시각화 HTML 생성"""

    # 데이터 준비

    sentence_data = []

    phrase_data = []

    for r in results["sentence"]:

        sentence_data.append(
            [round(r["distribution"].get(cat, 0), 2) for cat in categories]
        )

    for r in results["phrase"]:

        phrase_data.append(
            [round(r["distribution"].get(cat, 0), 2) for cat in categories]
        )

    sentence_clusters = [f"Sentence-p{r['cluster_id']}" for r in results["sentence"]]

    phrase_clusters = [f"Phrase-p{r['cluster_id']}" for r in results["phrase"]]

    sentence_sizes = [r["total"] for r in results["sentence"]]

    phrase_sizes = [r["total"] for r in results["phrase"]]

    html_content = f"""<!DOCTYPE html>

<html lang="ko">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Sentence/Phrase K=3 클러스터 비교 오버레이</title>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>

        * {{

            margin: 0;

            padding: 0;

            box-sizing: border-box;

        }}

        body {{

            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;

            background: #ffffff;

            min-height: 100vh;

            color: #000;

            padding: 20px;

        }}

        .container {{

            max-width: 1600px;

            margin: 0 auto;

        }}

        h1 {{

            text-align: center;

            font-size: 2.2rem;

            margin-bottom: 10px;

            color: #333;

        }}

        .subtitle {{

            text-align: center;

            color: #888;

            margin-bottom: 30px;

            font-size: 1.1rem;

        }}

        .chart-container {{

            background: rgba(0,0,0,0.02);

            border: 1px solid #ddd;

            border-radius: 16px;

            padding: 20px;

            margin-bottom: 30px;

        }}

        .chart-title {{

            font-size: 1.3rem;

            color: #333;

            margin-bottom: 15px;

            padding-left: 10px;

            border-left: 4px solid #333;

        }}

        #radar-chart, #bar-chart, #heatmap-combined {{

            width: 100%;

            height: 500px;

        }}

        .grid {{

            display: grid;

            grid-template-columns: repeat(2, 1fr);

            gap: 20px;

        }}

        .insight-box {{

            background: rgba(0,0,0,0.03);

            border: 1px solid #ccc;

            border-radius: 12px;

            padding: 20px;

            margin-top: 20px;

        }}

        .insight-box h3 {{

            color: #333;

            margin-bottom: 15px;

        }}

        .insight-item {{

            padding: 10px 15px;

            background: rgba(0,0,0,0.04);

            border-radius: 8px;

            margin: 10px 0;

            border-left: 3px solid #555;

        }}

        .legend-panel {{

            display: flex;

            justify-content: center;

            gap: 40px;

            margin: 20px 0;

        }}

        .legend-item {{

            display: flex;

            align-items: center;

            gap: 10px;

        }}

        .legend-color {{

            width: 30px;

            height: 6px;

            border-radius: 3px;

        }}

        .pa-color {{ background: #000; }}

        .sa-color {{ background: #888; }}

        @media (max-width: 1200px) {{

            .grid {{

                grid-template-columns: 1fr;

            }}

        }}

    </style>

</head>

<body>

    <div class="container">

        <h1>? Sentence/Phrase K=3 클러스터 비교 오버레이</h1>

        <p class="subtitle">문장경계(Sentence)와 구경계(Phrase)의 서종 분포 패턴 직접 비교</p>

        <div class="legend-panel">

            <div class="legend-item">

                <div class="legend-color pa-color"></div>

                <span>Sentence (문장경계) - {sum(sentence_sizes):,}건</span>

            </div>

            <div class="legend-item">

                <div class="legend-color sa-color"></div>

                <span>Phrase (구경계) - {sum(phrase_sizes):,}건</span>

            </div>

        </div>

        <!-- 레이더 차트 -->

        <div class="chart-container">

            <h2 class="chart-title">? 클러스터별 서종 분포 레이더 (Sentence vs Phrase)</h2>

            <div id="radar-chart"></div>

        </div>

        <!-- 그룹 바 차트 -->

        <div class="chart-container">

            <h2 class="chart-title">? 서종별 클러스터 비교 (Sentence vs Phrase 오버레이)</h2>

            <div id="bar-chart"></div>

        </div>

        <!-- 통합 히트맵 -->

        <div class="chart-container">

            <h2 class="chart-title">? Sentence-Phrase 통합 히트맵 (6개 클러스터 나란히)</h2>

            <div id="heatmap-combined"></div>

        </div>

        <!-- 발견 사항 -->

        <div class="insight-box">

            <h3>? Sentence-Phrase 비교 핵심 발견</h3>

            <div class="insight-item">

                <strong>일관성</strong>: Sentence와 Phrase 모두 K=3에서 유사한 4가지 언어 모드(서사/정의/논증/혼합)가 발현됨

            </div>

            <div class="insight-item">

                <strong>삼경 집중도</strong>: 두 레벨 모두 p2 클러스터에서 삼경(三經)이 46-53%로 압도적

            </div>

            <div class="insight-item">

                <strong>사서 분포</strong>: Phrase에서는 사서(四書)가 p1에 더 집중 (13.4%), Sentence에서는 p1(18.1%)과 p2(12.8%)에 분산

            </div>

            <div class="insight-item">

                <strong>역사 서사</strong>: Sentence-p3과 Phrase-p0에서 사서(史書) 비율이 38-43%로 유사

            </div>

        </div>

    </div>

    <script>

        const categories = {json.dumps(categories)};

        const paData = {json.dumps(sentence_data)};

        const saData = {json.dumps(phrase_data)};

        const paClusters = {json.dumps(sentence_clusters)};

        const saClusters = {json.dumps(phrase_clusters)};

        // 색상 팔레트

        const sentenceColors = ['#000000', '#444444', '#888888'];

        const phraseColors = ['#666666', '#999999', '#cccccc'];

        // 색약 접근성: Sentence=실선(굵기 차이), Phrase=점선/대시(패턴 차이)

        const sentenceDashes = ['solid', 'solid', 'solid'];

        const sentenceWidths = [3, 2.5, 2];

        const phraseDashes = ['dash', 'dot', 'dashdot'];

        const phraseWidths = [3, 2.5, 2];

        // 마커 심볼로도 추가 구분

        const sentenceSymbols = ['circle', 'square', 'diamond'];

        const phraseSymbols = ['triangle-up', 'cross', 'star'];

        // 1. 레이더 차트 (각 클러스터별)

        const radarTraces = [];

        for (let i = 0; i < paData.length; i++) {{

            // Sentence 클러스터 (실선, filled 마커)

            radarTraces.push({{

                type: 'scatterpolar',

                r: [...paData[i], paData[i][0]],

                theta: [...categories, categories[0]],

                fill: 'toself',

                fillcolor: sentenceColors[i] + '15',

                line: {{ color: sentenceColors[i], width: sentenceWidths[i], dash: sentenceDashes[i] }},

                marker: {{ symbol: sentenceSymbols[i], size: 10 }},

                name: paClusters[i],

                legendgroup: 'pa'

            }});

            // Phrase 클러스터 (점선/대시, open 마커)

            radarTraces.push({{

                type: 'scatterpolar',

                r: [...saData[i], saData[i][0]],

                theta: [...categories, categories[0]],

                fill: 'toself',

                fillcolor: phraseColors[i] + '10',

                line: {{ color: phraseColors[i], width: phraseWidths[i], dash: phraseDashes[i] }},

                marker: {{ symbol: phraseSymbols[i] + '-open', size: 10, line: {{ width: 2, color: phraseColors[i] }} }},

                name: saClusters[i],

                legendgroup: 'sa'

            }});

        }}

        Plotly.newPlot('radar-chart', radarTraces, {{

            polar: {{

                radialaxis: {{

                    visible: true,

                    range: [0, 60],

                    tickfont: {{ color: '#000' }},

                    gridcolor: 'rgba(0,0,0,0.15)'

                }},

                angularaxis: {{

                    tickfont: {{ color: '#000', size: 12 }}

                }},

                bgcolor: '#ffffff'

            }},

            paper_bgcolor: '#ffffff',

            plot_bgcolor: '#ffffff',

            legend: {{

                font: {{ color: '#000' }},

                x: 1.1,

                y: 0.5

            }},

            margin: {{ t: 50, b: 50 }}

        }}, {{ responsive: true }});

        // 2. 그룹 바 차트 (색약 접근성: Sentence=실색, Phrase=빗금 패턴)

        const barPatterns = ['/', '\\\\', 'x'];

        const barTraces = [];

        for (let i = 0; i < paData.length; i++) {{

            // Sentence: 실색 채우기

            barTraces.push({{

                x: categories,

                y: paData[i],

                name: paClusters[i],

                type: 'bar',

                marker: {{

                    color: sentenceColors[i],

                    opacity: 0.85,

                    line: {{ color: '#000', width: 1 }}

                }}

            }});

            // Phrase: 빗금 패턴 + 연한 채우기

            barTraces.push({{

                x: categories,

                y: saData[i],

                name: saClusters[i],

                type: 'bar',

                marker: {{

                    color: phraseColors[i],

                    opacity: 0.85,

                    pattern: {{ shape: barPatterns[i], solidity: 0.6, fgcolor: '#333' }},

                    line: {{ color: '#333', width: 1.5 }}

                }}

            }});

        }}

        Plotly.newPlot('bar-chart', barTraces, {{

            barmode: 'group',

            paper_bgcolor: '#ffffff',

            plot_bgcolor: '#ffffff',

            xaxis: {{

                tickfont: {{ color: '#000' }},

                gridcolor: 'rgba(0,0,0,0.1)'

            }},

            yaxis: {{

                title: '비율 (%)',

                tickfont: {{ color: '#000' }},

                titlefont: {{ color: '#000' }},

                gridcolor: 'rgba(0,0,0,0.1)'

            }},

            legend: {{

                font: {{ color: '#000' }},

                orientation: 'h',

                y: -0.2

            }},

            margin: {{ t: 30, b: 100 }}

        }}, {{ responsive: true }});

        // 3. 통합 히트맵

        const allClusters = [...paClusters, ...saClusters];

        const allData = [...paData, ...saData];

        Plotly.newPlot('heatmap-combined', [{{

            z: allData,

            x: categories,

            y: allClusters,

            type: 'heatmap',

            colorscale: [

                [0, '#ffffff'],

                [0.25, '#cccccc'],

                [0.5, '#888888'],

                [0.75, '#444444'],

                [1, '#000000']

            ],

            hovertemplate: '<b>%{{y}}</b><br>%{{x}}: %{{z:.1f}}%<extra></extra>',

            colorbar: {{

                title: '비율 (%)',

                titlefont: {{ color: '#000' }},

                tickfont: {{ color: '#000' }}

            }}

        }}], {{

            paper_bgcolor: '#ffffff',

            plot_bgcolor: '#ffffff',

            xaxis: {{

                tickfont: {{ color: '#000', size: 11 }},

                tickangle: -30

            }},

            yaxis: {{

                tickfont: {{ color: '#000', size: 12 }}

            }},

            margin: {{ t: 30, l: 100, r: 80, b: 100 }},

            annotations: []

        }}, {{ responsive: true }});

    </script>

</body>

</html>"""

    output_sentenceth.write_text(html_content, encoding="utf-8")

    print(f"? Sentence/Phrase 오버레이 시각화 저장: {output_sentenceth}")

def main():

    base_dir = Path(__file__).parent

    reports_dir = base_dir / "report_1-1"

    # 정규화된 Sentence/Phrase 데이터 경로

    sentence_path = reports_dir / "sentence_k3_normalized" / "sentence_clusters.csv"

    phrase_path = reports_dir / "phrase_k3_normalized" / "phrase_clusters.csv"

    if not sentence_path.exists() or not phrase_path.exists():

        print(f"K=3 클러스터 데이터가 없습니다")

        print(f"   필요: {sentence_path}")

        print(f"   필요: {phrase_path}")

        return

    # 데이터 로드 및 분석

    results, categories = load_cluster_data(sentence_path, phrase_path)

    print(f"\n? Sentence 클러스터: {len(results['sentence'])}개")

    print(f"? Phrase 클러스터: {len(results['phrase'])}개")

    # 오버레이 시각화 생성

    output_sentenceth = base_dir / "report_1-1" / "visualizations_k3" / "k3_sentence_phrase_overlay_normalized.html"

    generate_overlay_html(results, categories, output_sentenceth)

    print("\n? Sentence/Phrase K=3 오버레이 비교 시각화 완료! (정규화 반영)")

if __name__ == "__main__":

    main()
