"""

Sentence/Phrase K=3 임베딩 오버레이 시각화 (2D/3D UMAP)

Sentence(문장경계)와 Phrase(구경계)의 K=3 클러스터를 같은 공간에 겹쳐서

의미역(semantic field)이 어떻게 분포하는지 시각화합니다.

UMAP을 사용하여 고차원 임베딩을 2D 또는 3D로 축소합니다.

기본값: 흑백 인쇄용 그레이스케일 시각화 (마커 심볼로 클러스터 구분)

"""

import pandas as pd

import numpy as np

from pathlib import Path

import json

from tqdm import tqdm

import argparse

# 컬러 팔레트 정의

COLOR_PALETTES = {
    "color": {
        "sentence": ["#e94560", "#ff6b6b", "#ffc93c", "#ff8c42"],
        "phrase": ["#4ecca3", "#45b7aa", "#38a3a5", "#22577a"],
        "bg_gradient": "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
        "text": "#fff",
        "subtitle": "#888",
        "grid": "rgba(255,255,255,0.1)",
        "legend_bg": "rgba(0,0,0,0.3)",
    },
    "grayscale": {
        "sentence": ["#000000", "#333333", "#666666", "#999999"],
        "phrase": ["#CCCCCC", "#AAAAAA", "#888888", "#555555"],
        "bg_gradient": "#ffffff",
        "text": "#000",
        "subtitle": "#444",
        "grid": "rgba(0,0,0,0.15)",
        "legend_bg": "rgba(255,255,255,0.9)",
    },
}

# 흑백 마커 심볼 (구분용)

GRAYSCALE_SYMBOLS = {
    "sentence": ["circle", "square", "diamond", "cross"],
    "phrase": ["triangle-up", "triangle-down", "hexagon", "star"],
}

def load_embeddings_and_generate_umap(
    sentence_path: Path,
    phrase_path: Path,
    sample_size: int = 10000,
    n_components: int = 2,
    cache_npy: Path | None = None,
    sentence_cache_npy: Path | None = None,
    phrase_cache_npy: Path | None = None,
    allow_embed: bool = False,
):
    """Sentence/Phrase 데이터 로드, 임베딩 생성, UMAP 적용"""

    try:
        from umap import UMAP
    except ImportError:
        print("필요한 라이브러리: umap-learn")
        return None

    print("? 데이터 로드...")

    sentence_df = pd.read_csv(sentence_path)

    phrase_df = pd.read_csv(phrase_path)

    print(f"   Sentence: {len(sentence_df):,}개")

    print(f"   Phrase: {len(phrase_df):,}개")

    # 샘플링 (메모리 효율)

    if sample_size and len(sentence_df) > sample_size:

        sentence_sample = sentence_df.sample(n=sample_size, random_state=42)

    else:

        sentence_sample = sentence_df

    if sample_size and len(phrase_df) > sample_size:

        phrase_sample = phrase_df.sample(n=sample_size, random_state=42)

    else:

        phrase_sample = phrase_df

    print(f"   샘플: Sentence={len(sentence_sample):,}, Phrase={len(phrase_sample):,}")

    # 텍스트 컬럼 찾기

    text_col = None

    for col in ["원문", "left_sentence", "src_left", "src_L"]:

        if col in sentence_sample.columns:

            text_col = col

            break

    if text_col is None:

        print("텍스트 컬럼을 찾을 수 없습니다.")

        return None

    combined_expected = len(sentence_sample) + len(phrase_sample)
    combined_emb = None

    if sentence_cache_npy is not None or phrase_cache_npy is not None:
        if sentence_cache_npy is None or phrase_cache_npy is None:
            print("sentence/phrase 캐시는 둘 다 지정해야 합니다.")
            if not allow_embed:
                return None
        elif sentence_cache_npy.exists() and phrase_cache_npy.exists():
            print(f"\n? Sentence 캐시 사용: {sentence_cache_npy}")
            print(f"? Phrase 캐시 사용: {phrase_cache_npy}")
            sentence_emb = np.load(sentence_cache_npy)
            phrase_emb = np.load(phrase_cache_npy)

            if sentence_emb.shape[0] == len(sentence_df) and phrase_emb.shape[0] == len(
                phrase_df
            ):
                sentence_idx = sentence_sample.index.to_numpy()
                phrase_idx = phrase_sample.index.to_numpy()
                sentence_emb = sentence_emb[sentence_idx]
                phrase_emb = phrase_emb[phrase_idx]
                combined_emb = np.vstack([sentence_emb, phrase_emb])
            elif sentence_emb.shape[0] == len(sentence_sample) and phrase_emb.shape[
                0
            ] == len(phrase_sample):
                combined_emb = np.vstack([sentence_emb, phrase_emb])
            else:
                print(
                    "캐시 크기가 샘플 수와 맞지 않습니다. "
                    f"sentence_cache={sentence_emb.shape[0]}, expected={len(sentence_sample)}; "
                    f"phrase_cache={phrase_emb.shape[0]}, expected={len(phrase_sample)}"
                )
                if not allow_embed:
                    return None
        else:
            print("sentence/phrase 캐시 파일을 찾을 수 없습니다.")
            if not allow_embed:
                return None

    if combined_emb is None and cache_npy is not None and cache_npy.exists():
        print(f"\n? 임베딩 캐시 사용: {cache_npy}")
        combined_emb = np.load(cache_npy)
        if combined_emb.shape[0] == len(sentence_df) + len(phrase_df):
            sentence_idx = sentence_sample.index.to_numpy()
            phrase_idx = phrase_sample.index.to_numpy()
            sentence_part = combined_emb[: len(sentence_df)]
            phrase_part = combined_emb[len(sentence_df) :]
            combined_emb = np.vstack(
                [sentence_part[sentence_idx], phrase_part[phrase_idx]]
            )
        elif combined_emb.shape[0] != combined_expected:
            print(
                "캐시 크기가 샘플 수와 맞지 않습니다. "
                f"cache={combined_emb.shape[0]}, expected={combined_expected}"
            )
            if not allow_embed:
                return None
            combined_emb = None

    if combined_emb is None and not allow_embed:
        print("임베딩 캐시가 없고 재임베딩이 비활성화되어 있습니다.")
        return None

    if combined_emb is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            print("필요한 라이브러리: FlagEmbedding")
            return None

        # 임베딩 생성
        print(f"\n? BGE-M3 임베딩 생성 (컬럼: {text_col})...")

        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

        sentence_texts = sentence_sample[text_col].fillna("").tolist()

        phrase_texts = phrase_sample[text_col].fillna("").tolist()

        # 배치 처리

        batch_size = 256

        def encode_batch(texts, desc):

            embeddings = []

            for i in tqdm(range(0, len(texts), batch_size), desc=desc):

                batch = texts[i : i + batch_size]

                emb = model.encode(batch, max_length=512)["dense_vecs"]

                embeddings.append(emb)

            return np.vstack(embeddings)

        sentence_emb = encode_batch(sentence_texts, "Sentence 임베딩")

        phrase_emb = encode_batch(phrase_texts, "Phrase 임베딩")

        print(f"   Sentence 임베딩: {sentence_emb.shape}")

        print(f"   Phrase 임베딩: {phrase_emb.shape}")

        # 통합 임베딩

        combined_emb = np.vstack([sentence_emb, phrase_emb])

    labels = ["Sentence"] * len(sentence_emb) + ["Phrase"] * len(phrase_emb)

    cluster_ids = (
        sentence_sample["cluster_id"].tolist() + phrase_sample["cluster_id"].tolist()
    )

    print(f"\n? UMAP {n_components}D 차원 축소...")

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    coords = reducer.fit_transform(combined_emb)

    print(f"   UMAP 완료: {coords.shape}")

    # 결과 DataFrame 생성

    result_dict = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "boundary_type": labels,
        "cluster_id": cluster_ids,
    }

    if n_components == 3:

        result_dict["z"] = coords[:, 2]

    result_df = pd.DataFrame(result_dict)

    return result_df

def generate_2d_viz(
    result_df: pd.DataFrame, output_path: Path, grayscale: bool = False
):
    """Sentence/Phrase 임베딩 2D 오버레이 시각화 HTML 생성

    Args:

        grayscale: True일 경우 흑백 인쇄용 스타일 적용

    """

    palette = COLOR_PALETTES["grayscale"] if grayscale else COLOR_PALETTES["color"]

    sentence_colors = palette["sentence"]

    phrase_colors = palette["phrase"]

    traces_data = []

    for boundary_type in ["Sentence", "Phrase"]:

        colors = sentence_colors if boundary_type == "Sentence" else phrase_colors

        symbols = (
            GRAYSCALE_SYMBOLS["sentence" if boundary_type == "Sentence" else "phrase"]
            if grayscale
            else None
        )

        type_df = result_df[result_df["boundary_type"] == boundary_type]

        for i, cluster_id in enumerate(sorted(type_df["cluster_id"].unique())):

            cluster_df = type_df[type_df["cluster_id"] == cluster_id]

            trace = {
                "x": cluster_df["x"].tolist(),
                "y": cluster_df["y"].tolist(),
                "name": f"{boundary_type}-p{int(cluster_id)}",
                "color": colors[int(cluster_id) % len(colors)],
                "type": boundary_type,
            }

            if symbols:

                trace["symbol"] = symbols[i % len(symbols)]

            traces_data.append(trace)

    # 흑백 모드 여부에 따른 스타일 변수

    bg_style = palette["bg_gradient"]

    text_color = palette["text"]

    subtitle_color = palette["subtitle"]

    grid_color = palette["grid"]

    legend_bg = palette["legend_bg"]

    # 흑백 모드일 때 배경을 단색으로

    bg_css = f"background: {bg_style};" if not grayscale else f"background: {bg_style};"

    title_style = (
        "color: #333;"
        if grayscale
        else "background: linear-gradient(90deg, #e94560, #4ecca3); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;"
    )

    html_content = f"""<!DOCTYPE html>

<html lang="ko">

<head>

    <meta charset="UTF-8">

    <title>Sentence/Phrase K=3 임베딩 오버레이 2D{" (흑백)" if grayscale else ""}</title>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{

            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;

            {bg_css}

            min-height: 100vh;

            color: {text_color};

            padding: 20px;

        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        h1 {{

            text-align: center;

            font-size: 2rem;

            margin-bottom: 10px;

            {title_style}

        }}

        .subtitle {{ text-align: center; color: {subtitle_color}; margin-bottom: 20px; }}

        #scatter-plot {{ width: 100%; height: 700px; }}

        .controls {{

            display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap;

        }}

        .control-group {{

            background: rgba(0,0,0,0.03); border: 1px solid #ddd; padding: 12px 20px; border-radius: 10px;

        }}

        .control-group label {{ color: #333; font-weight: bold; margin-right: 10px; }}

        .control-group select, .control-group input {{

            background: #fff; border: 1px solid #ccc;

            color: #000; padding: 6px 12px; border-radius: 6px;

        }}

    </style>

</head>

<body>

    <div class="container">

        <h1>Sentence/Phrase K=3 임베딩 오버레이 (2D UMAP)</h1>

        <p class="subtitle">정규화 반영 | Sentence: 실선 마커, Phrase: 삼각형·별형 마커</p>

        <div class="controls">

            <div class="control-group">

                <label>포인트 크기:</label>

                <input type="range" id="point-size" min="2" max="14" value="6" onchange="updatePlot()">

            </div>

            <div class="control-group">

                <label>투명도:</label>

                <input type="range" id="opacity" min="10" max="100" value="80" onchange="updatePlot()">

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

                    const isSentence = t.type === 'Sentence';

                    const markerConfig = {{

                        size: pointSize,

                        symbol: t.symbol || 'circle',

                        opacity: isSentence ? opacity : opacity * 0.9,

                        color: isSentence ? t.color : 'rgba(255,255,255,0)',

                        line: {{ color: t.color, width: isSentence ? 1 : 2.5 }}

                    }};

                    traces.push({{

                        x: t.x, y: t.y, mode: 'markers', type: 'scatter', name: t.name,

                        marker: markerConfig,

                        hovertemplate: '<b>' + t.name + '</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'

                    }});

                }}

            }}

            Plotly.react('scatter-plot', traces, {{

                paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',

                xaxis: {{ title: 'UMAP-1', color: '#000', gridcolor: '{grid_color}' }},

                yaxis: {{ title: 'UMAP-2', color: '#000', gridcolor: '{grid_color}' }},

                legend: {{ font: {{ color: '#000' }}, bgcolor: '{legend_bg}' }},

                margin: {{ t: 30, b: 60, l: 60, r: 30 }}

            }}, {{ responsive: true }});

        }}

        updatePlot();

    </script>

</body>

</html>"""

    output_path.write_text(html_content, encoding="utf-8")

    print(f"? 2D 시각화 저장: {output_path}")

def generate_3d_viz(
    result_df: pd.DataFrame, output_path: Path, grayscale: bool = False
):
    """Sentence/Phrase 임베딩 3D 오버레이 시각화 HTML 생성 (실제 UMAP 3D)"""

    palette = COLOR_PALETTES["grayscale"] if grayscale else COLOR_PALETTES["color"]

    sentence_colors = palette["sentence"]

    phrase_colors = palette["phrase"]

    traces_data = []

    for boundary_type in ["Sentence", "Phrase"]:

        colors = sentence_colors if boundary_type == "Sentence" else phrase_colors

        symbols = (
            GRAYSCALE_SYMBOLS["sentence" if boundary_type == "Sentence" else "phrase"]
            if grayscale
            else None
        )

        type_df = result_df[result_df["boundary_type"] == boundary_type]

        for i, cluster_id in enumerate(sorted(type_df["cluster_id"].unique())):

            cluster_df = type_df[type_df["cluster_id"] == cluster_id]

            trace = {
                "x": cluster_df["x"].tolist(),
                "y": cluster_df["y"].tolist(),
                "z": cluster_df["z"].tolist(),
                "name": f"{boundary_type}-p{int(cluster_id)}",
                "color": colors[int(cluster_id) % len(colors)],
                "type": boundary_type,
            }

            if symbols:

                trace["symbol"] = symbols[i % len(symbols)]

            traces_data.append(trace)

    html_content = f"""<!DOCTYPE html>

<html lang="ko">

<head>

    <meta charset="UTF-8">

    <title>Sentence/Phrase K=3 임베딩 오버레이 3D</title>

    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{

            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;

            background: #ffffff;

            min-height: 100vh;

            color: #000;

            padding: 20px;

        }}

        .container {{ max-width: 1600px; margin: 0 auto; }}

        h1 {{

            text-align: center;

            font-size: 2rem;

            margin-bottom: 10px;

            color: #333;

        }}

        .subtitle {{ text-align: center; color: #666; margin-bottom: 20px; }}

        #scatter-3d {{ width: 100%; height: 800px; }}

        .controls {{

            display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap;

        }}

        .control-group {{

            background: rgba(0,0,0,0.03); border: 1px solid #ddd; padding: 12px 20px; border-radius: 10px;

        }}

        .control-group label {{ color: #333; font-weight: bold; margin-right: 10px; }}

        .control-group select, .control-group input {{

            background: #fff; border: 1px solid #ccc;

            color: #000; padding: 6px 12px; border-radius: 6px;

        }}

        .insight-box {{

            background: rgba(0,0,0,0.03);

            border: 1px solid #ccc;

            border-radius: 12px;

            padding: 20px;

            margin-top: 30px;

        }}

        .insight-box h3 {{ color: #333; margin-bottom: 15px; }}

        .insight-item {{

            padding: 10px 15px;

            background: rgba(0,0,0,0.04);

            border-radius: 8px;

            margin: 10px 0;

            border-left: 3px solid #555;

        }}

    </style>

</head>

<body>

    <div class="container">

        <h1>Sentence/Phrase K=3 임베딩 오버레이 (3D UMAP)</h1>

        <p class="subtitle">실제 3차원 UMAP 축소 | 마우스 드래그로 회전 | 그레이스케일 마커 구분</p>

        <div class="controls">

            <div class="control-group">

                <label>포인트 크기:</label>

                <input type="range" id="point-size" min="1" max="12" value="5" onchange="updatePlot()">

            </div>

            <div class="control-group">

                <label>투명도:</label>

                <input type="range" id="opacity" min="20" max="100" value="85" onchange="updatePlot()">

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

        <div id="scatter-3d"></div>

        <div class="insight-box">

            <h3>3D 시각화 해석</h3>

            <div class="insight-item">

                <strong>3개 클러스터 분리</strong>: 인위적 z축 없이도 UMAP 3D에서 자연스럽게 3개 영역이 형성됨

            </div>

            <div class="insight-item">

                <strong>Sentence-Phrase 대응</strong>: 유사한 의미역의 Sentence/Phrase 클러스터가 3D 공간에서 가까이 위치

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

                if (showMode === 'both' || (showMode === 'sentence' && t.type === 'Sentence') || (showMode === 'phrase' && t.type === 'Phrase')) {{

                    const isSentence = t.type === 'Sentence';

                    const markerConfig = {{

                        size: pointSize,

                        symbol: t.symbol || 'circle',

                        opacity: isSentence ? opacity : opacity * 0.9,

                        color: isSentence ? t.color : 'rgba(255,255,255,0)',

                        line: {{ color: t.color, width: isSentence ? 1 : 2.5 }}

                    }};

                    traces.push({{

                        x: t.x, y: t.y, z: t.z, mode: 'markers', type: 'scatter3d', name: t.name,

                        marker: markerConfig,

                        hovertemplate: '<b>' + t.name + '</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>'

                    }});

                }}

            }}

            Plotly.react('scatter-3d', traces, {{

                paper_bgcolor: '#ffffff',

                scene: {{

                    xaxis: {{ title: 'UMAP-1', color: '#000', gridcolor: 'rgba(0,0,0,0.1)' }},

                    yaxis: {{ title: 'UMAP-2', color: '#000', gridcolor: 'rgba(0,0,0,0.1)' }},

                    zaxis: {{ title: 'UMAP-3', color: '#000', gridcolor: 'rgba(0,0,0,0.1)' }},

                    bgcolor: '#ffffff'

                }},

                legend: {{ font: {{ color: '#000' }}, bgcolor: 'rgba(255,255,255,0.9)', x: 0.02, y: 0.98 }},

                margin: {{ t: 30, b: 30, l: 30, r: 30 }}

            }}, {{ responsive: true }});

        }}

        updatePlot();

    </script>

</body>

</html>"""

    output_path.write_text(html_content, encoding="utf-8")

    print(f"? 3D 시각화 저장: {output_path}")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sample", type=int, default=10000, help="샘플 크기")

    parser.add_argument("--dim", type=int, default=3, choices=[2, 3], help="UMAP 차원")

    parser.add_argument(
        "--grayscale",
        action="store_true",
        default=True,
        help="흑백 인쇄용 시각화 생성 (기본값: True)",
    )
    parser.add_argument(
        "--cache-npy",
        type=Path,
        default=Path("hyeonto") / "results" / "combined_embeddings_cache.npy",
        help="통합 임베딩 캐시 npy 경로",
    )
    parser.add_argument(
        "--sentence-cache-npy",
        type=Path,
        default=Path("hyeonto") / "results" / "sentence_embeddings_cache.npy",
        help="Sentence 임베딩 캐시 npy 경로",
    )
    parser.add_argument(
        "--phrase-cache-npy",
        type=Path,
        default=Path("hyeonto") / "results" / "phrase_embeddings_cache.npy",
        help="Phrase 임베딩 캐시 npy 경로",
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        choices=["1-1", "3-1"],
        default="1-1",
        help="가중치 모드: 1-1(균등) 또는 3-1(Canon3)",
    )
    parser.add_argument(
        "--allow-embed",
        action="store_true",
        help="캐시가 없거나 불일치 시 재임베딩 허용",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # 가중치 모드에 따른 경로 설정
    if args.weight_mode == "1-1":
        reports_dir = base_dir / "report_1-1"
        sentence_subdir = "sentence_k3_normalized"
        phrase_subdir = "phrase_k3_normalized"
        viz_subdir = "visualizations_k3"
    else:  # 3-1
        reports_dir = base_dir / "report_3-1"
        sentence_subdir = "sentence_k3_normalized"
        phrase_subdir = "phrase_k3_normalized"
        viz_subdir = "visualizations_k3"

    # 정규화된 K=3 클러스터 데이터
    sentence_path = reports_dir / sentence_subdir / "sentence_clusters.csv"
    phrase_path = reports_dir / phrase_subdir / "phrase_clusters.csv"

    if not sentence_path.exists() or not phrase_path.exists():
        print(f"K=3 클러스터 데이터가 없습니다 (가중치 모드: {args.weight_mode})")
        print(f"   필요: {sentence_path}")
        print(f"   필요: {phrase_path}")
        return

    mode_label = f" [{args.weight_mode}]"
    if args.grayscale:
        mode_label += " (흑백 인쇄용)"

    # 2D와 3D 모두 생성

    for dim in [2, 3]:

        print(f"\n{'='*60}")

        print(f"? {dim}D UMAP 실행{mode_label}")

        print(f"{'='*60}")

        sentence_cache = (
            args.sentence_cache_npy if args.sentence_cache_npy.exists() else None
        )
        phrase_cache = args.phrase_cache_npy if args.phrase_cache_npy.exists() else None
        cache_npy = args.cache_npy if args.cache_npy.exists() else None

        result_df = load_embeddings_and_generate_umap(
            sentence_path,
            phrase_path,
            args.sample,
            dim,
            cache_npy=cache_npy,
            sentence_cache_npy=sentence_cache,
            phrase_cache_npy=phrase_cache,
            allow_embed=args.allow_embed,
        )

        if result_df is None:

            continue

        # 출력 경로를 가중치 모드 폴더로
        output_dir = reports_dir / viz_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if dim == 2:

            output_path = output_dir / "k3_embedding_overlay_2d.html"

            generate_2d_viz(result_df, output_path, grayscale=args.grayscale)

            csv_path = output_dir / "k3_embedding_overlay_2d.csv"

        else:

            output_path = output_dir / "k3_embedding_overlay_3d.html"

            generate_3d_viz(result_df, output_path, grayscale=args.grayscale)

            csv_path = output_dir / "k3_embedding_overlay_3d.csv"

        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"? 좌표 데이터 저장: {csv_path}")

    print(f"\n? Sentence/Phrase K=3 임베딩 오버레이 시각화 완료! (2D + 3D){mode_label}")

if __name__ == "__main__":

    main()
