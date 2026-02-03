#!/usr/bin/env python3

"""

현토 분석 시각화 - 흑백 인쇄용 (Grayscale/Print-Friendly)



모든 시각화를 학술 논문 인쇄에 적합한 형태로 생성:

- 흑백/회색조 색상 팔레트

- 패턴/해치로 구분 (색맹 친화적)

- 고대비 선 스타일

- 마커 모양으로 카테고리 구분



생성 파일:

- UMAP 2D/3D 시각화 (흑백)

- 클러스터 분포 차트 (흑백)

- 히트맵 (흑백)

"""

import pandas as pd

import numpy as np

from pathlib import Path

from datetime import datetime

import json



# 시각화 라이브러리

import matplotlib

matplotlib.use('Agg')  # GUI 없이 실행

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from matplotlib.lines import Line2D



# 흑백 스타일 설정

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({

    'font.family': 'Malgun Gothic',

    'font.size': 10,

    'axes.labelsize': 11,

    'axes.titlesize': 12,

    'legend.fontsize': 9,

    'figure.dpi': 150,

    'savefig.dpi': 300,

    'savefig.bbox': 'tight',

    'axes.edgecolor': 'black',

    'axes.linewidth': 1.0,

})



BASE_DIR = Path(__file__).parent

REPORTS_DIR = BASE_DIR / "reports"



# 흑백 팔레트 (4개 클러스터용)

GRAYSCALE_COLORS = ['#000000', '#555555', '#999999', '#CCCCCC']

GRAYSCALE_MARKERS = ['o', 's', '^', 'D']  # 원, 사각, 삼각, 다이아몬드

GRAYSCALE_HATCHES = ['/', '\\', 'x', '.']





def load_cluster_data(boundary: str = 'sentence', k: int = 4) -> pd.DataFrame:

    """클러스터 데이터 로드"""

    path = REPORTS_DIR / f"{boundary}_k{k}_normalized" / f"{boundary}_clusters.csv"

    if not path.exists():

        print(f"? 파일 없음: {path}")

        return None

    return pd.read_csv(path)





def generate_cluster_distribution_chart(df: pd.DataFrame, boundary: str, output_dir: Path):

    """클러스터 분포 막대 차트 (흑백)"""

    print(f"  ? 클러스터 분포 차트 생성...")

    

    cluster_counts = df['cluster_id'].value_counts().sort_index()

    clusters = list(cluster_counts.index)

    counts = list(cluster_counts.values)

    

    fig, ax = plt.subplots(figsize=(10, 6))

    

    bars = ax.bar(range(len(clusters)), counts, 

                  color=GRAYSCALE_COLORS[:len(clusters)],

                  edgecolor='black', linewidth=1.2)

    

    # 해치 패턴 추가

    for bar, hatch in zip(bars, GRAYSCALE_HATCHES[:len(clusters)]):

        bar.set_hatch(hatch)

    

    ax.set_xlabel('클러스터 ID')

    ax.set_ylabel('데이터 수')

    ax.set_title(f'{boundary.capitalize()} 클러스터 분포 (K={len(clusters)})')

    ax.set_xticks(range(len(clusters)))

    ax.set_xticklabels([f'C{c}' for c in clusters])

    

    # 범례

    legend_elements = [

        mpatches.Patch(facecolor=color, edgecolor='black', hatch=hatch, 

                       label=f'C{c}: {count:,}건')

        for c, color, hatch, count in zip(clusters, GRAYSCALE_COLORS, GRAYSCALE_HATCHES, counts)

    ]

    ax.legend(handles=legend_elements, loc='upper right')

    

    # 저장

    output_path = output_dir / f"{boundary}_cluster_distribution_bw.png"

    plt.savefig(output_path)

    plt.close()

    print(f"    ? 저장: {output_path}")





def generate_book_distribution_heatmap(df: pd.DataFrame, boundary: str, output_dir: Path):

    """서적별 클러스터 분포 히트맵 (흑백)"""

    print(f"  ? 서적-클러스터 히트맵 생성...")

    

    # 서적별 클러스터 분포 계산

    pivot = pd.crosstab(df['book'], df['cluster_id'], normalize='index') * 100

    

    # 상위 15개 서적만 표시

    book_counts = df['book'].value_counts()

    top_books = book_counts.head(15).index.tolist()

    pivot = pivot.loc[pivot.index.isin(top_books)]

    pivot = pivot.reindex(top_books)

    

    fig, ax = plt.subplots(figsize=(10, 8))

    

    # 흑백 컬러맵

    im = ax.imshow(pivot.values, cmap='Greys', aspect='auto', vmin=0, vmax=50)

    

    # 축 설정

    ax.set_xticks(range(len(pivot.columns)))

    ax.set_xticklabels([f'C{c}' for c in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))

    ax.set_yticklabels(pivot.index)

    

    # 값 표시

    for i in range(len(pivot.index)):

        for j in range(len(pivot.columns)):

            val = pivot.iloc[i, j]

            color = 'white' if val > 25 else 'black'

            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 

                   color=color, fontsize=8)

    

    ax.set_xlabel('클러스터 ID')

    ax.set_ylabel('서적명')

    ax.set_title(f'{boundary.capitalize()} 서적-클러스터 분포 (%)')

    

    # 컬러바

    cbar = plt.colorbar(im, ax=ax)

    cbar.set_label('비율 (%)')

    

    # 저장

    output_path = output_dir / f"{boundary}_book_cluster_heatmap_bw.png"

    plt.savefig(output_path)

    plt.close()

    print(f"    ? 저장: {output_path}")





def generate_umap_visualization(df: pd.DataFrame, boundary: str, output_dir: Path):

    """UMAP 2D 시각화 (흑백, 샘플링)"""

    print(f"  ? UMAP 시각화 생성 중...")

    

    try:

        import umap

    except ImportError:

        print("    ?? umap-learn 미설치. 스킵.")

        return

    

    # 샘플링 (최대 5000개)

    sample_size = min(5000, len(df))

    df_sample = df.sample(n=sample_size, random_state=42)

    

    # 텍스트 임베딩 생성 (간단한 TF-IDF)

    try:

        from sklearn.feature_extraction.text import TfidfVectorizer

        

        texts = df_sample['left_sentence'].fillna('').tolist()

        vectorizer = TfidfVectorizer(max_features=500, analyzer='char', ngram_range=(1,3))

        embeddings = vectorizer.fit_transform(texts).toarray()

        

        # UMAP 차원 축소

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)

        coords = reducer.fit_transform(embeddings)

        

        df_sample['umap_x'] = coords[:, 0]

        df_sample['umap_y'] = coords[:, 1]

        

    except Exception as e:

        print(f"    ?? UMAP 실패: {e}")

        return

    

    # 시각화

    fig, ax = plt.subplots(figsize=(12, 10))

    

    clusters = sorted(df_sample['cluster_id'].unique())

    

    for i, cluster_id in enumerate(clusters):

        mask = df_sample['cluster_id'] == cluster_id

        ax.scatter(

            df_sample.loc[mask, 'umap_x'],

            df_sample.loc[mask, 'umap_y'],

            c=GRAYSCALE_COLORS[i % len(GRAYSCALE_COLORS)],

            marker=GRAYSCALE_MARKERS[i % len(GRAYSCALE_MARKERS)],

            s=30,

            alpha=0.6,

            edgecolors='black',

            linewidth=0.3,

            label=f'C{cluster_id}'

        )

    

    ax.set_xlabel('UMAP 1')

    ax.set_ylabel('UMAP 2')

    ax.set_title(f'{boundary.capitalize()} UMAP 시각화 (K={len(clusters)}, n={sample_size:,})')

    

    # 범례

    legend_elements = [

        Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,

               markeredgecolor='black', markersize=10, label=f'C{c}')

        for c, color, marker in zip(clusters, GRAYSCALE_COLORS, GRAYSCALE_MARKERS)

    ]

    ax.legend(handles=legend_elements, loc='upper right')

    

    # 저장

    output_path = output_dir / f"{boundary}_umap_2d_bw.png"

    plt.savefig(output_path)

    plt.close()

    print(f"    ? 저장: {output_path}")





def generate_marker_frequency_chart(df: pd.DataFrame, boundary: str, output_dir: Path):

    """현토 마커 빈도 차트 (흑백)"""

    print(f"  ? 현토 마커 빈도 차트 생성...")

    

    # 마커 추출

    if 'marker_normalized' in df.columns:

        all_markers = df['marker_normalized'].dropna().str.split(',').explode()

    elif 'marker' in df.columns:

        all_markers = df['marker'].dropna().str.split(',').explode()

    else:

        print("    ?? 마커 컬럼 없음. 스킵.")

        return

    

    # 상위 20개 마커

    marker_counts = all_markers.value_counts().head(20)

    

    fig, ax = plt.subplots(figsize=(12, 6))

    

    # 그라데이션 색상 생성

    n_bars = len(marker_counts)

    grays = [plt.cm.Greys(0.3 + 0.5 * i / n_bars) for i in range(n_bars)]

    

    bars = ax.barh(range(len(marker_counts)), marker_counts.values, 

                   color=grays, edgecolor='black', linewidth=0.8)

    

    ax.set_yticks(range(len(marker_counts)))

    ax.set_yticklabels(marker_counts.index)

    ax.set_xlabel('빈도')

    ax.set_ylabel('현토 마커')

    ax.set_title(f'{boundary.capitalize()} 상위 20 현토 마커')

    ax.invert_yaxis()

    

    # 값 라벨

    for i, (bar, count) in enumerate(zip(bars, marker_counts.values)):

        ax.text(bar.get_width() + max(marker_counts) * 0.01, bar.get_y() + bar.get_height()/2,

               f'{count:,}', va='center', fontsize=8)

    

    # 저장

    output_path = output_dir / f"{boundary}_marker_frequency_bw.png"

    plt.savefig(output_path)

    plt.close()

    print(f"    ? 저장: {output_path}")





def generate_cluster_marker_profile(df: pd.DataFrame, boundary: str, output_dir: Path):

    """클러스터별 마커 프로파일 (흑백)"""

    print(f"  ? 클러스터별 마커 프로파일 생성...")

    

    if 'marker_normalized' not in df.columns and 'marker' not in df.columns:

        print("    ?? 마커 컬럼 없음. 스킵.")

        return

    

    marker_col = 'marker_normalized' if 'marker_normalized' in df.columns else 'marker'

    clusters = sorted(df['cluster_id'].unique())

    

    # 클러스터별 top 5 마커

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    

    for idx, cluster_id in enumerate(clusters[:4]):

        cluster_df = df[df['cluster_id'] == cluster_id]

        markers = cluster_df[marker_col].dropna().str.split(',').explode()

        top_markers = markers.value_counts().head(10)

        

        ax = axes[idx]

        bars = ax.barh(range(len(top_markers)), top_markers.values,

                      color=GRAYSCALE_COLORS[idx], edgecolor='black', linewidth=0.8,

                      hatch=GRAYSCALE_HATCHES[idx])

        

        ax.set_yticks(range(len(top_markers)))

        ax.set_yticklabels(top_markers.index)

        ax.set_xlabel('빈도')

        ax.set_title(f'C{cluster_id} 상위 마커')

        ax.invert_yaxis()

    

    plt.suptitle(f'{boundary.capitalize()} 클러스터별 현토 마커 프로파일', fontsize=14)

    plt.tight_layout()

    

    # 저장

    output_path = output_dir / f"{boundary}_cluster_marker_profile_bw.png"

    plt.savefig(output_path)

    plt.close()

    print(f"    ? 저장: {output_path}")





def main():

    print("="*70)

    print("? 현토 분석 시각화 (흑백 인쇄용)")

    print("="*70)

    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    

    # 출력 디렉토리

    viz_dir = REPORTS_DIR / "visualizations_bw"

    viz_dir.mkdir(parents=True, exist_ok=True)

    

    # Sentence 시각화

    print("\n? Sentence 시각화")

    sentence_df = load_cluster_data('sentence', 4)

    if sentence_df is not None:

        generate_cluster_distribution_chart(sentence_df, 'sentence', viz_dir)

        generate_book_distribution_heatmap(sentence_df, 'sentence', viz_dir)

        generate_marker_frequency_chart(sentence_df, 'sentence', viz_dir)

        generate_cluster_marker_profile(sentence_df, 'sentence', viz_dir)

        generate_umap_visualization(sentence_df, 'sentence', viz_dir)

    

    # Phrase 시각화

    print("\n? Phrase 시각화")

    phrase_df = load_cluster_data('phrase', 4)

    if phrase_df is not None:

        generate_cluster_distribution_chart(phrase_df, 'phrase', viz_dir)

        generate_book_distribution_heatmap(phrase_df, 'phrase', viz_dir)

        generate_marker_frequency_chart(phrase_df, 'phrase', viz_dir)

        generate_cluster_marker_profile(phrase_df, 'phrase', viz_dir)

        generate_umap_visualization(phrase_df, 'phrase', viz_dir)

    

    print("\n" + "="*70)

    print("? 시각화 완료!")

    print(f"출력 디렉토리: {viz_dir}")

    print("="*70)





if __name__ == "__main__":

    main()

