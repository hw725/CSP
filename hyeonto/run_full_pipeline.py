#!/usr/bin/env python3
"""
현토 분석 전체 파이프라인 (원본에서 시작)

sentence_full.csv, phrase_full.csv에서 시작하여:
1. 임베딩 생성 (BGE-M3)
2. 클러스터링 (K=4, 14, 24)
3. 정규화 적용
4. 클러스터 프로파일 생성

입력: datasets/sentence_full.csv, datasets/phrase_full.csv
출력: reports/sentence_k4_normalized/, etc.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import regex
from tqdm import tqdm

# 현토 정규화 모듈
from hyeonto_normalizer import normalize_hyeonto_marker

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"


def extract_markers_from_text(text: str) -> list:
    """텍스트에서 현토 마커 추출"""
    if pd.isna(text):
        return []
    return regex.findall(r'\p{Hangul}+', str(text))


def normalize_markers(markers: list) -> str:
    """마커 리스트를 정규화하여 쉼표로 연결"""
    if not markers:
        return ''
    normalized = [normalize_hyeonto_marker(m) for m in markers]
    return ','.join(normalized)


def prepare_data_for_clustering(input_path: Path, output_dir: Path, boundary_type: str):
    """원본 데이터에서 클러스터링용 데이터 준비"""
    print(f"\n{'='*60}")
    print(f"📂 {boundary_type} 데이터 준비")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_path)
    print(f"  원본 행수: {len(df):,}")
    
    # 마커 추출 및 정규화
    if 'marker' not in df.columns:
        df['marker'] = df['원문'].apply(lambda x: ','.join(extract_markers_from_text(x)))
    
    df['marker_normalized'] = df['marker'].apply(
        lambda x: normalize_markers(x.split(',')) if pd.notna(x) and x else ''
    )
    
    # 잇고/잇가 보존 확인
    original_잇고 = df['원문'].astype(str).str.contains('잇고').sum()
    original_잇가 = df['원문'].astype(str).str.contains('잇가').sum()
    print(f"  잇고 보존: {original_잇고}")
    print(f"  잇가 보존: {original_잇가}")
    
    return df


def run_embedding_and_clustering(df: pd.DataFrame, k_values: list, output_dir: Path, boundary_type: str):
    """임베딩 생성 및 클러스터링"""
    print(f"\n{'='*60}")
    print(f"🔬 {boundary_type} 임베딩 & 클러스터링")
    print(f"{'='*60}")
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        print(f"❌ 필요한 라이브러리 없음: {e}")
        return None
    
    # 임베딩용 텍스트 준비 (원문 사용)
    texts = df['원문'].fillna('').tolist()
    
    print(f"\n📊 BGE-M3 임베딩 생성 중... ({len(texts):,}개)")
    
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # 배치 처리
    batch_size = 256
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="임베딩"):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, max_length=512)['dense_vecs']
        all_embeddings.append(embeddings)
    
    embeddings = np.vstack(all_embeddings)
    print(f"✅ 임베딩 완료: shape={embeddings.shape}")
    
    results = {}
    
    for k in k_values:
        print(f"\n🔹 K={k} 클러스터링...")
        
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
        labels = kmeans.fit_predict(embeddings)
        
        df_result = df.copy()
        df_result['cluster_id'] = labels
        
        # 클러스터 크기 계산
        cluster_sizes = df_result['cluster_id'].value_counts().to_dict()
        df_result['cluster_size'] = df_result['cluster_id'].map(cluster_sizes)
        
        # 저장
        k_dir = output_dir / f"{boundary_type.lower()}_k{k}_normalized"
        k_dir.mkdir(parents=True, exist_ok=True)
        
        result_path = k_dir / f"{boundary_type.lower()}_clusters.csv"
        df_result.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        print(f"  ✅ 저장: {result_path}")
        
        # 클러스터 분포
        dist = pd.Series(labels).value_counts().sort_index()
        print(f"  분포: {dict(dist)}")
        
        # 프로파일 생성
        generate_cluster_profile(df_result, k, output_dir, boundary_type)
        
        results[k] = df_result
    
    return results


def generate_cluster_profile(df: pd.DataFrame, k: int, output_dir: Path, boundary_type: str):
    """클러스터 프로파일 생성"""
    k_dir = output_dir / f"{boundary_type.lower()}_k{k}_normalized"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        f"# {boundary_type} K={k} 클러스터 프로파일",
        "",
        f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**총 행수**: {len(df):,}",
        "",
        "---",
        "",
        "## 클러스터 분포",
        "",
        "| Cluster | 크기 | 비율 |",
        "|:-------:|-----:|-----:|",
    ]
    
    total = len(df)
    for cluster_id in sorted(df['cluster_id'].unique()):
        count = len(df[df['cluster_id'] == cluster_id])
        pct = count / total * 100
        lines.append(f"| {cluster_id} | {count:,} | {pct:.1f}% |")
    
    profile_path = k_dir / f"{boundary_type.lower()}_cluster_profile.md"
    with open(profile_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  📝 프로파일: {profile_path}")


def main():
    parser = argparse.ArgumentParser(description="현토 분석 전체 파이프라인")
    parser.add_argument('--boundary', choices=['sentence', 'phrase', 'both'], default='both',
                        help="분석할 경계 유형")
    parser.add_argument('--sentence-k', type=int, nargs='+', default=[4, 14],
                        help="Sentence 클러스터 K 값들")
    parser.add_argument('--phrase-k', type=int, nargs='+', default=[4, 24],
                        help="Phrase 클러스터 K 값들")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 현토 분석 전체 파이프라인 (원본에서 시작)")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sentence 분석
    if args.boundary in ['sentence', 'both']:
        sentence_path = DATASETS_DIR / "sentence_full.csv"
        if sentence_path.exists():
            sentence_df = prepare_data_for_clustering(sentence_path, REPORTS_DIR, 'Sentence')
            
            # 정규화된 데이터 저장
            norm_path = DATASETS_DIR / "sentence_normalized.csv"
            sentence_df.to_csv(norm_path, index=False, encoding='utf-8-sig')
            print(f"  📁 정규화 저장: {norm_path}")
            
            run_embedding_and_clustering(sentence_df, args.sentence_k, REPORTS_DIR, 'Sentence')
        else:
            print(f"❌ 파일 없음: {sentence_path}")
    
    # Phrase 분석
    if args.boundary in ['phrase', 'both']:
        phrase_path = DATASETS_DIR / "phrase_full.csv"
        if phrase_path.exists():
            phrase_df = prepare_data_for_clustering(phrase_path, REPORTS_DIR, 'Phrase')
            
            # 정규화된 데이터 저장
            norm_path = DATASETS_DIR / "phrase_normalized.csv"
            phrase_df.to_csv(norm_path, index=False, encoding='utf-8-sig')
            print(f"  📁 정규화 저장: {norm_path}")
            
            run_embedding_and_clustering(phrase_df, args.phrase_k, REPORTS_DIR, 'Phrase')
        else:
            print(f"❌ 파일 없음: {phrase_path}")
    
    print("\n" + "=" * 70)
    print("✅ 파이프라인 완료!")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
