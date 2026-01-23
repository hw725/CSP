#!/usr/bin/env python3
"""배치 처리 결과 전체 집계 및 통계 분석"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np

def collect_all_evaluations(results_dir: Path) -> dict:
    """모든 책의 평가 결과 수집"""
    
    all_pa_evals = []
    all_sa_evals = []
    
    for book_dir in results_dir.iterdir():
        if not book_dir.is_dir():
            continue
        
        book_name = book_dir.name
        
        # PA 평가 결과 찾기
        pa_eval_files = list(book_dir.glob("*_PA_eval_*.xlsx"))
        for pa_file in pa_eval_files:
            try:
                df = pd.read_excel(pa_file, sheet_name='전체_요약')
                if not df.empty:
                    summary = df.iloc[0].to_dict()
                    summary['책이름'] = book_name
                    summary['평가파일'] = str(pa_file)
                    all_pa_evals.append(summary)
            except Exception as e:
                print(f"⚠️  PA 평가 읽기 실패: {book_name} - {e}")
        
        # SA 평가 결과 찾기
        sa_eval_files = list(book_dir.glob("*_SA_eval_*.xlsx"))
        for sa_file in sa_eval_files:
            try:
                df = pd.read_excel(sa_file, sheet_name='전체_요약')
                if not df.empty:
                    summary = df.iloc[0].to_dict()
                    summary['책이름'] = book_name
                    summary['평가파일'] = str(sa_file)
                    all_sa_evals.append(summary)
            except Exception as e:
                print(f"⚠️  SA 평가 읽기 실패: {book_name} - {e}")
    
    return {
        'pa_evaluations': all_pa_evals,
        'sa_evaluations': all_sa_evals
    }

def compute_statistics(df: pd.DataFrame, metric_cols: list) -> dict:
    """주요 지표 통계 계산"""
    
    stats = {}
    
    for col in metric_cols:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                stats[col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'q90': float(data.quantile(0.90)),
                    'count': int(len(data))
                }
    
    return stats

def generate_report(results_dir: Path, output_dir: Path) -> None:
    """전체 평가 결과 보고서 생성"""
    
    print("="*70)
    print("📊 배치 처리 결과 집계 시작")
    print("="*70)
    
    # 결과 수집
    data = collect_all_evaluations(results_dir)
    
    pa_df = pd.DataFrame(data['pa_evaluations'])
    sa_df = pd.DataFrame(data['sa_evaluations'])
    
    print(f"\n수집된 평가 결과:")
    print(f"  PA: {len(pa_df)}개 책")
    print(f"  SA: {len(sa_df)}개 책")
    
    # PA 통계
    pa_metrics = [
        'avg_exact_match', 'avg_segment_count_match', 'avg_text_match',
        'avg_source_text_match', 'avg_target_text_match',
        'avg_partial_match', 'avg_target_avg_similarity'
    ]
    
    pa_stats = compute_statistics(pa_df, pa_metrics)
    
    # SA 통계
    sa_metrics = [
        'avg_exact_match', 'avg_segment_count_match', 'avg_text_match',
        'avg_source_text_match', 'avg_target_text_match',
        'avg_partial_match', 'avg_target_avg_similarity'
    ]
    
    sa_stats = compute_statistics(sa_df, sa_metrics)
    
    # 통계 출력
    print("\n" + "="*70)
    print("📈 PA 평가 지표 통계")
    print("="*70)
    for metric, stat in pa_stats.items():
        print(f"\n{metric}:")
        print(f"  평균: {stat['mean']:.4f}, 중간값: {stat['median']:.4f}")
        print(f"  표준편차: {stat['std']:.4f}")
        print(f"  범위: [{stat['min']:.4f}, {stat['max']:.4f}]")
        print(f"  사분위: Q25={stat['q25']:.4f}, Q75={stat['q75']:.4f}, Q90={stat['q90']:.4f}")
    
    print("\n" + "="*70)
    print("📈 SA 평가 지표 통계")
    print("="*70)
    for metric, stat in sa_stats.items():
        print(f"\n{metric}:")
        print(f"  평균: {stat['mean']:.4f}, 중간값: {stat['median']:.4f}")
        print(f"  표준편차: {stat['std']:.4f}")
        print(f"  범위: [{stat['min']:.4f}, {stat['max']:.4f}]")
        print(f"  사분위: Q25={stat['q25']:.4f}, Q75={stat['q75']:.4f}, Q90={stat['q90']:.4f}")
    
    # 결과 저장
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # CSV 저장
    if not pa_df.empty:
        pa_csv = output_dir / 'pa_전체평가결과.csv'
        pa_df.to_csv(pa_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ PA 전체 결과 저장: {pa_csv}")
    
    if not sa_df.empty:
        sa_csv = output_dir / 'sa_전체평가결과.csv'
        sa_df.to_csv(sa_csv, index=False, encoding='utf-8-sig')
        print(f"✅ SA 전체 결과 저장: {sa_csv}")
    
    # 통계 JSON 저장
    report = {
        'generated_at': datetime.now().isoformat(),
        'pa_statistics': pa_stats,
        'sa_statistics': sa_stats,
        'summary': {
            'total_books_pa': len(pa_df),
            'total_books_sa': len(sa_df)
        }
    }
    
    json_path = output_dir / 'batch_statistics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 통계 보고서 저장: {json_path}")
    
    # 요약 엑셀 저장
    summary_path = output_dir / 'batch_summary.xlsx'
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        if not pa_df.empty:
            pa_df.to_excel(writer, sheet_name='PA_전체결과', index=False)
        if not sa_df.empty:
            sa_df.to_excel(writer, sheet_name='SA_전체결과', index=False)
        
        # 통계 시트
        if pa_stats:
            pa_stats_df = pd.DataFrame(pa_stats).T
            pa_stats_df.to_excel(writer, sheet_name='PA_통계')
        
        if sa_stats:
            sa_stats_df = pd.DataFrame(sa_stats).T
            sa_stats_df.to_excel(writer, sheet_name='SA_통계')
    
    print(f"✅ 요약 엑셀 저장: {summary_path}")
    
    print("\n" + "="*70)
    print("✅ 배치 처리 결과 집계 완료")
    print("="*70)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='배치 처리 결과 집계')
    parser.add_argument('--results-dir', default='xlsx_pipeline_results',
                       help='배치 결과 디렉터리 (기본: xlsx_pipeline_results)')
    parser.add_argument('--output-dir', default='analytics',
                       help='통계 출력 디렉터리 (기본: analytics)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"❌ 결과 디렉터리가 없습니다: {results_dir}")
        exit(1)
    
    generate_report(results_dir, output_dir)
