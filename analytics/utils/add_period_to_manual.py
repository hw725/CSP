#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
manual CSV에 시대 컬럼 추가 스크립트
작가명 기반 자동 추론 + 수동 편집 지원
"""

import pandas as pd
from book_metadata_extractor import BookMetadataExtractor

def add_period_to_manual_csv():
    # manual CSV 읽기
    manual_df = pd.read_csv('cumulative_analysis_results_manual.csv', encoding='utf-8-sig')
    
    print("현재 컬럼:", list(manual_df.columns))
    print("시대 컬럼 있는지:", '시대' in manual_df.columns)
    
    # BookMetadataExtractor로 시대 정보 추가
    extractor = BookMetadataExtractor()
    
    # 시대 정보를 작가명으로부터 추론
    period_list = []
    for idx, row in manual_df.iterrows():
        author = row['작가'] if pd.notna(row['작가']) else '미상'
        period = extractor.get_period_from_author(author)
        period_list.append(period)
        print(f"{row['책명']}: {author} → {period}")
    
    # 시대 컬럼이 이미 있으면 업데이트, 없으면 추가
    if '시대' in manual_df.columns:
        # 기존 시대 정보가 '미상'이 아닌 경우 보존 (수동 편집 우선)
        for idx, period in enumerate(period_list):
            if pd.isna(manual_df.iloc[idx]['시대']) or manual_df.iloc[idx]['시대'] == '미상':
                manual_df.iloc[idx, manual_df.columns.get_loc('시대')] = period
    else:
        # 4부분류 다음에 시대 컬럼 추가
        sibu_idx = manual_df.columns.get_loc('4부분류')
        manual_df.insert(sibu_idx + 1, '시대', period_list)
    
    # 원본 파일을 백업
    manual_df_original = pd.read_csv('cumulative_analysis_results_manual.csv', encoding='utf-8-sig')
    backup_filename = 'cumulative_analysis_results_manual_backup_with_period.csv'
    manual_df_original.to_csv(backup_filename, index=False, encoding='utf-8-sig')
    
    # 시대가 추가된 파일로 원본 덮어쓰기
    manual_df.to_csv('cumulative_analysis_results_manual.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 시대 정보가 추가되어 원본 파일 업데이트 완료!")
    print(f"총 {len(manual_df)}권 처리 완료")
    print(f"📁 백업 파일: {backup_filename}")
    
    # 시대 분포 확인
    period_counts = manual_df['시대'].value_counts()
    print("\n📊 시대 분포:")
    for period, count in period_counts.items():
        print(f"  {period}: {count}권")
    
    # 컬럼 순서 확인
    print(f"\n📋 최종 컬럼 순서:")
    for i, col in enumerate(manual_df.columns):
        print(f"  {i+1:2d}. {col}")
    
    return manual_df

if __name__ == "__main__":
    add_period_to_manual_csv()