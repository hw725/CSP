#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
manual CSV에 4부분류 추가 스크립트
"""

import pandas as pd
from book_metadata_extractor import BookMetadataExtractor

def add_sibu_to_manual_csv():
    # manual CSV 읽기
    manual_df = pd.read_csv('cumulative_analysis_results_manual.csv', encoding='utf-8-sig')
    
    print("현재 컬럼:", list(manual_df.columns))
    print("4부분류 컬럼 있는지:", '4부분류' in manual_df.columns)
    
    # BookMetadataExtractor로 4부분류 정보 추가
    extractor = BookMetadataExtractor()
    
    # 4부분류 정보를 추가
    sibu_list = []
    for book_name in manual_df['책명']:
        sibu = extractor.get_sibu_classification(book_name)
        sibu_list.append(sibu)
        print(f"{book_name}: {sibu}")
    
    # 4부분류 컬럼 추가
    manual_df['4부분류'] = sibu_list
    
    # 원본 파일을 백업
    manual_df_original = pd.read_csv('cumulative_analysis_results_manual.csv', encoding='utf-8-sig')
    manual_df_original.to_csv('cumulative_analysis_results_manual_backup.csv', index=False, encoding='utf-8-sig')
    
    # 4부분류가 추가된 파일로 원본 덮어쓰기
    manual_df.to_csv('cumulative_analysis_results_manual.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 4부분류가 추가되어 원본 파일 업데이트 완료!")
    print(f"총 {len(manual_df)}권 처리 완료")
    print("📁 백업 파일: cumulative_analysis_results_manual_backup.csv")
    
    # 4부분류 분포 확인
    sibu_counts = manual_df['4부분류'].value_counts()
    print("\n📊 4부분류 분포:")
    for sibu, count in sibu_counts.items():
        print(f"  {sibu}: {count}권")
    
    return manual_df

if __name__ == "__main__":
    add_sibu_to_manual_csv()