#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 구병렬 파일에 문단식별자 컬럼 추가
- 대응하는 문장병렬 파일과 조인하여 문장식별자 → 문단식별자 매핑
- 구병렬 파일의 맨 앞에 문단식별자 컬럼 삽입
"""

import pandas as pd
from pathlib import Path
import sys

def add_paragraph_id_to_gubyeollyeol(output_base_dir: str = './xlsx'):
    """모든 구병렬 파일에 문단식별자 추가"""
    
    output_base_dir = Path(output_base_dir)
    
    if not output_base_dir.exists():
        print(f"❌ 디렉토리 없음: {output_base_dir}")
        return False
    
    # 모든 구병렬 파일 찾기
    gubyeol_files = sorted(output_base_dir.glob('*/*_구병렬.xlsx'))
    
    if not gubyeol_files:
        print(f"⚠️ {output_base_dir}에 *_구병렬.xlsx 파일이 없습니다.")
        return False
    
    print(f"📂 디렉토리: {output_base_dir}")
    print(f"📋 발견된 구병렬 파일: {len(gubyeol_files)}개\n")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    
    for gubyeol_file in gubyeol_files:
        file_name = gubyeol_file.name
        book_name = file_name.replace('_구병렬.xlsx', '')
        print(f"\n처리 중: {file_name}")
        
        try:
            # 대응하는 문장병렬 파일 찾기
            sentence_file = gubyeol_file.parent / f'{book_name}_문장병렬.xlsx'
            
            if not sentence_file.exists():
                print(f"  ⚠️ 대응 문장병렬 파일 없음: {sentence_file.name}")
                fail_count += 1
                continue
            
            # 파일 읽기
            print(f"  📖 구병렬 읽기...")
            gubyeol_df = pd.read_excel(gubyeol_file, engine='openpyxl')
            
            print(f"  📖 문장병렬 읽기...")
            sentence_df = pd.read_excel(sentence_file, engine='openpyxl')
            
            # 문장식별자 → 문단식별자 매핑 생성
            # 문장병렬에서 문장식별자와 문단식별자의 매핑 추출
            id_mapping = sentence_df[['문장식별자', '문단식별자']].drop_duplicates()
            
            if id_mapping.empty:
                print(f"  ⚠️ 문장식별자/문단식별자 매핑 실패")
                fail_count += 1
                continue
            
            # 구병렬의 문장식별자로 문단식별자 추가
            print(f"  🔗 문장식별자로 조인...")
            gubyeol_merged = gubyeol_df.merge(
                id_mapping,
                on='문장식별자',
                how='left'
            )
            
            # 컬럼 순서 조정 (문단식별자를 맨 앞에)
            cols = list(gubyeol_merged.columns)
            if '문단식별자' in cols:
                cols.remove('문단식별자')
                cols = ['문단식별자'] + cols
                gubyeol_merged = gubyeol_merged[cols]
            
            # 저장
            print(f"  💾 저장 중...")
            gubyeol_merged.to_excel(gubyeol_file, index=False, engine='openpyxl')
            
            # 통계
            nan_count = gubyeol_merged['문단식별자'].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️ 주의: {nan_count}개 행의 문단식별자가 NaN")
            
            print(f"  ✅ 완료: {len(gubyeol_merged):,}개 행, 컬럼 {len(gubyeol_merged.columns)}개")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            fail_count += 1
    
    print("\n" + "=" * 80)
    print(f"\n📊 처리 완료:")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {fail_count}개")
    print(f"  총 처리: {len(gubyeol_files)}개")
    
    return fail_count == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='모든 구병렬 파일에 문단식별자 컬럼을 추가합니다.'
    )
    parser.add_argument('--dir', type=str, default='./xlsx',
                        help='xlsx 디렉토리 경로 (기본값: ./xlsx)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🔧 구병렬 문단식별자 추가 도구")
    print("=" * 80 + "\n")
    
    success = add_paragraph_id_to_gubyeollyeol(output_base_dir=args.dir)
    sys.exit(0 if success else 1)
