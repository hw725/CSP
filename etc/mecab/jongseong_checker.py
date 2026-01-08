#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
종성 유무 판별 스크립트
merged.csv 파일의 '읽기' 열에서 마지막 글자의 종성 유무를 판별하여 새 컬럼 추가
"""

import pandas as pd
import os
from pathlib import Path

def check_jongseong(text):
    """
    텍스트의 마지막 글자가 종성을 가지고 있는지 확인하는 함수
    
    Args:
        text (str): 확인할 텍스트
        
    Returns:
        str: 종성이 있으면 'T', 없으면 'F'
    """
    if pd.isna(text) or text == '' or not isinstance(text, str):
        return 'F'
    
    # 마지막 글자 추출
    last_char = text.strip()[-1]
    
    # 유니코드 값 추출
    unicode_value = ord(last_char)
    
    # 한글 범위 확인 (가: 44032, 힣: 55203)
    if 44032 <= unicode_value <= 55203:
        # (유니코드값 - 44032) % 28이 0이 아니면 종성 있음
        remainder = (unicode_value - 44032) % 28
        return 'T' if remainder != 0 else 'F'
    else:
        # 한글이 아닌 경우 종성 없음으로 처리
        return 'F'

def main():
    """메인 실행 함수"""
    # 현재 디렉토리에서 hanja_hybrid_complete_edited.csv 파일 찾기
    input_file = 'user_handic.csv'
    output_file = 'user_handic_jongseong.csv'
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"오류: {input_file} 파일을 찾을 수 없습니다.")
        print(f"현재 디렉토리: {os.getcwd()}")
        return
    
    try:
        # CSV 파일 읽기 - 파싱 오류 해결을 위한 옵션 추가
        print(f"CSV 파일 읽는 중: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8', 
                        on_bad_lines='skip',    # 잘못된 줄 건너뛰기
                        quoting=1,              # 따옴표 처리 강화
                        skipinitialspace=True)  # 공백 처리
        
        print(f"성공적으로 읽은 행 수: {len(df)}")
        
        # 컬럼 확인
        if '읽기' not in df.columns:
            print("오류: '읽기' 컬럼을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return
        
        print(f"총 {len(df)}개의 행을 처리합니다...")
        
        # '종성 유무' 컬럼 생성
        df['종성 유무'] = df['읽기'].apply(check_jongseong)
        
        # 결과를 새 CSV 파일로 저장
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("처리 완료!")
        print(f"결과 파일: {output_file}")
        
        # 결과 확인용 출력
        print("\n처리 결과 샘플:")
        sample_df = df[['읽기', '종성 유무']].head(10)
        for idx, row in sample_df.iterrows():
            print(f"  {row['읽기']} -> {row['종성 유무']}")
        
        # 종성 유무 통계
        print(f"\n종성 유무 통계:")
        stats = df['종성 유무'].value_counts()
        total = len(df)
        for value, count in stats.items():
            percentage = (count / total) * 100
            print(f"  {value}: {count:,}개 ({percentage:.1f}%)")
        
        # 파일 크기 정보
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"\n출력 파일 크기: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
