"""
문장병렬 Excel 파일들의 행 번호를 다시 매기기
"""

import pandas as pd
from pathlib import Path

def renumber_excel_files(output_base_dir):
    """Excel 파일들의 문단식별자를 순차 번호로 재지정
    
    같은 식별자가 연속되는 동안은 같은 번호, 
    식별자가 바뀌면 새로운 번호를 부여한다.
    """
    
    output_base_dir = Path(output_base_dir)
    
    # 모든 서브디렉토리에서 문장병렬 Excel 파일 찾기
    excel_files = list(output_base_dir.glob('*/*_문장병렬.xlsx'))
    
    print(f"총 {len(excel_files)}개 파일 발견\n")
    
    for excel_file in sorted(excel_files):
        try:
            # Excel 파일 읽기
            df = pd.read_excel(excel_file, engine='openpyxl')
            
            # 식별자 그룹 번호 부여: 연속된 같은 식별자는 같은 번호
            group_numbers = []
            current_group = 0
            prev_identifier = None
            
            for identifier in df['문단식별자']:
                if identifier != prev_identifier:
                    current_group += 1
                    prev_identifier = identifier
                group_numbers.append(current_group)
            
            df['문단식별자'] = group_numbers
            
            # 다시 저장
            df.to_excel(excel_file, index=False, engine='openpyxl')
            
            print(f"✓ {excel_file.parent.name}/{excel_file.name} - {len(df)}개 행, {current_group}개 문단")
            
        except Exception as e:
            print(f"✗ 오류: {excel_file.name} - {str(e)}")
    
    print(f"\n완료!")

if __name__ == '__main__':
    output_directory = '/workspace/xlsx'
    renumber_excel_files(output_directory)
