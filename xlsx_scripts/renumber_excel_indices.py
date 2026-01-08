"""
문장병렬 Excel 파일들의 행 번호를 다시 매기기
"""

import pandas as pd
from pathlib import Path

def renumber_excel_files(output_base_dir):
    """Excel 파일들의 문단식별자를 식별자 값의 변화에 따라 누적 번호로 다시 매기기
    
    주의: 같은 식별자 그룹은 각각 고유한 번호를 할당해야 함
    단순히 식별자가 바뀔 때만 번호를 증가시키면, 
    같은 식별자의 여러 문장이 모두 같은 번호로 병합될 수 있음
    """
    
    output_base_dir = Path(output_base_dir)
    
    # 모든 서브디렉토리에서 문장병렬 Excel 파일 찾기
    excel_files = list(output_base_dir.glob('*/*_문장병렬.xlsx'))
    
    print(f"총 {len(excel_files)}개 파일 발견\n")
    
    for excel_file in sorted(excel_files):
        try:
            # Excel 파일 읽기
            df = pd.read_excel(excel_file, engine='openpyxl')
            
            # 원본 식별자 유지 (문단번호로 재설정하지 않음)
            # 대신 문단 그룹별로 순차 번호 할당
            new_para_id = []
            current_para_num = 0
            prev_identifier = None
            
            for idx, row in df.iterrows():
                current_identifier = row['문단식별자']
                
                # 식별자가 바뀌면 번호 증가
                if current_identifier != prev_identifier:
                    current_para_num += 1
                    prev_identifier = current_identifier
                
                new_para_id.append(current_para_num)
            
            df['문단식별자'] = new_para_id
            
            # 다시 저장
            df.to_excel(excel_file, index=False, engine='openpyxl')
            
            print(f"✓ {excel_file.parent.name}/{excel_file.name} - {len(df)}개 행, {current_para_num}개 문단")
            
        except Exception as e:
            print(f"✗ 오류: {excel_file.name} - {str(e)}")
    
    print(f"\n완료!")

if __name__ == '__main__':
    output_directory = '/workspace/tsv_output'
    renumber_excel_files(output_directory)
