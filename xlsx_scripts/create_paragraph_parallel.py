"""
문장병렬 Excel 파일을 문단식별자 기준으로 묶어서 문단병렬 파일 생성
컬럼: 문단식별자, 원문, 번역문
NaN 값은 그대로 보존
"""

import pandas as pd
from pathlib import Path

def create_paragraph_parallel(output_base_dir):
    """문장병렬 파일을 문단병렬로 변환 (NaN 보존)"""

    output_base_dir = Path(output_base_dir)

    # 모든 서브디렉토리에서 문장병렬 Excel 파일 찾기
    excel_files = list(output_base_dir.glob("*/*_문장병렬.xlsx"))

    print(f"총 {len(excel_files)}개 파일 처리 중\n")

    for excel_file in sorted(excel_files):
        try:
            # 문장병렬 Excel 파일 읽기
            df = pd.read_excel(excel_file, engine="openpyxl")

            # 문단식별자 기준으로 그룹화
            def join_texts(texts):
                """NaN을 제외하고 공백으로 연결"""
                filtered = []
                has_nan = False
                for t in texts:
                    if pd.isna(t):
                        has_nan = True
                    else:
                        filtered.append(str(t).strip())

                # 모든 값이 NaN이면 NaN 반환
                if not filtered and has_nan:
                    return pd.NA

                return " ".join(filtered) if filtered else ""

            grouped = df.groupby("문단식별자", as_index=False).agg(
                {"원문": join_texts, "번역문": join_texts}
            )

            # 컬럼 순서 정렬
            grouped = grouped[["문단식별자", "원문", "번역문"]]

            # 문단병렬 Excel 파일로 저장 (같은 디렉토리에)
            book_name = excel_file.parent.name
            output_file = excel_file.parent / f"{book_name}_문단병렬.xlsx"

            grouped.to_excel(output_file, index=False, engine="openpyxl")

            print(f"✓ {book_name}_문단병렬.xlsx - {len(grouped)}개 문단")

        except Exception as e:
            print(f"✗ 오류: {excel_file.name} - {str(e)}")

    print(f"\n완료!")

if __name__ == "__main__":
    output_directory = "/workspace/xlsx"
    create_paragraph_parallel(output_directory)
