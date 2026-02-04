"""
모든 문장병렬 파일에서 NaN 값을 찾아 로그로 남기기
NaN은 그대로 두고, 직전행+직후행 최소 3행을 기록
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def log_nan_values():
    """NaN 값이 있는 부분을 로그로 기록"""

    output_base_dir = Path("/workspace/tsv_output")
    log_file = Path("/workspace/nan_log.txt")

    # 모든 서브디렉토리에서 문장병렬 Excel 파일 찾기
    excel_files = list(output_base_dir.glob("*/*_문장병렬.xlsx"))

    print(f"총 {len(excel_files)}개 파일 검사 중\n")

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"NaN 값 로그\n")
        log.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*100}\n\n")

        total_nan_count = 0

        for excel_file in sorted(excel_files):
            try:
                # 문장병렬 Excel 파일 읽기
                df = pd.read_excel(excel_file, engine="openpyxl")

                # NaN 값 찾기
                nan_rows_원문 = df[df["원문"].isna()].index.tolist()
                nan_rows_번역문 = df[df["번역문"].isna()].index.tolist()
                nan_indices = sorted(set(nan_rows_원문 + nan_rows_번역문))

                if nan_indices:
                    book_name = excel_file.parent.name
                    log.write(f"파일: {book_name}_문장병렬.xlsx\n")
                    log.write(f"{'-'*100}\n")

                    for idx in nan_indices:
                        total_nan_count += 1

                        # 직전행, 현재행, 직후행 추출
                        start = max(0, idx - 1)
                        end = min(len(df), idx + 2)

                        log.write(f"\n▶ NaN 발견 (행 {idx}):\n")

                        for i in range(start, end):
                            row = df.iloc[i]
                            marker = " ← NaN" if i == idx else ""

                            원문_preview = (
                                str(row["원문"])[:60]
                                if pd.notna(row["원문"])
                                else "[NaN]"
                            )
                            번역문_preview = (
                                str(row["번역문"])[:60]
                                if pd.notna(row["번역문"])
                                else "[NaN]"
                            )

                            log.write(
                                f"  행{i}: 문단={row['문단식별자']}, 문장={row['문장식별자']}\n"
                            )
                            log.write(f"        원문: {원문_preview}\n")
                            log.write(f"        번역문: {번역문_preview}{marker}\n")

                    log.write(f"\n{'-'*100}\n\n")

            except Exception as e:
                print(f"✗ 오류: {excel_file.name} - {str(e)}")

        log.write(f"\n{'='*100}\n")
        log.write(f"총 NaN 값 개수: {total_nan_count}개\n")

    print(f"✓ 로그 저장: {log_file}")
    print(f"✓ 총 NaN 값: {total_nan_count}개\n")

if __name__ == "__main__":
    log_nan_values()
