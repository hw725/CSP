import argparse
from pathlib import Path
import pandas as pd

"""
구병렬(.xlsx) 파일에서 '문단식별자' 컬럼을 일괄 제거하는 스크립트

대상: xlsx/<책이름>/<책이름>_구병렬.xlsx ('.bak' 파일은 건너뜀)
동작:
- 시트의 컬럼에 '문단식별자'가 있으면 제거 후 저장
- 기본 컬럼은 유지: ['문장식별자','구식별자','원문','번역문'] 순서 권장
- 처리 로그를 콘솔에 출력
"""


def remove_paragraph_id(output_base_dir: str = './xlsx') -> bool:
    base = Path(output_base_dir)
    if not base.exists():
        print(f"경로 없음: {base}")
        return False

    target_files = []
    for book_dir in sorted(base.glob('*')):
        if not book_dir.is_dir():
            continue
        # *_구병렬.xlsx (bak 제외)
        for f in book_dir.glob('*_구병렬.xlsx'):
            if f.name.endswith('.bak'):
                continue
            target_files.append(f)

    print(f"발견된 구병렬 파일: {len(target_files)}개")

    processed = 0
    removed = 0
    skipped = 0

    for xlsx_path in target_files:
        try:
            df = pd.read_excel(xlsx_path)
            cols = [str(c) for c in df.columns]
            has_para = any(c == '문단식별자' or 'paragraph' in str(c).lower() for c in cols)

            if not has_para:
                print(f"- 건너뜀(문단식별자 없음): {xlsx_path.name}")
                skipped += 1
                continue

            # '문단식별자' 정확히 명명된 컬럼 우선 제거, 없으면 영문 추정 제거
            drop_cols = []
            for c in df.columns:
                if str(c) == '문단식별자' or 'paragraph' in str(c).lower():
                    drop_cols.append(c)
            if drop_cols:
                df = df.drop(columns=drop_cols)

            # 권장 컬럼 순서로 정렬(존재하는 것만 유지)
            preferred = ['문장식별자', '구식별자', '원문', '번역문']
            ordered = [c for c in preferred if c in df.columns]
            # 나머지 컬럼도 뒤에 보존
            rest = [c for c in df.columns if c not in ordered]
            df = df[ordered + rest] if ordered else df

            # 저장(덮어쓰기)
            df.to_excel(xlsx_path, index=False)
            print(f"✓ 제거 완료: {xlsx_path.name} (삭제 컬럼: {', '.join(map(str, drop_cols))})")
            removed += 1
            processed += 1
        except Exception as e:
            print(f"✗ 처리 실패: {xlsx_path.name} -> {e}")

    print("\n요약:")
    print(f"- 처리: {processed}개")
    print(f"- 제거: {removed}개")
    print(f"- 건너뜀: {skipped}개")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='구병렬에서 문단식별자 컬럼 제거')
    parser.add_argument('--dir', default='./xlsx', help='기본 폴더 (기본: ./xlsx)')
    args = parser.parse_args()
    remove_paragraph_id(output_base_dir=args.dir)
