"""
데이터셋 익명화 스크립트
번역문을 SHA-256 해시로 처리하여 원문 보호

Implementation based on: anonymization_and_reproducibility.md
- Algorithm: SHA-256
- Output: Truncated to first 16 hexadecimal characters
- Target Column: 번역문 (Translation)
- LLM judgment columns preserved for reproducibility
"""

import pandas as pd
import hashlib
from pathlib import Path


def anonymize_translation(text: str) -> str:
    """번역문을 SHA-256 해시로 변환 (16자 truncate)"""
    if pd.isna(text) or text == "":
        return ""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def anonymize_dataset(input_path: str, output_path: str, llm_results_path: str = None):
    """
    데이터셋 익명화 처리
    
    Args:
        input_path: 원본 데이터셋 경로
        output_path: 익명화된 데이터셋 저장 경로
        llm_results_path: LLM 판정 결과 JSON 경로 (선택)
    """
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Total rows: {len(df):,}")
    
    # LLM 판정 결과 병합 (있는 경우)
    if llm_results_path and Path(llm_results_path).exists():
        import json
        with open(llm_results_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        print(f"LLM results loaded from: {llm_results_path}")
    
    # 번역문 컬럼 해시 처리
    if '번역문' in df.columns:
        print("Hashing '번역문' column...")
        original_count = df['번역문'].notna().sum()
        df['번역문'] = df['번역문'].apply(anonymize_translation)
        hashed_count = df['번역문'].apply(lambda x: len(x) == 16 if x else True).sum()
        print(f"  Processed: {original_count:,} -> {hashed_count:,} hashed")
    
    # 보존되는 컬럼 확인
    preserved_cols = ['원문', 'book_name', 'marker_normalized', 'marker_final', 
                      '문장식별자', '구식별자', 'genre']
    existing_preserved = [c for c in preserved_cols if c in df.columns]
    print(f"Preserved columns: {existing_preserved}")
    
    # 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_path}")
    
    # 해시 충돌 검사
    if '번역문' in df.columns:
        unique_hashes = df['번역문'].nunique()
        total_non_empty = (df['번역문'] != '').sum()
        collision_rate = 1 - (unique_hashes / total_non_empty) if total_non_empty > 0 else 0
        print(f"Hash collision check: {unique_hashes:,} unique / {total_non_empty:,} total")
        print(f"  Collision rate: {collision_rate:.2e}")


def main():
    base_dir = Path('datasets')
    
    # phrase_normalized.csv 처리
    input_file = base_dir / 'phrase_normalized.csv'
    output_file = base_dir / 'phrase_normalized_anonymized.csv'
    llm_results = Path('reports/phase4/dansa_full_survey.json')
    
    if input_file.exists():
        anonymize_dataset(
            str(input_file), 
            str(output_file),
            str(llm_results) if llm_results.exists() else None
        )
    else:
        print(f"File not found: {input_file}")


if __name__ == "__main__":
    main()