?"""

?곗��?곗�� ?듬��???ㅽ�щ┰���

踰����臾몄�� SHA-256 ?댁��濡?泥�由�?���� ???�沅� 蹂댄��



Implementation based on: anonymization_and_reproducibility.md

- Algorithm: SHA-256

- Output: Truncated to first 16 hexadecimal characters

- Target Column: 踰����臾?(Translation)

- LLM judgment columns preserved for reproducibility

"""



import pandas as pd

import hashlib

from pathlib import Path





def anonymize_translation(text: str) -> str:

    """踰����臾몄�� SHA-256 ?댁��濡?蹂???(16??truncate)"""

    if pd.isna(text) or text == "":

        return ""

    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]





def anonymize_dataset(input_path: str, output_path: str, llm_results_path: str = None):

    """

    ?곗��?곗�� ?듬��??泥�由�

    

    Args:

        input_path: ?�蹂� ?곗��?곗�� 寃쎈��

        output_path: ?듬��?���� ?곗��?곗�� ????寃쎈��

        llm_results_path: LLM ?���� 寃곌낵 JSON 寃쎈�� (?����)

    """

    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path)

    

    print(f"Total rows: {len(df):,}")

    

    # LLM ?���� 寃곌낵 蹂���� (?���� 寃쎌��)

    if llm_results_path and Path(llm_results_path).exists():

        import json

        with open(llm_results_path, 'r', encoding='utf-8') as f:

            llm_data = json.load(f)

        # TODO: LLM 寃곌낵瑜??곗��?고��?����??蹂����

        print(f"LLM results loaded from: {llm_results_path}")

    

    # 踰����臾?而щ�� ?댁�� 泥�由�

    if '踰����臾? in df.columns:

        print("Hashing '踰����臾? column...")

        original_count = df['踰����臾?].notna().sum()

        df['踰����臾?] = df['踰����臾?].apply(anonymize_translation)

        hashed_count = df['踰����臾?].apply(lambda x: len(x) == 16 if x else True).sum()

        print(f"  Processed: {original_count:,} ??{hashed_count:,} hashed")

    

    # 蹂댁〈?���� 而щ�� ?����

    preserved_cols = ['?�臾�', 'book', 'marker_normalized', 'marker_final', 

                      '臾몄��?�蹂�??, '援ъ��蹂����', 'genre']

    existing_preserved = [c for c in preserved_cols if c in df.columns]

    print(f"Preserved columns: {existing_preserved}")

    

    # ????
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Saved: {output_path}")

    

    # ?댁�� 異⑸�� 寃???
    if '踰����臾? in df.columns:

        unique_hashes = df['踰����臾?].nunique()

        total_non_empty = (df['踰����臾?] != '').sum()

        collision_rate = 1 - (unique_hashes / total_non_empty) if total_non_empty > 0 else 0

        print(f"Hash collision check: {unique_hashes:,} unique / {total_non_empty:,} total")

        print(f"  Collision rate: {collision_rate:.2e}")





def main():

    base_dir = Path('datasets')

    

    # phrase_normalized.csv 泥�由�

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

h *cascade08h?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08"(c9bfe0b46f5f7097b29a8f99d3ee94a0e38df5c92Qfile:///c:/Users/junto/Downloads/head-repo/hw725/CSP/hyeonto/anonymize_dataset.py:4file:///c:/Users/junto/Downloads/head-repo/hw725/CSP