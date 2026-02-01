?"""
?곗씠?곗뀑 ?듬챸???ㅽ겕由쏀듃
踰덉뿭臾몄쓣 SHA-256 ?댁떆濡?泥섎━?섏뿬 ??묎텒 蹂댄샇

Implementation based on: anonymization_and_reproducibility.md
- Algorithm: SHA-256
- Output: Truncated to first 16 hexadecimal characters
- Target Column: 踰덉뿭臾?(Translation)
- LLM judgment columns preserved for reproducibility
"""

import pandas as pd
import hashlib
from pathlib import Path


def anonymize_translation(text: str) -> str:
    """踰덉뿭臾몄쓣 SHA-256 ?댁떆濡?蹂??(16??truncate)"""
    if pd.isna(text) or text == "":
        return ""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def anonymize_dataset(input_path: str, output_path: str, llm_results_path: str = None):
    """
    ?곗씠?곗뀑 ?듬챸??泥섎━
    
    Args:
        input_path: ?먮낯 ?곗씠?곗뀑 寃쎈줈
        output_path: ?듬챸?붾맂 ?곗씠?곗뀑 ???寃쎈줈
        llm_results_path: LLM ?먯젙 寃곌낵 JSON 寃쎈줈 (?좏깮)
    """
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Total rows: {len(df):,}")
    
    # LLM ?먯젙 寃곌낵 蹂묓빀 (?덈뒗 寃쎌슦)
    if llm_results_path and Path(llm_results_path).exists():
        import json
        with open(llm_results_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        # TODO: LLM 寃곌낵瑜??곗씠?고봽?덉엫??蹂묓빀
        print(f"LLM results loaded from: {llm_results_path}")
    
    # 踰덉뿭臾?而щ읆 ?댁떆 泥섎━
    if '踰덉뿭臾? in df.columns:
        print("Hashing '踰덉뿭臾? column...")
        original_count = df['踰덉뿭臾?].notna().sum()
        df['踰덉뿭臾?] = df['踰덉뿭臾?].apply(anonymize_translation)
        hashed_count = df['踰덉뿭臾?].apply(lambda x: len(x) == 16 if x else True).sum()
        print(f"  Processed: {original_count:,} ??{hashed_count:,} hashed")
    
    # 蹂댁〈?섎뒗 而щ읆 ?뺤씤
    preserved_cols = ['?먮Ц', 'book_name', 'marker_normalized', 'marker_final', 
                      '臾몄옣?앸퀎??, '援ъ떇蹂꾩옄', 'genre']
    existing_preserved = [c for c in preserved_cols if c in df.columns]
    print(f"Preserved columns: {existing_preserved}")
    
    # ???
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {output_path}")
    
    # ?댁떆 異⑸룎 寃??
    if '踰덉뿭臾? in df.columns:
        unique_hashes = df['踰덉뿭臾?].nunique()
        total_non_empty = (df['踰덉뿭臾?] != '').sum()
        collision_rate = 1 - (unique_hashes / total_non_empty) if total_non_empty > 0 else 0
        print(f"Hash collision check: {unique_hashes:,} unique / {total_non_empty:,} total")
        print(f"  Collision rate: {collision_rate:.2e}")


def main():
    base_dir = Path('datasets')
    
    # phrase_normalized.csv 泥섎━
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