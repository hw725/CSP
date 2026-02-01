?"""
'?섎떎' 留덉빱 ?λⅤ蹂?遺꾩꽍 (湲곗궗吏??寃利?
=========================================
Level 3: 湲곗궗吏??鼇섇뤁阿뗦뼴) - ??궗???뱀쑀??怨듭쟻 湲곕줉 醫낃껐??

Implementation based on: genre_classification_logic.md
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json


def classify_genre(book_name: str) -> str:
    """Classifies books into genres based on content keywords."""
    if pd.isna(book_name):
        return '湲고?'
    
    book_name = str(book_name)
    
    if '?먯튂?듦컧' in book_name or '異섏텛醫뚯뵪?? in book_name:
        return '??궗??
    elif '?뱀넚?붾?媛' in book_name:
        return '臾몄쭛'
    elif '?덇린' in book_name:
        return '寃쎌쟾'
    elif '?뱀떆?쇰갚?? in book_name:
        return '??
    else:
        return '湲고?'


def analyze_hada_by_genre():
    """?λⅤ蹂?'?섎떎' 留덉빱 遺꾪룷 遺꾩꽍"""
    print("=" * 60)
    print("Level 3: 湲곗궗吏??鼇섇뤁阿뗦뼴) - '?섎떎' ?λⅤ蹂?遺꾩꽍")
    print("=" * 60)
    
    # ?곗씠??濡쒕뱶
    df = pd.read_csv('datasets/phrase_normalized.csv')
    print(f"珥??곗씠?? {len(df):,}嫄?)
    
    # ?λⅤ 遺꾨쪟
    df['genre'] = df['book_name'].apply(classify_genre)
    
    # '?섎떎' 留덉빱 異붿텧
    hada_mask = df['marker_final'].str.endswith('?섎떎', na=False)
    df['is_hada'] = hada_mask
    
    print(f"\n'?섎떎' 留덉빱 ?꾩껜: {hada_mask.sum():,}嫄?)
    
    # ?λⅤ蹂?吏묎퀎
    print("\n" + "-" * 60)
    print("?λⅤ蹂?'?섎떎' 遺꾪룷")
    print("-" * 60)
    
    results = []
    for genre in ['??궗??, '臾몄쭛', '寃쎌쟾', '??, '湲고?']:
        genre_df = df[df['genre'] == genre]
        total = len(genre_df)
        hada_count = genre_df['is_hada'].sum()
        ratio = (hada_count / total * 100) if total > 0 else 0
        
        results.append({
            'genre': genre,
            'total': int(total),
            'hada_count': int(hada_count),
            'ratio': float(ratio)
        })
        
        print(f"{genre:8s}: {total:>8,}嫄?以?{hada_count:>6,}嫄?({ratio:>5.2f}%)")
    
    # Chi-squared 寃??(??궗??vs 鍮꾩뿭?ъ꽌)
    history = df[df['genre'] == '??궗??]
    non_history = df[df['genre'] != '??궗??]
    
    table = np.array([
        [history['is_hada'].sum(), len(history) - history['is_hada'].sum()],
        [non_history['is_hada'].sum(), len(non_history) - non_history['is_hada'].sum()]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    print("\n" + "-" * 60)
    print("?듦퀎 寃?? ??궗??vs 鍮꾩뿭?ъ꽌")
    print("-" * 60)
    print(f"?짼 = {chi2:.2f}, p = {p_value:.2e}")
    print(f"寃곕줎: {'H? 湲곌컖 ??(??궗?쒖뿉???좎쓽誘명븯寃??믪쓬)' if p_value < 0.05 else 'H? 湲곌컖 ?ㅽ뙣'}")
    
    # 寃곌낵 ???
    output = {
        'analysis': 'Level 3: 湲곗궗吏??(?섎떎 ?λⅤ蹂?遺꾩꽍)',
        'total_records': len(df),
        'total_hada': int(hada_mask.sum()),
        'by_genre': results,
        'chi2_test': {
            'history_vs_non_history': {
                'chi2': float(chi2),
                'p_value': float(p_value),
                'reject_h0': bool(p_value < 0.05)
            }
        }
    }
    
    output_dir = Path('reports/phase4')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'hada_genre_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n寃곌낵 ??? reports/phase4/hada_genre_analysis.json")
    
    return output


if __name__ == "__main__":
    analyze_hada_by_genre()
 *cascade08'*cascade08': *cascade08:;*cascade08;= *cascade08=f*cascade08fh *cascade08hm*cascade08mn *cascade08no*cascade08oq *cascade08qt*cascade08tw *cascade08w?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08? *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?*cascade08? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?*cascade08 *cascade08?*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08?*cascade08? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08??*cascade08?? *cascade08"(c9bfe0b46f5f7097b29a8f99d3ee94a0e38df5c92Ufile:///c:/Users/junto/Downloads/head-repo/hw725/CSP/hyeonto/analyze_hada_by_genre.py:4file:///c:/Users/junto/Downloads/head-repo/hw725/CSP