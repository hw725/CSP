"""
단사(斷辭) 전수조사 - GPT-5-nano LLM 분석
=====================================================
Level 1: 유사이단 '로다' 전체 (952건)
Level 2: 쾌절/미절 '니라' 전체 (4,233건) vs '라' (4,233건 샘플)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm
import time

# API 설정
client = OpenAI()
MODEL = "gpt-5-nano"

def load_data():
    df = pd.read_csv('datasets/phrase_normalized.csv')
    return df

def analyze_batch_with_llm(texts, prompt_template, batch_size=20):
    """LLM으로 번역문 뉘앙스 분석"""
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="LLM 분석"):
        batch = texts[i:i+batch_size]
        
        prompt = prompt_template.format(
            texts="\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])
        )
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            
            for line in content.strip().split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    parts = line.split('.')
                    if len(parts) >= 2:
                        judgment = parts[1].strip().upper()
                        results.append('O' in judgment or 'Y' in judgment or '예' in judgment)
                    else:
                        results.append(False)
        except Exception as e:
            print(f"Error: {e}")
            results.extend([False] * len(batch))
        
        time.sleep(0.05)
    
    return results

# ============================================================
# Level 1: 유사이단 '로다' 전수조사
# ============================================================
def verify_level1_full(df):
    """로다 전체 952건 분석"""
    print("\n" + "="*60)
    print("Level 1: 유사이단 '로다' 전수조사")
    print("="*60)
    
    roda = df[df['marker_final'].str.endswith('로다', na=False)]
    control = df[df['marker_final'] == '라'].sample(n=len(roda), random_state=42)
    
    print(f"로다 전체: {len(roda)}건")
    print(f"라 대조군: {len(control)}건")
    
    prompt = """다음 번역문들이 **"감탄이나 여운을 남기는"** 뉘앙스를 담고 있는지 판단해주세요.

"감탄이나 여운"이란:
- 감정적 고양 (탄복, 감탄, 찬탄, 탄식)
- 시적 여운, 열린 마무리
- 정서적 반응을 유발하는 표현

해당되면 O, 아니면 X로 답해주세요.

{texts}

각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""
    
    print("\n로다 그룹 분석 중...")
    roda_judgments = analyze_batch_with_llm(roda['번역문'].tolist(), prompt)
    
    print("라 그룹 분석 중...")
    control_judgments = analyze_batch_with_llm(control['번역문'].tolist(), prompt)
    
    roda_positive = sum(roda_judgments)
    control_positive = sum(control_judgments)
    
    table = np.array([
        [roda_positive, len(roda_judgments) - roda_positive],
        [control_positive, len(control_judgments) - control_positive]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    print(f"\n분할표 (LLM 판정):")
    print(f"         감탄O  감탄X")
    print(f"로다     {table[0,0]:5}  {table[0,1]:5}  ({table[0,0]/len(roda_judgments)*100:.1f}%)")
    print(f"라       {table[1,0]:5}  {table[1,1]:5}  ({table[1,0]/len(control_judgments)*100:.1f}%)")
    print(f"\nχ² = {chi2:.2f}, p = {p_value:.2e}")
    print(f"결론: {'H₀ 기각 ✅' if p_value < 0.05 else 'H₀ 기각 실패'}")
    
    return {
        'level': 'Level 1',
        'marker': '로다',
        'category': '유사이단',
        'n_target': len(roda_judgments),
        'n_control': len(control_judgments),
        'target_positive': int(roda_positive),
        'control_positive': int(control_positive),
        'target_rate': roda_positive / len(roda_judgments) * 100,
        'control_rate': control_positive / len(control_judgments) * 100,
        'chi2': float(chi2),
        'p_value': float(p_value),
        'reject_h0': bool(p_value < 0.05)
    }

# ============================================================
# Level 2: 쾌절 vs 미절 전수조사
# ============================================================
def verify_level2_full(df):
    """니라 전체 4233건 vs 라 4233건 샘플"""
    print("\n" + "="*60)
    print("Level 2: 쾌절 vs 미절 전수조사")
    print("="*60)
    
    nira = df[df['marker_final'] == '니라']
    ra = df[df['marker_final'] == '라'].sample(n=len(nira), random_state=42)
    
    print(f"니라 전체: {len(nira)}건")
    print(f"라 샘플: {len(ra)}건")
    
    prompt = """다음 번역문들이 **"단호하게 결론짓는"** 뉘앙스를 가지는지 판단해주세요.

"단호한 종결"이란:
- 확정적 단언, 최종 결론
- 더 이상 논의가 필요 없는 완결된 진술
- 강한 확신을 담은 결정적 서술

"약한 종결"이란:
- 확신이 약한 추측, 추정
- 단순 사실 나열, 경과 보고

"단호한 종결"에 해당하면 O, "약한 종결"에 해당하면 X로 답해주세요.

{texts}

각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""
    
    print("\n니라 그룹 분석 중...")
    nira_judgments = analyze_batch_with_llm(nira['번역문'].tolist(), prompt)
    
    print("라 그룹 분석 중...")
    ra_judgments = analyze_batch_with_llm(ra['번역문'].tolist(), prompt)
    
    nira_positive = sum(nira_judgments)
    ra_positive = sum(ra_judgments)
    
    table = np.array([
        [nira_positive, len(nira_judgments) - nira_positive],
        [ra_positive, len(ra_judgments) - ra_positive]
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    print(f"\n분할표 (LLM 판정):")
    print(f"         단호O  단호X")
    print(f"니라     {table[0,0]:5}  {table[0,1]:5}  ({table[0,0]/len(nira_judgments)*100:.1f}%)")
    print(f"라       {table[1,0]:5}  {table[1,1]:5}  ({table[1,0]/len(ra_judgments)*100:.1f}%)")
    print(f"\nχ² = {chi2:.2f}, p = {p_value:.2e}")
    print(f"결론: {'H₀ 기각 ✅' if p_value < 0.05 else 'H₀ 기각 실패'}")
    
    return {
        'level': 'Level 2',
        'marker': '니라 vs 라',
        'category': '쾌절 vs 미절',
        'n_nira': len(nira_judgments),
        'n_ra': len(ra_judgments),
        'nira_positive': int(nira_positive),
        'ra_positive': int(ra_positive),
        'nira_rate': nira_positive / len(nira_judgments) * 100,
        'ra_rate': ra_positive / len(ra_judgments) * 100,
        'chi2': float(chi2),
        'p_value': float(p_value),
        'reject_h0': bool(p_value < 0.05)
    }

# ============================================================
# 메인 실행
# ============================================================
def main():
    print("="*60)
    print("단사(斷辭) 전수조사 - GPT-5-nano LLM 분석")
    print("="*60)
    
    df = load_data()
    print(f"총 데이터: {len(df):,}건")
    
    results = []
    
    # Level 1 전수조사
    r1 = verify_level1_full(df)
    results.append(r1)
    
    # Level 2 전수조사
    r2 = verify_level2_full(df)
    results.append(r2)
    
    # 결과 저장
    output_dir = Path('reports/phase4')
    output_dir.mkdir(exist_ok=True)
    
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj
    
    with open(output_dir / 'dansa_full_survey.json', 'w', encoding='utf-8') as f:
        json.dump(convert_types(results), f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("전수조사 완료! 결과: reports/phase4/dansa_full_survey.json")
    print("="*60)
    
    print("\n📊 전수조사 결과 요약")
    print("-"*60)
    for r in results:
        status = "✅ H₀ 기각" if r['reject_h0'] else "❌ H₀ 기각 실패"
        print(f"{r['level']}: {r['marker']} ({r['category']})")
        print(f"  결과: {status} (p={r['p_value']:.4e})")
        if 'target_rate' in r:
            print(f"  비율: 타겟 {r['target_rate']:.1f}% vs 대조군 {r['control_rate']:.1f}%")
        else:
            print(f"  비율: 니라 {r['nira_rate']:.1f}% vs 라 {r['ra_rate']:.1f}%")
        print()
    
    return results

if __name__ == "__main__":
    main()
