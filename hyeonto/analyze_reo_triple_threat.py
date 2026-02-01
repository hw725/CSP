"""
회상상(-러-) 마커 LLM 기반 Triple-Threat 검증
- 영가설: 통계 검정 (카이제곱, Fisher's exact, 효과 크기)
- 반대가설: 다른 의미 범주 분석
- 대립가설: -더- 마커와 비교
"""
import pandas as pd
import requests
import json
import time
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-v3.2:cloud"  # 주력 모델

SYSTEM_PROMPT_RETROSPECTIVE = """당신은 국어학 전문가입니다.
번역문에서 '회상(回想)' 표현 여부를 판단합니다.

회상 표현의 특징:
- 화자가 과거에 경험/목격한 사실을 현재 시점에서 떠올리며 서술
- "~였는데", "~했는데" (과거 배경 제시 후 전환)
- "~더니" (과거 경험 후 결과)
- 과거 상황과 현재/후속 상황의 대비 구조

판단 기준:
- 단순 과거 사실 나열: 비회상
- 과거 경험을 떠올리며 서술 + 상황 전환/대비: 회상"""

SYSTEM_PROMPT_CATEGORY = """당신은 국어학 전문가입니다.
번역문의 주된 의미 범주를 분석합니다.

가능한 범주:
1. 회상: 과거 경험을 현재 시점에서 떠올리며 서술
2. 추측: 불확실한 사실에 대한 추정
3. 완료: 행위의 완결/결과 상태
4. 단순과거: 객관적인 과거 사실 나열
5. 기타: 위 범주에 해당하지 않음"""


def analyze_retrospective(translation: str) -> dict:
    """회상 여부 분석"""
    prompt = f"""다음 번역문이 '회상'을 나타내는지 분석하세요.

번역문: {translation}

답변 형식:
판정: [회상/비회상]
근거: [한 문장]"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "system": SYSTEM_PROMPT_RETROSPECTIVE, 
                  "stream": False, "options": {"temperature": 0.1}},
            timeout=60
        )
        if response.status_code == 200:
            text = response.json().get("response", "")
            is_retro = "회상" in text.split("판정:")[-1].split("\n")[0] and "비회상" not in text.split("판정:")[-1].split("\n")[0]
            return {"is_retrospective": is_retro, "raw": text, "success": True}
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_category(translation: str) -> dict:
    """의미 범주 분석 (반대가설용)"""
    prompt = f"""다음 번역문의 주된 의미 범주를 분석하세요.

번역문: {translation}

답변 형식:
범주: [회상/추측/완료/단순과거/기타]
근거: [한 문장]"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "system": SYSTEM_PROMPT_CATEGORY,
                  "stream": False, "options": {"temperature": 0.1}},
            timeout=60
        )
        if response.status_code == 200:
            text = response.json().get("response", "")
            category_line = text.split("범주:")[-1].split("\n")[0].strip()
            categories = ["회상", "추측", "완료", "단순과거", "기타"]
            detected = next((c for c in categories if c in category_line), "기타")
            return {"category": detected, "raw": text, "success": True}
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_statistics(reo_retro, reo_total, ctrl_retro, ctrl_total):
    """통계 검정 수행"""
    # 분할표
    table = np.array([
        [reo_retro, reo_total - reo_retro],
        [ctrl_retro, ctrl_total - ctrl_retro]
    ])
    
    # 카이제곱 검정
    chi2, p_chi2, dof, expected = chi2_contingency(table)
    
    # Fisher's exact test
    odds_ratio, p_fisher = fisher_exact(table)
    
    # 효과 크기 (Cram?r's V)
    n = table.sum()
    cramers_v = np.sqrt(chi2 / (n * min(table.shape[0]-1, table.shape[1]-1)))
    
    # 효과 크기 (Cohen's h)
    p1 = reo_retro / reo_total
    p2 = ctrl_retro / ctrl_total
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    return {
        "chi2": chi2,
        "p_chi2": p_chi2,
        "odds_ratio": odds_ratio,
        "p_fisher": p_fisher,
        "cramers_v": cramers_v,
        "cohens_h": cohens_h,
        "dof": dof
    }


def main():
    print("=" * 70)
    print("회상상(-러-) LLM 기반 Triple-Threat 검증")
    print(f"모델: {MODEL}")
    print("=" * 70)
    
    # 데이터 로드
    df = pd.read_csv('datasets/sentence_normalized.csv')
    
    # 마커별 추출
    reo_df = df[df['marker_normalized'].str.contains('러', na=False)].copy()
    deo_df = df[df['marker_normalized'].str.contains('더', na=False) & 
                ~df['marker_normalized'].str.contains('러', na=False)].copy()
    ctrl_df = df[~df['marker_normalized'].str.contains('러|더', na=False, regex=True)].copy()
    
    print(f"\n-러- 포함: {len(reo_df)}개")
    print(f"-더- 포함 (러 제외): {len(deo_df)}개")
    print(f"대조군 (러/더 미포함): {len(ctrl_df)}개")
    
    SAMPLE_SIZE = 50  # 시간 절약
    
    # 샘플링
    reo_sample = reo_df.sample(n=min(SAMPLE_SIZE, len(reo_df)), random_state=42)
    deo_sample = deo_df.sample(n=min(SAMPLE_SIZE, len(deo_df)), random_state=42)
    ctrl_sample = ctrl_df.sample(n=min(SAMPLE_SIZE, len(ctrl_df)), random_state=42)
    
    results = {"reo": [], "deo": [], "ctrl": []}
    
    # ========== 1. 영가설 검증: 회상 비율 비교 ==========
    print("\n" + "=" * 70)
    print("[1] 영가설 검증: -러- vs 대조군 회상 비율")
    print("=" * 70)
    
    for group_name, sample in [("reo", reo_sample), ("ctrl", ctrl_sample)]:
        print(f"\n분석 중: {group_name}...")
        for i, (idx, row) in enumerate(sample.iterrows()):
            result = analyze_retrospective(str(row['번역문']))
            results[group_name].append(result)
            if (i + 1) % 10 == 0:
                success = sum(1 for r in results[group_name] if r.get('success'))
                retro = sum(1 for r in results[group_name] if r.get('is_retrospective'))
                print(f"  {i+1}/{SAMPLE_SIZE} | 성공: {success} | 회상: {retro}")
            time.sleep(0.1)
    
    # 통계 계산
    reo_success = [r for r in results["reo"] if r.get("success")]
    ctrl_success = [r for r in results["ctrl"] if r.get("success")]
    
    reo_retro = sum(1 for r in reo_success if r.get("is_retrospective"))
    ctrl_retro = sum(1 for r in ctrl_success if r.get("is_retrospective"))
    
    stats = calculate_statistics(reo_retro, len(reo_success), ctrl_retro, len(ctrl_success))
    
    print(f"\n[영가설 검정 결과]")
    print(f"  -러- 회상 비율: {reo_retro}/{len(reo_success)} ({reo_retro/len(reo_success)*100:.1f}%)")
    print(f"  대조군 회상 비율: {ctrl_retro}/{len(ctrl_success)} ({ctrl_retro/len(ctrl_success)*100:.1f}%)")
    print(f"  χ² = {stats['chi2']:.2f}, p = {stats['p_chi2']:.2e}")
    print(f"  Fisher's exact OR = {stats['odds_ratio']:.2f}, p = {stats['p_fisher']:.2e}")
    print(f"  Cram?r's V = {stats['cramers_v']:.3f}")
    print(f"  Cohen's h = {stats['cohens_h']:.3f}")
    
    null_verdict = "REJECT" if stats['p_chi2'] < 0.05 else "FAIL_TO_REJECT"
    print(f"  → 영가설 {null_verdict} (α=0.05)")
    
    # ========== 2. 반대가설 검증: 다른 범주 분석 ==========
    print("\n" + "=" * 70)
    print("[2] 반대가설 검증: -러- 문장의 의미 범주 분포")
    print("=" * 70)
    
    category_results = []
    reo_sample_small = reo_sample.head(30)  # 시간 절약
    
    for i, (idx, row) in enumerate(reo_sample_small.iterrows()):
        result = analyze_category(str(row['번역문']))
        category_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/30 완료")
        time.sleep(0.1)
    
    # 범주 집계
    categories = {}
    for r in category_results:
        if r.get("success"):
            cat = r.get("category", "기타")
            categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n[범주 분포]")
    total_cat = sum(categories.values())
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}개 ({count/total_cat*100:.1f}%)")
    
    inverse_verdict = "REJECT" if categories.get("회상", 0) >= categories.get("단순과거", 0) else "SUPPORT"
    print(f"  → 반대가설 {inverse_verdict}")
    
    # ========== 3. 대립가설 검증: -더- vs -러- ==========
    print("\n" + "=" * 70)
    print("[3] 대립가설 검증: -더- vs -러- 회상 비율 비교")
    print("=" * 70)
    
    print("\n분석 중: -더- 샘플...")
    for i, (idx, row) in enumerate(deo_sample.iterrows()):
        result = analyze_retrospective(str(row['번역문']))
        results["deo"].append(result)
        if (i + 1) % 10 == 0:
            success = sum(1 for r in results["deo"] if r.get('success'))
            retro = sum(1 for r in results["deo"] if r.get('is_retrospective'))
            print(f"  {i+1}/{SAMPLE_SIZE} | 성공: {success} | 회상: {retro}")
        time.sleep(0.1)
    
    deo_success = [r for r in results["deo"] if r.get("success")]
    deo_retro = sum(1 for r in deo_success if r.get("is_retrospective"))
    
    print(f"\n[대립가설 검정 결과]")
    print(f"  -러- 회상 비율: {reo_retro}/{len(reo_success)} ({reo_retro/len(reo_success)*100:.1f}%)")
    print(f"  -더- 회상 비율: {deo_retro}/{len(deo_success)} ({deo_retro/len(deo_success)*100:.1f}%)")
    
    alt_stats = calculate_statistics(reo_retro, len(reo_success), deo_retro, len(deo_success))
    print(f"  χ² = {alt_stats['chi2']:.2f}, p = {alt_stats['p_chi2']:.2e}")
    
    alt_verdict = "REJECT" if alt_stats['p_chi2'] < 0.05 else "FAIL_TO_REJECT"
    print(f"  → 대립가설 {alt_verdict}")
    
    # ========== 결과 저장 ==========
    output_dir = Path('reports/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "model": MODEL,
        "sample_size": SAMPLE_SIZE,
        "null_hypothesis": {
            "reo_retrospective": reo_retro,
            "reo_total": len(reo_success),
            "ctrl_retrospective": ctrl_retro,
            "ctrl_total": len(ctrl_success),
            "statistics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in stats.items()},
            "verdict": null_verdict
        },
        "inverse_hypothesis": {
            "category_distribution": categories,
            "verdict": inverse_verdict
        },
        "alternative_hypothesis": {
            "deo_retrospective": deo_retro,
            "deo_total": len(deo_success),
            "statistics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in alt_stats.items()},
            "verdict": alt_verdict
        }
    }
    
    with open(output_dir / 'LLM_TRIPLE_THREAT_ANALYSIS.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {output_dir / 'LLM_TRIPLE_THREAT_ANALYSIS.json'}")
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("Triple-Threat 검증 최종 요약")
    print("=" * 70)
    print(f"  [1] 영가설 (H0: -러-와 회상 무관): {null_verdict}")
    print(f"  [2] 반대가설 (H_inv: -러-는 다른 의미): {inverse_verdict}")
    print(f"  [3] 대립가설 (H_alt: -더-가 더 우수): {alt_verdict}")


if __name__ == "__main__":
    main()
