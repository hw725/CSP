"""
회상상(-러-) 마커 LLM 기반 번역 패턴 분석
- Ollama gemma3-pro-preview 모델 사용
- 번역문에서 실제 "회상" 표현 여부를 LLM이 판단
"""
import pandas as pd
import requests
import json
import time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-v3.1:671b-cloud"

SYSTEM_PROMPT = """당신은 국어학 전문가입니다.
번역문에서 '회상(回想)' 표현 여부를 판단합니다.

회상 표현의 특징:
- 화자가 과거에 경험/목격한 사실을 현재 시점에서 떠올리며 서술
- "~였는데", "~했는데" (과거 배경 제시 후 전환)
- "~더니" (과거 경험 후 결과)
- 과거 상황과 현재/후속 상황의 대비 구조

판단 기준:
- 단순 과거 사실 나열: 비회상
- 과거 경험을 떠올리며 서술 + 상황 전환/대비: 회상"""

def analyze_with_llm(translation: str) -> dict:
    """LLM으로 회상 여부 분석"""
    prompt = f"""다음 번역문이 '회상(回想)'을 나타내는지 분석하세요.

번역문: {translation}

아래 형식으로만 답변하세요:
판정: [회상/비회상]
근거: [한 문장으로 설명]"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("response", "")
            
            # 판정 추출
            is_retrospective = "회상" in text.split("판정:")[-1].split("\n")[0] and "비회상" not in text.split("판정:")[-1].split("\n")[0]
            
            return {
                "raw_response": text,
                "is_retrospective": is_retrospective,
                "success": True
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 60)
    print("회상상(-러-) LLM 기반 번역 패턴 분석")
    print(f"모델: {MODEL}")
    print("=" * 60)
    
    # 데이터 로드
    df = pd.read_csv('datasets/sentence_normalized.csv')
    
    # -러- 포함 문장 추출
    reo_df = df[df['marker_normalized'].str.contains('러', na=False)].copy()
    non_reo_df = df[~df['marker_normalized'].str.contains('러', na=False)].copy()
    
    print(f"\n총 -러- 포함 문장: {len(reo_df)}개")
    print(f"대조군 (-러- 미포함): {len(non_reo_df)}개")
    
    # 샘플링 (각 100개)
    SAMPLE_SIZE = 100
    reo_sample = reo_df.sample(n=min(SAMPLE_SIZE, len(reo_df)), random_state=42)
    non_reo_sample = non_reo_df.sample(n=min(SAMPLE_SIZE, len(non_reo_df)), random_state=42)
    
    print(f"\n샘플 크기: 각 {SAMPLE_SIZE}개")
    print("\nLLM 분석 시작...")
    
    # -러- 포함 문장 분석
    reo_results = []
    print("\n[실험군: -러- 포함 문장]")
    for i, (idx, row) in enumerate(reo_sample.iterrows()):
        result = analyze_with_llm(str(row['번역문']))
        result['원문'] = str(row['원문'])[:50]
        result['번역문'] = str(row['번역문'])[:100]
        result['marker'] = str(row['marker_normalized'])[:30]
        reo_results.append(result)
        
        if (i + 1) % 10 == 0:
            success_count = sum(1 for r in reo_results if r.get('success'))
            retro_count = sum(1 for r in reo_results if r.get('is_retrospective'))
            print(f"  진행: {i+1}/{SAMPLE_SIZE} | 성공: {success_count} | 회상: {retro_count}")
        
        time.sleep(0.1)  # Rate limiting
    
    # 대조군 분석
    non_reo_results = []
    print("\n[대조군: -러- 미포함 문장]")
    for i, (idx, row) in enumerate(non_reo_sample.iterrows()):
        result = analyze_with_llm(str(row['번역문']))
        result['원문'] = str(row['원문'])[:50]
        result['번역문'] = str(row['번역문'])[:100]
        result['marker'] = str(row['marker_normalized'])[:30]
        non_reo_results.append(result)
        
        if (i + 1) % 10 == 0:
            success_count = sum(1 for r in non_reo_results if r.get('success'))
            retro_count = sum(1 for r in non_reo_results if r.get('is_retrospective'))
            print(f"  진행: {i+1}/{SAMPLE_SIZE} | 성공: {success_count} | 회상: {retro_count}")
        
        time.sleep(0.1)
    
    # 결과 집계
    print("\n" + "=" * 60)
    print("LLM 분석 결과")
    print("=" * 60)
    
    reo_success = [r for r in reo_results if r.get('success')]
    reo_retro = sum(1 for r in reo_success if r.get('is_retrospective'))
    reo_ratio = reo_retro / len(reo_success) * 100 if reo_success else 0
    
    non_reo_success = [r for r in non_reo_results if r.get('success')]
    non_reo_retro = sum(1 for r in non_reo_success if r.get('is_retrospective'))
    non_reo_ratio = non_reo_retro / len(non_reo_success) * 100 if non_reo_success else 0
    
    print(f"\n[실험군: -러- 포함 문장]")
    print(f"  분석 성공: {len(reo_success)}개")
    print(f"  회상 판정: {reo_retro}개 ({reo_ratio:.1f}%)")
    
    print(f"\n[대조군: -러- 미포함 문장]")
    print(f"  분석 성공: {len(non_reo_success)}개")
    print(f"  회상 판정: {non_reo_retro}개 ({non_reo_ratio:.1f}%)")
    
    if non_reo_ratio > 0:
        ratio_diff = reo_ratio / non_reo_ratio
        print(f"\n[비율 비교]")
        print(f"  -러- 포함 문장은 대조군 대비 {ratio_diff:.2f}배 더 회상으로 판정됨")
    
    # 결과 저장
    output_dir = Path('reports/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    results = {
        "model": MODEL,
        "sample_size": SAMPLE_SIZE,
        "reo_group": {
            "total": len(reo_success),
            "retrospective": reo_retro,
            "ratio": reo_ratio
        },
        "control_group": {
            "total": len(non_reo_success),
            "retrospective": non_reo_retro,
            "ratio": non_reo_ratio
        },
        "ratio_difference": reo_ratio / non_reo_ratio if non_reo_ratio > 0 else None,
        "reo_samples": reo_results[:10],
        "control_samples": non_reo_results[:10]
    }
    
    with open(output_dir / 'LLM_RETROSPECTIVE_ANALYSIS.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {output_dir / 'LLM_RETROSPECTIVE_ANALYSIS.json'}")


if __name__ == "__main__":
    main()
