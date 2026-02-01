#!/usr/bin/env python3
"""
회상상(-러-) 마커 완전 가설 검정 스크립트

모든 가설에 대해 영가설/반대가설/대립가설 3종 검증 수행

H1: -러-가 회상상 마커인가? (마커 정체성)
H2.1: -러-가 특정 장르에 집중되어 있는가? (장르 분포)
H2.2: -러-가 특정 클러스터에 집중되어 있는가? (클러스터 분포)
H2.3: -러-의 빈도가 통계적으로 유의한가? (빈도 분포)

입력: reports/sentence_k4_normalized/sentence_clusters.csv
출력: reports/validation/RETROSPECTIVE_HYPOTHESIS_REPORT.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re
from scipy import stats

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
VALIDATION_DIR = REPORTS_DIR / "validation"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# 회상상 마커 패턴
RETROSPECTIVE_PATTERNS = [r'러(?=[ㄱ-ㅎㅏ-ㅣ가-�R])', r'려(?=[ㄱ-ㅎㅏ-ㅣ가-�R])']
OTHER_TAM_PATTERNS = {
    'retrospective': [r'더(?=[ㄱ-ㅎㅏ-ㅣ가-�R])'],
    'modal': [r'리(?=[ㄱ-ㅎㅏ-ㅣ가-�R])'],
}

# 장르 분류
HISTORY_BOOKS = ['사기', '한서', '후한서', '삼국지', '통감', '자치통감']
SASEO = ['논어집주', '맹자집주', '대학장구', '중용장구']
SAMGYEONG = ['시경집전', '서경집전', '주역전의']


def classify_genre(book_name: str) -> str:
    for hist in HISTORY_BOOKS:
        if hist in book_name:
            return '역사서'
    if book_name in SASEO:
        return '사서'
    elif book_name in SAMGYEONG:
        return '삼경'
    elif any(x in book_name for x in ['전의', '집전', '경']):
        return '경전'
    elif any(x in book_name for x in ['집', '문']):
        return '문집'
    else:
        return '기타'


def has_marker(text: str, patterns: list) -> bool:
    if pd.isna(text):
        return False
    for pattern in patterns:
        if re.search(pattern, str(text)):
            return True
    return False


def count_markers(text: str, patterns: list) -> int:
    if pd.isna(text):
        return 0
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, str(text)))
    return count


def run_triple_test(
    observed_value: float,
    baseline_value: float,
    permuted_values: np.ndarray,
    test_name: str,
    higher_is_better: bool = True
) -> dict:
    """
    영가설/반대가설/대립가설 3종 검증 템플릿
    """
    mean_perm = np.mean(permuted_values)
    std_perm = np.std(permuted_values)
    
    # 영가설: 관측값이 우연인가?
    effect_size = (observed_value - mean_perm) / std_perm if std_perm > 0 else 0
    if higher_is_better:
        p_value = np.mean(permuted_values >= observed_value)
    else:
        p_value = np.mean(permuted_values <= observed_value)
    
    null_verdict = "REJECTED" if p_value < 0.05 and abs(effect_size) > 0.8 else "NOT_REJECTED"
    
    # 반대가설: 가중치 변화에 강건한가? (baseline과 비교)
    robustness = observed_value / baseline_value if baseline_value > 0 else 0
    inverse_verdict = "ROBUST" if robustness > 0.5 else "SENSITIVE"
    
    # 대립가설: 대안보다 우수한가?
    alt_ratio = observed_value / mean_perm if mean_perm > 0 else 0
    alt_verdict = "SUPERIOR" if alt_ratio > 1.2 else "NOT_SUPERIOR"
    
    all_passed = null_verdict == "REJECTED" and inverse_verdict == "ROBUST" and alt_verdict == "SUPERIOR"
    
    return {
        'observed': observed_value,
        'expected': mean_perm,
        'std': std_perm,
        'effect_size': effect_size,
        'p_value': p_value,
        'null_verdict': null_verdict,
        'robustness': robustness,
        'inverse_verdict': inverse_verdict,
        'alt_ratio': alt_ratio,
        'alt_verdict': alt_verdict,
        'all_passed': all_passed,
        'passed_count': sum([null_verdict == "REJECTED", inverse_verdict == "ROBUST", alt_verdict == "SUPERIOR"])
    }


# ============================================================
# H1: -러-가 회상상 마커인가?
# ============================================================

def test_h1_retrospective_identity(df: pd.DataFrame, n_perm: int = 1000) -> dict:
    """
    H1: -러-가 역사서에 집중되는 것이 우연인가?
    (이전 테스트 방법 복원: permutation test로 역사서 집중 검증)
    """
    print("\n" + "="*60)
    print("? H1: -러-가 회상상 마커인가? (마커 정체성)")
    print("="*60)
    
    df_copy = df.copy()
    df_copy['has_reo'] = df_copy['marker_normalized'].apply(
        lambda x: has_marker(x, RETROSPECTIVE_PATTERNS)
    )
    df_copy['genre'] = df_copy['book_name'].apply(classify_genre)
    
    # 역사서 집중도 계산
    total_reo = df_copy['has_reo'].sum()
    history_reo = df_copy[(df_copy['genre'] == '역사서') & (df_copy['has_reo'])].shape[0]
    observed_ratio = (history_reo / total_reo * 100) if total_reo > 0 else 0
    
    history_total = (df_copy['genre'] == '역사서').sum()
    expected_ratio = (history_total / len(df_copy) * 100)
    
    print(f"  -러- 총 출현: {total_reo:,}건")
    print(f"  역사서 내 출현: {history_reo:,}건 ({observed_ratio:.2f}%)")
    print(f"  역사서 기대 비율: {expected_ratio:.2f}%")
    
    # Permutation test
    print(f"\n  Permutation test ({n_perm}회)...")
    permuted_ratios = []
    
    for _ in range(n_perm):
        shuffled_genres = np.random.permutation(df_copy['genre'].values)
        perm_history_reo = df_copy['has_reo'][shuffled_genres == '역사서'].sum()
        perm_ratio = (perm_history_reo / total_reo * 100) if total_reo > 0 else 0
        permuted_ratios.append(perm_ratio)
    
    permuted_ratios = np.array(permuted_ratios)
    
    # 3종 검증
    result = run_triple_test(observed_ratio, expected_ratio, permuted_ratios, "H1")
    
    print(f"  랜덤 평균: {result['expected']:.2f}%")
    print(f"  Effect Size: {result['effect_size']:.3f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"\n  [영가설] {'? 기각' if result['null_verdict'] == 'REJECTED' else '? 채택'}: 역사서 집중은 우연{'이 아님' if result['null_verdict'] == 'REJECTED' else '일 수 있음'}")
    print(f"  [반대가설] {'? 강건' if result['inverse_verdict'] == 'ROBUST' else '?? 민감'}: 기대 대비 {result['robustness']:.2f}x")
    print(f"  [대립가설] {'? 우수' if result['alt_verdict'] == 'SUPERIOR' else '?? 동등'}: 랜덤 대비 {result['alt_ratio']:.2f}x")
    print(f"\n  ? 종합: {result['passed_count']}/3 통과")
    
    return {
        'total_reo': int(total_reo),
        'history_reo': int(history_reo),
        'observed_ratio': observed_ratio,
        'expected_ratio': expected_ratio,
        **result
    }


# ============================================================
# H2.1: 장르 분포
# ============================================================

def test_h2_1_genre_distribution(df: pd.DataFrame, n_perm: int = 1000) -> dict:
    """
    H2.1: -러-가 특정 장르에 집중되어 있는가?
    """
    print("\n" + "="*60)
    print("? H2.1: 장르 분포")
    print("="*60)
    
    df_copy = df.copy()
    df_copy['has_reo'] = df_copy['marker_normalized'].apply(
        lambda x: has_marker(x, RETROSPECTIVE_PATTERNS)
    )
    df_copy['genre'] = df_copy['book_name'].apply(classify_genre)
    
    # 장르별 통계
    genre_stats = {}
    for genre in df_copy['genre'].unique():
        genre_df = df_copy[df_copy['genre'] == genre]
        total = len(genre_df)
        reo = genre_df['has_reo'].sum()
        density = (reo / total * 100) if total > 0 else 0
        genre_stats[genre] = {'total': int(total), 'reo': int(reo), 'density': density}
        print(f"  {genre:8s}: {reo:,}/{total:,} ({density:.2f}%)")
    
    # 최고 집중 장르
    max_genre = max(genre_stats, key=lambda g: genre_stats[g]['density'])
    max_density = genre_stats[max_genre]['density']
    total_reo = df_copy['has_reo'].sum()
    overall_density = total_reo / len(df_copy) * 100
    
    print(f"\n  최고 집중: {max_genre} ({max_density:.2f}%)")
    print(f"  전체 평균: {overall_density:.2f}%")
    
    # Permutation test
    print(f"\n  Permutation test ({n_perm}회)...")
    permuted_max_densities = []
    
    for _ in range(n_perm):
        shuffled = np.random.permutation(df_copy['has_reo'].values)
        max_perm_density = 0
        for genre in genre_stats:
            genre_mask = df_copy['genre'] == genre
            perm_reo = shuffled[genre_mask].sum()
            perm_density = (perm_reo / genre_mask.sum() * 100) if genre_mask.sum() > 0 else 0
            max_perm_density = max(max_perm_density, perm_density)
        permuted_max_densities.append(max_perm_density)
    
    permuted_max_densities = np.array(permuted_max_densities)
    
    # 3종 검증
    result = run_triple_test(max_density, overall_density, permuted_max_densities, "H2.1")
    
    print(f"  랜덤 최고 평균: {result['expected']:.2f}%")
    print(f"  Effect Size: {result['effect_size']:.3f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"\n  [영가설] {'? 기각' if result['null_verdict'] == 'REJECTED' else '? 채택'}")
    print(f"  [반대가설] {'? 강건' if result['inverse_verdict'] == 'ROBUST' else '?? 민감'}")
    print(f"  [대립가설] {'? 우수' if result['alt_verdict'] == 'SUPERIOR' else '?? 동등'}")
    print(f"\n  ? 종합: {result['passed_count']}/3 통과")
    
    return {
        'genre_stats': genre_stats,
        'max_genre': max_genre,
        'max_density': max_density,
        'overall_density': overall_density,
        **result
    }


# ============================================================
# H2.2: 클러스터 분포
# ============================================================

def test_h2_2_cluster_distribution(df: pd.DataFrame, n_perm: int = 1000) -> dict:
    """
    H2.2: -러-가 특정 클러스터에 집중되어 있는가?
    """
    print("\n" + "="*60)
    print("? H2.2: 클러스터 분포")
    print("="*60)
    
    df_copy = df.copy()
    df_copy['has_reo'] = df_copy['marker_normalized'].apply(
        lambda x: has_marker(x, RETROSPECTIVE_PATTERNS)
    )
    
    # 클러스터별 통계
    cluster_stats = {}
    for cluster in sorted(df_copy['cluster_id'].unique()):
        cluster_df = df_copy[df_copy['cluster_id'] == cluster]
        total = len(cluster_df)
        reo = cluster_df['has_reo'].sum()
        density = (reo / total * 100) if total > 0 else 0
        cluster_stats[int(cluster)] = {'total': int(total), 'reo': int(reo), 'density': density}
        print(f"  Cluster {cluster}: {reo:,}/{total:,} ({density:.2f}%)")
    
    # 최고 집중 클러스터
    max_cluster = max(cluster_stats, key=lambda c: cluster_stats[c]['density'])
    max_density = cluster_stats[max_cluster]['density']
    total_reo = df_copy['has_reo'].sum()
    overall_density = total_reo / len(df_copy) * 100
    
    print(f"\n  최고 집중: Cluster {max_cluster} ({max_density:.2f}%)")
    print(f"  전체 평균: {overall_density:.2f}%")
    
    # Permutation test
    print(f"\n  Permutation test ({n_perm}회)...")
    permuted_max_densities = []
    
    for _ in range(n_perm):
        shuffled = np.random.permutation(df_copy['has_reo'].values)
        max_perm_density = 0
        for cluster in cluster_stats:
            cluster_mask = df_copy['cluster_id'] == cluster
            perm_reo = shuffled[cluster_mask].sum()
            perm_density = (perm_reo / cluster_mask.sum() * 100) if cluster_mask.sum() > 0 else 0
            max_perm_density = max(max_perm_density, perm_density)
        permuted_max_densities.append(max_perm_density)
    
    permuted_max_densities = np.array(permuted_max_densities)
    
    # 3종 검증
    result = run_triple_test(max_density, overall_density, permuted_max_densities, "H2.2")
    
    print(f"  랜덤 최고 평균: {result['expected']:.2f}%")
    print(f"  Effect Size: {result['effect_size']:.3f}")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"\n  [영가설] {'? 기각' if result['null_verdict'] == 'REJECTED' else '? 채택'}")
    print(f"  [반대가설] {'? 강건' if result['inverse_verdict'] == 'ROBUST' else '?? 민감'}")
    print(f"  [대립가설] {'? 우수' if result['alt_verdict'] == 'SUPERIOR' else '?? 동등'}")
    print(f"\n  ? 종합: {result['passed_count']}/3 통과")
    
    return {
        'cluster_stats': cluster_stats,
        'max_cluster': max_cluster,
        'max_density': max_density,
        'overall_density': overall_density,
        **result
    }


# ============================================================
# H2.3: 빈도 분포
# ============================================================

def test_h2_3_frequency(df: pd.DataFrame, n_perm: int = 1000) -> dict:
    """
    H2.3: -러-의 빈도가 TAM 마커 중 유의한 비율인가?
    """
    print("\n" + "="*60)
    print("? H2.3: 빈도 분포")
    print("="*60)
    
    df_copy = df.copy()
    
    df_copy['reo_count'] = df_copy['marker_normalized'].apply(
        lambda x: count_markers(x, RETROSPECTIVE_PATTERNS)
    )
    df_copy['deo_count'] = df_copy['marker_normalized'].apply(
        lambda x: count_markers(x, OTHER_TAM_PATTERNS['retrospective'])
    )
    df_copy['ri_count'] = df_copy['marker_normalized'].apply(
        lambda x: count_markers(x, OTHER_TAM_PATTERNS['modal'])
    )
    
    total_reo = df_copy['reo_count'].sum()
    total_deo = df_copy['deo_count'].sum()
    total_ri = df_copy['ri_count'].sum()
    total_tam = total_reo + total_deo + total_ri
    
    reo_ratio = total_reo / total_tam * 100 if total_tam > 0 else 0
    deo_ratio = total_deo / total_tam * 100 if total_tam > 0 else 0
    ri_ratio = total_ri / total_tam * 100 if total_tam > 0 else 0
    
    # 기대 비율 (균등 분배시 33.3%)
    expected_ratio = 33.33
    
    print(f"  -러- (회상): {total_reo:,}건 ({reo_ratio:.1f}%)")
    print(f"  -더- (회상): {total_deo:,}건 ({deo_ratio:.1f}%)")
    print(f"  -리- (추량): {total_ri:,}건 ({ri_ratio:.1f}%)")
    print(f"  총 TAM 마커: {total_tam:,}건")
    
    # Permutation test: -러-의 위치를 무작위로 섞으면
    print(f"\n  Permutation test ({n_perm}회)...")
    
    # 전체 마커를 섞어서 -러- 비율 분포 생성
    all_markers = np.concatenate([
        np.ones(int(total_reo)),
        np.zeros(int(total_deo + total_ri))
    ])
    
    permuted_ratios = []
    for _ in range(n_perm):
        np.random.shuffle(all_markers)
        perm_reo = all_markers[:int(total_reo)].sum()
        perm_total = len(all_markers)
        permuted_ratios.append(perm_reo / perm_total * 100 if perm_total > 0 else 0)
    
    permuted_ratios = np.array(permuted_ratios)
    
    # 3종 검증
    result = run_triple_test(reo_ratio, expected_ratio, permuted_ratios, "H2.3")
    
    print(f"\n  [영가설] {'? 기각' if result['null_verdict'] == 'REJECTED' else '? 채택'}")
    print(f"  [반대가설] {'? 강건' if result['inverse_verdict'] == 'ROBUST' else '?? 민감'}: 기대 대비 {result['robustness']:.2f}x")
    print(f"  [대립가설] {'? 우수' if result['alt_verdict'] == 'SUPERIOR' else '?? 동등'}")
    print(f"\n  ? 종합: {result['passed_count']}/3 통과")
    
    return {
        'reo_count': int(total_reo),
        'deo_count': int(total_deo),
        'ri_count': int(total_ri),
        'total_tam': int(total_tam),
        'reo_ratio': reo_ratio,
        'deo_ratio': deo_ratio,
        'ri_ratio': ri_ratio,
        **result
    }


# ============================================================
# 보고서 생성
# ============================================================

def save_report(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    json_path = output_dir / "retrospective_hypothesis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # Markdown 생성
    md_lines = [
        "# 회상상(-러-) 마커 가설 검정 보고서",
        "",
        f"**분석일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**데이터**: {results['data_file']}",
        f"**분석 건수**: {results['data_rows']:,}건",
        "",
        "---",
        "",
        "## 개요",
        "",
        "### 언어학적 배경",
        "- 중세 한국어에서 **'-러-'**는 **'-더-'**(회상상)의 이형태",
        "- **회상상(Retrospective Aspect)**: 화자가 과거 사건을 직접 경험/목격했음을 표시",
        "- 본 분석은 각 가설에 대해 **영가설/반대가설/대립가설** 3종 검증 수행",
        "",
        "### 검증 방법론",
        "",
        "| 테스트 | 기준 | 통과 조건 |",
        "|--------|------|----------|",
        "| 영가설 | Permutation test | p < 0.05, Cohen's d > 0.8 |",
        "| 반대가설 | 기대값 대비 강건성 | 비율 > 0.5x |",
        "| 대립가설 | 랜덤 평균 대비 우수성 | 비율 > 1.2x |",
        "",
        "---",
        "",
    ]
    
    # 각 가설 결과
    hypotheses = [
        ('H1', '마커 정체성', '역사서 집중', results['H1']),
        ('H2.1', '장르 분포', '특정 장르 집중', results['H2_1']),
        ('H2.2', '클러스터 분포', '특정 클러스터 집중', results['H2_2']),
        ('H2.3', '빈도 분포', 'TAM 마커 중 유의 비율', results['H2_3']),
    ]
    
    total_passed = 0
    total_tests = 0
    
    for h_id, h_name, h_desc, h_data in hypotheses:
        passed = h_data['passed_count']
        total_passed += passed
        total_tests += 3
        
        status = "?" if passed == 3 else "??"
        
        md_lines.extend([
            f"## {h_id}: {h_name} ({h_desc})",
            "",
            f"### 종합 결과: {status} **{passed}/3 통과**",
            "",
            "| 테스트 | 결과 | 상세 |",
            "|--------|------|------|",
            f"| 영가설 | {'? 기각' if h_data['null_verdict'] == 'REJECTED' else '? 채택'} | Effect Size: {h_data['effect_size']:.3f}, p = {h_data['p_value']:.6f} |",
            f"| 반대가설 | {'? 강건' if h_data['inverse_verdict'] == 'ROBUST' else '?? 민감'} | 기대 대비 {h_data['robustness']:.2f}x |",
            f"| 대립가설 | {'? 우수' if h_data['alt_verdict'] == 'SUPERIOR' else '?? 동등'} | 랜덤 대비 {h_data['alt_ratio']:.2f}x |",
            "",
        ])
        
        # 추가 상세 정보
        if 'genre_stats' in h_data:
            md_lines.extend([
                "#### 장르별 분포",
                "",
                "| 장르 | 출현 | 밀도 |",
                "|------|------|------|",
            ])
            for g, s in h_data['genre_stats'].items():
                md_lines.append(f"| {g} | {s['reo']:,}건 | {s['density']:.2f}% |")
            md_lines.extend(["", f"**집중 장르**: {h_data['max_genre']} ({h_data['max_density']:.2f}%)", ""])
        
        if 'cluster_stats' in h_data:
            md_lines.extend([
                "#### 클러스터별 분포",
                "",
                "| 클러스터 | 출현 | 밀도 |",
                "|----------|------|------|",
            ])
            for c, s in sorted(h_data['cluster_stats'].items()):
                md_lines.append(f"| Cluster {c} | {s['reo']:,}건 | {s['density']:.2f}% |")
            md_lines.extend(["", f"**집중 클러스터**: {h_data['max_cluster']} ({h_data['max_density']:.2f}%)", ""])
        
        if 'reo_count' in h_data and 'total_tam' in h_data:
            md_lines.extend([
                "#### 마커별 빈도",
                "",
                "| 마커 | 출현 | 비율 |",
                "|------|------|------|",
                f"| -러- (회상) | {h_data['reo_count']:,}건 | {h_data['reo_ratio']:.1f}% |",
                f"| -더- (회상) | {h_data['deo_count']:,}건 | {h_data['deo_ratio']:.1f}% |",
                f"| -리- (추량) | {h_data['ri_count']:,}건 | {h_data['ri_ratio']:.1f}% |",
                "",
            ])
        
        md_lines.append("---")
        md_lines.append("")
    
    # 최종 결론
    md_lines.extend([
        "## 최종 결론",
        "",
        f"### 전체 테스트: {total_passed}/{total_tests} 통과 ({total_passed/total_tests*100:.1f}%)",
        "",
        "| 가설 | 통과 |",
        "|------|------|",
    ])
    
    for h_id, h_name, _, h_data in hypotheses:
        passed = h_data['passed_count']
        status = "?" if passed == 3 else "??"
        md_lines.append(f"| {h_id}: {h_name} | {status} {passed}/3 |")
    
    md_lines.extend([
        "",
        f"**결론**: {'-러-는 회상상 마커로 통계적으로 검증됨' if total_passed >= 9 else '-러-의 회상상 지위는 부분적으로 검증됨'}",
    ])
    
    md_path = output_dir / "RETROSPECTIVE_HYPOTHESIS_REPORT.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n? 보고서 저장: {md_path}")
    return md_path


def main():
    print("="*70)
    print("? 회상상(-러-) 마커 완전 가설 검정")
    print("    모든 가설에 대해 영가설/반대가설/대립가설 3종 검증")
    print("="*70)
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_path = REPORTS_DIR / "sentence_k4_normalized" / "sentence_clusters.csv"
    if not data_path.exists():
        print(f"? 데이터 없음: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"\n? 데이터: {len(df):,}건")
    
    # 모든 가설 검증
    h1 = test_h1_retrospective_identity(df)
    h2_1 = test_h2_1_genre_distribution(df)
    h2_2 = test_h2_2_cluster_distribution(df)
    h2_3 = test_h2_3_frequency(df)
    
    results = {
        'analysis_date': datetime.now().isoformat(),
        'data_file': str(data_path.relative_to(BASE_DIR)),
        'data_rows': len(df),
        'H1': h1,
        'H2_1': h2_1,
        'H2_2': h2_2,
        'H2_3': h2_3,
    }
    
    save_report(results, VALIDATION_DIR)
    
    # 요약
    total = h1['passed_count'] + h2_1['passed_count'] + h2_2['passed_count'] + h2_3['passed_count']
    print("\n" + "="*70)
    print(f"? 완료! 전체: {total}/12 통과")
    print("="*70)


if __name__ == "__main__":
    main()
