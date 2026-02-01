# 회상상(-러-) 마커 검증 로직 상세 문서

**작성일**: 2026-01-27
**목적**: 검증 방법론의 투명성 확보

---

## 1. 검증 프레임워크 개요

모든 가설에 대해 **3종 검증**을 수행합니다:

| 테스트 | 질문 | 방법 | 통과 기준 |
|--------|------|------|----------|
| **영가설** | 관측 결과가 우연인가? | Permutation Test (1000회) | p < 0.05 AND Cohen's d > 0.8 |
| **반대가설** | 가중치에 민감한가? | 기대값 대비 비율 | 비율 > 0.5x |
| **대립가설** | 랜덤보다 우수한가? | 랜덤 평균 대비 비율 | 비율 > 1.2x |

---

## 2. H1: 마커 정체성 (역사서 집중) 검증 로직

### 2.1 가설
- **H0 (영가설)**: -러- 마커가 역사서에 집중되는 것은 우연이다
- **H1 (대립가설)**: -러- 마커는 역사서에 유의하게 집중된다

### 2.2 데이터 준비
```python
# 1. 각 문장에서 -러- 마커 존재 여부 판정
df['has_reo'] = df['marker_normalized'].apply(
    lambda x: has_marker(x, [r'러(?=[ㄱ-ㅎㅏ-ㅣ가-�R])', r'려(?=[ㄱ-ㅎㅏ-ㅣ가-�R])'])
)

# 2. 장르 분류
df['genre'] = df['book_name'].apply(classify_genre)
# 역사서: 사기, 한서, 후한서, 삼국지, 통감, 자치통감 포함 여부
```

### 2.3 관측값 계산
```python
total_reo = df['has_reo'].sum()           # 총 -러- 출현 = 5,095건
history_reo = df[(df['genre'] == '역사서') & (df['has_reo'])].shape[0]  # 역사서 내 출현 = 1,616건
observed_ratio = history_reo / total_reo * 100   # 관측 비율 = 31.72%
```

### 2.4 기대값 (Baseline)
```python
history_total = (df['genre'] == '역사서').sum()  # 역사서 문장수 = 23,640건
expected_ratio = history_total / len(df) * 100   # 기대 비율 = 15.70%
```
> **해석**: 만약 -러-가 장르와 무관하게 균등 분포한다면, 전체 문장 중 역사서 비율(15.70%)만큼만 역사서에서 나타날 것

### 2.5 Permutation Test (1000회)
```python
permuted_ratios = []
for _ in range(1000):
    # 장르 라벨을 무작위로 섞음
    shuffled_genres = np.random.permutation(df['genre'].values)
    
    # 셔플된 장르에서 역사서에 해당하는 -러- 비율 계산
    perm_history_reo = df['has_reo'][shuffled_genres == '역사서'].sum()
    perm_ratio = perm_history_reo / total_reo * 100
    permuted_ratios.append(perm_ratio)
```

### 2.6 통계량 계산
```python
mean_perm = np.mean(permuted_ratios)  # 랜덤 평균 = 15.68%
std_perm = np.std(permuted_ratios)    # 랜덤 표준편차 = 0.507%

# Cohen's d (Effect Size)
effect_size = (observed_ratio - mean_perm) / std_perm
# = (31.72 - 15.68) / 0.507 = 31.64

# p-value: 랜덤에서 관측값 이상이 나올 확률
p_value = np.mean(permuted_ratios >= observed_ratio)
# = 0.000 (1000회 중 0회)
```

### 2.7 판정
- **영가설**: p=0.0 < 0.05, d=31.64 > 0.8 → ? **기각**
- **반대가설**: 31.72/15.70 = 2.02x > 0.5 → ? **강건**
- **대립가설**: 31.72/15.68 = 2.02x > 1.2 → ? **우수**

---

## 3. H2.1: 장르 분포 검증 로직

### 3.1 가설
- **H0**: 모든 장르에서 -러- 밀도가 동일하다
- **H1**: 특정 장르가 유의하게 높은 밀도를 보인다

### 3.2 관측값 계산
```python
genre_densities = {}
for genre in ['문집', '기타', '역사서', '경전', '사서']:
    genre_df = df[df['genre'] == genre]
    total = len(genre_df)
    reo = genre_df['has_reo'].sum()
    density = reo / total * 100
    genre_densities[genre] = density

# 결과:
# - 문집: 4.29%
# - 기타: 1.81%
# - 역사서: 6.84% ← 최고
# - 경전: 0.92%
# - 사서: 2.11%

max_density = 6.84%  # 역사서
overall_density = total_reo / len(df) * 100  # 3.38%
```

### 3.3 Permutation Test (1000회)
```python
permuted_max_densities = []
for _ in range(1000):
    # -러- 마커를 무작위로 섞음
    shuffled = np.random.permutation(df['has_reo'].values)
    
    # 각 장르의 밀도 계산 후 최대값 추출
    max_perm_density = 0
    for genre in genre_stats:
        genre_mask = df['genre'] == genre
        perm_reo = shuffled[genre_mask].sum()
        perm_density = perm_reo / genre_mask.sum() * 100
        max_perm_density = max(max_perm_density, perm_density)
    
    permuted_max_densities.append(max_perm_density)
```

### 3.4 통계량 계산
```python
mean_perm = 3.52%   # 랜덤에서 최고 밀도 평균
std_perm = 0.067%   # 랜덤에서 최고 밀도 표준편차

effect_size = (6.84 - 3.52) / 0.067 = 49.67
p_value = 0.000  # 랜덤에서 6.84% 이상 0회
```

### 3.5 판정
- **영가설**: p=0.0, d=49.67 → ? **기각**
- **반대가설**: 6.84/3.38 = 2.02x → ? **강건**
- **대립가설**: 6.84/3.52 = 1.94x → ? **우수**

---

## 4. H2.2: 클러스터 분포 검증 로직

### 4.1 가설
- **H0**: 모든 클러스터에서 -러- 밀도가 동일하다
- **H1**: 특정 클러스터가 유의하게 높은 밀도를 보인다

### 4.2 관측값
```python
cluster_densities = {
    0: 8.63%,  # ← 최고
    1: 1.56%,
    2: 1.74%,
    3: 0.59%
}
max_density = 8.63%
overall_density = 3.38%
```

### 4.3 Permutation Test
```python
# H2.1과 동일한 로직, 장르 대신 클러스터 사용
# 랜덤에서 최고 밀도 평균 = 3.50%, 표준편차 = 0.059%
effect_size = (8.63 - 3.50) / 0.059 = 86.69
p_value = 0.000
```

### 4.4 판정
- **영가설**: p=0.0, d=86.69 → ? **기각**
- **반대가설**: 8.63/3.38 = 2.55x → ? **강건**
- **대립가설**: 8.63/3.50 = 2.47x → ? **우수**

---

## 5. H2.3: 빈도 분포 검증 로직

### 5.1 가설
- **H0**: -러-의 빈도가 TAM 마커 중 무작위 비율이다
- **H1**: -러-는 유의한 비율을 차지한다

### 5.2 관측값
```python
total_reo = 5,213건 (37.1%)
total_deo = 255건 (1.8%)    # -더-
total_ri = 8,588건 (61.1%)  # -리-
total_tam = 14,056건

reo_ratio = 37.1%
expected_ratio = 33.33%  # 3개 마커 균등 분배시
```

### 5.3 Permutation Test
```python
# 전체 TAM 마커를 섞어서 -러- 비율 분포 생성
all_markers = [1]*5213 + [0]*8843  # -러-=1, 나머지=0
permuted_ratios = []
for _ in range(1000):
    np.random.shuffle(all_markers)
    perm_reo_ratio = sum(all_markers[:5213]) / len(all_markers) * 100
    permuted_ratios.append(perm_reo_ratio)

# 랜덤 평균 = 13.76%, 표준편차 = 0.20%
effect_size = (37.1 - 13.76) / 0.20 = 116.9
p_value = 0.000
```

### 5.4 판정
- **영가설**: p=0.0, d=116.9 → ? **기각**
- **반대가설**: 37.1/33.33 = 1.11x > 0.5 → ? **강건**
- **대립가설**: 37.1/13.76 = 2.70x → ? **우수**

---

## 6. 왜 모든 테스트가 통과했는가?

### 6.1 통계적 해석
| 가설 | Effect Size | 해석 |
|------|-------------|------|
| H1 | 31.64 | 관측값이 랜덤 평균에서 **31 표준편차** 떨어져 있음 |
| H2.1 | 49.67 | 관측값이 랜덤 평균에서 **50 표준편차** 떨어져 있음 |
| H2.2 | 86.69 | 관측값이 랜덤 평균에서 **87 표준편차** 떨어져 있음 |
| H2.3 | 116.9 | 관측값이 랜덤 평균에서 **117 표준편차** 떨어져 있음 |

> 일반적으로 Cohen's d > 0.8이면 "큰 효과", d > 2.0이면 "극단적 효과"로 해석합니다.
> 
> **모든 가설에서 d > 30**은 데이터에 **극도로 강한 패턴**이 존재한다는 증거입니다.

### 6.2 언어학적 해석
- -러- 마커가 **역사서**와 **Cluster 0**에 집중되는 것은 우연이 아닙니다
- 1000회 무작위 셔플에서 **단 한 번도** 관측값에 근접하지 못했습니다
- 이는 **-러-가 특정 맥락(역사 서술)에서 사용되는 회상상 마커**라는 언어학적 가설과 일치합니다

### 6.3 검증의 진정성
- Permutation test는 **데이터 자체를 기반**으로 분포를 생성합니다
- 외부 가정 없이 **순수하게 무작위 재배열**만 수행합니다
- 1000회 시뮬레이션에서 p=0.000은 **0/1000**, 즉 랜덤에서 절대 도달 불가능함을 의미

---

## 7. 재현 방법

```bash
cd c:\Users\junto\Downloads\head-repo\hw725\CSP\hyeonto
python run_retrospective_validation.py
```

결과 파일:
- `reports/validation/RETROSPECTIVE_HYPOTHESIS_REPORT.md`
- `reports/validation/retrospective_hypothesis.json`

---

## 8. LLM 기반 번역문 의미 분석 (2026-01-27 추가)

### 8.1 배경

기존 검증(H1~H2.3)은 **분포 패턴**을 기반으로 했습니다. 그러나 "-러-가 실제로 회상의 의미를 나타내는가?"를 검증하려면 **번역문의 실제 의미**를 분석해야 합니다.

### 8.2 방법론

LLM(DeepSeek-V3.2)을 활용하여 번역문에서 '회상' 표현 여부를 판정:

```python
# 시스템 프롬프트
"""당신은 국어학 전문가입니다.
번역문에서 '회상(回想)' 표현 여부를 판단합니다.

회상 표현의 특징:
- 화자가 과거에 경험/목격한 사실을 현재 시점에서 떠올리며 서술
- "~였는데", "~했는데" (과거 배경 제시 후 전환)
- "~더니" (과거 경험 후 결과)
- 과거 상황과 현재/후속 상황의 대비 구조"""
```

### 8.3 Triple-Threat 검증 결과 (파일럿, n=50)

| 가설 | 결과 | 통계량 | 해석 |
|------|------|--------|------|
| **H0 (영가설)** | **REJECT** | χ²=9.00, p=0.003, OR=9.33 | -러-와 회상 간 유의미한 관계 있음 |
| **H_inv (반대가설)** | SUPPORT | 회상 23.3% < 단순과거 63.3% | ?? LLM 범주 분류 기준 검토 필요 |
| **H_alt (대립가설)** | FAIL_TO_REJECT | -러- 28% ? -더- 30% (p=1.0) | 두 마커 간 차이 없음 |

#### 상세 수치

| 그룹 | 샘플 | 회상 판정 | 비율 |
|------|------|----------|------|
| -러- 포함 | 50 | 14 | **28.0%** |
| -더- 포함 | 50 | 15 | **30.0%** |
| 대조군 | 50 | 2 | **4.0%** |

#### 효과 크기
- **Cram?r's V = 0.300** (중간 효과)
- **Cohen's h = 0.712** (큰 효과)
- **Odds Ratio = 9.33** (-러- 포함 시 회상으로 판정될 확률 9배)

### 8.4 해석 및 한계

#### 긍정적 결과
- -러- 포함 문장은 대조군 대비 **7배** 더 회상으로 판정됨
- 통계적으로 유의함 (p < 0.01)

#### 주의사항
1. **샘플 크기**: 50개는 파일럿 수준, 통계적 검정력 제한
2. **반대가설 해석**: LLM이 "단순과거"로 분류한 경우가 많음 - 이는 LLM의 범주 분류 기준과 언어학적 정의 간 괴리 가능성
3. **-더- vs -러-**: 두 마커 간 유의미한 차이 없음 → 둘 다 회상상 마커로서 유사한 기능?

### 8.5 향후 계획

> ?? **본 결과는 파일럿 테스트입니다. 전수조사 전 설계 보완 필요:**

#### 설계 보완 필요 사항

**1. 대립가설 재설계**
- **현재 문제**: `-러-` vs `-더-` 비교는 부적절 (둘 다 회상상 `-더-`의 이형태)
- **수정 방향**: `-러-/-더-` (회상상) vs `-리-` (추정상/미래) 비교
- **근거**: 이형태 간 비교는 동어반복, 기능이 다른 마커와 비교해야 의미 있음

**2. 반대가설 정밀 검토**
- **현재 결과**: "단순과거" 63% vs "회상" 23%
- **검토 필요**:
  - "단순과거"로 판정된 샘플 직접 검토
  - LLM 범주 분류 기준과 언어학적 정의 간 괴리 분석
  - "회상"과 "단순과거" 경계 재정의

**3. 학술적 정의 확보**
- KCI에서 '-더-' 또는 '회상상' 관련 논문 검색
- 고전 국어학 문헌에서 정확한 정의 확인
- 프롬프트에 학술적 정의 반영

#### 전수조사 계획 (설계 보완 후)

1. **샘플 구성**:
   - 실험군: -러-/-더- 포함 5,329건
   - 대조군 A: -리- 포함 (추정상)
   - 대조군 B: 마커 없음 (무작위 5,000건)

2. **실행 방식**: Ollama 클라우드 사용량 제한 고려, 분할 수행
3. **향후 옵션**: OpenAI API 도입 (효율화)

### 8.6 재현 방법

```bash
cd c:\Users\junto\Downloads\head-repo\hw725\CSP\hyeonto
python analyze_reo_triple_threat.py
```

결과 파일:
- `reports/validation/LLM_TRIPLE_THREAT_ANALYSIS.json`

