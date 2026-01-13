# F1 0.80 → 0.90 달성 로드맵
**분석 기준**: test_results/multitest_seed1_10_markerbonusA_skipfixA (seed 1~10, 1000 케이스)  
**현재 성능**: F1 0.87 (2026-01-13 기준), Similarity 0.94
**목표**: F1 0.90 달성 (Supar Bonus + Ensemble Voting)

---

## 📊 현황 분석

### 핵심 발견
1. **35.7% (357/1000건)의 케이스에서 1등/2등 점수 차이 < 0.10**
   - 근소한 차이 (<0.03): 152건 (15.2%)
   - 중간 차이 (0.03~0.10): 205건 (20.5%)
   - **이 케이스들은 가중치 조정만으로 결과가 바뀔 가능성이 큼**

2. **현재 선택 분포**
   - Boundary: 55.1% (평균 점수 0.77)
   - Supar: 30.3% (평균 점수 0.48)
   - Whitespace_DP: 14.6% (평균 점수 0.62)
   
3. **개선 여지**
   - Boundary가 과선택될 가능성 (55.1%는 높은 편)
   - Supar 점수가 낮아서 구조적으로 좋아도 선택 안 되는 케이스 존재
   - 근소한 차이 케이스에서 prior_bonus(0.15)가 결정적 영향

---

## 🎯 5대 개선 전략

### 1️⃣ Prior Bonus (현토 마커) 가중치 최적화
**현재**: boundary 후보에 평균 0.15 보너스  
**문제**: 근소한 차이 케이스에서 ±0.15가 결과를 뒤집을 수 있음  
**실험**:
- A: 보너스 0.10 (약화) → boundary 과선택 방지
- B: 보너스 0.20 (강화) → 현토 신호 더 신뢰
- C: 동적 보너스 (유사도 기반 0.10~0.20 가변)

**예상 효과**: +2% F1

---

### 2️⃣ Boundary Model Threshold 조정
**현재**: threshold 0.70, boundary 선택 55.1%  
**문제**: 낮은 threshold로 노이즈가 많은 boundary 후보 생성  
**실험**:
- A: threshold 0.75 (보수적) → 고품질 boundary만
- B: threshold 0.65 (공격적) → 더 많은 boundary 탐색
- C: Confidence 기반 필터링 (top-k만 사용)

**예상 효과**: +1% F1

---

### 3️⃣ Supar Weight 증대
**현재**: supar 선택 30.3%, 평균 점수 0.48 (낮음)  
**문제**: 구조적으로 좋아도 점수가 낮아 선택 안 됨  
**실험**:
- A: supar base score에 +0.05 보정
- B: supar base score에 +0.10 보정
- C: supar 구조 일치도를 별도 가중치로 반영

**예상 효과**: +1% F1

---

### 5️⃣ Ensemble Voting (신규 접근)
**현재**: 단순 최고 점수 선택  
**문제**: 근소한 차이 케이스(152건)에서 단일 지표 의존  
**실험**:
- A: `margin < 0.03`이면 similarity가 더 높은 쪽 선택
- B: 근소한 차이일 때 두 후보 결과를 실제로 비교해 더 나은 쪽 선택
- C: 3개 후보 전체의 가중 평균 (soft voting)

**예상 효과**: +4% F1 (152건 중 70% 개선 시)

---

## 📈 예상 개선 효과

### 보수적 추정: F1 0.80 → 0.85 (+6.25%)
| 개선 항목 | 기여도 |
|-----------|--------|
| Prior bonus 최적화 | +2% |
| Boundary threshold 조정 | +1% |
| Supar weight 증대 | +1% |
| **합계** | **+6%** |

### 낙관적 추정: F1 0.80 → 0.90 (+12.5%)
- 위 4가지 개선: +6%
- **Ensemble voting**: +4% (근소한 차이 152건 중 70% 개선)
- 시너지 효과: +2.5%

---

## 🚀 3단계 실행 계획

### Phase 1: 빠른 Grid Search (소요: 1일)
**목표**: Prior bonus 최적값 탐색

```bash
python scripts/grid_search_pa_weights.py \
  --prior-bonus 0.10,0.15,0.20 \
  --seeds 1,2,3 \
  --output-dir test_results/grid_search_phase1
```

**실험 횟수**: 3 × 3 seeds = 9회  
**검증 지표**: micro_f1_tgt_exact 평균  
**의사결정**: 최고 F1을 달성한 조합을 Phase 2에 적용

---

### Phase 2: 정밀 튜닝 (소요: 1~2일)
**목표**: Phase 1 최선 조합 + Threshold/Supar 조정

```bash
# 예: Phase 1에서 prior_bonus=0.15가 최선이라면
python scripts/grid_search_pa_weights.py \
  --prior-bonus 0.15 \
  --boundary-threshold 0.65,0.70,0.75 \
  --supar-bonus 0.0,0.05,0.10 \
  --seeds 1-10 \
  --output-dir test_results/grid_search_phase2
```

**실험 횟수**: 1 × 1 × 3 × 3 × 10 seeds = 90회  
**검증 지표**: F1 + boundary 선택 비율 + supar 선택 비율  
**의사결정**: F1 0.85 이상 달성 시 Phase 3 진행, 미달 시 Phase 1 재조정

---

### Phase 3: Ensemble 고도화 (선택, 소요: 1일)
**목표**: 근소한 차이 케이스 특수 처리

**구현 방향**:
1. `pa/sentence_splitter.py`에 ensemble voting 로직 추가:
   ```python
   if best_score < second_score + 0.03:
       # 근소한 차이 → similarity 비교
       if second_similarity > best_similarity + 0.05:
           return second_candidate
   return best_candidate
   ```

2. 실험:
   ```bash
   python scripts/run_multitest.py \
     --seeds 1-10 \
     --config best_from_phase2.json \
     --ensemble-mode similarity_tiebreak \
     --output-dir test_results/phase3_ensemble
   ```

**검증**: 근소한 차이 152건에서 개선율 측정 → 70% 이상 개선 시 0.90 달성 가능

---

## 📋 체크리스트

### Phase 1 완료 기준 (완료)
- [x] 27회 실험 완료 (2026-01-11)
- [x] 결과 집계 스크립트 실행 (`scripts/summarize_grid_search.py`)
- [x] 최선 조합 CSV 생성
- [x] F1 개선 확인 (0.80 → 0.84)

### Phase 2 완료 기준 (완료/진행중)
- [x] Grid Search 완료 (Supar Bonus 도입)
- [x] F1 0.85 이상 달성 (0.87 달성 완료!)
- [ ] Boundary 선택 비율 45~50% 범위 (분석 중)
- [ ] Supar 선택 비율 35~40% 범위 (분석 중)
- [ ] 전/후 비교 리포트 생성

### Phase 3 완료 기준 (선택)
- [ ] Ensemble voting 로직 구현
- [ ] 근소한 차이 케이스 152건 중 100건 이상 개선
- [ ] F1 0.90 달성
- [ ] 최종 리포트 생성

---

## 🔧 필요한 스크립트

### 1. Grid Search 러너 ✅
**파일**: `scripts/grid_search_pa_selection_params.py`  
**상태**: 작성 완료  
**기능**: PA 후보 선택 점수에 직접 연결되는 레버(예: prior bonus, threshold, supar bonus, pa_selection_params)를 조합 실험

### 2. Grid Search 집계
**파일**: `scripts/summarize_grid_search.py`  
**기능**:
- 모든 실험 결과 CSV를 읽어 F1 평균/표준편차 계산
- 최선 조합 자동 선택

### 3. 설정 기반 PA 실행
**파일**: `pa/main.py` 수정 (필요 시)  
**기능**:
- `--config` 인자로 JSON 설정 파일 로드
- (필요 시) 실험용 설정을 설정 파일에서 읽어 적용

---

## 💡 추가 고려사항

### 1. 실험 병렬화
- Grid search 27~90회 실험은 시간이 오래 걸릴 수 있음
- 가능하다면 멀티프로세싱으로 병렬 실행 고려

### 2. 중간 체크포인트
- Phase 1 중간에 9회 실험마다 결과 확인
- 명백히 나쁜 조합은 조기 중단 가능

### 3. Ablation 자동화 (관측성 매뉴얼 원칙)
- 각 개선 항목의 독립적 기여도 측정
- 예: `prior_bonus=0.10 vs 0.15` 단독 비교
- 통계적 유의성 검증 (t-test)

---

## 📚 참고 문서

- [SKIPFIX_IMPACT_REPORT.md](test_results/SKIPFIX_IMPACT_REPORT.md): 이전 실험 결과 (considered==1 해결)
- [OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md](OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md): 관측성 우선 개발 원칙
- 분석 스크립트: `scripts/deep_analysis_for_0.9.py`

---

## 🎯 최종 목표

**F1 0.90 달성 시**:
- 실용 수준의 정렬 품질 확보
- 학습 모델(boundary/supar)의 완전한 활용
- 현토 단서의 최적 가중치 확정
- 향후 프로젝트에서 재사용 가능한 가중치 세트 확보

**달성 실패 시**:
- F1 0.85 이상이면 실용 가능 (타협안)
- 추가 접근: 모델 자체 재학습 (boundary/supar threshold 재조정)
- 근본적 개선: 더 많은 학습 데이터 수집

---

**다음 액션**: Phase 1 Grid Search 실행  
**예상 소요**: 3일 (Phase 1~2), 선택적 Phase 3 +1일  
**성공 확률**: 80% (F1 0.85+), 60% (F1 0.90+)

---

## 📊 상세 분석 결과 (2026-01-08 기준)

**분석 대상**: test_results/multitest_seed1_10/20260104_192816 (Seed 1-10)

### Family 선택 분포 (950개 문단)
```
Boundary:      888개 (93.5%) - 평균 점수 0.7005
Whitespace_dp:  44개 ( 4.6%) - 평균 점수 0.5677
Supar:          18개 ( 1.9%) - 평균 점수 0.6894
```

**문제점**:
- Boundary 과다 선택 (93.5%) - prior bonus 0.15 효과
- **Supar Under-utilization**: 높은 점수(0.6894)임에도 1.9%만 선택

### 점수 차이 분포
```
근소한 차이 (<0.03):     29개 (3.1%) ← 최우선 타겟
중간 차이 (0.03~0.10):   58개 (6.1%) ← 개선 가능
명확한 우승 (≥0.10):    863개 (90.8%) ← 안정적
```

**핵심**: 87개(9.2%) 케이스에서 가중치 조정만으로 결과 변경 가능

### Prior Bonus 영향력
- **근소 케이스 29개 중 16개(55.2%)에서 prior bonus가 결정적**
- 현재 설정:
  - Boundary: 0.010 (base) + 최대 0.006 (style) = 최대 0.016
  - Supar: 0.015 (매우 낮음)

---

## 🚀 즉시 실행 가능한 명령어

### 1단계: Phase 1 Grid Search 시작
```bash
# 작업 디렉토리로 이동
cd c:\Users\junto\Downloads\head-repo\hw725\CSP

# Grid search 실행 (약 6시간 소요)
python scripts/grid_search_pa_weights.py \
  --prior-bonus 0.005,0.010,0.015,0.020 \
  --supar-bonus 0.01,0.05,0.10,0.15 \
  --seeds 1,2,3 \
  --output-dir test_results/grid_search_phase1_prior \
  --sample-size 100 \
  --yes

# 결과 요약
python scripts/summarize_grid_search.py \
  --input-dir test_results/grid_search_phase1_prior \
  --metric micro_f1_tgt_exact \
  --output results/phase1_best_config.csv
```

### 2단계: 최선 조합 확인
```bash
# CSV에서 최고 F1 조합 확인
cat results/phase1_best_config.csv | sort -t, -k2 -nr | head -5
```

### 3단계: Phase 2 실행
```bash
# 최선 조합으로 threshold 탐색 (약 8시간 소요)
python scripts/grid_search_pa_weights.py \
  --prior-bonus 0.012 \
  --supar-bonus 0.08 \
  --boundary-threshold 0.68,0.70,0.72,0.75,0.78 \
  --seeds 1-10 \
  --output-dir test_results/grid_search_phase2_threshold \
  --sample-size 100 \
  --yes
```

---

**문서 통합일**: 2026-01-08
**통합 내역**: PA_OPTIMIZATION_GUIDE.md → ROADMAP_TO_F1_0.9.md 병합
