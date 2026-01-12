# Hyeonto 연구 다음 단계 (Next Steps)
**작성일**: 2026-01-11 (v6 업데이트)
**목적**: v6 분석 완료 후 학술 발표/논문 작성을 위한 로드맵

---

## 🎯 현재 상태 (As of 2026-01-11)

### ✅ 완료된 작업

1. **V6 데이터 분석 완료** (PA 87,943 + SA 294,889건)
   - 번역문(tgt) 포함 BGE-M3 임베딩
   - 16개 클러스터 프로파일링 (PA/SA 각각)
   - 가중치 민감도 분석 (사서 5.0x, 서경 3.0x 등)
   - PA→SA 위계적 흐름 분석 (Sankey Diagram)

4. **편향 검증 완료 (v6)**
   - **영가설(랜덤) 기각**: Cohen's d = 79.5 (p < 0.0001)
   - **반대가설(역가중치) 기각**: 클러스터 구성의 데이터 내재성 증명
   - **대립가설(삼경) 기각**: 사서의 절대적 중심성 확인

5. **현토 패턴 체계적 검증 완료 (v6)**
   - **-러니(회상)만 3단계 검증 통과**
   - -하나니: 영가설 통과(z=11.0)되었으나 반대가설 기각 실패 (-니라도 9.6%)
   - -리오 & -잇가: 둘 다 의문 계열로 공유
   - 영가설/반대가설 검증으로 학술적 엄밀성 확보

6. **논문 Methods 섹션 초안 작성 중**
   - `reports/paper/METHODS_SECTION_DRAFT.md` 생성
   - 3단계 가설 검증 프레임워크 포함
   - 장르별 차이: 사서는 현재 중심, 역사서는 과거 우세

### ⚠️ 남은 작업 (Critical Path)

1. **논문 Methods 섹션 작성** (2주)
2. **학술 발표 준비** (2주)

---

## 📅 단계별 실행 계획

### Week 1: 시제 형태소 분석 통합 (1/13 - 1/17)

#### Day 1-2: kiwipiepy 설치 및 시제 분석

```bash
# Docker 환경에서 실행
docker exec -it csp-workspace bash

# kiwipiepy 설치
pip install kiwipiepy

# 시제 형태소 분석 실행
python scripts/analyze_tense_morphemes.py \
    --csv hyeonto/reports/pa_boundary_v6_full/boundary_clusters.csv \
    --out-dir hyeonto/reports/tense_analysis \
    --min-count 50

# 예상 소요 시간: 30~60분
```

**예상 출력**:
- `tense_morpheme_markers.csv`: 시제 포함 현토 목록
- `normalization_impact.csv`: 정규화 영향 분석
- `tense_analysis_report.md`: 마크다운 리포트

#### Day 3-5: 교차 검증 (선택)

```bash
# 사서 제외 실험
python scripts/cross_validate_saseo.py \
    --csv hyeonto/datasets/pa_merged_v2.csv \
    --mode exclude_saseo \
    --out-dir hyeonto/reports/bias_validation/cross_validation

# 사서만 실험
python scripts/cross_validate_saseo.py \
    --csv hyeonto/datasets/pa_merged_v2.csv \
    --mode only_saseo \
    --out-dir hyeonto/reports/bias_validation/cross_validation
```

---

### Week 2-3: 논문 Methods 섹션 작성 (1/20 - 1/31)

#### 논문 구조 (예시)

```latex
\section{Methods}

\subsection{Data Collection and Preprocessing}
- 382,832 boundary instances (PA 87,943 + SA 294,889)
- Korean translation included in embedding (V6 methodology)
- BGE-M3 multilingual embeddings

\subsection{Hierarchical Weighting}
- Saseo (Four Books): 5.0x
- Samgyeong (Three Classics): 3.0x
- Others: 1.0x-2.0x
- Rationale: Reflects Joseon-era canonical hierarchy

\subsection{Bias Validation}
Bias Score = 0.286 (below threshold 0.3)
- Alternative hypothesis tests: Cohen's d = 282.018
- Random label permutation: significant difference confirmed

\subsection{Hierarchical Flow Analysis}
- PA clusters represent sentence-level stylistic identity
- SA clusters represent phrase-level grammatical resources
- Cross-level flow analysis via Sankey diagram (74,998 matches)
```

---

### Week 4: 학술 발표 준비 (2/3 - 2/7)

#### 발표 자료 구성 (20~30분 발표 기준)

**슬라이드 구성**:

1. **Title & Motivation** (2분)
   - 연구 배경: 현토의 중요성
   - v6 혁신: 번역문 포함 임베딩

2. **Data & Methods** (5분)
   - 데이터: 38만 건 (PA + SA)
   - 가중치: 사서 5x (+ 민감도 분석)
   - 편향 검증: Bias Score 0.286

3. **Key Findings** (10분)
   - 발견 1: 사서는 문법적 북극점 (Canonicity 48.8%)
   - 발견 2: PA→SA 위계적 흐름 (Sankey)
   - 발견 3: 다의성 자동 탐지 (Entropy)
   - 시각화: Joint Embedding, Heatmap

4. **Applications** (5분)
   - 한문 교육: 현토 암기 순서 최적화
   - NLP: 한문 번역 모델 개선

5. **Q&A** (5분)

---

## 🎓 학회 제출 전략

### 국내 학회 (2026년 상반기)

#### 1차 목표: 한국한문학회 춘계 학술대회

**제출 마감**: 2026년 3월 중순 (예상)
**발표일**: 2026년 5월 중순 (예상)

**초록 예시**:
```
현토의 위계적 구조 분석: 번역문 통합 임베딩 기반 38만 건 분석

본 연구는 사서삼경 등 10종 경전에서 추출한 382,832건의 현토 데이터를
대상으로, 번역문을 포함한 다국어 임베딩(BGE-M3)과 클러스터링을 통해
현토의 통사적 기능을 분석하였다.

주요 발견 사항: (1) 사서는 전체 현토 체계의 "문법적 북극점" 역할을 하며,
PA-12 클러스터에서 48.8%의 집중도를 보임. (2) 문장 단위(PA)의 문체적
정체성이 구 단위(SA)의 문법 자원으로 74% 수렴 (Sankey 분석). (3) 번역문
포함으로 화용론적 뉘앙스 포착 가능.
```

### 국제 학회 (2026년 하반기)

#### 1차 목표: COLING 2026

**제출 마감**: 2026년 6월 중순 (예상)
**제출 형식**: Long Paper (8 pages + references)

**차별화 포인트**:
- 대규모 데이터 (38만 건)
- 번역문 통합 임베딩 (v6 신규)
- 위계적 흐름 분석 (Sankey)

---

## 📝 체크리스트

### Phase 1: 시제 분석 (1주)
- [ ] kiwipiepy 설치 및 시제 분석 실행
- [ ] 시제 정보 손실 평가 (>5%면 재처리)

### Phase 2: 논문 작성 (2주)
- [ ] Methods 섹션 작성 (LaTeX)
- [ ] Results 섹션 작성
- [ ] 그림/표 준비 (Sankey, Joint Embedding 등)

### Phase 3: 발표 준비 (1주)
- [ ] 발표 슬라이드 작성 (20~30장)
- [ ] 시각 자료 준비 (고해상도 PNG)
- [ ] Q&A 예상 질문 준비

---

## 🚀 장기 목표 (2027년 이후)

### 1. 저널 논문 게재
- Language Resources and Evaluation (Springer)
- Digital Scholarship in the Humanities (Oxford)

### 2. 오픈 소스 공개
```
GitHub Repository: hyeonto-analysis
├── data/              # 데이터셋 (CSV)
├── scripts/           # 전체 스크립트
├── docs/              # 문서화
└── LICENSE            # MIT or CC-BY
```

---

**작성 완료**: 2026-01-11 (v6)
**책임자**: CSP Research Team

**"데이터로 증명하고, 투명하게 공개하라."** 🔬✨
