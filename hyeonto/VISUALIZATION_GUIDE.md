# Hyeonto 시각화 해석 가이드 (v6 + Multi-Resolution)

본 문서는 hyeonto 프로젝트에서 생성된 다양한 시각화 자료를 올바르게 해석하기 위한 지침을 제공합니다. v6 분석에서는 한문 원문과 한국어 번역문이 결합되어 화용론적 뉘앙스가 반영되었으며, Multi-Resolution 분석(K=4/14/24)이 추가되었습니다.

---

## 🌉 1. PA ↔ SA 클러스터 Sankey Diagram - ⭐ BEST

**경로**: `reports/exploratory/pa_sa_sankey/*.html`

### 생성된 Sankey 다이어그램 종류

| 파일명 | 설명 | 용도 |
|:---|:---|:---|
| `pa_k4_sa_k4_sankey.html` | PA(K=4) ↔ SA(K=4) | 거시적 장르 흐름 분석 |
| `pa_k4_sa_k24_sankey.html` | PA(K=4) ↔ SA(K=24) | 거시→미시 분화 분석 |
| `pa_k14_sa_k24_sankey.html` | PA(K=14) ↔ SA(K=24) | 세밀한 구문 기능 매핑 |

### 해석 방법

- **왼쪽 노드 (PA)**: 문장의 종결 방식 및 문체적 특성 (예: 사서 논증, 문집 서사 등)
- **오른쪽 노드 (SA)**: 구 단위의 논리 관계 및 조사/접속 기능 (예: 조건 구문, 시간격 등)
- **연결선 굵기**: 동일 문장 ID를 공유하는 PA-SA 쌍의 개수

**핵심 해석**:
- **두꺼운 연결선**: PA의 정체성이 SA에서도 강력하게 유지됨 (예: 사서 핵심부 계승)
- **분산된 연결선**: 문장 층위에서는 같은 스타일이지만, 내부적으로 다양한 문법 자원 활용

**실제 분석 결과**:
- 총 매칭: **222,548쌍** (PA 87,943건 × SA 294,889건 중 sentence_id 기반 정밀 매핑)
- PA K=4 → SA K=4: 주요 흐름 PA p1 → SA p1 (사서 논증형 계승)

---

## 🔀 2. 클러스터 분화 Sankey (K=4 → K=14/24)

**경로**: 
- `reports/exploratory/cluster_flow_p2s/cluster_flow_k4_to_k14.html`
- `reports/exploratory/cluster_flow_s2p/cluster_flow_k4_to_k24.html`

### 해석 방법

- **왼쪽 노드**: 거시적 클러스터 (K=4)
- **오른쪽 노드**: 미시적 클러스터 (K=14/24)
- **연결선 굵기**: 분화 비율 (해당 K=4 클러스터가 어떤 K=14/24 클러스터로 분화되는가)

**SA K=4 → K=24 분화 패턴**:

| K=4 클러스터 | 주요 분화처 | 해석 |
|:---:|:---|:---|
| p0 (역사서) | p8(16%), p23(15%) | 전쟁서술/편년체로 세분화 |
| p1 (사서 논증) | p20(20%), p11(17%) | 조건문/의문문으로 분화 |
| p2 (정의형) | p5(25%), p7(13%) | 주역/시경 전문으로 분화 |
| p3 (설명형) | p21(24%), p9(11%) | 인과논증/예시제시로 분화 |

---

## 📊 3. 고급 클러스터 산점도 (Convex Hull + 추세선)

**경로**: `reports/*_full/visualization/advanced_cluster_viz.html`

t-SNE 차원 축소를 통해 클러스터를 2D 공간에 배치하고, 다음 요소를 추가합니다:

### 시각적 요소

1. **Convex Hull (다각형 경계)**: 각 클러스터의 외곽선. 클러스터 간 분리 정도를 직관적으로 파악.
2. **클러스터 라벨 (p0, p1, ...)**: 각 클러스터의 중심에 라벨 표시.
3. **Canonicity 추세선**: 사서 비율에 따른 그라데이션. 진한 색 = 사서 비율 높음.

### 지형 해석

- **표준성의 핵심부**: 중앙에 밀집된 사서 특화 클러스터 (`는`, `라`, `니라` 마커)
- **역사서 벨트**: 한쪽으로 길게 형성된 띠 (`하니`, `하야`, `한대` 마커)
- **문학적 주변부**: 외곽에 산재된 클러스터 (당시, 팔대가문초 등)

---

## 💎 4. Joint Embedding 시각화

**경로**: `reports/pa_boundary_v6_full/joint_embedding/joint_embedding_normal_markers_2d.html`

클러스터(다이아몬드)와 현토 마커(원형)를 동일 좌표계에 배치하여 상관관계를 시각화합니다.

### 해석

- **다이아몬드 근처의 원형**: 해당 클러스터에서 빈번하게 사용되는 마커
- **중앙 밀집 영역**: 사서 핵심부 + 정의형 마커 (`는`, `라`, `니라`)
- **외곽 분산 영역**: 문집/역사서 전용 마커 (`하야`, `한대`, `어늘`)

---

## 🌡️ 5. Marker Distribution Heatmap

**경로**: `reports/pa_boundary_v6_full/marker_distribution/marker_distribution_heatmap.html`

특정 현토가 어떤 클러스터에 얼마나 집중되어 있는지 보여주는 통계 지도입니다.

### 해석

| 색상 | 의미 |
|:---:|:---|
| 진한 빨간색 | 해당 현토가 특정 클러스터의 '전담' 마커 |
| 연한 색 (행 전체) | '범용(Universal)' 마커 - 모든 클러스터에서 사용 |
| 진한 색 (한두 칸) | '장르 특화(Specialized)' 마커 |

---

## 📊 6. Marker Entropy Chart

**경로**: `reports/pa_boundary_v6_full/marker_distribution/marker_entropy_chart.html`

현토의 범용성 지수(Normalized Entropy)를 바차트로 보여줍니다.

| Entropy | 색상 | 의미 |
|:---:|:---:|:---|
| 높음 | 초록색 | 모든 장르에서 고르게 쓰이는 범용 현토 (`라`, `는`, `에`) |
| 낮음 | 빨간색 | 특정 장르에서만 쓰이는 특화 현토 (시경의 `요`, 역사서의 `하야`) |

---

## 📈 7. Syntactic Profile Scatter

**경로**: `reports/pa_boundary_v6_full/syntactic_analysis/syntactic_profile_scatter.html`

현토의 문법적 성격을 종결 비율과 다의성(Silhouette) 지표로 분석합니다.

| 축 | 의미 |
|:---:|:---|
| X축 (종결비율) | 1.0 = 문장 끝에만 쓰이는 종결어미, 0.0 = 문장 중간의 접속사 |
| Y축 (Silhouette) | 높을수록 맥락에 따라 의미가 변하는 다의적 현토 |
| 원 크기 | 데이터셋의 전체 빈도 |

---

## 📂 8. 통합 대시보드

**경로**: `reports/dashboard.html`

모든 분석 결과를 한눈에 탐색할 수 있는 인터랙티브 대시보드입니다.

### 구성

- **PA 분석 결과**: K=4/K=14 프로파일, 클러스터 분화 Sankey
- **SA 분석 결과**: K=4/K=24 프로파일, 클러스터 분화 Sankey, 심층 프로파일
- **PA ↔ SA 비교**: 3종 Sankey (K=4-4, K=4-24, K=14-24)
- **탐색적 분석**: 이상치, N-gram, 공기 네트워크, 음운 패턴

---

## 🎨 9. SA K=24 심층 프로파일 해석

**경로**: `reports/sa_boundary_k24_full/sa_deep_profile.md`

일반 프로파일 이상의 심층 지표를 제공합니다.

### 주요 지표

| 지표 | 설명 | 해석 |
|:---|:---|:---|
| **Lift** | 도서별 특이도 (전역 대비 클러스터 내 비율) | Lift > 2.0 = 해당 도서가 특별히 집중됨 |
| **Entropy** | 마커 다양성 (Shannon bits) | 낮을수록 특정 마커에 집중, 높을수록 다양 |
| **Syntactic Guess** | 구문 기능 추정 (Rule-based) | Topic/Conditional/Declarative 등 |

### 대표적 K=24 클러스터 해석

| 클러스터 | Syntactic | Top Lift Book | Entropy | 해석 |
|:---:|:---:|:---|:---:|:---|
| **p5** | Topic (53%) | 시경 (x3.8) | 3.24 | 주제 표지 + 정의문 |
| **p7** | Mixed | 주역 (x7.1) | 4.51 | 정의/해설 혼합 |
| **p17** | Other (45%) | 당시삼백수 (x9.4) | 4.29 | 시적 운율/생략 |
| **p22** | Other (82%) | 춘추좌씨전 (x10.0) | 3.15 | 시간/장소격 집중 |

---

**마지막 업데이트**: 2026-01-13 (v6 + Multi-Resolution)
