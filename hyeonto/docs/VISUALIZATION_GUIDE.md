# Hyeonto 시각화 해석 가이드 (v7.0 K=3)

본 문서는 hyeonto 프로젝트에서 생성된 다양한 시각화 자료를 올바르게 해석하기 위한 지침을 제공합니다.

> **? 분석 단위 (v7.0 K=3)**
> - **Sentence**: 문장 단위 클러스터링 (183,322건)
> - **Phrase**: 구 단위 클러스터링 (643,357건)

---

## ? 1. Sentence ↔ Phrase 클러스터 Sankey Diagram - ? BEST

**경로**: 
- 균등(1:1): `../report_1-1/visualizations_k3/p2s_k3_s2p_k3_sankey.html`
- 균등↔가중 전환: `../report_3-1/sentence_uniform_vs_weighted_sankey.html`

### 해석 방법

- **왼쪽 노드 (Sentence)**: 문장의 종결 방식 및 문체적 특성 (K=3)
- **오른쪽 노드 (Phrase)**: 구 단위의 논리 관계 및 조사/접속 기능 (K=3)
- **연결선 굵기**: 동일 문장 ID를 공유하는 Sentence-Phrase 쌍의 개수

**핵심 해석**:
- **두꺼운 연결선**: Sentence의 정체성이 Phrase에서도 강력하게 유지됨 (예: 사서 핵심부 계승)
- **분산된 연결선**: 문장 층위에서는 같은 스타일이지만, 내부적으로 다양한 문법 자원 활용

**실제 분석 결과**:
- 균등(1:1): `../report_1-1/visualizations_k3/`

---

## ? 2. 3D/2D UMAP 임베딩 오버레이

**경로**:
- `../report_1-1/visualizations_k3/k3_embedding_overlay_3d.html` (3D - 권장)
- `../report_1-1/visualizations_k3/k3_embedding_overlay_2d.html` (2D)

### 해석 방법

- **점 색상**: K=3 클러스터 소속 (p0, p1, p2)
- **점 형태**: Sentence(●) vs Phrase(▲)
- **밀집 영역**: 유사한 문법 패턴을 공유하는 경계들

### 지형 해석

| 영역 | 특징 | 해석 |
|:---|:---|:---|
| **중앙 밀집** | p1 클러스터 | 문장/구 패턴 중심부 |
| **상단 띠** | p0 클러스터 | 서종 분포 상 삼경 비중 높음 |
| **하단 분산** | p2 클러스터 | 역사서/기타 비중 높음 |

---

## ? 3. 한자-현토 공기 네트워크

**경로**: (레거시 산출물 — 현재 레포 미포함)

### 해석 방법

- **중앙 큰 노드**: 빈도가 높은 핵심 한자/현토
- **연결선 굵기**: PMI(Pointwise Mutual Information) 점수 - 강한 공기 관계
- **노드 색상**: 한자(파란색) vs 현토(녹색)

### 테마 전환

v6.9.4에서는 **Print-Ready Grayscale 모드**가 추가되었습니다:
- 다크 모드: 기본 화면 표시용
- 라이트(흑백) 모드: 인쇄 및 논문 삽입용
- `localStorage`로 테마 상태 저장

---

## ? 4. 서종 분포 비교 차트

**경로**: `../report_1-1/visualizations_k3/k3_sentence_phrase_overlay_normalized.html`

### 해석 방법

- **X축**: 클러스터 (p0, p1, p2)
- **Y축**: 서종(書種)별 비율
- **막대 색상**: Sentence(청색) vs Phrase(녹색)

**핵심 발견**:
- Sentence/ Phrase K=3 분포는 `k3_book_distribution_analysis.md` 참조

---

## ? 5. 대시보드 사용법

**경로**: `../../analytics/monitoring_dashboard.html`

### 브라우저에서 직접 열기 (권장)

**HTML 시각화 파일**은 로컬 서버 없이 브라우저에서 바로 열 수 있습니다:
- `k4_embedding_overlay_3d.html`
- `k4_embedding_overlay_2d.html`
- `sankey_sentence4_phrase4.html`
- `cooccurrence_network_normalized.html`

### 마크다운 뷰어 사용 시 (로컬 서버 필요)

마크다운 문서(`.md`)를 뷰어에서 보려면 로컬 서버가 필요합니다:

```bash
cd hyeonto/reports
	python -m http.server 8080
	# 브라우저에서 http://localhost:8080/analytics/monitoring_dashboard.html 접속
```

**대안**: 마크다운 파일은 텍스트 에디터(VS Code, Typora 등)에서 직접 열어서 읽을 수 있습니다.

---

## ? 6. 클러스터 프로파일 해석

**경로**:
- `../report_1-1/sentence_k3_normalized/sentence_cluster_profile.md`
- `../report_1-1/phrase_k3_normalized/phrase_cluster_profile.md`

### 주요 지표

| 지표 | 설명 | 해석 |
|:---|:---|:---|
| **Canonicity** | 사서 비율 (%) | 높을수록 사서 원본 현토에 가까움 |
| **Entropy** | 마커 다양성 (Shannon bits) | 낮을수록 특정 마커에 집중 |
| **Top Markers** | 상위 빈도 현토 | 클러스터의 문법적 정체성 |

### K=3 클러스터별 해석

- 세부 수치는 `sentence_cluster_profile.md`, `phrase_cluster_profile.md` 참고
- 서종 분포 요약은 각 `k3_book_distribution_analysis.md` 참고

---

## ?? 7. 부가 분석 결과

### 1인칭 표지 분석

**경로**: (레거시 산출물 — 현재 레포 미포함)

- **총 화자 표현**: 8,232 instances
- **고유 마커 유형**: 178 distinct types
- **상위 마커**: `라하고`(1,454), `호되`(1,168), `노라`(857)

### 장르별 예문 분석

**경로**: (레거시 산출물 — 현재 레포 미포함)

- 문집(37.7%), 역사서(26.6%), 삼경(22.2%), 사서(9.6%), 예학(3.9%)
- 각 장르별 대표 예문 및 현토 패턴

---

**마지막 업데이트**: 2026-02-04 (K=3 Production Sync)
