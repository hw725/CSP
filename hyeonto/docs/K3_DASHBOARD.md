# K=3 분석 결과 통합 대시보드

**생성일**: 2026-02-04  
**데이터**: Sentence 183,322건 / Phrase 643,357건

---

## 📌 가중치 체계 선택

### 균등 가중치 (1:1)
**목적**: 데이터의 자연스러운 구조 발견 (클러스터 안정성 검증)  
**폴더**: [`report_1-1/`](../report_1-1/)

### Canon3 가중치 (3:1)
**목적**: 사서+삼경 중심 관점으로 평가 (해석 변경)  
**폴더**: [`report_3-1/`](../report_3-1/)

---

## 📊 주요 시각화

| 시각화 | 균등(1:1) | Canon3(3:1) | 설명 |
|:---|:---:|:---:|:---|
| **3D UMAP** | [링크](../report_1-1/visualizations_k3/k3_embedding_overlay_3d.html) | - | 입체 공간 클러스터 분포 |
| **2D UMAP** | [링크](../report_1-1/visualizations_k3/k3_embedding_overlay_2d.html) | - | 평면 투영 패턴 |
| **Sentence↔Phrase** | [링크](../report_1-1/visualizations_k3/k3_sentence_phrase_overlay_normalized.html) | - | 문장/구 클러스터 비교 |
| **P2S↔S2P Sankey** | [링크](../report_1-1/visualizations_k3/p2s_k3_s2p_k3_sankey.html) | - | 클러스터 흐름 다이어그램 |
| **균등↔가중 Sankey** | - | [Sentence](../report_3-1/sentence_uniform_vs_weighted_sankey.html) / [Phrase](../report_3-1/phrase_uniform_vs_weighted_sankey.html) | 1:1 ↔ 3:1 라벨 전환 |
| **Sentence K 최적화** | [링크](../report_1-1/visualizations_k3/sentence_k_optimization_visualization.png) | - | K=3 선택 근거 |
| **Phrase K 최적화** | [링크](../report_1-1/visualizations_k3/phrase_k_optimization_visualization.png) | - | K=3 선택 근거 |

---

## 📈 클러스터 프로파일

### Sentence (문장 단위)

| 지표 | 균등(1:1) | Canon3(3:1) |
|:---|:---:|:---:|
| **클러스터 프로파일** | [링크](../report_1-1/sentence_k3_normalized/sentence_cluster_profile.md) | - |
| **서종 분포 분석** | [링크](../report_1-1/sentence_k3_normalized/k3_book_distribution_analysis.md) | - |
| **마커 통사 기능 분석(균등)** | [Sentence CSV](../report_1-1/syntactic_function/sentence_uniform_syntax.csv) | - |

### Phrase (구 단위)

| 지표 | 균등(1:1) | Canon3(3:1) |
|:---|:---:|:---:|
| **클러스터 프로파일** | [링크](../report_1-1/phrase_k3_normalized/phrase_cluster_profile.md) | - |
| **서종 분포 분석** | [링크](../report_1-1/phrase_k3_normalized/k3_book_distribution_analysis.md) | - |
| **마커 통사 기능 분석(균등)** | [Phrase CSV](../report_1-1/syntactic_function/phrase_uniform_syntax.csv) | - |

---

## 🔍 가중치 민감도 분석 (메타 분석)

**목적**: 클러스터 **안정성** 검증 (가중치 변화에 대한 강건성 측정)

| 분석 | 링크 | 설명 |
|:---|:---:|:---|
| **Canon3 (3:1)** | [Sentence](../report_3-1/weight_sensitivity/weight_sensitivity_canon3/sentence_WEIGHT_SENSITIVITY_REPORT.md) / [Phrase](../report_3-1/weight_sensitivity/weight_sensitivity_canon3/phrase_WEIGHT_SENSITIVITY_REPORT.md) | Canon3 시나리오 집중 분석 |
| **전체 시나리오 (v6)** | [Sentence](../report_3-1/weight_sensitivity/weight_sensitivity_v6_sentence/) / [Phrase](../report_3-1/weight_sensitivity/weight_sensitivity_v6_phrase/) | 6가지 가중치 시나리오 비교 |

---

## 🔗 문서 링크

- [시각화 해석 가이드](VISUALIZATION_GUIDE.md)
- [핵심 발견 사항](KEY_FINDINGS.md)
- [편향 검증 보고서](BIAS_VALIDATION.md)
- [재현 가이드](REPRODUCE.md)

---

## 💡 사용 팁

1. **먼저 볼 것**: [균등(1:1) 3D UMAP](../report_1-1/visualizations_k3/k3_embedding_overlay_3d.html) → 데이터 구조 파악
2. **비교 분석**: 균등 vs Canon3 클러스터 프로파일 → 가중치 효과 확인
3. **안정성 검증**: [가중치 민감도 분석](#-가중치-민감도-분석-메타-분석) → 클러스터 강건성 평가

**접근 방법**: 마크다운 뷰어나 브라우저에서 직접 열어보세요.
