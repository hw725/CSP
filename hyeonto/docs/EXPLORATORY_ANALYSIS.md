# 탐색적 분석 가이드

> **Version**: v7.0 K=3
> **Data Scale**: Sentence 183,322 / Phrase 643,357

이 문서는 핵심 K=3 분석 이외에 추가로 수행 가능한 탐색적 분석들을 설명합니다.
스크립트는 `hyeonto/scripts/` 폴더에 위치합니다.

> **알림**: 탐색적 산출물은 실행 후 `report_1-1/exploratory/` 하위에 생성됩니다.

---

## 목차

1. [이상치 분석 (Outlier Detection)](#1-이상치-분석-outlier-detection)
2. [N-gram 시퀀스 분석](#2-n-gram-시퀀스-분석)
3. [공기어 분석 (Co-occurrence Network)](#3-공기어-분석-co-occurrence-network)
4. [음운 패턴 분석 (Phonetic Pattern)](#4-음운-패턴-분석-phonetic-pattern)
5. [고급 시각화](#5-고급-시각화)

---

## 1. 이상치 분석 (Outlier Detection)

클러스터 중심에서 가장 멀리 떨어진 경계 토큰을 식별합니다.

### 실행

```bash
cd scripts
python detect_outliers_boundary.py --level sentence --k 3
```

### 출력 위치
- `report_1-1/exploratory/outliers_sentence/`

### 주요 출력
| 파일 | 설명 |
|:-----|:-----|
| `outlier_analysis_sentence.md` | 이상치 상세 보고서 |
| `outlier_stats.json` | 통계 데이터 |

### 해석 포인트
- 이상치가 특정 장르에 집중 → 해당 장르의 현토 체계가 비정형적
- 이상치가 고어(古語) 마커를 포함 → 클러스터링이 언어 변화를 감지

---

## 2. N-gram 시퀀스 분석

현토 마커의 연속 패턴을 분석하여 장르별 문법 구조를 파악합니다.

### 실행

```bash
cd scripts
python analyze_ngram_sequences.py --level phrase --n 3
```

### 출력 위치
- `report_1-1/exploratory/ngram_phrase/`

### 주요 출력
| 파일 | 설명 |
|:-----|:-----|
| `ngram_analysis_phrase.md` | N-gram 빈도 분석 보고서 |
| `top_ngrams.json` | 상위 N-gram 목록 (JSON) |

### 해석 포인트
- 장르별 특이 N-gram이 존재 → 각 장르의 문법적 선호도를 반영
- 반복되는 패턴 → 정형화된 구문 구조 (예: `는,요,는,라` = 나열 구조)

---

## 3. 공기어 분석 (Co-occurrence Network)

한자와 현토 마커의 공기 관계를 네트워크로 시각화합니다.

### 실행

```bash
cd scripts
python analyze_cooccurrence_normalized.py --level sentence
```

### 출력 위치
- `report_1-1/exploratory/cooccurrence_normalized/` (정규화 버전, 권장)
- `report_1-1/exploratory/cooccurrence_phrase/` (Phrase 수준)

### 주요 출력
| 파일 | 설명 |
|:-----|:-----|
| `cooccurrence_analysis_normalized.md` | 공기 분석 보고서 |
| `cooccurrence_network_normalized.html` | 인터랙티브 네트워크 |

### 해석 포인트
- 강한 공기 관계 → 의미-문법 연결의 규범화
- 허브 노드 → 다양한 문맥에서 사용되는 범용 마커

### 특수 기능
- **인쇄용 흑백 모드**: 브라우저에서 인쇄(Ctrl+P) 시 자동으로 흑백 변환됨

---

## 4. 음운 패턴 분석 (Phonetic Pattern)

현토 마커의 초성/종성 분포를 분석하여 음운론적 특성을 파악합니다.

### 실행

```bash
cd scripts
python analyze_phonetic_patterns.py --level sentence --level phrase
```

### 출력 위치
- `report_1-1/exploratory/phonetic/`

### 주요 출력
| 파일 | 설명 |
|:-----|:-----|
| `phonetic_analysis_sentence+phrase.md` | 음운 패턴 종합 보고서 |
| `phonetic_heatmap.png` | 초성×종성 히트맵 |

### 해석 포인트
- 특정 초성/종성의 빈도 편중 → 현토의 음운론적 조화 원리
- 클러스터별 음운 분포 차이 → 장르별 발화 특성 반영

---

## 5. 고급 시각화

추가적인 분석적 시각화를 제공합니다.

### 출력 위치
- `report_1-1/exploratory/viz_advanced_sentence/`

### 주요 시각화
| 파일 | 설명 |
|:-----|:-----|
| `advanced_cluster_viz.html` | Sentence 클러스터 고급 산점도 (밀도 기반) |

---

## 분석 의존 관계

```
[run_full_pipeline.py]
    ↓
embedding_cache.pkl (필수 선행)
    ↓
[exploratory scripts]
    ↓
report_1-1/exploratory/*
```

> **참고**: 탐색적 분석 스크립트는 모두 `embedding_cache.pkl`이 생성된 후에 실행해야 합니다. 캐시 생성에는 약 40분(GPU 기준) / 6시간(CPU 기준)이 소요됩니다.
