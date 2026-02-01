# 단사(斷辭) 연구: Phase 5 통계 검증 가이드

## 개요

본 연구는 조선시대 현토(懸吐) 분석 프로젝트의 일환으로, 전근대 문헌 기준을 적용한 단사(斷辭) 분류 체계를 검증합니다.

**연구 주제**: 전근대 원전 문헌의 종결어미(단사) 분류 및 통계적 검증
**코퍼스**: 364,007건 (익명화된 병렬 코퍼스)

---

## 1. 단사 6단계 체계

임규직 《구두해법》, 이삼환 《구두지남》, 박문호 《이두해》 기준

| 단계 | 범주 | 한문 | 의미 | 대표 마커 |
|:---:|------|----------|------|----------|
| 1 | 유사이단사 | 游辭以斷 | 감탄, 여운 | `로다`, `하놋다` |
| 2a | 쾌절지단사 | 夬絶 | 단호한 결정 | `니라`, `이니라` |
| 2b | 미절지단사 | 微絶 | 약한 종결 | `라`, `이라` |
| 3 | 기사지단사 | 記史之斷 | 공적 기록체 | `하다` |
| 4 | 서술지단사 | 敍述之斷 | 서술/이야기체 | `하더라`, `러라` |
| 5 | 범론이단사 | 汎論이斷 | 일반적 진술 | `하나니라` |
| 6 | 인용단사 | 引用斷辭 | 인용 종결 | `라하다` |

---

## 2. 통계 검증 결과 요약

### 2.1 Level 1: 유사이단 (`로다`) 검증
- **가설**: `로다` 마커가 감탄·여운을 표현함
- **결과**: χ² = 114.16, p < 10⁻²⁶ ✅

### 2.2 Level 2: 쾌절 vs 미절 (`니라` vs `라`)
- **가설**: `니라`가 `라`보다 단호한 결정을 표현함
- **결과**: χ² = 208.90, p < 10⁻⁴⁷ ✅

### 2.3 Level 3: 기사지단 (`하다`) 장르별 검증
- **가설**: `하다`가 역사서에 편중됨
- **결과**: χ² = 9211.88, p ≈ 0 ✅

---

## 3. 파일 구조

```
hyeonto/
├── scripts/                    # 분석 스크립트
│   ├── phase4_premodern_classify.py  # 664종 마커 분류 엔진
│   ├── dansa_full_survey.py          # Level 1, 2 전수조사 (MDA Protocol)
│   ├── analyze_hada_by_genre.py      # Level 3 장르별 통계 분석
│   ├── hyeonto_normalizer.py         # 형태 정규화 로직
│   └── anonymize_dataset.py          # 데이터셋 익명화 (SHA-256)
├── data/                       # 연구 데이터
│   └── phrase_normalized_anonymized.csv  # 익명화된 분석 데이터 (364,007건)
├── results/                    # 분석 결과
│   └── dansa_full_survey.json        # 통계 검정 결과
└── reports/phase4/             # 리포트
    └── CLASSIFIED_MARKERS.md         # 마커 분류 상세 보고서
```

---

## 4. 실행 방법

### 환경 설정
```bash
# Python 3.9 이상
pip install pandas numpy scipy tqdm openai
```

### OpenAI API (Level 1, 2 검증용)
```bash
# Windows
set OPENAI_API_KEY=sk-...

# Linux/Mac
export OPENAI_API_KEY=sk-...
```

### 스크립트 실행
```bash
# 마커 재분류
python scripts/phase4_premodern_classify.py

# 단사 전수조사 (LLM 사용)
python scripts/dansa_full_survey.py

# 장르별 하다 분석
python scripts/analyze_hada_by_genre.py
```

---

## 5. 문헌 기준

- **임규직 《구두해법》**: 16가지 기능적 종결어미 정의
- **이삼환 《구두지남》**: 한자(曰, 則, 當)와 종결어미 연결
- **박문호 《이두해》 30번**: 탄사(嘆辭) 및 감격 표현 정의

---

## 6. 데이터 가용성

익명화된 데이터셋은 `data/phrase_normalized_anonymized.csv`에서 제공됩니다.
원본 번역문은 SHA-256 해시로 대체되어 연구 재현성을 보장합니다.

---

*Last Updated: 2026-02-01 (Phase 5 Baseline)*�*cascade08"(2813ac07a953d0584d8df76d4b7067e9d304d5f42Lfile:///c:/Users/junto/Downloads/head-repo/hw725/CSP/hyeonto/DANSA_README.md:4file:///c:/Users/junto/Downloads/head-repo/hw725/CSP