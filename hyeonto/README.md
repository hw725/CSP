# Dansa (斷辭) Research: Phase 5 Statistical Validation

**현토(懸吐) 종결어미의 전근대 문법 체계 검증 연구**

> **Branch**: `research/dansa-phase5`  
> **Last Verified**: 2026-02-01

---

## 📊 연구 결과 요약

| Level | 마커 | 가설 | χ² | p-value | 결과 |
|-------|------|------|---:|--------:|------|
| 1 | 로다 (유사이단) | 감탄/여운 뉘앙스 | 114.16 | 1.2e-26 | ✅ |
| 2 | 니라 vs 라 | 쾌절 vs 미절 | 208.90 | 2.4e-47 | ✅ |
| 3 | 하다 (기사지단) | 역사서 편중 | 9211.88 | ≈0 | ✅ |

---

## 🔄 재현 가이드 (Reproduction Guide)

### 1. 브랜치 클론
```bash
git clone -b research/dansa-phase5 https://github.com/<repo>/CSP.git
cd CSP/hyeonto
```

### 2. 환경 요구사항

| 항목 | 요구사항 |
|------|----------|
| **Python** | 3.10.11+ |
| **Docker** | ❌ 불필요 |
| **GPU** | ❌ 불필요 |
| **OS** | Windows / macOS / Linux |

```bash
# 패키지 설치
pip install -r requirements.txt

# 또는 직접 설치
pip install pandas scipy tqdm numpy openai
```

### 3. 핵심 검증 실행

```bash
# Level 1, 2 검증 (로다/니라 LLM 분석)
# ⚠️ LLM API 필요 (OpenAI 또는 로컬 Ollama)
cd hyeonto
python scripts/dansa_full_survey.py

# Level 3 검증 (하다 장르별 통계)
python scripts/analyze_hada_by_genre.py

# 마커 분류 재생성 (664종)
python scripts/phase4_premodern_classify.py
```

### 4. 예상 결과
| 항목 | 예상값 |
|------|--------|
| 총 데이터 | 364,007건 |
| 미분류 마커 | 515건 |
| 단사_미절 (라) | 29,728건 |
| 하다 역사서 비율 | 7.53% |

---

## 📁 디렉토리 구조

```
hyeonto/
├── data/
│   └── phrase_normalized_anonymized.csv  # 익명화 데이터 (364,007건)
├── scripts/
│   ├── dansa_full_survey.py              # Level 1, 2 전수조사
│   ├── analyze_hada_by_genre.py          # Level 3 장르 분석
│   ├── phase4_premodern_classify.py      # 664종 마커 분류
│   └── hyeonto_normalizer.py             # 형태 정규화
├── results/
│   ├── dansa_full_survey.json            # Level 1, 2 통계 결과
│   ├── hada_genre_analysis.json          # Level 3 통계 결과
│   └── classified_markers.json           # 마커 분류 결과
└── reports/
    ├── phase4/                           # 상세 보고서
    └── validation/                       # LLM 검증 결과
```

---

## 📚 문헌 기준

| 문헌 | 저자 | 역할 |
|------|------|------|
| 《구두지남》 | 이삼환 | 쾌절/미절 구분 |
| 《구두해법》 | 임규직 | 유사이단/기사지단 정의 |
| 《이두해》 | 박문호 | 역사적 용례 참조 |

---

## ⚠️ 주의사항

1. **경로**: 모든 스크립트는 `Path(__file__)`로 상대경로 사용. 어디서든 실행 가능.
2. **LLM**: Level 1, 2 검증은 LLM API 필요. 결과는 `results/dansa_full_survey.json`에 저장됨.
3. **익명화**: 원문/번역문은 SHA-256 해시 처리됨. 원본 복원 불가.

---

## 📄 라이선스

연구 목적 사용 허용. 상업적 이용 시 문의 필요.
