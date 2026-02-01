# Dansa (斷辭) Research: Phase 5 Statistical Validation

**현토(懸吐) 종결어미의 전근대 문법 체계 검증 연구**

## 연구 결과 요약

| Level | 마커 | 가설 | χ² | p-value | 결과 |
|-------|------|------|---:|--------:|------|
| 1 | 로다 (유사이단) | 감탄/여운 뉘앙스 | 114.16 | 1.2e-26 | ✅ |
| 2 | 니라 vs 라 | 쾌절 vs 미절 | 208.90 | 2.4e-47 | ✅ |
| 3 | 하다 (기사지단) | 역사서 편중 | 9211.88 | ≈0 | ✅ |

---

## 심 파일

### 분석 스크립트
| 파일 | 설명 |
|------|------|
| `dansa_full_survey.py` | Level 1, 2 전수조사 (MDA Protocol) |
| `analyze_hada_by_genre.py` | Level 3 장르별 통계 분석 |
| `phase4_premodern_classify.py` | 664종 마커 분류 엔진 |
| `hyeonto_normalizer.py` | 형태 정규화 로직 |
| `anonymize_dataset.py` | 데이터셋 익명화 (SHA-256) |

### 연구 데이터
| 파일 | 설명 |
|------|------|
| `datasets/phrase_normalized_anonymized.csv` | 익명화된 분석 데이터 (364,007건) |
| `reports/phase4/dansa_full_survey.json` | 통계 검정 결과 |
| `reports/phase4/CLASSIFIED_MARKERS.md` | 마커 분류 상세 |

### 문헌 기준
- 이삼환 《구두지남》
- 임규직 《구두해법》  
- 박문호 《이두해》

---

## 재현 방법

```bash
# 환경 설정
pip install pandas scipy openai tqdm

# Level 1, 2 검증 (LLM 필요)
python dansa_full_survey.py

# Level 3 검증 (통계만)
python analyze_hada_by_genre.py

# 마커 분류 재생성
python phase4_premodern_classify.py
```

---

## 라이선스

연구 목적 사용 허용. 상업적 이용 시 문의 필요.
