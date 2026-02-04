# Dansa (斷辭) Research: Phase 5 Statistical Validation

**현토(懸吐) 종결어미의 전근대 문법 체계 검증 연구**

---

## 중요: Docker 명령 사용 가이드

**임베딩 관련 작업은 반드시 Docker 환경에서 실행하세요:**

```bash
# 가중 클러스터링
docker compose exec csp python scripts/cluster_with_weights.py [args]

# 라벨 변화 분석
docker compose exec csp python scripts/analyze_cluster_label_changes.py [args]

# 임베딩 생성/분석
docker compose exec csp python [script_path]
```

**이유**: 임베딩 캐시(.npy), 대용량 데이터 처리는 Docker 컨테이너 내부 경로 필요

---

## 연구 결과 요약

| Level | 마커 | 가설 | χ² | p-value | 결과 |
|-------|------|------|---:|--------:|------|
| 1 | 로다 (유사이단) | 감탄/여운 뉘앙스 | 114.16 | 1.2e-26 | ✅ |
| 2 | 니라 vs 라 | 쾌절 vs 미절 | 208.90 | 2.4e-47 | ✅ |
| 3 | 하다 (기사지단) | 역사서 편중 | 9211.88 | ≈0 | ✅ |

---

## 핵심 파일

### 분석 스크립트
| 파일 | 설명 |
|------|------|
| `scripts/dansa_full_survey.py` | Level 1, 2 전수조사 (MDA Protocol) |
| `phase4_premodern_classify.py` | 664종 마커 분류 엔진 |
| `hyeonto_normalizer.py` | 형태 정규화 로직 |

### 연구 데이터
| 파일 | 설명 |
|------|------|
| `datasets/phrase_normalized_anonymized.csv` | 익명화된 분석 데이터 (364,007건) |
| `results/dansa_full_survey.json` | 통계 검정 결과 |
| `results/CLASSIFIED_MARKERS.md` | 마커 분류 상세 |

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
python hyeonto/scripts/dansa_full_survey.py

# 마커 분류 재생성
python hyeonto/phase4_premodern_classify.py
```

---

## 라이선스

연구 목적 사용 허용. 상업적 이용 시 문의 필요.
