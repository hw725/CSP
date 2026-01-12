# 데이터 접근 및 재현성 안내 (Data Availability)

## 데이터 출처

본 연구에 사용된 현토 데이터는 다음 DB에서 제공하는 
사서삼경 및 기타 유교 경전의 현토본을 기반으로 합니다.

| DB | URL |
|:---|:---|
| **동양고전종합DB** | https://db.juntong.or.kr |
| **동양고전번역용례** | https://db.juntong.or.kr/example |

## 저작권 및 재현성 안내

⚠️ **일부 텍스트는 저작권 문제로 공개 접근이 제한되어 있습니다.**

공개된 텍스트만으로도 **유사한 결과 재현이 가능**합니다.
가중치 민감도 테스트 결과, 클러스터 구성은 데이터 부분집합에서도 안정적이었습니다:

- 사서 클러스터 분리 현상
- 정의형 마커 우세 패턴  
- PA→SA 위계적 흐름

## 재현성 검증

### 방법 1: 공개 데이터로 부분 재현

DB에서 공개된 텍스트만으로도 본 연구의 핵심 발견을 검증할 수 있습니다:
- 사서 클러스터 분리 현상
- 정의형 마커 우세 패턴
- PA→SA 위계적 흐름

### 방법 2: 코드 실행 테스트

```bash
# 1. 코드 클론
git clone https://github.com/hw725/CSP

# 2. 데이터 배치
# DB에서 취득한 데이터를 hyeonto/datasets/ 경로에 배치

# 3. 파이프라인 실행
docker exec csp-workspace python scripts/cluster_pa_boundary_functions.py \
    --input hyeonto/datasets/pa_merged_v2.csv \
    --out-dir hyeonto/reports/pa_boundary_v6_full \
    --k 16 --use-src --use-tgt
```

### 데이터 스키마

분석에 필요한 CSV 컬럼 구조:

| 컬럼명 | 설명 | 예시 |
|:---|:---|:---|
| `src_l` | 좌측 한문 원문 | 子曰學而時習之 |
| `src_r` | 우측 한문 원문 | 不亦說乎 |
| `tgt_l` | 좌측 번역문 | 공자께서 말씀하시길 |
| `tgt_r` | 우측 번역문 | 기쁘지 아니한가 |
| `marker` | 현토 마커 | 하시니 |
| `book` | 도서명 | 논어집주 |

---

**작성일**: 2026-01-12
