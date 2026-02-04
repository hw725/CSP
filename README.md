# 📚 CSP (Corpus Split Parallel) Pipeline

> **한문 고전 원문-번역문 정렬 파이프라인**

## 📋 프로젝트 개요

CSP는 한문 고전 문헌을 **문단→문장으로 분할(P2S)** 하고 **문장→구로 분할하여 1:1 정렬(S2P)** 하는 작업을 자동화하는 시스템입니다.

### 주요 기능
- **P2S (Paragraph to Sentence)**: 문단을 문장으로 분할
- **S2P (Sentence to Phrase)**: 문장을 구로 분할하고 1:1 정렬 (경계 모델 v3 기본 적용)
- **무결성 보장**: 원문 문자 100% 보존 (공백 외 손실 없음)
- **GPU 가속**: CUDA 기반 고속 처리

---

## 🚀 빠른 시작

### Docker 환경 실행
```bash
docker-compose up -d
docker-compose exec csp bash
```

### P2S 파이프라인
```bash
python p2s/main.py <input.csv> <output.xlsx>
```

### S2P 파이프라인
```bash
python s2p/main.py <input.csv> <output.xlsx> [--batch-size 32]
```

---

## 📊 모니터링 & 분석

### 마크다운 대시보드 (로컬 서버)

**자동 시작 (권장):**
```bash
# Windows (CMD) - 최상위 디렉토리에서
start_md_server.bat

# Windows (PowerShell) - hyeonto 디렉토리에서
.\hyeonto\start_md_server.ps1

# 또는 빠른 열기
open_dashboard.bat
```

**또는 수동 실행:**
```bash
cd hyeonto
python md_server.py --port 8080
# 브라우저에서 http://127.0.0.1:8080/dashboard.html 열기
```

### 대시보드 기능
- **K=3 클러스터 분석**: 현토 마커 & 서종 분포 분석
- **임베딩 시각화**: 2D/3D UMAP 오버레이 (흑백 인포그래픽)
- **Sankey 다이어그램**: P2S ↔ S2P 흐름 분석 (흑백)
- **검증 분석**: 공기어 & 이상치 탐지
- **라벨 변화**: 1:1 vs 3:1 가중치 비교
- **마크다운 뷰어**: 모든 분석 리포트 렌더링

### 분석 리포트
- 모니터링 대시보드: [analytics/monitoring_dashboard.html](analytics/monitoring_dashboard.html)
- 최신 분석 요약: [hyeonto/KEY_FINDINGS.md](hyeonto/KEY_FINDINGS.md)
- 시각화 해석 가이드: [hyeonto/VISUALIZATION_GUIDE.md](hyeonto/VISUALIZATION_GUIDE.md)

---

## 📊 처리 예시

### P2S (문단 → 문장 분할)

**입력**:

| 문단(원문) | 문단(번역문) |
|:---------|:----------|
| 公子開方事君 不歸視死父. 衛懿公好鶴 不恤死國. 齊桓公得子亂國 | 공자개방이 군주를 섬기며 죽은 아버지를 돌아보지 않았다. 위의공은 학을 좋아하여 나라가 죽는 것을 돌보지 않았다. 제환공은 자란국을 얻었다 |

**출력**:

| 문단ID | 문장ID | 원문(분할) | 번역문(분할) |
|:-----|:-----|:---------|:----------|
| 1 | 1 | 公子開方事君 不歸視死父 | 공자개방이 군주를 섬기며 죽은 아버지를 돌아보지 않았다 |
| 1 | 2 | 衛懿公好鶴 不恤死國 | 위의공은 학을 좋아하여 나라가 죽는 것을 돌보지 않았다 |
| 1 | 3 | 齊桓公得子亂國 | 제환공은 자란국을 얻었다 |

---

### S2P (문장 → 구 분할)

**입력**:

| 원문(샘플) | 번역문(샘플) |
|:---------|:----------|
| 公子開方事君 不歸視死父 | 공자개방이 군주를 섬기며 죽은 아버지를 돌아보지 않았다 |

**출력 (구병렬)**:

| 문장식별자 | 구식별자 | 원문구 | 번역구 |
|:---------|:-------|:------|:------|
| 1 | 1 | 公子開方 | 공자개방이 |
| 1 | 2 | 事君 | 군주를 섬기며 |
| 1 | 3 | 不歸視死父 | 죽은 아버지를 돌아보지 않았다 |

---

## 📁 디렉토리 구조
```
CSP/
├── p2s/           # Paragraph to Sentence 모듈
├── s2p/           # Sentence to Phrase 모듈
├── common/        # 공통 유틸리티 (embedders, tokenizers 등)
├── accuracy/      # 평가 스크립트
├── datasets/      # 입력 데이터셋
│   ├── paragraph/ # 문단 단위 (P2S 입력)
│   ├── sentence/  # 문장 단위 (P2S 정답 / S2P 입력)
│   └── phrase/    # 구 단위 (S2P 정답)
├── models/        # 학습된 경계 모델
└── test_results/  # 테스트 결과물
```

---

## 📈 성능 (2026-01-23 기준)

### P2S (Paragraph → Sentence)
| 지표 | 값 |
|------|-----|
| **F1** | 0.8724 (87.24%) |
| **원문 유사도** | 0.9174 (91.74%) (번역문 일치 문장 기준) |

### S2P (Sentence → Phrase)
| 지표 | 값 |
|------|-----|
| **F1** | 0.8091 (80.91%) |
| **번역문 유사도** | 0.8323 (83.23%) |

---

## �️ 무결성 (Integrity) 검증 (2026-01-23)

| 파이프라인 | 전역 무결성 | 설명 |
|----------|-----------|------|
| **P2S** | **PASS** | 원문 텍스트 100% 보존 (정규화 기준) |
| **S2P** | **FAIL** (-1 char) | 1글자(`曰`) 누락 확인됨. 그 외 99.999% 일치. |

---

## �🔄 마이그레이션 기록 (2026-01-19)

### PA/SA → P2S/S2P 전환
- **폴더명 변경**: `pa/` → `p2s/`, `sa/` → `s2p/`
- **데이터셋 재구성**: `datasets/paragraph`, `datasets/sentence`, `datasets/phrase`

### S2P 설정 수정
- **기본값 변경**: `--use-boundary-model`을 기본 `True`로 설정 (F1 0.8315 재현용)
- **인자 추가**: `--batch-size` 인자 추가 (기본 32)
- **무결성 체크 개선**: 공백/개행/탭 무시하여 오탐 해결

### 검증 결과 (10개 샘플)
- **P2S**: F1 0.85, 번역문 일치율 100%
- **S2P**: 테스트 중 (경계 모델 v3 정상 동작)

---

## 🛠️ 핵심 기술

- **임베더**: BGE-M3 FlagModel (GPU 가속)
- **경계 모델**: Cross-Attention 기반 Boundary Tagger v3
- **구문분석**: SuPar-Kanbun (한문) + Stanza (한국어)
- **토크나이저**: SikuBERT (한문) + Kiwipiepy (현토)

---

## 🐳 개발 환경

### Docker (권장)
GPU/CUDA 의존성과 재현성을 위해 Docker 사용을 권장합니다.

```bash
# 컨테이너 빌드 및 실행
docker-compose up -d
docker-compose exec csp bash

# 헬퍼 스크립트 (Windows)
./docker.ps1 python scripts/example.py
```

### 패키지 관리
- **Docker**: `uv`를 사용한 고속 패키지 설치 (pip 대비 2-10배 빠름)
- **로컬**: `.venv` + `requirements.txt` (torch 2.9.1)
- **버전 차이**: Docker는 `torch==2.6.0` (공식 이미지 기준), 로컬은 `torch==2.9.1`

### 보안 점검
```powershell
# 로컬 환경 취약점 점검
./scripts/safety_check.ps1

# Docker 환경 취약점 점검
./scripts/safety_check.ps1 -Docker

# 자동 수정 시도
./scripts/safety_check.ps1 -Fix
```

---

**최종 업데이트**: 2026년 02월 04일 - 모니터링 대시보드 및 K=3 분석 문서 정합화

