# CSP Documentation Guide

> **Documentation index for the CSP (Corpus Split Parallel) pipeline** — an automated system for aligning Classical Chinese texts with Korean translations at paragraph, sentence, and phrase levels.
>
> P2S achieves F1 = 0.9384 (4,934 paragraphs), S2P v2.1 achieves F1 = 0.8555 (446 sentences).

---

## Quick Navigation

| Document | Description | Reading Time |
|----------|-------------|-------------|
| [P2S_MECHANISM.md](P2S_MECHANISM.md) | P2S algorithm: target-anchored splitting, BGE refinement | 20 min |
| [P2S_CODE_ANATOMY.md](P2S_CODE_ANATOMY.md) | P2S code walkthrough with Mermaid diagrams | 15 min |
| [S2P_MECHANISM.md](S2P_MECHANISM.md) | S2P algorithm: BiLSTM phrase alignment, Viterbi decoding | 20 min |
| [S2P_CODE_ANATOMY.md](S2P_CODE_ANATOMY.md) | S2P v2.1 architecture and model internals | 15 min |
| [WORKFLOW.md](WORKFLOW.md) | End-to-end pipeline workflow | 15-20 min |
| [DATA_PREPARATION.md](DATA_PREPARATION.md) | XML to XLSX data conversion pipeline | 10-15 min |
| [PERFORMANCE.md](PERFORMANCE.md) | Benchmarks and optimization guide | 10-15 min |
| [CLOUD_GPU_TESTING.md](CLOUD_GPU_TESTING.md) | RunPod H200 deployment and testing | 10 min |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions | As needed |
| [MULTIVECTOR_VS_DENSE.md](MULTIVECTOR_VS_DENSE.md) | BGE-M3 embedding mode comparison | 10 min |
| [OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md](OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md) | AI agent collaboration design manual | 15 min |

---

## 문서 로드맵

### 처음 시작하는 사용자

1. **[README.md](../README.md)** - 5분
   - 프로젝트 개요, 빠른 시작, 기본 명령어

2. **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - 10분 (선택)
   - XML 원본 데이터 구조, XLSX 정제 파이프라인

3. **[WORKFLOW.md](WORKFLOW.md)** - 15분
   - P2S/S2P 파이프라인 상세, 알고리즘 원리, 데이터 흐름

4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 필요시
   - 자주 발생하는 문제, 진단 방법, 해결책

### 개발자/시스템 관리자

1. **[P2S_CODE_ANATOMY.md](P2S_CODE_ANATOMY.md)** + **[S2P_CODE_ANATOMY.md](S2P_CODE_ANATOMY.md)** - 코드 구조 이해
2. **[PERFORMANCE.md](PERFORMANCE.md)** - 시스템 요구사항, 성능 튜닝, 벤치마크
3. **[CLOUD_GPU_TESTING.md](CLOUD_GPU_TESTING.md)** - RunPod GPU 환경 구축
4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 로그 분석, 데이터 검증

### 결과 분석가

1. **[analytics/monitoring_dashboard.html](../analytics/monitoring_dashboard.html)** - 모니터링 대시보드
2. **[hyeonto/KEY_FINDINGS.md](../hyeonto/KEY_FINDINGS.md)** - 최신 핵심 발견
3. **[WORKFLOW.md](WORKFLOW.md)** - 평가 시스템 섹션

---

## 파일 구조

```
docs/
├── INDEX.md                          ← 이 파일 (문서 가이드)
├── P2S_MECHANISM.md                  ← P2S 메커니즘 상세 (F1=0.9384)
├── P2S_CODE_ANATOMY.md               ← P2S 코드 해부 (경계 모델 아키텍처)
├── S2P_MECHANISM.md                  ← S2P 메커니즘 상세 (F1=0.8555, v2.1)
├── S2P_CODE_ANATOMY.md               ← S2P 코드 해부 (v2.1 Phrase Alignment)
├── DATA_PREPARATION.md               ← 데이터 처리 과정
├── WORKFLOW.md                       ← 전체 워크플로우
├── PERFORMANCE.md                    ← 성능 벤치마크
├── CLOUD_GPU_TESTING.md              ← RunPod GPU 테스트 가이드
├── TROUBLESHOOTING.md                ← 문제 해결
├── MULTIVECTOR_VS_DENSE.md           ← Multi-Vector vs Dense 임베딩 비교
└── OBSERVABILITY_FIRST_PROMPT_DESIGN_MANUAL.md ← 관측성 우선 설계 매뉴얼
```

---

## 주제별 빠른 검색

### P2S 알고리즘
- 메커니즘: [P2S_MECHANISM.md](P2S_MECHANISM.md)
- 코드 구조: [P2S_CODE_ANATOMY.md](P2S_CODE_ANATOMY.md)
- 문제 해결: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 최적화: [PERFORMANCE.md](PERFORMANCE.md)

### S2P 알고리즘
- 메커니즘: [S2P_MECHANISM.md](S2P_MECHANISM.md)
- 코드 구조: [S2P_CODE_ANATOMY.md](S2P_CODE_ANATOMY.md)
- 문제 해결: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 최적화: [PERFORMANCE.md](PERFORMANCE.md)

### 배치 처리
- 워크플로우: [WORKFLOW.md](WORKFLOW.md)
- 문제 해결: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 최적화: [PERFORMANCE.md](PERFORMANCE.md)

---

## 문서 업데이트 이력

### 2026년 2월 14일 (Code Wiki 최적화)
- 전체 문서 영문 Abstract 추가
- INDEX.md PA/SA → P2S/S2P 명칭 통일
- README.md 영한 병기 전면 재작성

### 2026년 2월 10일 (v2.1 업데이트)
- **S2P F1=0.8555** (v2.1 Phrase Alignment, 446문장) 반영
- S2P_CODE_ANATOMY.md 전면 재작성
- S2P_MECHANISM.md v2.1 반영

### 2026년 2월 10일 (초기)
- **P2S F1=0.9384** (4,934문단 전체 테스트 완료) 반영
- BGE Refinement v3 (3-pass) 반영
- P2S_CODE_ANATOMY.md 경계 모델 아키텍처 상세 추가

### 2025년 12월 19일
- XLSX 기반 완전 재정리, 초기 문서 작성

---

**마지막 업데이트**: 2026년 2월 14일
**문서 버전**: 4.0 (Code Wiki 최적화, P2S F1=0.9384, S2P F1=0.8555 v2.1)
