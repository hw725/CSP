# 📚 CSP 프로젝트

> **XLSX 기반 문단 정렬(PA) 및 문장 정렬(SA) 파이프라인**

## 🎯 프로젝트 개요

CSP는 한문 고전 문헌의 **문단을 문장으로 분할(Paragraph Alignment, PA)**하고 **문장을 구로 분할하여 1:1 대응(Corpus split parallel, SA)**하는 작업을 자동화하는 시스템입니다.

### 주요 기능

- **문단을 문장으로 분할** - PA가 문단 분할 자동화
- **문장을 구로 분할하고 1:1 대응** - SA가 Corpus split parallel 자동화
- **🚀 병렬 처리 기본 활성화**: BGE/OpenAI 모두 지원 (옵션으로 4배 이상 향상)
- **🐳 Docker 기반 안정성 및 재현성 보장**
- **구문분석기 통합 (공통)**:
  - 원문(한문): SuPar-Kanbun (GPU)
  - 번역문(한국어): Stanza (GPU)
  - 구문 기반 분할(정확한 문장 분할)
- **하이브리드 토크나이저 (공통)**:
  - 원문: SikuBERT (한문) + Kiwipiepy (현토)
  - 번역문: RoBERTa-Korean-Hanja + Kiwipiepy
- **벡터 임베더**: BGE-M3 FlagModel(기본) + OpenAI(병렬 최적화)
- **한글 토씨 매칭**: 한자/한글 분리 후 최적 토크나이저 적용
- **실시간 무결성 검증 시스템**
- **무결성 보장**: 원문 문자 100% 보존 (공백 외 손실 없음)
- **토큰 기반 n:m 평가**: 가변 행 개수 처리 가능
- **정확도 평가**: 자동화된 성능 검증

---

## 📂 프로젝트 구조

```
CSP/
├── xlsx/                          # 입력 데이터 (문단병렬)
│   └── {책이름}/
│       ├── {책이름}_문단병렬.xlsx  # PA 입력 (구병렬 정보)
│       ├── {책이름}_문장병렬.xlsx  # GT 문장병렬 (평가용)
│       └── {책이름}_구병렬.xlsx    # GT 구병렬 (SA 평가용)
│
├── xlsx_pipeline_results/         # 배치 처리 결과
│   └── {책이름}/
│       ├── {책이름}_PA_문장병렬.xlsx     # PA 출력
│       ├── {책이름}_PA_eval_row.xlsx    # PA 평가 결과
│       ├── {책이름}_SA.xlsx             # SA 출력
│       └── {책이름}_SA_eval_row.xlsx    # SA 평가 결과
│
├── pa/                            # 문단 정렬 모듈
│   ├── main.py                    # PA 진입점
│   ├── processor.py               # PA 처리 엔진
│   ├── aligner.py                 # 의미 기반 정렬
│   └── dynamic_processor.py        # 동적 처리
│
├── sa/                            # 문장 정렬 모듈
│   ├── main.py                    # SA 진입점
│   ├── processor.py               # SA 처리 엔진
│   └── advanced_segmentation.py   # 고급 분할
│
├── common/                        # 공유 모듈
│   ├── integrity_verifier.py      # 무결성 검증
│   ├── text_processor.py          # 텍스트 처리
│   └── config.py                  # 설정
│
├── accuracy/                      # 정확도 평가
│   ├── accuracy_evaluator.py      # 평가 엔진
│   ├── pa_gt/                     # PA 정답 데이터
│   └── sa_gt/                     # SA 정답 데이터
│
├── analytics/                     # 분석 및 시각화
│   ├── aggregate_batch_results.py # 배치 결과 집계
│   └── 무결성_리포트.csv           # 무결성 검증 리포트
│
├── batch_43books.py               # 43권 배치 처리 스크립트
├── integrity_report.py            # 무결성 리포트 생성
├── docker-compose.yml             # Docker 설정
├── Dockerfile                     # 컨테이너 이미지
└── requirements.txt               # Python 의존성
```

---

## 🚀 빠른 시작

### 1. 환경 구성

```bash
# Docker 환경에서 실행 (권장: torch/GPU/파서 포함)
docker compose up -d

# 로컬 환경(선택): 일부 경량 스크립트만 가능
# - torch/suparkanbun/stanza 등이 없으면 PA 관련 모듈은 경고가 발생할 수 있습니다.
# - 정확한 재현/평가/trace는 아래 "도커에서만" 섹션 커맨드를 사용하세요.
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 단일 책 처리 (PA)

```bash
# 문단병렬 입력 → 문장병렬 출력
docker compose run --rm csp python pa/main.py \
  xlsx/당송팔대가문초한유3/당송팔대가문초한유3_문단병렬.xlsx \
  output.xlsx \
  --embedder bge
```

### 3. 43권 전체 배치 처리

```bash
# 모든 책 자동 처리 (PA + SA + 평가)
docker compose run --rm csp python batch_43books.py
```

---

## 🐳 PA 재현/평가(도커에서만)

로컬 Python 환경에 `torch`/`suparkanbun`/`stanza`가 없으면 PA 관련 경고가 발생할 수 있습니다. **성능/재현/trace/평가는 아래처럼 도커 컨테이너에서만 실행**하는 것을 기준으로 합니다.

### (중요) 실험 산출물 정리 규칙

- 대량 산출물은 기본적으로 `test_results/`, `logs/`에 생성됩니다.
- 작업 중 누적된 산출물이 너무 많아지면, 로컬 전용 `trash/<timestamp>/`로 **이동(삭제 아님)** 해서 폴더를 비웁니다.
- `trash/`와 `logs/`는 로컬 전용이며 git에 포함하지 않습니다.
- 번역문 분할의 **종결부호 뒤 공백 관련 미세 튜닝은 현재 보류**(추후 필요 시 실험 브랜치에서 진행)합니다.

### 1) (선택) 도커에서 의존성 로드 확인

```bash
docker compose run --rm csp python -c "import torch; print('torch', torch.__version__); from pa.processor import process_paragraph_alignment_with_boundary_model; print('import ok')"
```

### (자주씀) 이미 떠있는 컨테이너에서 한 줄 실행 (exec)

```bash
# PA 실행(예: PD test100, strict) - 결정론 재현(권장)
# - PYTHONHASHSEED까지 고정해야 결과 drift가 줄어듭니다.
docker compose exec -e PYTHONHASHSEED=1 csp python -u pa/main.py \
  datasets/pd/test_100.csv \
  test_results/repro_det_thr070_len200_seed1.csv \
  --embedder bge \
  --use-boundary-model \
  --boundary-threshold 0.70 \
  --boundary-min-len 20 \
  --max-length 200 \
  --seed 1 \
  --deterministic

# gold 비교 리포트 (정답: datasets/pa/test_100_from_pd.csv)
docker compose exec csp python -u integrity_report.py \
  --input test_results/repro_det_thr070_len200_seed1.csv \
  --gold datasets/pa/test_100_from_pd.csv
```

### 2) PA 실행 + stage trace(JSONL) 생성 (권장 기본값)

아래 커맨드는 현재 작업 기준으로 **자주 사용하는 기본값**을 고정합니다:

- `--boundary-threshold 0.70`
- `--seed 1 --deterministic`
- `PYTHONHASHSEED=1` (파이썬 해시 랜덤화 고정)
- trace 파일명 규칙: `logs/pa_stage_trace_bthr{bthr}_ml{ml}_seed{seed}.jsonl`

```bash
# adjacent refine ON (권장)
docker compose run -e PYTHONHASHSEED=1 --rm csp python pa/main.py \
  <input.xlsx> \
  <output.xlsx> \
  --embedder bge \
  --use-boundary-model \
  --boundary-threshold 0.70 \
  --boundary-min-len 20 \
  --trace-stages-jsonl logs/pa_stage_trace_bthr0.70_ml10_seed1.jsonl \
  --seed 1 \
  --deterministic

# adjacent refine OFF (비교/ablation)
docker compose run -e PYTHONHASHSEED=1 --rm csp python pa/main.py \
  <input.xlsx> \
  <output.xlsx> \
  --embedder bge \
  --use-boundary-model \
  --boundary-threshold 0.70 \
  --boundary-min-len 20 \
  --disable-adjacent-boundary-refine \
  --trace-stages-jsonl logs/pa_stage_trace_bthr0.70_ml10_seed1_noAdjRef.jsonl \
  --seed 1 \
  --deterministic
```

### 3) stage drift 분석(ground truth + trace)

```bash
docker compose run --rm csp python scripts/analyze_stage_drift.py \
  --gt-xlsx datasets/pa/test_100_from_pd.csv \
  --trace-jsonl logs/pa_stage_trace_bthr0.70_ml20_seed1.jsonl \
  --out-csv test_results/stage_drift_bthr0.70_ml20_seed1.csv

docker compose run --rm csp python scripts/analyze_stage_drift.py \
  --gt-xlsx datasets/pa/test_100_from_pd.csv \
  --trace-jsonl logs/pa_stage_trace_bthr0.70_ml20_seed1_noAdjRef.jsonl \
  --out-csv test_results/stage_drift_bthr0.70_ml20_seed1_noAdjRef.csv
```

### PowerShell 예시(줄바꿈은 백틱 ` 사용)

```powershell
$env:PYTHONHASHSEED = 1

docker compose run --rm csp python pa/main.py `
  <input.xlsx> `
  <output.xlsx> `
  --embedder bge `
  --use-boundary-model `
  --boundary-threshold 0.70 `
  --boundary-min-len 20 `
  --trace-stages-jsonl logs/pa_stage_trace_bthr0.70_ml20_seed1.jsonl `
  --seed 1 `
  --deterministic

docker compose run --rm csp python scripts/analyze_stage_drift.py `
  --gt-xlsx datasets/pa/test_100_from_pd.csv `
  --trace-jsonl logs/pa_stage_trace_bthr0.70_ml20_seed1.jsonl `
  --out-csv test_results/stage_drift_bthr0.70_ml20_seed1.csv

docker compose run --rm csp python scripts/analyze_stage_drift.py `
  --gt-xlsx datasets/pa/test_100_from_pd.csv `
  --trace-jsonl logs/pa_stage_trace_bthr0.70_ml20_seed1_noAdjRef.jsonl `
  --out-csv test_results/stage_drift_bthr0.70_ml20_seed1_noAdjRef.csv
```

---

## 📊 처리 파이프라인

### PA (문단 정렬 파이프라인)

```
입력: 문단병렬 (원문 + 번역문 문단 쌍)
  ↓
1. 목표 분할 (Target Split)
  - 마지막 문장 상태 감지
  - 종료 기호("\", """) 기준 문장 추출
  ↓
2. 동적 프로그래밍 (DP Word-boundary Allocation)
  - 단어 경계 기준 최적 배치
  - 각 문장 → 스팬(start, end) 계산
  ↓
3. 단어 스팬 슬라이싱 (Word Span Slicing)
  - src_text[word_spans[a][0]:word_spans[b-1][1]]
  - 100% 무결성 보장 (문자 손실 0)
  ↓
4. 무결성 검증 (Integrity Verification)
  - 원문 길이 비교: 입력 vs 출력
  - 손실 문자 상세 분석
  ↓
출력: 문장병렬 (분할된 원문 + 번역문)
```

### SA (문장 정렬 파이프라인)

```
입력: 문장병렬 (원문 문장 + 번역문 문장)
  ↓
1. 임베딩 계산 (BGE Model)
  - 각 문장 → 벡터 표현
  ↓
2. 의미 기반 매칭 (Semantic Matching)
  - 원문 문장 ↔ 번역문 문장 유사도 계산
  - 최적 매칭 찾기
  ↓
3. 구병렬 분할 (Phrase Annotation)
  - 문장 내 의미 단위 분할
  ↓
출력: 구병렬 (원문 구 + 번역문 구)
```

---

## 📈 정확도 평가 시스템

### 평가 지표

#### 1. 행 수준 평가 (Row-level)

- **매칭 전략**
  - PA: 소스 기반 스마트 매칭 (source-based smart matching)
  - 같은 원문을 참조하는 행들 매칭
  - 행 분할 차이 자동 처리

#### 2. 토큰 레벨 n:m 매칭 (Token-level n:m Matching)

- **DP 기반 최적화**
  - 동적 프로그래밍으로 최적 매칭 찾기
  - 1:n, n:1, n:m 모든 관계 처리

#### 3. 유사도 계산

- **Jaccard 유사도**
  - 토큰 집합의 교집합 / 합집합
  - 0~1 범위 (1이 완벽 일치)

### 평가 결과 해석

```
F1 Score: 정밀도와 재현율의 조화평균
- 80% 이상: 우수
- 70-80%: 양호
- 50-70%: 개선 필요
- 50% 이하: 심각한 문제

정밀도(Precision): 예측 데이터의 정확성
재현율(Recall): 정답 데이터의 포함도
```

---

## 🔧 주요 모듈 설명

### PA (문단 정렬)

**목적**: 문단을 문장으로 분할 (문단병렬 → 문장병렬)

**핵심 알고리즘**:

1. **Target Split**: 마지막 문장 추출
2. **Dynamic Programming**: 단어 경계 기준 최적 배치
3. **양방향 유사도 측정**: 원문 문장 구조 + 번역문 문장 구조 동시 고려
   - 원문을 문장 단위로 분할하여 유사도 계산 (50% 가중치)
   - 번역문 문장과의 유사도 계산 (50% 가중치)
   - 결합된 유사도로 최적 정렬 수행
   - PA는 원문과 번역문 둘 다 분할하므로 균등 가중치 적용
4. **Word Span Slicing**: 무결성 보존 추출

**실행**:

```bash
python pa/main.py <input.xlsx> <output.xlsx> --embedder bge
```

**입력/출력 예시**:

| 문단(원문)                                                                | 문단(번역문)                                                                                                                                                             |
| :------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 子曰 學而時習之 不亦說乎. 有朋自遠方來 不亦樂乎. 人不知而不慍 不亦君子乎. | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가. 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가. 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가. |

**출력**:

| 문단ID | 문장ID | 원문(분할)               | 번역문(분할)                                                   |
| :----- | :----- | :----------------------- | :------------------------------------------------------------- |
| 1      | 1      | 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 1      | 2      | 有朋自遠方來 不亦樂乎    | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가                   |
| 1      | 3      | 人不知而不慍 不亦君子乎  | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가      |

---

### SA (문장 정렬)

**목적**: 문장병렬 → 구병렬 (Corpus split parallel)

**핵심 알고리즘**:

1. **임베딩**: BGE 모델로 의미 벡터화
2. **매칭**: 코사인 유사도 기반 최적 매칭
3. **분할**: 의미 경계 기반 구 분할

**실행**:

```bash
python sa/main.py <input.xlsx> <output.xlsx> --embedder bge
```

**입력/출력 예시**:

**입력**:

| 원문(샘플)               | 번역문(샘플)                                                   |
| :----------------------- | :------------------------------------------------------------- |
| 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 有朋自遠方來 不亦樂乎    | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가                   |
| 人不知而不慍 不亦君子乎  | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가      |

**출력 (구병렬)**:

| 문장식별자 | 구식별자 | 원문구       | 번역구                             |
| :--------- | :------- | :----------- | :--------------------------------- |
| 1          | 1        | 子曰         | 공자께서 말씀하셨다                |
| 1          | 2        | 學而時習之   | 배우고 때때로 익히면               |
| 1          | 3        | 不亦說乎     | 또한 기쁘지 아니한가               |
| 2          | 1        | 有朋自遠方來 | 벗이 먼 곳에서 찾아오면            |
| 2          | 2        | 不亦樂乎     | 또한 즐겁지 아니한가               |
| 3          | 1        | 人不知而不慍 | 남이 알아주지 않아도 성내지 않으면 |
| 3          | 2        | 不亦君子乎   | 또한 군자가 아니겠는가             |

---

### 무결성 검증 (Integrity Verifier)

**기능**:

- 원문/번역문 길이 비교
- 손실 문자 위치 및 문맥 기록
- 공백 손실 vs 실제 손실 구분

**사용**:

```python
from common.integrity_verifier import verify_global_integrity

passed, losses_df, analysis = verify_global_integrity(
    input_df, output_df, 
    source_col='원문', 
    target_col='원문'
)
```

---

## 📊 배치 처리

### 43권 자동 처리

```bash
docker-compose run csp python batch_43books.py
```

**처리 순서 (책당)**:

1. PA 실행 (문단병렬 → 문장병렬)
2. PA 평가 (정답 있으면)
3. SA 실행 (문장병렬 → 구병렬)
4. SA 평가 (정답 있으면)

**결과**: `xlsx_pipeline_results/{책이름}/`

---

## 📋 결과 분석

### 무결성 리포트 생성

```bash
python integrity_report.py
```

**출력**: `analytics/무결성_리포트.csv`

**포함 내용**:

- 입력 vs 출력 텍스트 길이 비교
- 손실/추가 문자 분석
- 행 분할 차이
- 책별 종합 평가

*참고: 43권 전체 배치 처리 완료 후 분석 결과를 확인하세요.*

---

## 🐛 트러블슈팅

### PA 무결성 경고

```
❌ 무결성 경고: 원문 손실 2.84%
```

**원인**: 공백 손실 (정상)
**해결**: 실제 문자 손실이 있는지 `integrity_losses` 시트 확인

### PA 평가 F1 점수 낮음

```
F1: 34.6% with 98.7% source_mismatch
```

**원인**: 행 매칭 전략 오류
**해결**: 이미 수정됨 (소스 기반 스마트 매칭으로 개선)

### SA 입력 파일 없음

```
⚠️ SA 입력 파일 없음
```

**원인**: PA 출력 또는 GT 문장병렬 파일 부재
**해결**: 파일 경로 확인 (`xlsx/{책이름}/{책이름}_문장병렬.xlsx`)

---

## � 핵심 기술

### 구문분석기 시스템 (공통)

- **SuPar-Kanbun**: 한문 전용 구문분석기, 고전 한문 구문 구조 정확 분석
- **Stanza**: 다국어 구문분석기, 한국어 구문 구조 분석
- **GPU 가속**: CUDA 12.4 최적화로 고속 처리
- **구문 기반 분할**: 의미 단위가 아닌 구문 구조 기반 정확한 문장 분할
- **BGE-M3 FlagModel**: 구문 분할된 문장들의 의미적 매칭

### 하이브리드 토크나이저 시스템 (공통)

- **원문**: SikuBERT (한문) + Kiwipiepy (현토)
- **번역문**: RoBERTa-Korean-Hanja + Kiwipiepy
- **AnchiBERT**: 고전 한문 BERT, 백업 토크나이저
- **통합 인터페이스**: `common/tokenizers/` 모듈로 일관된 토큰화
- **GPU 배치 처리**: 배치 크기 32로 최적화

### 한글 토씨 매칭 시스템

- **Kiwipiepy 기반**: 직접 연동으로 고성능 분석
- **고전 한문 번역체 지원**: 고어 토씨 패턴 인식
- **통합 모듈**: `common/korean_particle_matcher.py`로 SA/PA 공용
- **무결성 보장**: None 값 검증 및 안전한 토큰 처리

### 임베더

- **BGE-M3 FlagModel**: FlagEmbedding 1.1.7, 다국어 지원
- **GPU 가속**: CUDA 최적화로 고속 처리
- **안정화된 의존성**: transformers 4.36.0 호환성 보장
- **고품질 임베딩**: 0.3~0.8 유사도 범위로 정확한 의미 매칭

### 정렬 알고리즘

- **PA**: 구문분석 + BGE-M3 의미 매칭 하이브리드 방식
- **SA**: 의미 기반 매칭 (코사인 유사도 + 한국어 조사 매칭)
- **동적 프로그래밍**: 최적 정렬 경로 탐색
- **무결성 보장**: 문자 손실 최소화 알고리즘, 순차적 텍스트 처리

---

## ⚙️ 설정 (csp_config.json)

모든 프로젝트 설정은 **루트 디렉토리의 `csp_config.json` 파일 하나**에서 관리됩니다.

1. `csp_config.json.example`을 복사하여 `csp_config.json`로 사용하세요.
2. 필요한 값만 수정하면 됩니다. 생략된 값은 기본값이 적용됩니다.

**주요 설정 항목**:

- `results_dir`: XLSX 출력물 저장 경로 (기본: `xlsx_pipeline_results`)
- `embedder`: 공통 임베더 (기본: `bge-m3`)
- `pa_embedder`, `sa_embedder`: 분석별 임베더 (선택)
- `device`: 장치 지정 (`cuda:0`, `cpu` 등)
- `openai_api_key`: OpenAI 임베딩 사용 시 키

**Windows(cmd) 환경변수 예시**:

```bat
set CSP_XLSX_RESULTS=C:\path\to\xlsx_pipeline_results
set CSP_EMBEDDER=bge-m3
set CSP_DEVICE=cuda:0
```

---

## 📊 성능 지표 (Docker 환경 + 병렬 처리)

### SA 처리 결과 (병렬 처리)

- **입력**: 1,846개 문장
- **출력**: 5,906개 구 쌍
- **처리시간**: ~5-8초 (4개 워커 병렬)
- **성능 향상**: 기존 대비 약 4배 빠른 속도
- **무결성**: 성공 5,906, 실패 0

### PA 처리 결과 (병렬 처리)

- **입력**: 201개 문단
- **출력**: 정확한 문장 분할 및 정렬
- **구문분석기**: SuPar-Kanbun + Stanza + SikuBERT
- **성능 향상**: 병렬 처리로 대폭 개선

---

### GPU 활용

```bash
# GPU 활성화 (Docker에서 자동)
docker-compose run --gpus all csp python batch_43books.py
```

### 메모리 절약

```bash
# 배치 크기 조절
python pa/main.py input.xlsx output.xlsx --batch-size 32
```

### 병렬 처리

```bash
# 워커 수 조절
python pa/main.py input.xlsx output.xlsx --max-workers 4
```

## 📝 주요 업데이트

### 2025-12-19: XLSX 기반 문서 완전 재정리

- ✅ **문서 시스템 재구축**: XML 파이프라인 제거, XLSX 기반으로 완전 통일
- ✅ **5개 신규 문서**: INDEX, README, WORKFLOW, TROUBLESHOOTING, PERFORMANCE
- ✅ **명칭 통일**: Corpus split parallel 표준화
- ✅ **배치 처리**: batch_43books.py로 43권 자동 처리
- ✅ **정확도 평가 개선**: 소스 기반 스마트 매칭 (F1: 34.6% → 79.0%)

### 2025-12-14: 원본 텍스트 보존 및 정렬 품질 개선

- ✅ **토큰 단위 보존**: 한자 괄호 augmentation이 토큰 개수를 유지하도록 수정
- 🔒 **출력 크기 보장**: SA가 정확히 N개 항목 반환 보장
- 📊 **유사도 점수 통합**: SA 결과물에도 PA처럼 코사인 유사도 열 추가
- 🧹 **코드 정리**: IntegrityManager 레거시 시스템 완전 제거
- 🛡️ **검증 강화**: augmentation 전후 토큰 개수 일치 검증 추가

### 2025-08-28: PA 병렬 인자 전달 버그 수정

- ✅ PA에서 `max_workers`/`batch_size`가 올바르게 전달되도록 수정
- ✅ CLI 옵션이 OpenAI 임베더까지 정상 반영

### 2025-08-23: Docker 환경 마이그레이션

- 🐳 **Docker 기반 환경 구축**: 의존성 문제 완전 해결
- ⚡ **PyTorch 2.6.0+cu124**: CUDA 성능 최적화
- 🛡️ **3단계 보호 체계**: constraints.txt + 환경변수 + 패키지 고정

### 2025-08-21: 구문분석기 통합

- 🧠 **SuPar-Kanbun**: 한문 전용 구문분석기 추가
- 🌐 **Stanza**: 다국어 구문분석 파이프라인 통합
- 🔀 **하이브리드 토크나이저**: SikuBERT + RoBERTa-Hanja + Kiwipiepy

---

## 📚 참고 문서

- **[정확도 평가 가이드](../accuracy/README.md)** - 평가 지표 및 방법론
- **[새로운 문서 체계](./INDEX.md)** - 전체 문서 가이드
- **[XML to XLSX 정제](./DATA_PREPARATION.md)** - 원본 데이터 정제 프로세스
- **[워크플로우 상세](./WORKFLOW.md)** - PA/SA 알고리즘 상세 설명
- **[문제 해결](./TROUBLESHOOTING.md)** - FAQ 및 문제 진단
- **[성능 최적화](./PERFORMANCE.md)** - 튜닝 가이드

---

**최근 업데이트**: 2025년 12월 19일 - XLSX 기반 완전 재정리
