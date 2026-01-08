## CSP - 구문분석 기반 병렬 정렬

한문-한국어 번역 텍스트의 자동 정렬 CLI 도구 (SA/PA)

주요 기능
- 문장을 구로 분할하고 1:1 대응
- 🚀 병렬 처리 기본 활성화: BGE/OpenAI 모두 지원(옵션으로 4배 이상 향상)
- 🐳 Docker 기반 안정성 및 재현성 보장
- 구문분석기 통합(공통):
  - 원문(한문): SuPar-Kanbun (GPU)
  - 번역문(한국어): Stanza (GPU)
  - 구문 기반 분할(정확한 문장 분할)
- 하이브리드 토크나이저(공통):
  - 중국어: SikuBERT (GPU)
  - 한국어: RoBERTa-Korean-Hanja + Kiwipiepy
  - 공통 모듈: `common/tokenizers/`
  - 한글 토씨 매칭: `common/korean_particle_matcher.py`
- 벡터 임베더: BGE-M3 FlagModel(기본) + OpenAI(병렬 최적화)
- 한국어 처리: 한자/한글 분리 후 최적 토크나이저 적용
- 실시간 무결성 검증 시스템
- 환경 안정성: PyTorch 2.6.0+cu124, 의존성 고정

### 🚀 Docker 환경 실행 (권장)

#### 환경 요구사항
- **Docker Desktop** (Windows/Mac/Linux)
- **NVIDIA GPU** + **CUDA 12.4** 지원
- **8GB+ RAM** 권장

#### 1️⃣ 프로젝트 시작
```bash
# 컨테이너 빌드 및 시작
cd CSP
docker-compose up -d

# 컨테이너 접속
docker exec -it csp-workspace /bin/bash
```

#### 2️⃣ 환경 확인
```bash
# PyTorch CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 설치된 패키지 확인
pip list | grep -E "(torch|transformers|supar|stanza|openai|bge)"

# SuPar-Kanbun 테스트
python -c "import supar; print('SuPar-Kanbun:', supar.__version__)"

# Stanza 테스트  
python -c "import stanza; print('Stanza:', stanza.__version__)"

# SikuBERT 테스트
python -c "from transformers import BertTokenizer; print('SikuBERT 토크나이저 로드 성공')"
```

#### 3️⃣ 스크립트 실행

**SA (문장/구 정렬) - 의미 기반 (BGE 기본, 병렬 처리 기본 활성화)**
```bash
cd sa
python main.py input.xlsx output.xlsx
# 기본값: --embedder bge --max-workers 4

# OpenAI 사용: python main.py input.xlsx output.xlsx --embedder openai
# 순차 분할 (임베더 미사용): python main.py input.xlsx output.xlsx --embedder none
# 병렬 처리 조정: python main.py input.xlsx output.xlsx --max-workers 8 --chunk-size 200
```

**PA (문단→문장 정렬) - 구문분석 기반 (BGE 기본, 병렬 처리 기본 활성화)**
```bash
cd pa  
python main.py input.xlsx output.xlsx
# 기본값: --embedder bge --max-workers 4

# OpenAI 사용: python main.py input.xlsx output.xlsx --embedder openai
# 순차 분할 (임베더 미사용): python main.py input.xlsx output.xlsx --embedder none
# 병렬 처리 조정: python main.py input.xlsx output.xlsx --max-workers 8 --batch-size 100
```

#### 4️⃣ 컨테이너 관리
```bash
# 상태 확인
docker-compose ps

# 로그 확인  
docker-compose logs csp

# 종료
docker-compose down

# 완전 정리 (볼륨까지 삭제)
docker-compose down -v
```

### 🪟 Windows (cmd.exe) 빠른 실행

```bat
REM SA 실행 (기본 임베더: bge)
cd sa
python main.py input.xlsx output.xlsx --max-workers 4

REM OpenAI 임베더 사용 (환경변수 OPENAI_API_KEY 필요)
cd sa
python main.py input.xlsx output.xlsx --embedder openai --max-workers 8

REM PA 실행 (구문분석 기반)
cd ..\pa
python main.py input.xlsx output.xlsx --max-workers 4 --batch-size 100

REM 정확도 평가 (행 모드 + 자동 시프트)
cd ..\accuracy
python accuracy_evaluator.py --mode row --row-auto-shift --input sa01.xlsx --pred sa01_eval.xlsx --project SA
```

### 📊 성능 지표 (Docker 환경 + 병렬 처리)

#### SA 처리 결과 (OpenAI 병렬 처리)
- **입력**: 1,846개 문장 
- **출력**: 5,906개 구 쌍
- **처리시간**: ~5-8초 (OpenAI 4개 워커 병렬)
- **성능 향상**: 기존 대비 약 4배 빠른 속도
- **무결성**: 성공 5,906, 실패 0

#### PA 처리 결과 (OpenAI 병렬 처리)
- **입력**: 201개 문단
- **출력**: 정확한 문장 분할 및 정렬
- **구문분석기**: SuPar-Kanbun + Stanza + SikuBERT
- **처리시간**: OpenAI 병렬 처리로 대폭 개선
- **성능 향상**: 기존 대비 약 4배 빠른 속도

## 📁 프로젝트 구조

```
CSP/
├── accuracy/                    # 평가 도구(정확도/무결성)
│   ├── README.md                # 정확도 평가 가이드(정본)
│   ├── accuracy_evaluator.py    # 세그먼트 기반 평가(행/문장, 자동 보정 등)
│   ├── compute_thresholds.py    # 퍼센타일/임계값 산출 보조
│   ├── thresholds_config.py     # 프로젝트별 임계값 설정
│   └── ...
├── pa/                          # 문단→문장 정렬 (구문분석 기반)
│   ├── main.py                  # PA 메인 실행
│   ├── processor.py             # 문단 처리 로직
│   ├── aligner.py               # 문장 정렬 알고리즘
│   ├── sentence_splitter.py     # 의미적 분할
│   └── ...
├── sa/                          # 문장/구 정렬 (의미 기반)
│   ├── main.py                  # SA 메인 실행
│   ├── io_manager.py            # 메인 처리 로직(병렬)
│   ├── sa_aligner.py            # 의미 기반 정렬 알고리즘
│   ├── punctuation.py           # 무결성 검증
│   └── ...
├── common/                      # 공통 모듈
│   ├── io_utils.py              # 파일 입출력
│   ├── korean_particle_matcher.py
│   ├── progress_manager.py      # 진행률/상태 관리
│   ├── new_parsers.py           # 구문분석기 통합(SuPar-Kanbun + Stanza)
│   ├── tokenizers/              # 하이브리드 토크나이저(공통)
│   └── embedders/               # 임베딩 모델(BGE 등)
├── documents/                   # 가이드/플로우차트 모음
│   └── ...
└── pyproject.toml               # Poetry 설정
```

## 🔧 핵심 기술

### 구문분석기 시스템 (공통)

- **SuPar-Kanbun**: 한문 전용 구문분석기, 고전 중국어 구문 구조 정확 분석
- **Stanza**: 다국어 구문분석기, 한국어 구문 구조 분석
- **GPU 가속**: CUDA 12.4 최적화로 고속 처리
- **구문 기반 분할**: 의미 단위가 아닌 구문 구조 기반 정확한 문장 분할
- **BGE-M3 FlagModel**: 구문 분할된 문장들의 의미적 매칭

### 하이브리드 토크나이저 시스템 (공통)

- **SikuBERT**: 사고전서 코퍼스 훈련, 고전 중국어 특화
- **AnchiBERT**: 고전 중국어 BERT, 백업 토크나이저
- **RoBERTa-Korean-Hanja**: 한자 포함 한국어 토크나이저
- **Kiwipiepy**: 한글 형태소 분석, 고어 인식 가능
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

## 📋 입출력 형식

### PA(문단→문장 정렬) 샘플 입력/출력 예시

**입력 파일(Excel, xlsx):**

| 문단(원문)                                                                | 문단(번역문)                                                                                                                                                             |
| :------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 子曰 學而時習之 不亦說乎. 有朋自遠方來 不亦樂乎. 人不知而不慍 不亦君子乎. | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가. 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가. 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가. |

**출력 파일(Excel, xlsx):**

| 문단ID | 문장ID | 원문(분할)               | 번역문(분할)                                                   |
| :----- | :----- | :----------------------- | :------------------------------------------------------------- |
| 1      | 1      | 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 1      | 2      | 有朋自遠方來 不亦樂乎    | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가                   |
| 1      | 3      | 人不知而不慍 不亦君子乎  | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가      |

### SA(문장/구 정렬) 샘플 입력/출력 예시

**입력 파일(Excel, xlsx):**

| 원문(샘플)               | 번역문(샘플)                                                   |
| :----------------------- | :------------------------------------------------------------- |
| 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 有朋自遠方來 不亦樂乎    | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가                   |
| 人不知而不慍 不亦君子乎  | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가      |

**출력 파일(Excel, xlsx):**

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

## 🐳 Docker 마이그레이션 (2025-08-23)

### 마이그레이션 배경
- **CUDA 기능 상실**: GPU 가속 불가능으로 성능 저하
- **환경 의존성**: 개발 환경마다 다른 결과

### 해결된 문제들
- ✅ **간접 의존성 공격 차단**: PyTorch CUDA 버전 완전 보호
- ✅ **환경 일관성**: 모든 환경에서 동일한 결과 보장  
- ✅ **성능 최적화**: PyTorch 2.6.0+cu124로 업그레이드
- ✅ **의존성 안정성**: constraints.txt로 모든 패키지 고정

### Docker 환경 장점
- **완전 격리**: Poetry와 시스템 패키지 분리
- **GPU 가속**: NVIDIA RTX 3070 Ti 완전 활용
- **재현 가능**: 개발/운영 환경 100% 일치
- **보안**: 3단계 보호 체계로 의존성 충돌 방지

자세한 문서는 아래 "문서 모음" 섹션을 참고하세요.

---

## 📚 문서 모음

### 가이드(Guide)
- ✅ [정확도 평가 가이드](./accuracy/README.md)
- 📈 [임계값/퍼센타일 산출 스크립트](./accuracy/compute_thresholds.py)
- 🧭 [임계값 설정](./accuracy/thresholds_config.py)
- 🐳 [Docker 설치 가이드](./documents/DOCKER_설치_가이드.md)
- 🚀 [Docker 환경 가이드](./documents/DOCKER_환경_가이드.md)
- 🎯 [SA 튜닝 가이드](./documents/튜닝_가이드.md)

### 플로우차트(Flowchart)
- 🗺️ [CSP 전체 워크플로우](./documents/CSP_전체_워크플로우.md)
- 📁 디렉토리별 플로우차트
  - [SA 디렉토리 플로우차트](./documents/SA_디렉토리_플로우차트.md)
  - [PA 디렉토리 플로우차트](./documents/PA_디렉토리_플로우차트.md)
  - [Common 디렉토리 플로우차트](./documents/Common_디렉토리_플로우차트.md)
  - [Accuracy 디렉토리 플로우차트](./documents/Accuracy_디렉토리_플로우차트.md)
  - [Mermaid 원본(.mmd)](./documents/)

## 🎯 SA 튜닝 초간단 가이드

빠르게 품질을 조정할 때는 아래 옵션만 우선 확인하세요.

- 경계 어긋남 완화: --boundary-bonus, --particle-bonus (dp-window 2~3 권장)
- 매칭이 모호함: --sim-gamma (권장 1.4~1.5)

예시 (Windows cmd):

```bat
python sa\main.py input.xlsx output.xlsx --boundary-bonus 0.35 --particle-bonus 0.25 --dp-window 2 --sim-gamma 1.5
```

## 📝 주요 업데이트

### 2025-08-28: PA 병렬 인자 전달 버그 수정
- ✅ PA에서 `split_source_by_whitespace_and_align()`로 `max_workers`/`batch_size`가 전달되지 않아 발생하던 오류를 수정했습니다.
- ✅ CLI의 `--max-workers`, `--batch-size` 옵션이 OpenAI 임베더까지 정상적으로 반영됩니다.
- 🔧 내부 시그니처 정리: `pa/sentence_splitter.py` 함수 시그니처 확장, `pa/processor.py`와 `pa/aligner.py`에서 인자 전달 경로 정합화.

### 2025-08-23: Docker 환경 마이그레이션
- 🐳 **Docker 기반 환경 구축**: 의존성 문제 완전 해결
- ⚡ **PyTorch 2.6.0+cu124**: torch.load 보안 문제 해결
- 🛡️ **3단계 보호 체계**: constraints.txt + 환경변수 + 패키지 고정
- 🚀 **성능 향상**: SikuBERT + SuPar + Stanza 모두 정상 작동

### 2025-08-21: 구문분석기 통합
- 🧠 **SuPar-Kanbun**: 한문 전용 구문분석기 추가
- 🌐 **Stanza**: 다국어 구문분석 파이프라인 통합
- 🔀 **하이브리드 토크나이저**: SikuBERT + RoBERTa-Hanja + Kiwipiepy

 
