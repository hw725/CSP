## CSP - 구문분석 기반 병렬 정렬 (2025-08-21 최신)

한문-한국어 번역 텍스트의 자동 정렬 CLI 도구 (SA/PA)

- 문장을 구로 분할 및 1:1 대응
- **구문분석기 통합** (PA):
  - **원문 (한문)**: SuPar-Kanbun (GPU 가속, 한문 전용 구문분석)
  - **번역문 (한국어)**: Stanza (GPU 가속, 다국어 구문분석)
  - **구문 기반 분할**: 구문 구조를 반영한 정확한 문장 분할
- **하이브리드 토크나이저** (SA):
  - **중국어**: SikuBERT, AnchiBERT (GPU 가속)
  - **한국어**: RoBERTa-Korean-Hanja (한자) + Kiwipiepy (한글)
  - **공통 모듈**: `common/tokenizers/` 디렉토리로 통합
  - **한글 토씨 매칭**: `common/korean_particle_matcher.py` (Kiwipiepy 기반)
- **벡터 임베더**: BGE-M3 FlagModel (GPU 최적화, 다국어 임베딩)
- **한국어 처리**: 한자/한글 분리 → 각각 최적 토크나이저 적용
- 실시간 무결성 검증 시스템 완비
- **의존성 안정화**: transformers 4.36.0, FlagEmbedding 1.1.7 호환

### 실행 순서 (GPU 환경 권장)

#### Poetry 환경 (권장)

```bash
# 1️⃣ 기본 환경 설정 (순서 중요!)
cd CSP
poetry lock && poetry install && poetry update

# 2️⃣ 수동 설치 필수 패키지들 (Poetry로 불가능)
poetry shell

# PyTorch CUDA 11.8 (RTX 3070 Ti 최적화)
poetry run pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# BGE-M3 + NLP 도구들
poetry run pip install FlagEmbedding>=1.2.11 transformers>=4.34.0 openai>=1.0.0

# 3️⃣ 설치 검증
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
poetry run python -c "from FlagEmbedding import FlagModel; print('BGE-M3 ✅')"
```

**⚠️ 중요**: `poetry show`는 Poetry 관리 패키지만 표시. 실제 환경 확인은 `poetry run pip list` 사용!

# 🆕 구문분석기 설치 (PA용)
poetry add supar  # SuPar-Kanbun 한문 구문분석
poetry add stanza  # Stanza 다국어 구문분석

# 🆕 하이브리드 의존성 설치 (SA용)
poetry add kiwipiepy  # 한글 형태소 분석
poetry add transformers==4.36.0  # 안정화된 버전
poetry add FlagEmbedding==1.1.7  # BGE-M3 FlagModel

# GPU PyTorch 설치 (CUDA 11.8 권장)
poetry run pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU\"}')"
```

#### 기존 venv 환경

```bash
python -m venv venv
# Windows  
venv\Scripts\activate
# Linux/WSL
source venv/bin/activate
pip install -r requirements.txt
pip install kiwipiepy  # 🆕 SA용 한글 형태소 분석
pip install supar stanza  # 🆕 PA용 구문분석기
pip install transformers==4.36.0 FlagEmbedding==1.1.7  # 🆕 안정화된 의존성
```

### 실행 예시

#### PA (문단→문장 정렬) - 구문분석 기반

```bash
poetry run python pa\main.py pa\input.xlsx pa\output.xlsx
# 출력: PA: 구문분석기 초기화 완료 (원문: SuPar-Kanbun, 번역문: Stanza, 임베더: BGE-M3 FlagModel)
# 결과: output.xlsx (201개 문단 → 1,365개 문장 쌍)
# 처리시간: ~60초 (GPU), 무결성: 99.97%
# 특징: 구문 구조 기반 정확한 문장 분할
```

#### SA (문장/구 정렬) - 의미 기반

```bash
poetry run python sa\main.py sa\input.xlsx sa\output.xlsx
# 출력: SA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)  
# 결과: output.xlsx (1,846개 문장 → 5,906개 구 쌍)
# 처리시간: ~19초 (GPU), 무결성: 원문 96.4%, 번역문 99.8%
# 특징: BGE-M3 FlagModel 기반 의미적 정렬
```

#### 무결성 검증

```bash
# PA 무결성 체크 (상세 분석)
poetry run python analyze_text_loss.py

# SA 무결성 체크 (상세 분석)  
poetry run python analyze_sa_text_loss.py
```

### Poetry 환경 설정

```bash
# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -

# 프로젝트 설정
cd CSP
poetry install

# 환경 활성화
poetry shell
```

### GPU 환경 설정

```bash
# CUDA PyTorch 설치 (안정화된 버전)
poetry run pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 버전 고정 (호환성 보장)
poetry add transformers==4.36.0 tokenizers==0.15.0 huggingface_hub==0.19.4
poetry add FlagEmbedding==1.1.7  # BGE-M3 FlagModel 지원

# GPU 확인
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"
```

## 📁 프로젝트 구조

```
CSP/
├── pa/                           # 문단→문장 정렬 (구문분석 기반)
│   ├── main.py                   # PA 메인 실행
│   ├── processor.py              # 문단 처리 로직
│   ├── aligner.py               # 문장 정렬 알고리즘
│   ├── sentence_splitter.py     # BGE-M3 기반 의미적 분할
│   ├── input.xlsx               # 입력 파일 (201개 문단)
│   └── output.xlsx              # 최종 출력
├── sa/                          # 문장/구 정렬 (의미 기반)
│   ├── main.py                  # SA 메인 실행
│   ├── io_manager.py           # 메인 처리 로직 (병렬 처리)
│   ├── sa_aligner.py           # 의미 기반 정렬 알고리즘
│   ├── punctuation.py          # 무결성 검증 모듈
│   ├── input.xlsx            # 입력 파일 (1,846개 문장)
│   └── output.xlsx           # 최종 출력
├── common/                     # 공통 모듈
│   ├── io_utils.py            # 파일 입출력
│   ├── korean_particle_matcher.py  # 통합 한글 토씨 매칭 (Kiwipiepy)
│   ├── new_parsers.py         # 🆕 구문분석기 통합 (SuPar-Kanbun + Stanza)
│   ├── tokenizers/            # 하이브리드 토크나이저 모듈 (SA용)
│   │   ├── __init__.py       # 토크나이저 인터페이스
│   │   ├── siku_tokenizer.py # SikuBERT 토크나이저
│   │   ├── anchi_tokenizer.py # AnchiBERT 토크나이저
│   │   ├── hybrid_korean_tokenizer.py # 한국어 하이브리드
│   │   ├── roberta_hanja_tokenizer.py # RoBERTa 한자
│   │   └── kiwi_tokenizer.py # Kiwipiepy 한글
│   └── embedders/             # 임베딩 모델
│       └── bge.py            # 🆕 BGE-M3 FlagModel 통합
├── models/                    # 사전 훈련 모델
│   └── bge-m3/               # BGE-M3 임베더
├── test_5_paragraphs.py      # 🆕 PA 5문단 테스트 스크립트
├── analyze_text_loss.py      # PA 무결성 검증
├── analyze_sa_text_loss.py   # SA 무결성 검증
└── pyproject.toml            # Poetry 설정
```

## 🔧 핵심 기술

### 구문분석기 시스템 (PA 전용)

- **SuPar-Kanbun**: 한문 전용 구문분석기, 고전 중국어 구문 구조 정확 분석
- **Stanza**: 다국어 구문분석기, 한국어 구문 구조 분석
- **GPU 가속**: CUDA 11.8 최적화로 고속 처리
- **구문 기반 분할**: 의미 단위가 아닌 구문 구조 기반 정확한 문장 분할
- **BGE-M3 FlagModel**: 구문 분할된 문장들의 의미적 매칭

### 하이브리드 토크나이저 시스템 (SA 전용)

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
