## CSP - 하이브리드 토크나이저 기반 병렬 정렬 (2025-08-18 최신)

한문-한국어 번역 텍스트의 자동 정렬 CLI 도구 (SA/PA)

- 문장을 구로 분할 및 1:1 대응
- **🆕 하이브리드 토크나이저 통합**: 
  - **중국어**: SikuBERT, AnchiBERT (GPU 가속)
  - **한국어**: RoBERTa-Korean-Hanja (한자) + Kiwipiepy (한글)
  - **공통 모듈**: `common/tokenizers/` 디렉토리로 통합
  - **한글 토씨 매칭**: `common/korean_particle_matcher.py` (Kiwipiepy 기반)
- 벡터 임베더 : BGE-M3 (GPU 최적화)
- **한국어 처리**: 한자/한글 분리 → 각각 최적 토크나이저 적용
- 실시간 무결성 검증 시스템 완비
- **SA와 PA 동일한 토크나이저**: 경고 메시지 통일, 출력 형식 일관성

### 실행 순서 (GPU 환경 권장)

#### Poetry 환경 (권장)
```bash
# Poetry 설치 및 환경 설정
curl -sSL https://install.python-poetry.org | python3 -
cd CSP
poetry install
poetry shell

# 🆕 하이브리드 의존성 설치
poetry add kiwipiepy  # 한글 형태소 분석
poetry add transformers  # RoBERTa Korean-Hanja

# GPU PyTorch 설치 (CUDA 사용 시)
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 기존 venv 환경
```bash
python -m venv venv
# Windows  
venv\Scripts\activate
# Linux/WSL
source venv/bin/activate
pip install -r requirements.txt
pip install kiwipiepy  # 🆕 추가 설치
```

### 실행 예시

#### PA (문단→문장 정렬)
```bash
cd pa
poetry run python main.py input.xlsx output.xlsx
# 출력: PA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)
# 결과: output.xlsx (409개 문단 → 2,191개 문장 쌍)
# 처리시간: ~60초 (GPU), 무결성: 99.97%
```

#### SA (문장/구 정렬) 
```bash
cd sa
poetry run python main.py input01.xlsx output01.xlsx
# 출력: SA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)  
# 결과: output01.xlsx (1,846개 문장 → 5,906개 구 쌍)
# 처리시간: ~19초 (GPU), 무결성: 원문 96.4%, 번역문 99.8%
```

#### 무결성 검증
```bash
# PA 무결성 체크 (상세 분석)
poetry run python analyze_text_loss.py

# SA 무결성 체크 (상세 분석)  
poetry run python analyze_sa_text_loss.py
```

### CLI 고급 옵션

#### SA 세부 조정
```bash
# 보수적 분할 (긴 구 선호, 무결성 우선)  
poetry run python sa/main.py --min-src-tokens 5 --max-src-tokens 20

# 세밀한 분할 (짧은 구 선호)
poetry run python sa/main.py --min-src-tokens 2 --max-src-tokens 8

# OpenAI 임베더 사용
poetry run python sa/main.py --embedder openai --openai-model text-embedding-3-large --openai-api-key sk-xxxx
```

#### PA 옵션  
```bash
# 기본 실행 (GPU 자동 감지)
poetry run python pa/main.py

# CPU 강제 모드
CUDA_VISIBLE_DEVICES="" poetry run python pa/main.py
```

### 성능 지표

#### PA (문단→문장 정렬)
- **입력**: 409개 문단
- **출력**: 2,191개 문장 쌍 (`output.xlsx`)  
- **무결성**: 99.97% (39자 손실)
- **처리시간**: ~60초 (GPU 가속)
- **토크나이저**: 하이브리드 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)

#### SA (문장/구 정렬)
- **입력**: 1,846개 문장 (24,805자 원문, 62,420자 번역문)
- **출력**: 5,906개 구 쌍 (`output01.xlsx`)
- **무결성**: 원문 96.4%, 번역문 99.8%  
- **처리시간**: ~19초 (GPU 가속)
- **토크나이저**: 하이브리드 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)

---

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
# CUDA PyTorch 설치
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU 확인
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"
```

## 🚀 실행 방법

### PA (문단→문장 정렬)
```bash
cd pa
poetry run python main.py input.xlsx output.xlsx
```

**출력**: `output.xlsx` (2,191개 문장 쌍)

### SA (문장/구 정렬)
```bash
cd sa
poetry run python main.py input01.xlsx output01.xlsx
```

**출력**: `output01.xlsx` (5,906개 구 쌍)

### 무결성 검증
```bash
# PA 무결성 체크
poetry run python analyze_text_loss.py

# SA 무결성 체크  
poetry run python analyze_sa_text_loss.py
```

## 📁 프로젝트 구조

```
CSP/
├── pa/                           # 문단→문장 정렬
│   ├── main.py                   # PA 메인 실행
│   ├── processor.py              # 문단 처리 로직
│   ├── aligner.py               # 문장 정렬 알고리즘
│   ├── input.xlsx               # 입력 파일 (409개 문단)
│   └── output.xlsx              # 최종 출력
├── sa/                          # 문장/구 정렬
│   ├── main.py                  # SA 메인 실행
│   ├── io_manager.py           # 메인 처리 로직 (병렬 처리)
│   ├── punctuation.py          # 무결성 검증 모듈
│   ├── korean_particle_matcher.py  # 한글 토씨 매칭 (래퍼)
│   ├── input01.xlsx            # 입력 파일 (1,846개 문장)
│   └── output01.xlsx           # 최종 출력
├── common/                     # 공통 모듈
│   ├── io_utils.py            # 파일 입출력
│   ├── korean_particle_matcher.py  # 통합 한글 토씨 매칭 (Kiwipiepy)
│   ├── tokenizers/            # 하이브리드 토크나이저 모듈
│   │   ├── __init__.py       # 토크나이저 인터페이스
│   │   ├── siku_tokenizer.py # SikuBERT 토크나이저
│   │   ├── anchi_tokenizer.py # AnchiBERT 토크나이저
│   │   ├── hybrid_korean_tokenizer.py # 한국어 하이브리드
│   │   ├── roberta_hanja_tokenizer.py # RoBERTa 한자
│   │   └── kiwi_tokenizer.py # Kiwipiepy 한글
│   └── embedders/             # 임베딩 모델
├── models/                    # 사전 훈련 모델
│   └── bge-m3/               # BGE-M3 임베더
├── analyze_text_loss.py      # PA 무결성 검증
├── analyze_sa_text_loss.py   # SA 무결성 검증
└── pyproject.toml            # Poetry 설정
```

## 🔧 핵심 기술

### 하이브리드 토크나이저 시스템
- **SikuBERT**: 사고전서 코퍼스 훈련, 고전 중국어 특화
- **AnchiBERT**: 고전 중국어 BERT, 백업 토크나이저
- **RoBERTa-Korean-Hanja**: 한자 포함 한국어 토크나이저
- **Kiwipiepy**: 한글 형태소 분석, 고어 인식 가능
- **통합 인터페이스**: `common/tokenizers/` 모듈로 SA/PA 공통 사용
- **GPU 배치 처리**: 배치 크기 32로 최적화

### 한글 토씨 매칭 시스템
- **Kiwipiepy 기반**: 직접 연동으로 고성능 분석
- **고전 한문 번역체 지원**: 고어 토씨 패턴 인식
- **통합 모듈**: `common/korean_particle_matcher.py`로 SA/PA 공용
- **무결성 보장**: None 값 검증 및 안전한 토큰 처리

### 임베더
- **BGE-M3**: FlagEmbedding, 다국어 지원
- **GPU 가속**: CUDA 최적화로 고속 처리

### 정렬 알고리즘
- **의미 기반 매칭**: 코사인 유사도 + 한국어 조사 매칭
- **동적 프로그래밍**: 최적 정렬 경로 탐색
- **무결성 보장**: 문자 손실 최소화 알고리즘

## 📈 무결성 분석

### PA 손실 패턴
```
총 손실: 39자 (0.03%)
주요 원인: 공백 정규화, 문장부호 처리
해결책: 공백 패턴 보존 알고리즘 적용
```

### SA 손실 패턴
```
원문 손실: 883자 (3.6%) - 주로 공백, 구두점
번역문 손실: 145자 (0.2%) - 거의 무손실
주요 원인: 문단 끝 공백 제거, nan 값 처리
```

## 🎛️ 고급 설정

### PA 최적화 옵션
```bash
# 보수적 설정 (무결성 우선)
poetry run python pa/main.py --conservative

# 성능 우선 설정
poetry run python pa/main.py --fast
```

### SA 토큰 길이 조정
```bash
# 보수적 분할 (긴 구 선호)
poetry run python sa/main.py --min-src-tokens 5 --max-src-tokens 20

# 세밀한 분할 (짧은 구 선호)  
poetry run python sa/main.py --min-src-tokens 2 --max-src-tokens 8
```

## 🚨 문제 해결

### GPU 인식 실패
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 재설치
poetry run pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu118
```

### 메모리 부족
```bash
# 배치 크기 감소
export BATCH_SIZE=16

# 또는 CPU 모드 강제
export CUDA_VISIBLE_DEVICES=""
```

### 토크나이저 오류
```bash
# 모델 재다운로드
poetry run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('SIKU-BERT/sikubert', force_download=True)"
```

## 📋 입출력 형식

### PA(문단→문장 정렬) 샘플 입력/출력 예시

**입력 파일(Excel, xlsx):**

| 문단(원문) | 문단(번역문) |
|:-----------|:------------|
| 子曰 學而時習之 不亦說乎. 有朋自遠方來 不亦樂乎. 人不知而不慍 不亦君子乎. | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가. 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가. 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가. |

**출력 파일(Excel, xlsx):**

| 문단ID | 문장ID | 원문(분할) | 번역문(분할) |
|:-------|:-------|:-----------|:-------------|
| 1 | 1 | 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 1 | 2 | 有朋自遠方來 不亦樂乎 | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가 |
| 1 | 3 | 人不知而不慍 不亦君子乎 | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가 |

### SA(문장/구 정렬) 샘플 입력/출력 예시

**입력 파일(Excel, xlsx):**

| 원문(샘플) | 번역문(샘플) |
|:-----------|:------------|
| 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 有朋自遠方來 不亦樂乎 | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가 |
| 人不知而不慍 不亦君子乎 | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가 |

**출력 파일(Excel, xlsx):**

| 문장식별자 | 구식별자 | 원문구 | 번역구 |
|:----------|:--------|:-------|:-------|
| 1 | 1 | 子曰 | 공자께서 말씀하셨다 |
| 1 | 2 | 學而時習之 | 배우고 때때로 익히면 |
| 1 | 3 | 不亦說乎 | 또한 기쁘지 아니한가 |
| 2 | 1 | 有朋自遠方來 | 벗이 먼 곳에서 찾아오면 |
| 2 | 2 | 不亦樂乎 | 또한 즐겁지 아니한가 |
| 3 | 1 | 人不知而不慍 | 남이 알아주지 않아도 성내지 않으면 |
| 3 | 2 | 不亦君子乎 | 또한 군자가 아니겠는가 |

---

**CLI 환경에서 대용량 한중문 정렬, 고품질 임베딩 연동, 환경별 설치/실행법을 모두 지원합니다.**
