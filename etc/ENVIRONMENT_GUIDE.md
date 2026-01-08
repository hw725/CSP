# CSP 환경 설정 가이드

## 개요
CSP 프로젝트의 완벽한 환경 설정을 위한 종합 가이드입니다.

## 1. 기본 Poetry 환경

### 핵심 의존성 (pyproject.toml)
```toml
[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
openpyxl = "^3.1.0"
xlsxwriter = "^3.1.0"
spacy = "^3.7.0"
kiwipiepy = "^0.17.0"
```

### Poetry 환경 초기화
```bash
cd c:\Users\junto\Downloads\head-repo\CSP
poetry install
```

## 2. 수동 설치 패키지

### PyTorch CUDA 환경
- **버전**: PyTorch 2.1.1 + CUDA 11.8
- **GPU**: RTX 3070 Ti 최적화
- **설치**: `poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### FlagEmbedding (BGE-M3)
- **버전**: FlagEmbedding >= 1.2.11
- **기능**: Multi-vector embeddings (dense + sparse + colbert)
- **설치**: `poetry run pip install FlagEmbedding>=1.2.11`

### Transformers
- **버전**: transformers == 4.34.1
- **호환성**: BGE-M3와 완벽 호환
- **설치**: `poetry run pip install transformers==4.34.1`

### OpenAI API (선택사항)
- **버전**: openai >= 1.0.0
- **용도**: GPT 임베딩 비교 분석
- **설치**: `poetry run pip install openai>=1.0.0`

## 3. 프로젝트 구조

### 핵심 모듈
```
CSP/
├── common/
│   ├── embedders/
│   │   └── bge.py                    # BGE-M3 multi-vector
│   └── tokenizers/
│       ├── hybrid_korean.py          # RoBERTa-Hanja + Kiwipiepy
│       └── siku_tokenizer.py         # SikuBERT 중국어
├── pa/                               # Paragraph Aligner
│   ├── aligner.py
│   ├── sentence_splitter.py
│   └── processor.py
├── sa/                               # Sentence Aligner
│   ├── sa_aligner.py
│   └── main.py
└── accuracy/                         # 평가 모듈
    └── accuracy_evaluator.py
```

## 4. 환경 검증

### Poetry 패키지 확인
```bash
poetry show
```

### Poetry 의존성 트리 (참고용)
```
black 25.1.0 The uncompromising code formatter.
├── click >=8.0.0
│   └── colorama *
├── mypy-extensions >=0.4.3
├── packaging >=22.0
├── pathspec >=0.9.0
├── platformdirs >=2
├── tomli >=1.1.0
└── typing-extensions >=4.0.1
ipykernel 6.30.1 IPython Kernel for Jupyter
├── appnope >=0.1.2
├── comm >=0.1.1
├── debugpy >=1.6.5
├── ipython >=7.23.1
│   ├── colorama *
│   ├── decorator *
│   ├── exceptiongroup *
│   ├── jedi >=0.16
│   ├── matplotlib-inline *
│   ├── pexpect >4.3
│   ├── prompt-toolkit >=3.0.41,<3.1.0
│   ├── pygments >=2.4.0
│   ├── stack-data *
│   ├── traitlets >=5.13.0
│   └── typing-extensions >=4.6
├── jupyter-client >=8.0.0
├── jupyter-core >=4.12,<5.0.dev0 || >=5.1.dev0
├── matplotlib-inline >=0.1
├── nest-asyncio >=1.4
├── packaging >=22
├── psutil >=5.7
├── pyzmq >=25
├── tornado >=6.2
└── traitlets >=5.4.0
isort 6.0.1 A Python utility / library to sort Python imports.
jupyter 1.1.1 Jupyter metapackage. Install all the Jupyter components in one go.
├── ipykernel *
├── ipywidgets *
├── jupyter-console *
├── jupyterlab *
├── nbconvert *
└── notebook *
kiwipiepy 0.21.0 Kiwi, the Korean Tokenizer for Python
├── kiwipiepy-model >=0.21,<0.22
├── numpy *
└── tqdm *
numpy 1.26.4 Fundamental package for array computing in Python
openpyxl 3.1.5 A Python library to read/write Excel 2010 xlsx/xlsm files
└── et-xmlfile *
pandas 2.3.2 Powerful data structures for data analysis, time series, and statistics
├── numpy >=1.22.4
├── python-dateutil >=2.8.2
├── pytz >=2020.1
└── tzdata >=2022.7
pytest 8.4.1 pytest: simple powerful testing with Python
├── colorama >=0.4
├── exceptiongroup >=1
├── iniconfig >=1
├── packaging >=20
├── pluggy >=1.5,<2
├── pygments >=2.7.2
└── tomli >=1
pytest-cov 6.2.1 Pytest plugin for measuring coverage.
├── coverage >=7.5
├── pluggy >=1.2
└── pytest >=6.2.5
regex 2025.7.34 Alternative regular expression module, to replace re.
stanza 1.10.1 A Python NLP Library for Many Human Languages, by the Stanford NLP Group
├── emoji *
├── networkx *
├── numpy *
├── protobuf >=3.15.0
├── requests *
├── tomli *
├── torch >=1.3.0 (GPU 패키지 - pip으로 설치됨)
└── tqdm *
supar 1.1.4 Syntactic/Semantic Parsing Models
├── dill *
├── nltk *
├── stanza *
├── torch >=1.7.1 (GPU 패키지 - pip으로 설치됨)
└── transformers >=4.0.0 (GPU 패키지 - pip으로 설치됨)
tqdm 4.67.1 Fast, Extensible Progress Meter
└── colorama *
```

### 수동 설치 패키지 확인
```bash
poetry run pip list | findstr "torch\|transformers\|FlagEmbedding\|openai"
```

### PyTorch CUDA 검증
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용가능: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 5. 문제 해결

### 패키지 충돌 해결
1. **Poetry 캐시 초기화**: `poetry cache clear --all .`
2. **Lock 파일 재생성**: `poetry lock --no-update`
3. **의존성 재설치**: `poetry install --sync`

### transformers 패키지 손상
1. **손상된 패키지 제거**: `poetry run pip uninstall transformers -y`
2. **캐시 정리**: `poetry run pip cache purge`
3. **재설치**: `poetry run pip install transformers==4.34.1`

### Poetry vs pip 패키지 관리
- **Poetry 관리**: `poetry show` (pyproject.toml 패키지)
- **pip 관리**: `poetry run pip list` (전체 패키지 포함)
- **권장**: 핵심 패키지는 Poetry, CUDA 관련은 pip 수동 설치

## 6. 성능 최적화

### BGE-M3 Multi-vector 설정
```python
from common.embedders.bge import BGEEmbedder

embedder = BGEEmbedder()
# dense + sparse + colbert 벡터 모두 활용
embeddings = embedder.embed_texts(texts)
```

### GPU 메모리 최적화
- **모델 캐싱**: BGE-M3 모델 자동 캐싱
- **배치 처리**: 대용량 텍스트 처리 시 배치 크기 조정
- **혼합 정밀도**: FP16 사용으로 메모리 절약

## 7. 운영 환경

### 개발 환경
- **Python**: 3.10+
- **OS**: Windows 10/11
- **GPU**: RTX 3070 Ti
- **CUDA**: 11.8

### 프로덕션 고려사항
- **모델 다운로드**: BGE-M3 모델 (약 2.3GB) 자동 다운로드
- **메모리 요구사항**: 최소 8GB GPU 메모리 권장
- **디스크 공간**: 모델 캐시용 5GB 여유 공간

## 8. 유지보수

### 정기 점검 항목
1. **의존성 업데이트**: `poetry update` (신중하게)
2. **보안 취약성**: `poetry audit` 
3. **호환성 검증**: BGE-M3 + transformers 버전 호환성
4. **성능 모니터링**: GPU 메모리 사용량 추적

### 백업 권장사항
- **환경 설정**: `poetry export -f requirements.txt`
- **프로젝트 상태**: Git 버전 관리
- **모델 캐시**: BGE-M3 모델 파일 백업

---

**참고**: 이 가이드는 CSP 프로젝트의 안정적인 운영을 위해 지속적으로 업데이트됩니다.
