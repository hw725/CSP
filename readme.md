## 20250616 의미 기반 병렬 구분할

한문-한국어 번역 텍스트의 자동 정렬 CLI 도구 (SA/PA)

- 소규모 테스트 완료
- 문장을 구로 분할 및 1:1 대응
- 토크나이저 : mecab, jieba 등
- 벡터 임베더 : bge-m3 등
- tokenizers.py, embedders.py에 여러 라이브러리를 클래스로 추가해서 교체 가능
- 실행 순서

<<<<<<< HEAD
  ```
  python -m venv venv
      # 명령 프롬프트나 conda에서 가상 환경 생성(권장)
    venv\scripts\activate
      # 가상 환경 활성화(권장) windows 명령 프롬프트 기준
    pip install -r requirements.txt
      # pip install --upgrade pip 필요할 수 있음
    python main.py input.xlsx output.xlsx --parallel --workers 4
      # 메모리가 부족하면 workers 수를 낮출 것
  ```
- CLI 예시

  ```
  # 모든 기본값 사용
  python main.py input.xlsx output.xlsx

  # 임베더만 변경
  python main.py input.xlsx output.xlsx --embedder openai

  # 토크나이저만 변경  
  python main.py input.xlsx output.xlsx --source-tokenizer kiwi --target-tokenizer kiwi

  # 여러 옵션 조합
  python main.py input.xlsx output.xlsx --embedder cohere --source-tokenizer mecab --parallel --workers 4
  ```
- 도움말 확인

  ```
  python main.py --help

  --source-tokenizer    원문 토크나이저 타입 (기본값: jieba)
  --target-tokenizer    번역문 토크나이저 타입 (기본값: mecab)  
  --embedder           임베더 타입 (기본값: bge-m3)
  ```
=======
## 주요 특징

- **SA**: 문장/구 단위 정렬 (원문: jieba, 번역문: mecab)
- **PA**: 문단→문장 분할 및 정렬 (spaCy 기반)
<<<<<<< HEAD
- 다양한 임베더: SentenceTransformer, BGE-M3, OpenAI(모델/키 직접 선택)
- 실시간 진행률, 캐시, 상세 로그 지원
- CLI 환경에서 대용량/고품질 정렬에 최적화
>>>>>>> a69c32a (CLI 롤백)
=======
- 지원 임베더: BGE-M3, OpenAI(모델/키 직접 선택)
- 실시간 진행률(터미널 tqdm/GUI progress bar), 캐시, 상세 로그 지원
- CLI/GUI 환경에서 대용량/고품질 정렬에 최적화
- **경량화**: 성능이 우수한 임베더(BGE, OpenAI)와 토크나이저(jieba, mecab)만 남기고 경량화.
>>>>>>> 2ab721c (readme update, 환경파일 반영)


<<<<<<< HEAD

### 구조
- tokenizer : 원문(공백), 번역문(spacy) 의미 기반 분할
- punctuation : 마스킹, 언마스킹 함수 및 관련 부호 상세 설정
- embedder : 벡터 임베딩
- aligner : 의미 대응 배열
- io_manager : 엑셀 입출력
- main : 모듈 실행
=======
## CLI 사용법 (권장)

```bash
# SA 예시 (문장/구 단위 정렬)
python sa/main.py input.xlsx output.xlsx --tokenizer mecab --embedder bge --min-tokens 2 --max-tokens 10

# OpenAI 임베더 사용
python sa/main.py input.xlsx output.xlsx --embedder openai --openai-model text-embedding-3-large --openai-api-key sk-xxxx

# PA 예시 (문단→문장 정렬)
python pa/main.py input.xlsx output.xlsx --embedder bge --max-length 180 --threshold 0.35
```

---

## 환경설정 및 설치 (SA/PA 임베딩 연동 완벽 지원)

### 1. 가상환경 생성 및 활성화

#### (A) venv (Windows/Linux/WSL)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/WSL
source venv/bin/activate
```

#### (B) conda (권장: 대규모/ML 환경, 환경파일 제공)
```bash
conda env create -f environment.yml  # pa/sa 폴더의 environment.yml 사용
conda activate csp-pa  # 또는 csp-sa
```

### 2. 필수 패키지 설치

#### (venv 사용 시)
```bash
pip install -r requirements.txt
```
#### (conda 환경은 environment.yml로 자동 설치)

- **Windows**: mecab-python3가 반드시 필요 (requirements.txt에 포함)
- **Linux/WSL**: mecab-ko, mecab-ko-dic, mecab-python3 모두 설치 권장
- **GPU 사용**: torch/torchvision/torchaudio는 CUDA 버전에 맞게 설치 필요

### 3. mecab 사용자 사전/한자어 지원
- 표준국어대사전 기반 한자어 mecab 사용자사전 자동 생성/적용 기능은 추후 릴리즈 예정
- **stuser.dic은 표준국어대사전에서 한자어만 추출하여 만든 mecab 사용자 사전입니다.**
- **직접 생성한 사용자 사전(stuser.dic) 적용 방법:**
    1. 사용자 사전 csv를 mecab-dict-index로 컴파일하여 stuser.dic 생성
    2. Python 코드에서 아래와 같이 -u 옵션으로 경로를 지정
        ```python
        tagger = MeCab.Tagger('-d <mecab-ko-dic 경로> -u <stuser.dic 경로>')
        # 예시:
        # tagger = MeCab.Tagger('-d c:/.../mecab-ko-dic -u c:/.../stuser.dic')
        ```
    3. 여러 사용자 사전을 함께 쓰고 싶으면 csv를 미리 병합하여 하나의 dic로 컴파일
    4. stuser.dic을 mecab-ko-dic 폴더에 복사하면 -u stuser.dic처럼 파일명만 지정해도 됨
- 현재는 기본 mecab-ko-dic 또는 직접 생성한 사용자 사전만 사용 가능

### 4. spaCy 모델 설치 (최초 1회)
```bash
python -m spacy download ko_core_news_lg
python -m spacy download zh_core_web_lg
```

---

## 개발/실행 예시

```bash
# SA/PA 폴더에서 직접 실행
python sa/main.py ...
python pa/main.py ...
```

- 입력/출력은 Excel(xlsx) 파일만 지원
- OpenAI 임베더 사용 시 모델명/키를 CLI 옵션으로 직접 입력
- **진행률/에러/로그는 CLI(터미널 tqdm) 및 GUI(진행률 바) 모두 실시간 표시**

---

## 실행파일/GUI 안내
- pa_gui.exe, sa_gui.exe 등 GUI/실행파일 버전도 제공 (sa_gui.py 직접 실행 가능)
- GUI에서는 진행률 바(progress bar)로 실시간 진행 상황 확인 가능
- 최신 실행파일 및 한자어 mecab 사용자사전 지원은 [Releases](https://github.com/hw725/CSP/releases)에서 추후 확인

---

## 문제 해결
- 오류 발생 시 [Issues](https://github.com/hw725/CSP/issues)로 문의
- spaCy 모델은 최초 실행 시 자동 다운로드 (수동 설치 권장)
- mecab 관련 ImportError 발생 시 mecab-python3 설치 여부 확인

---

**CLI 환경에서 대용량 한중문 정렬, 고품질 임베딩 연동, 환경별 설치/실행법을 모두 지원합니다.**
>>>>>>> a69c32a (CLI 롤백)
