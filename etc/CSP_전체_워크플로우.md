# CSP 프로젝트 전체 워크플로우 (2025-08-18 하이브리드 토크나이저 통합)

## 프로젝트 개요
CSP(Chinese-Korean Sentence Pairing)는 한문-한국어 번역 텍스트의 자동 정렬 시스템입니다. SA(문장/구 정렬)와 PA(문단→문장 정렬) 두 가지 모듈로 구성되어 있으며, **하이브리드 토크나이저 시스템**을 공통으로 사용합니다.

**🆕 2025-08-18 주요 업데이트:**
- **하이브리드 토크나이저 통합**: 중국어(SikuBERT/AnchiBERT) + 한국어(RoBERTa-Hanja+Kiwipiepy)
- **공통 모듈 아키텍처**: `common/tokenizers/` 디렉토리로 SA/PA 통합
- **Kiwipiepy 직접 연동**: `common/korean_particle_matcher.py`로 고성능 한글 분석
- **일관된 출력 형식**: SA/PA 모두 동일한 토크나이저 초기화 메시지

---

## 전체 시스템 아키텍처 플로우차트

```mermaid
flowchart TB
    %% 스타일 정의 (글자가 잘 보이도록)
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef module fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef config fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 CSP 시스템 시작]:::startEnd
    B[📊 입력 데이터<br/>Excel 파일]:::data
    C{처리 방식<br/>선택}:::decision
    
    %% SA 모듈 (문장/구 정렬)
    D[📝 SA 모듈<br/>문장/구 단위 정렬]:::module
    E[🏮 SA 하이브리드 토크나이저<br/>중국어: SikuBERT/AnchiBERT<br/>한국어: RoBERTa-Hanja+Kiwipiepy]:::process
    F[🧠 SA 임베더<br/>BGE-M3 / OpenAI]:::process
    G[🎯 SA 정렬기<br/>의미 기반 매칭 + 한글 토씨]:::process
    H[📄 SA 출력<br/>구별 정렬 결과]:::data
    
    %% PA 모듈 (문단→문장 정렬)
    I[📑 PA 모듈<br/>문단→문장 정렬]:::module
    IE[🏮 PA 하이브리드 토크나이저<br/>중국어: SikuBERT/AnchiBERT<br/>한국어: RoBERTa-Hanja+Kiwipiepy]:::process
    J[✂️ PA 하이브리드 분할기<br/>🆕 임베딩 기반 의미적 분할<br/>폴백: spaCy]:::process
    K[🧠 PA 임베더<br/>BGE-M3 / OpenAI]:::process
    L[🎯 PA 정렬기<br/>품질 검증 + 재분할]:::process
    M[📄 PA 출력<br/>문장별 정렬 결과]:::data
    
    %% 공통 구성요소
    N[⚙️ 환경 설정<br/>config.json]:::config
    O[📋 로그 시스템<br/>진행률 + 상세 로그]:::process
    P[💾 캐시 시스템<br/>임베딩 재사용]:::process
    Q[✅ 최종 결과<br/>Excel 출력]:::startEnd
    
    A --> B
    B --> C
    C --> D
    C --> I
    
    %% SA 플로우
    D --> E
    E --> F
    F --> G
    G --> H
    H --> Q
    
    %% PA 플로우  
    I --> IE
    IE --> J
    J --> K
    K --> L
    L --> M
    M --> Q
    
    %% 공통 연결
    N -.-> D
    N -.-> I
    O -.-> D
    O -.-> I
    P -.-> F
    P -.-> K
```

---

## SA 모듈 상세 워크플로우

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef tokenizer fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef embedder fill:#fafafa,stroke:#424242,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 SA 모듈 시작<br/>문장/구 단위 정렬]:::startEnd
    B[📊 Excel 입력 파일 읽기<br/>원문과 번역문 열]:::data
    
    %% 토크나이징 단계
    C[🏮 원문 하이브리드 토크나이징<br/>SikuBERT/AnchiBERT 우선]:::tokenizer
    D[🏮 번역문 하이브리드 토크나이징<br/>RoBERTa-Hanja+Kiwipiepy]:::tokenizer
    E[🎭 구두점 마스킹<br/>특수문자 임시 처리]:::process
    
    %% 임베딩 단계
    F{임베더<br/>선택}:::decision
    G[🤖 BGE-M3 임베딩<br/>로컬 모델]:::embedder
    H[🌐 OpenAI 임베딩<br/>API 호출]:::embedder
    
    %% 정렬 단계
    I[🎯 의미 유사도 계산<br/>코사인 유사도]:::process
    J[📐 최적 정렬 찾기<br/>헝가리안 알고리즘]:::process
    K[🎭 구두점 언마스킹<br/>특수문자 복원]:::process
    
    %% 출력 단계
    L[📊 결과 검증<br/>정렬 품질 확인]:::process
    M[📄 Excel 출력 생성<br/>문장식별자와 구식별자와 원문구와 번역구]:::data
    N[✅ SA 처리 완료]:::startEnd
    
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    
    %% 캐시 시스템
    O[💾 임베딩 캐시<br/>중복 계산 방지]:::data
    P[📋 진행률 표시<br/>tqdm / GUI 진행률 바]:::process
    
    G -.-> O
    H -.-> O
    I -.-> P
```

---

## PA 모듈 상세 워크플로우

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef spacy fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef embedder fill:#fafafa,stroke:#424242,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 PA 모듈 시작<br/>문단→문장 정렬]:::startEnd
    B[📊 Excel 입력 파일 읽기<br/>문단 원문과 문단 번역문 열]:::data
    
    %% 문장 분할 단계
    C[✂️ 원문 문단 분할<br/>spaCy 중국어 모델]:::spacy
    D[✂️ 번역문 문단 분할<br/>spaCy 한국어 모델]:::spacy
    E[📏 문장 길이 검증<br/>최소/최대 길이 확인]:::process
    
    %% 임베딩 단계
    F{임베더<br/>선택}:::decision
    G[🤖 BGE-M3 임베딩<br/>문장 벡터화]:::embedder
    H[🌐 OpenAI 임베딩<br/>API 기반 벡터화]:::embedder
    
    %% 정렬 단계
    I[🎯 문장 간 유사도 계산<br/>벡터 내적 / 코사인 유사도]:::process
    J[📊 유사도 매트릭스 생성<br/>M×N 행렬]:::process
    K[🎯 최적 정렬 찾기<br/>임계값 기반 매칭]:::process
    
    %% 출력 단계
    L[📋 정렬 결과 검증<br/>매칭 품질 확인]:::process
    M[📄 Excel 출력 생성<br/>문단ID와 문장ID와 원문분할과 번역문분할]:::data
    N[✅ PA 처리 완료]:::startEnd
    
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    
    %% 지원 시스템
    O[⚙️ 설정 관리<br/>최대 길이, 임계값]:::data
    P[📊 실시간 진행률<br/>문단별 처리 상황]:::process
    Q[🚨 오류 처리<br/>spaCy 모델 누락 등]:::process
    
    E -.-> O
    I -.-> P
    C -.-> Q
    D -.-> Q
```

---

## 환경 설정 및 의존성 플로우차트

```mermaid
flowchart TB
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef python fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000,font-weight:bold
    classDef model fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 환경 설정 시작]:::startEnd
    
    %% 가상환경 설정
    B{가상환경<br/>선택}:::decision
    C[🐍 venv 환경<br/>python -m venv venv]:::python
    D[🐍 conda 환경<br/>conda env create -f environment.yml]:::python
    
    %% 패키지 설치
    E[📦 기본 패키지 설치<br/>pip install -r requirements.txt]:::process
    F[📦 conda 패키지 설치<br/>environment.yml로 자동 설치]:::process
    
    %% 모델 설치
    G[🤖 spaCy 모델 설치<br/>ko_core_news_lg]:::model
    H[🤖 spaCy 모델 설치<br/>zh_core_web_lg]:::model
    I[🤖 BGE-M3 모델<br/>로컬 다운로드]:::model
    
    %% 사전 설정
    J[📚 하이브리드 토크나이저<br/>SikuBERT/AnchiBERT/RoBERTa-Hanja/Kiwipiepy 설정]:::data
    K[🔧 OpenAI API 키<br/>환경변수 설정]:::data
    
    %% 검증
    L[✅ 환경 검증<br/>모든 모듈 import 테스트]:::process
    M[🎯 설정 완료]:::startEnd
    
    A --> B
    B --> C
    B --> D
    C --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    
    %% 조건부 설치
    N[🪟 Windows 전용<br/>torch+CUDA]:::python
    O[🐧 Linux 전용<br/>torch+CUDA]:::python
    P[🎮 GPU 사용<br/>CUDA 버전 torch]:::python
    
    E -.-> N
    E -.-> O
    E -.-> P
```

---

## CLI 사용법 플로우차트

```mermaid
flowchart LR
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef command fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef option fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef file fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[💻 CLI 시작]:::startEnd
    
    %% SA 명령어
    B[📝 SA 실행<br/>python sa/main.py]:::command
    C[📄 입력 파일<br/>input.xlsx]:::file
    D[📄 출력 파일<br/>output.xlsx]:::file
    
    %% SA 옵션들
    E[🏮 하이브리드 토크나이저<br/>--tokenizer hybrid]:::option
    F[🧠 임베더<br/>--embedder bge]:::option
    G[📏 토큰 제한<br/>--min-tokens 2<br/>--max-tokens 10]:::option
    
    %% PA 명령어
    H[📑 PA 실행<br/>python pa/main.py]:::command
    I[📏 길이 제한<br/>--max-length 180]:::option
    J[🎯 임계값<br/>--threshold 0.7]:::option
    
    %% 결과
    K[✅ 처리 완료]:::startEnd
    
    A --> B
    A --> H
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> K
    
    H --> C
    H --> I
    I --> J
    J --> K
    
    %% OpenAI 옵션
    L[🌐 OpenAI 설정<br/>--embedder openai<br/>--openai-model text-embedding-3-large<br/>--openai-api-key sk-xxxx]:::option
    
    F -.-> L
    L -.-> K
```

---

## 주요 특징

### ✨ 핵심 기능
- **SA 모듈**: 문장/구 단위 정밀 정렬 (하이브리드 토크나이저)
- **PA 모듈**: 문단→문장 분할 및 정렬 (하이브리드 토크나이저 + spaCy)
- **다중 임베더**: BGE-M3 (로컬) + OpenAI (API) 지원
- **실시간 모니터링**: CLI tqdm + GUI 진행률 바
- **캐시 시스템**: 임베딩 결과 재사용으로 성능 최적화

### 🔧 기술 스택
- **토크나이저**: SikuBERT/AnchiBERT (중국어), RoBERTa-Hanja+Kiwipiepy (한국어)
- **문장 분할**: spaCy (ko_core_news_lg, zh_core_web_lg) + 하이브리드 토크나이저
- **임베딩**: BGE-M3, OpenAI text-embedding-3-large
- **정렬 알고리즘**: 헝가리안 알고리즘, 임계값 기반 매칭
- **입출력**: Excel (xlsx) 파일 지원

### 📊 처리 성능
- **대용량 처리**: 수천 문장 단위 배치 처리
- **고품질 정렬**: 의미 기반 유사도 매칭
- **환경 최적화**: venv/conda 환경별 최적화
- **멀티플랫폼**: Windows/Linux/WSL 지원

### 🚀 사용 시나리오
- **학술 연구**: 한문 고전 번역 정렬
- **대용량 코퍼스**: 병렬 말뭉치 구축
- **번역 품질 평가**: 원문-번역문 대응 분석
- **자동화 파이프라인**: CLI 기반 배치 처리
