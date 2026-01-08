## CSP 프로젝트 전체 워크플로우 (2025-08-28 업데이트)

## 프로젝트 개요
CSP(Chinese-Korean Sentence Pairing)는 한문-한국어 번역 텍스트의 자동 정렬 시스템입니다. SA(문장/구 정렬)와 PA(문단→문장 정렬) 두 가지 모듈로 구성되어 있으며, **차별화된 접근 방식**을 사용합니다.

**🆕 2025-08-28 주요 업데이트:**
- **PA 병렬 옵션 전달 버그 수정**: `split_source_by_whitespace_and_align()` 시그니처 확장으로 `--max-workers`, `--batch-size`가 정상 반영됩니다(OpenAI 전용).
- **문서 정리**: Poetry 관련 내용 제거, Docker 중심 가이드로 일원화.

**🆕 2025-08-24 주요 업데이트:**
- **🐳 Docker 환경**: PyTorch 2.6.0+cu124 안정화, 일관된 재현성 확보
- **PA**: 구문분석 기반 (SuPar-Kanbun + Stanza + OpenAI/BGE-M3 하이브리드)
- **SA**: 의미 기반 (하이브리드 토크나이저 + OpenAI/BGE-M3 하이브리드)
- **OpenAI 통합**: text-embedding-3-large 모델 완전 지원
- **정확도 평가**: 문장식별자↔문단식별자 호환, 원문 기준 매칭 강화
- **무결성 시스템**: SA 순차적 텍스트 처리로 단어 순서 보장

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
    classDef parser fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef config fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 CSP 시스템 시작<br/>하이브리드 정렬 시스템]:::startEnd
    B[📊 입력 데이터<br/>Excel 파일]:::data
    C{처리 방식<br/>선택}:::decision
    
    %% SA 모듈 (문장/구 정렬) - 의미 기반
    D[📝 SA 모듈<br/>의미 기반 문장/구 정렬]:::module
    E[🏮 SA 하이브리드 토크나이저<br/>중국어: SikuBERT<br/>한국어: RoBERTa-Hanja+Kiwipiepy]:::process
    F[🧠 OpenAI/BGE-M3 하이브리드<br/>🆕 text-embedding-3-large + FlagModel]:::process
    G[🎯 SA 순차적 정렬기<br/>🆕 무결성 보장 + 의미 매칭]:::process
    H[📄 SA 출력<br/>구별 정렬 결과 무결성 100%]:::data
    
    %% PA 모듈 (문단→문장 정렬) - 구문분석 기반
    I[📑 PA 모듈<br/>구문분석 기반 문단→문장 정렬]:::module
    IE[🏮 PA 구문분석기<br/>🆕 원문: SuPar-Kanbun<br/>번역문: Stanza]:::parser
    J[✂️ PA 구문분석 분할기<br/>구문 구조 기반 정확한 문장 분할]:::parser
    K[🧠 OpenAI/BGE-M3 하이브리드<br/>🆕 구문+의미 하이브리드 임베딩]:::process
    L[🎯 PA 하이브리드 정렬기<br/>구문구조 + 의미유사도]:::parser
    M[📄 PA 출력<br/>문장별 정렬 결과 고품질]:::data
    
    %% 공통 구성요소
    N[🐳 Docker 환경<br/>🆕 PyTorch 2.6.0+cu124 안정화<br/>Poetry 의존성 지옥 해결]:::config
    O[📋 로그 시스템<br/>진행률 + 상세 로그]:::process
    P[💾 하이브리드 임베딩 캐시<br/>🆕 OpenAI + BGE-M3 통합 캐시]:::process
    PAcc[📊 정확도 평가 시스템<br/>🆕 문장식별자↔문단식별자 호환<br/>원문 기준 매칭 강화]:::process
    Q[✅ 최종 결과<br/>Excel 출력]:::startEnd
    
    A --> B
    B --> C
    C -->|SA 의미기반| D
    C -->|PA 구문기반| I
    
    %% SA 플로우 (의미 기반)
    D --> E
    E --> F
    F --> G
    G --> H
    H --> Q
    
    %% PA 플로우 (구문분석 기반)
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
    C[🏮 원문 하이브리드 토크나이징<br/>SikuBERT+Kiwipiepy]:::tokenizer
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
    classDef parser fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef embedder fill:#fafafa,stroke:#424242,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 PA 모듈 시작<br/>문단→문장 정렬]:::startEnd
    B[📊 Excel 입력 파일 읽기<br/>문단 원문과 문단 번역문 열]:::data
    
    %% 구문분석 기반 문장 분할 단계
    C[✂️ 원문 문단 분할<br/>SuPar-Kanbun 한문 구문분석]:::parser
    D[✂️ 번역문 문단 분할<br/>Stanza 한국어 구문분석]:::parser
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
    Q[🚨 오류 처리<br/>구문분석기 모델 누락 등]:::process
    
    E -.-> O
    I -.-> P
    C -.-> Q
    D -.-> Q
```

---

## 환경 설정 및 의존성 플로우차트 (Docker 중심)

```mermaid
flowchart TB
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef model fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold

    A["🚀 환경 설정 시작"]:::startEnd
    B["🐳 Docker 빌드/실행<br/>docker-compose up -d"]:::process
    C["🔐 환경변수 설정<br/>OPENAI_API_KEY"]:::data
    D["🎮 CUDA 확인<br/>torch.cuda.is_available()"]:::process
    E["🧪 모델 로딩 확인<br/>SuPar/Stanza/BGE-M3"]:::model
    F["✅ 설정 완료"]:::startEnd

    A --> B --> C --> D --> E --> F
```

---

## CLI 사용법 플로우차트 (PA 병렬 옵션 반영)

```mermaid
flowchart LR
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef command fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef option fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef file fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A["💻 CLI 시작"]:::startEnd
    
    %% SA 명령어
    B["📝 SA 실행<br/>python sa/main.py"]:::command
    C["📄 입력 파일<br/>input.xlsx"]:::file
    D["📄 출력 파일<br/>output.xlsx"]:::file
    
    %% SA 옵션들
    E["🏮 하이브리드 토크나이저<br/>--tokenizer hybrid"]:::option
    F["🧠 임베더<br/>--embedder bge"]:::option
    G["📏 토큰 제한<br/>--min-tokens 2<br/>--max-tokens 10"]:::option
    
    %% PA 명령어
    H["📑 PA 실행<br/>python pa/main.py"]:::command
    I["📏 길이 제한<br/>--max-length 180"]:::option
    J["🎯 임계값<br/>--threshold 0.7"]:::option
    M2["🚀 병렬 옵션 OpenAI 전용<br/>--max-workers 4<br/>--batch-size 50"]:::option
    
    %% 결과
    K["✅ 처리 완료"]:::startEnd
    
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
    J --> M2
    M2 --> K
    
    %% OpenAI 옵션
    L["🌐 OpenAI 설정<br/>--embedder openai<br/>--openai-model text-embedding-3-large<br/>--openai-api-key sk-xxxx"]:::option
    
    F -.-> L
    L -.-> K
```

---

## 주요 특징

### ✨ 핵심 기능
- **SA 모듈**: 문장/구 단위 정밀 정렬 (하이브리드 토크나이저)
- **PA 모듈**: 문단→문장 분할 및 정렬 (SuPar-Kanbun + Stanza + BGE-M3)
- **다중 임베더**: BGE-M3 (로컬) + OpenAI (API) 지원
- **실시간 모니터링**: CLI tqdm + GUI 진행률 바
- **캐시 시스템**: 임베딩 결과 재사용으로 성능 최적화

### 🔧 기술 스택
- **토크나이저**: SikuBERT (중국어), RoBERTa-Hanja+Kiwipiepy (한국어)
- **구문분석**: SuPar-Kanbun (한문), Stanza (한국어) + 하이브리드 토크나이저
- **임베딩**: BGE-M3, OpenAI text-embedding-3-large
- **정렬 알고리즘**: 헝가리안 알고리즘, 임계값 기반 매칭
- **입출력**: Excel (xlsx) 파일 지원

### 📊 처리 성능
- **대용량 처리**: 수천 문장 단위 배치 처리
- **고품질 정렬**: 의미 기반 유사도 매칭
- **환경 최적화**: Poetry 의존성 관리 및 가상환경
- **멀티플랫폼**: Windows/Linux/WSL 지원

### 🚀 사용 시나리오
- **학술 연구**: 한문 고전 번역 정렬
- **대용량 코퍼스**: 병렬 말뭉치 구축
- **번역 품질 평가**: 원문-번역문 대응 분석
- **자동화 파이프라인**: CLI 기반 배치 처리

---

## Common 디렉토리 공통 모듈 플로우차트

```mermaid
flowchart TB
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef core fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef embedder fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef tokenizer fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef parser fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef utility fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef cache fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 Common 모듈 시작<br/>공통 기능 라이브러리]:::startEnd
    
    %% 임베더 모듈
    B[🧠 embedders/ 디렉토리<br/>임베딩 엔진 모음]:::embedder
    C[🤖 bge.py<br/>BGE-M3 FlagModel 래퍼<br/>안정화된 임베딩 엔진]:::embedder
    D[🌐 openai_embedder.py<br/>OpenAI API 래퍼<br/>text-embedding-3-large]:::embedder
    
    %% 토크나이저 모듈
    E[🏮 tokenizers/ 디렉토리<br/>하이브리드 토크나이저 모음]:::tokenizer
    F[📚 siku_bert.py<br/>SikuBERT 토크나이저<br/>사고전서 코퍼스 훈련]:::tokenizer
    H[📚 roberta_hanja.py<br/>RoBERTa-Hanja 토크나이저<br/>한자 포함 한국어]:::tokenizer
    I[📚 kiwipiepy_wrapper.py<br/>Kiwipiepy 래퍼<br/>한글 형태소 분석]:::tokenizer
    
    %% 구문분석기 모듈
    J[🏮 parsers/ 디렉토리<br/>🆕 구문분석기 모음]:::parser
    K[🏮 supar_kanbun.py<br/>🆕 SuPar-Kanbun 래퍼<br/>한문 구문분석기]:::parser
    L[🏮 stanza_korean.py<br/>🆕 Stanza 래퍼<br/>한국어 구문분석기]:::parser
    M[🏮 new_parsers.py<br/>🆕 통합 구문분석기<br/>SuPar + Stanza 연동]:::parser
    
    %% 유틸리티 모듈
    N[🔧 io_utils.py<br/>Excel 입출력 유틸리티<br/>pandas 래퍼]:::utility
    O[🏮 korean_particle_matcher.py<br/>한글 토씨 매칭<br/>Kiwipiepy 연동]:::utility
    P[📊 progress_manager.py<br/>진행률 관리<br/>tqdm + 로깅]:::utility
    
    %% 캐시 시스템
    Q[💾 cache/ 디렉토리<br/>임베딩 캐시 관리]:::cache
    R[💾 embedding_cache.py<br/>BGE-M3 캐시 관리<br/>pickle 기반 저장]:::cache
    S[💾 model_cache.py<br/>모델 캐시 관리<br/>transformers 캐시]:::cache
    
    A --> B
    A --> E
    A --> J
    A --> N
    A --> Q
    
    %% 임베더 관계
    B --> C
    B --> D
    
    %% 토크나이저 관계
    E --> F
    E --> G
    E --> H
    E --> I
    
    %% 구문분석기 관계
    J --> K
    J --> L
    J --> M
    
    %% 유틸리티 관계
    N --> O
    N --> P
    
    %% 캐시 관계
    Q --> R
    Q --> S
```

---

## 모듈 간 의존성 플로우차트

```mermaid
flowchart LR
    %% 스타일 정의
    classDef sa fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef pa fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,color:#000,font-weight:bold
    classDef common fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    %% 외부 의존성
    A[🤖 transformers 4.36.0<br/>BGE-M3 호환]:::external
    B[🧠 FlagEmbedding 1.1.7<br/>안정화된 버전]:::external
    C[🏮 SuPar<br/>구문분석기]:::external
    D[🏮 Stanza<br/>다언어 구문분석]:::external
    F[🔤 Kiwipiepy<br/>한글 형태소 분석]:::external
    
    %% Common 모듈
    G[🧠 common/embedders/<br/>BGE-M3 + OpenAI]:::common
    H[🏮 common/tokenizers/<br/>하이브리드 토크나이저]:::common
    I[🏮 common/parsers/<br/>🆕 구문분석기 통합]:::common
    J[🔧 common/io_utils.py<br/>Excel 입출력]:::common
    K[🏮 common/korean_particle_matcher.py<br/>한글 토씨 매칭]:::common
    
    %% SA 모듈
    L[📝 SA 모듈<br/>의미 기반 정렬]:::sa
    M[🎭 sa/punctuation.py<br/>무결성 관리]:::sa
    N[🎯 sa/sa_aligner.py<br/>순차적 정렬]:::sa
    
    %% PA 모듈
    O[📑 PA 모듈<br/>구문분석 기반]:::pa
    P[✂️ pa/sentence_splitter.py<br/>구문분석 분할]:::pa
    Q[🎯 pa/aligner.py<br/>하이브리드 정렬]:::pa
    
    %% 데이터 플로우
    R[📊 input.xlsx<br/>원문과 번역문]:::data
    S[📊 SA output.xlsx<br/>구별 정렬 결과]:::data
    T[📊 PA output.xlsx<br/>문장별 정렬 결과]:::data
    
    %% 의존성 관계
    A --> B
    B --> G
    C --> I
    D --> I
    E --> P
    F --> H
    F --> K
    
    %% Common → SA
    G --> L
    H --> L
    J --> L
    K --> L
    
    %% Common → PA
    G --> O
    H --> O
    I --> O
    J --> O
    
    %% SA 내부
    L --> M
    L --> N
    M --> N
    
    %% PA 내부
    O --> P
    O --> Q
    P --> Q
    
    %% 데이터 플로우
    R --> L
    R --> O
    L --> S
    O --> T
```

---

## 성능 최적화 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef optimization fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef cache fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef parallel fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 성능 최적화 시작]:::startEnd
    
    %% 캐시 최적화
    B[💾 임베딩 캐시 확인<br/>기존 벡터 재사용]:::cache
    C{캐시<br/>적중?}:::decision
    D[⚡ 캐시에서 로드<br/>계산 시간 단축]:::cache
    E[🧠 새로운 임베딩<br/>BGE-M3 계산]:::process
    F[💾 캐시에 저장<br/>다음 사용을 위해]:::cache
    
    %% 병렬 처리 최적화
    G[🚀 병렬 처리 설정<br/>CPU 코어 수 감지]:::parallel
    H{병렬 처리<br/>활성화?}:::decision
    I[👷 멀티프로세싱<br/>ProcessPoolExecutor]:::parallel
    J[📋 단일 스레드<br/>순차 처리]:::process
    
    %% 메모리 최적화
    K[🧹 메모리 관리<br/>배치 크기 조정]:::optimization
    L[📊 GPU 메모리 확인<br/>CUDA 사용 가능 여부]:::optimization
    M{GPU<br/>사용 가능?}:::decision
    N[🎮 GPU 가속<br/>CUDA 연산]:::optimization
    O[💻 CPU 연산<br/>기본 처리]:::process
    
    %% 모델 최적화
    P[🤖 모델 로딩 최적화<br/>한 번만 초기화]:::optimization
    Q[📏 토큰 길이 최적화<br/>불필요한 토큰 제거]:::optimization
    R[🎯 임계값 조정<br/>품질 vs 속도 균형]:::optimization
    
    %% 결과
    S[📊 성능 모니터링<br/>처리 시간 측정]:::process
    T[✅ 최적화 완료<br/>향상된 성능]:::startEnd
    
    A --> B
    B --> C
    C -->|Yes| D
    C -->|No| E
    E --> F
    D --> G
    F --> G
    
    G --> H
    H -->|Yes| I
    H -->|No| J
    I --> K
    J --> K
    
    K --> L
    L --> M
    M -->|Yes| N
    M -->|No| O
    N --> P
    O --> P
    
    P --> Q
    Q --> R
    R --> S
    S --> T
```

---

## Accuracy 디렉토리 평가 모듈 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef evaluation fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef metrics fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 정확도 평가 시작<br/>accuracy 디렉토리]:::startEnd
    
    %% 입력 데이터
    B[📊 평가 데이터 로드<br/>관자1_구병렬.xlsx<br/>관자3_문장병렬.xlsx]:::data
    C[📊 시스템 출력 로드<br/>SA/PA 처리 결과]:::data
    
    %% 평가 모듈별 처리
    D[🔧 accuracy_evaluator.py<br/>전체 정확도 평가 엔진]:::evaluation
    E[🔧 row_pair_evaluator.py<br/>행별 쌍 매칭 평가]:::evaluation
    F[🔧 reconstructed_evaluator.py<br/>재구성 품질 평가]:::evaluation
    
    %% 평가 메트릭
    G[📊 정확도 계산<br/>Precision, Recall, F1]:::metrics
    H[📊 매칭 품질<br/>1:1, 1:N, N:1 매칭]:::metrics
    I[📊 텍스트 손실 분석<br/>누락된 텍스트 검출]:::metrics
    
    %% 텍스트 손실 분석
    J[🔍 analyze_text_loss.py<br/>일반 텍스트 손실 분석]:::evaluation
    K[🔍 analyze_sa_text_loss.py<br/>SA 전용 텍스트 손실 분석]:::evaluation
    L{텍스트 손실<br/>감지?}:::decision
    M[⚠️ 손실 보고서 생성<br/>누락 위치와 내용]:::process
    
    %% 결과 생성
    N[📄 종합 평가 보고서<br/>accuracy_results_improved.xlsx]:::data
    O[📄 행별 평가 결과<br/>row_pair_results.xlsx]:::data
    P[📊 평가 통계<br/>정확도 메트릭 요약]:::process
    
    %% 품질 검증
    Q{평가 품질<br/>기준 통과?}:::decision
    R[✅ 평가 통과<br/>시스템 품질 확인]:::startEnd
    S[⚠️ 개선 필요<br/>문제점 분석 보고]:::process
    
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    
    D --> G
    E --> H
    F --> I
    
    G --> J
    H --> K
    I --> L
    
    L -->|Yes| M
    L -->|No| N
    M --> N
    
    N --> O
    O --> P
    P --> Q
    
    Q -->|Pass| R
    Q -->|Fail| S
    S --> R
```

---

## 전체 시스템 통합 플로우차트 (Docker 중심)

```mermaid
flowchart TB
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef module fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef quality fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 CSP 통합 시스템 시작]:::startEnd
    
    %% 환경 설정
    B[🐳 Docker 컨테이너 준비<br/>의존성/모델 일괄 관리]:::process
    C[🤖 모델 로딩<br/>BGE-M3 + 구문분석기 + 토크나이저]:::process
    
    %% 입력 처리
    D[📊 입력 데이터 검증<br/>Excel 파일 형식 확인]:::data
    E{처리 모드<br/>선택}:::decision
    
    %% 처리 모듈
    F[📝 SA 모듈 실행<br/>의미 기반 문장/구 정렬<br/>무결성 보장]:::module
    G[📑 PA 모듈 실행<br/>구문분석 기반 문단→문장 정렬<br/>고품질 분할]:::module
    
    %% 병렬 처리
    H[🚀 병렬 처리 엔진<br/>멀티프로세싱 최적화]:::process
    I[💾 캐시 시스템<br/>임베딩 재사용]:::process
    
    %% 품질 보장
    J[🔍 실시간 품질 검증<br/>무결성 + 정확도 모니터링]:::quality
    K[📊 진행률 모니터링<br/>tqdm + 로깅]:::process
    
    %% 결과 처리
    L[📄 결과 통합<br/>Excel 출력 생성]:::data
    M[🎯 정확도 평가<br/>accuracy 모듈 실행]:::quality
    N[📊 종합 보고서<br/>처리 통계 + 품질 메트릭]:::data
    
    %% 최종 검증
    O{품질 기준<br/>충족?}:::decision
    P[✅ 처리 완료<br/>고품질 정렬 결과]:::startEnd
    Q[🔄 재처리<br/>파라미터 조정]:::process
    
    A --> B
    B --> C
    C --> D
    D --> E
    
    E -->|SA 모드| F
    E -->|PA 모드| G
    
    F --> H
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    
    O -->|Yes| P
    O -->|No| Q
    Q --> E
```

---

## 📋 종합 요약

### 🎯 CSP 시스템 특징
1. **이중 모듈 아키텍처**: SA(의미 기반) + PA(구문분석 기반)
2. **하이브리드 접근법**: 토크나이저 + 구문분석기 + 임베딩 통합
3. **무결성 보장**: 순차적 텍스트 처리로 단어 순서 유지
4. **성능 최적화**: 병렬 처리 + 캐시 시스템 + GPU 가속
5. **품질 평가**: 정확도 모듈로 실시간 품질 검증

### 🔧 핵심 기술 스택
- **BGE-M3 FlagModel**: 안정화된 1024차원 임베딩
- **SuPar-Kanbun**: 한문 구문분석 (PA 모듈)
- **Stanza**: 한국어 구문분석 (PA 모듈)  
- **하이브리드 토크나이저**: SikuBERT + RoBERTa-Hanja + Kiwipiepy
- **Poetry**: 의존성 관리 및 가상환경

### 📊 처리 플로우
1. **입력**: Excel 파일 (원문과 번역문)
2. **전처리**: 구두점 마스킹 + 데이터 검증
3. **분할/토크나이징**: 모듈별 차별화된 접근법
4. **임베딩**: BGE-M3 벡터화 (캐시 활용)
5. **정렬**: 의미/구문 기반 매칭
6. **후처리**: 구두점 복원 + 무결성 검증
7. **출력**: 정렬 결과 Excel + 품질 평가 보고서

### 🚀 확장 가능성
- **다언어 지원**: 추가 언어 모델 통합
- **GUI 인터페이스**: 웹/데스크톱 UI 개발
- **API 서비스**: RESTful API 제공
- **클라우드 배포**: Docker + Kubernetes 지원
- **실시간 처리**: 스트리밍 데이터 처리

이 문서는 CSP 프로젝트의 완전한 워크플로우를 시각화하여 개발자와 사용자 모두가 시스템을 이해하고 활용할 수 있도록 구성되었습니다.
```
```
