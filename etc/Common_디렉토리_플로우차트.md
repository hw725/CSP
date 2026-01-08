# Common 디렉토리 상세 플로우차트

## Common 모듈 구조 및 워크플로우 - 공통 라이브러리 (2025-08-21 최신)

---

## Common 디렉토리 전체 아키텍처 플로우차트

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
    G[📚 anchi_bert.py<br/>AnchiBERT 토크나이저<br/>고전 중국어 BERT]:::tokenizer
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

## BGE-M3 FlagModel 임베더 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef cache fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef gpu fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 BGE-M3 FlagModel 초기화]:::startEnd
    
    %% 환경 확인
    B[🔍 환경 확인<br/>FlagEmbedding 1.1.7 버전 체크]:::process
    C{FlagEmbedding<br/>설치 확인?}:::decision
    D[❌ FlagEmbedding 미설치<br/>설치 안내 메시지]:::error
    
    %% GPU 확인
    E[🎮 GPU 환경 확인<br/>CUDA 사용 가능 여부]:::gpu
    F{CUDA<br/>사용 가능?}:::decision
    G[🎮 GPU 모드<br/>cuda device 설정]:::gpu
    H[💻 CPU 모드<br/>cpu device 설정]:::process
    
    %% 모델 로딩
    I[🤖 BGE-M3 모델 로드<br/>BAAI/bge-m3 from HuggingFace]:::model
    J{모델 로딩<br/>성공?}:::decision
    K[❌ 모델 로딩 실패<br/>네트워크 또는 디스크 오류]:::error
    
    %% 토크나이저 로딩
    L[🏮 BGE-M3 토크나이저 로드<br/>AutoTokenizer 초기화]:::model
    M[⚙️ 모델 설정<br/>max_length, batch_size 등]:::process
    
    %% 임베딩 처리 준비
    N[🧠 임베딩 함수 준비<br/>encode 메서드 래핑]:::process
    O[💾 캐시 시스템 연결<br/>중복 계산 방지]:::cache
    P[✅ BGE-M3 초기화 완료<br/>임베딩 서비스 준비]:::startEnd
    
    A --> B
    B --> C
    C -->|성공| E
    C -->|실패| D
    E --> F
    F -->|Yes| G
    F -->|No| H
    G --> I
    H --> I
    I --> J
    J -->|성공| L
    J -->|실패| K
    L --> M
    M --> N
    N --> O
    O --> P
    
    D -.-> P
    K -.-> P
```

---

## 하이브리드 토크나이저 통합 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef chinese fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef korean fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef validation fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 하이브리드 토크나이저 통합 시작]:::startEnd
    
    %% 언어 감지
    B[🔍 언어 감지<br/>중국어 vs 한국어 구분]:::process
    C{텍스트 언어<br/>판별}:::decision
    
    %% 중국어 토크나이저 경로
    D[🇨🇳 중국어 텍스트 처리]:::chinese
    E[📚 SikuBERT 우선 적용<br/>사고전서 코퍼스 특화]:::chinese
    F{SikuBERT<br/>가용성?}:::decision
    G[📚 AnchiBERT 폴백<br/>고전 중국어 BERT]:::chinese
    H[✂️ 중국어 토크나이징 완료]:::chinese
    
    %% 한국어 토크나이저 경로
    I[🇰🇷 한국어 텍스트 처리]:::korean
    J[🔍 한자/한글 혼재 확인<br/>문자 유형별 분리]:::korean
    K[📚 한자 구간: RoBERTa-Hanja<br/>한자 포함 한국어 특화]:::korean
    L[📚 한글 구간: Kiwipiepy<br/>형태소 분석 + 고어 처리]:::korean
    M[🔗 혼재 토큰 통합<br/>한자+한글 시퀀스 결합]:::korean
    
    %% 토큰 검증 및 정제
    N[🔍 토큰 길이 검증<br/>min/max tokens 범위 확인]:::validation
    O{길이 범위<br/>적합?}:::decision
    P[✂️ 토큰 조정<br/>truncation 또는 padding]:::validation
    
    %% 최종 출력
    Q[🧹 토큰 정제<br/>특수문자, 공백 처리]:::output
    R[📋 하이브리드 토큰 리스트<br/>언어별 최적화된 토큰]:::output
    S[✅ 하이브리드 토크나이징 완료]:::startEnd
    
    A --> B
    B --> C
    C -->|중국어| D
    C -->|한국어| I
    C -->|혼재| D
    
    %% 중국어 플로우
    D --> E
    E --> F
    F -->|가용| H
    F -->|불가| G
    G --> H
    H --> N
    
    %% 한국어 플로우
    I --> J
    J --> K
    J --> L
    K --> M
    L --> M
    M --> N
    
    %% 검증 플로우
    N --> O
    O -->|적합| Q
    O -->|부적합| P
    P --> Q
    Q --> R
    R --> S
```

---

## 구문분석기 통합 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef supar fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef stanza fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef validation fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 구문분석기 통합 시작<br/>🆕 SuPar + Stanza]:::startEnd
    
    %% 초기화 단계
    B[🔍 구문분석기 환경 확인<br/>SuPar, Stanza 설치 여부]:::process
    C{구문분석기<br/>가용성 확인}:::decision
    D[❌ 구문분석기 미설치<br/>설치 안내 및 폴백]:::process
    
    %% SuPar-Kanbun 초기화
    E[🏮 SuPar-Kanbun 초기화<br/>한문 전용 구문분석기]:::supar
    F{SuPar-Kanbun<br/>로드 성공?}:::decision
    G[🏮 SuPar-Kanbun 준비<br/>한문 구문분석 가능]:::supar
    H[⚠️ SuPar-Kanbun 실패<br/>한문 분석 제한]:::process
    
    %% Stanza 초기화
    I[🏮 Stanza 초기화<br/>한국어 전용 구문분석기]:::stanza
    J{Stanza<br/>로드 성공?}:::decision
    K[🏮 Stanza 준비<br/>한국어 구문분석 가능]:::stanza
    L[⚠️ Stanza 실패<br/>한국어 분석 제한]:::process
    
    %% 통합 구문분석 함수
    M[🔧 통합 분석 함수<br/>언어별 자동 라우팅]:::process
    N[🔍 텍스트 언어 감지<br/>중국어 vs 한국어]:::validation
    O{언어<br/>판별}:::decision
    
    %% 언어별 구문분석
    P[🏮 중국어: SuPar-Kanbun<br/>한문 구문 구조 분석]:::supar
    Q[🏮 한국어: Stanza<br/>한국어 구문 구조 분석]:::stanza
    
    %% 구문분석 결과 통합
    R[🔗 구문분석 결과 통합<br/>언어별 결과 표준화]:::validation
    S[📊 구문 기반 문장 분할<br/>구문 경계 기준 분할]:::output
    T[✅ 구문분석 통합 완료<br/>고품질 분할 결과]:::startEnd
    
    A --> B
    B --> C
    C -->|성공| E
    C -->|실패| D
    
    E --> F
    F -->|성공| G
    F -->|실패| H
    G --> I
    H --> I
    
    I --> J
    J -->|성공| K
    J -->|실패| L
    K --> M
    L --> M
    
    M --> N
    N --> O
    O -->|중국어| P
    O -->|한국어| Q
    P --> R
    Q --> R
    R --> S
    S --> T
    
    D -.-> T
```

---

## IO 유틸리티 플로우차트

```mermaid
flowchart LR
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef validation fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef excel fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef pandas fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,font-weight:bold
    
    %% Excel 읽기 플로우
    A[📂 Excel 입력 파일]:::excel
    B[🔧 pandas.read_excel<br/>데이터 로딩]:::pandas
    C[🔍 데이터 검증<br/>컬럼 구조 확인]:::validation
    D[📊 DataFrame 반환<br/>표준화된 데이터]:::pandas
    
    %% Excel 쓰기 플로우
    E[📊 DataFrame 입력<br/>처리 결과 데이터]:::pandas
    F[🔧 pandas.to_excel<br/>Excel 형식 변환]:::pandas
    G[💾 Excel 출력 파일<br/>결과 저장]:::excel
    
    %% 오류 처리
    H[❌ 파일 오류<br/>FileNotFoundError 등]:::error
    I[❌ 데이터 오류<br/>컬럼 누락, 형식 오류 등]:::error
    
    %% 데이터 플로우
    A --> B
    B --> C
    C --> D
    E --> F
    F --> G
    
    %% 오류 처리
    B -.-> H
    C -.-> I
    
    %% 유틸리티 기능들
    J[🔧 컬럼 이름 표준화<br/>원문, 번역문 등]:::process
    K[🔧 인덱스 재설정<br/>연속된 번호 부여]:::process
    L[🔧 빈 행 제거<br/>데이터 정제]:::process
    
    C --> J
    J --> K
    K --> L
    L --> D
```

---

## 캐시 시스템 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef cache fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef file fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef hit fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000,font-weight:bold
    classDef miss fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 캐시 시스템 시작]:::startEnd
    
    %% 캐시 키 생성
    B[🔑 캐시 키 생성<br/>텍스트 + 모델명 해시]:::process
    C[🔍 캐시 파일 확인<br/>기존 임베딩 존재 여부]:::cache
    D{캐시<br/>적중?}:::decision
    
    %% 캐시 히트
    E[⚡ 캐시 히트<br/>기존 임베딩 로드]:::hit
    F[📊 임베딩 벡터 반환<br/>계산 시간 단축]:::hit
    
    %% 캐시 미스
    G[🧠 캐시 미스<br/>새로운 임베딩 계산]:::miss
    H[💾 임베딩 캐시 저장<br/>다음 사용을 위해]:::miss
    I[📊 계산된 임베딩 반환]:::miss
    
    %% 캐시 관리
    J[🧹 캐시 정리<br/>오래된 캐시 제거]:::cache
    K[📊 캐시 통계<br/>히트율, 용량 등]:::cache
    L[💾 캐시 파일 관리<br/>pickle 직렬화]:::file
    
    A --> B
    B --> C
    C --> D
    D -->|Hit| E
    D -->|Miss| G
    E --> F
    G --> H
    H --> I
    
    %% 백그라운드 관리
    F -.-> J
    I -.-> J
    J --> K
    K --> L
    L -.-> C
```

---

## Common 모듈 통합 테스트 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef test fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef success fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000,font-weight:bold
    classDef failure fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,font-weight:bold
    classDef module fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 Common 모듈 통합 테스트 시작]:::startEnd
    
    %% 임베더 테스트
    B[🧠 BGE-M3 임베더 테스트<br/>초기화 및 임베딩 수행]:::test
    C{BGE-M3<br/>테스트 통과?}:::test
    D[✅ BGE-M3 테스트 성공<br/>임베딩 품질 확인]:::success
    E[❌ BGE-M3 테스트 실패<br/>모델 로딩 또는 임베딩 오류]:::failure
    
    %% 토크나이저 테스트
    F[🏮 하이브리드 토크나이저 테스트<br/>중국어/한국어 토크나이징]:::test
    G{토크나이저<br/>테스트 통과?}:::test
    H[✅ 토크나이저 테스트 성공<br/>언어별 토큰 품질 확인]:::success
    I[❌ 토크나이저 테스트 실패<br/>모델 로딩 또는 토큰화 오류]:::failure
    
    %% 구문분석기 테스트
    J[🏮 구문분석기 테스트<br/>SuPar + Stanza 통합]:::test
    K{구문분석기<br/>테스트 통과?}:::test
    L[✅ 구문분석기 테스트 성공<br/>구문 분석 품질 확인]:::success
    M[❌ 구문분석기 테스트 실패<br/>모델 로딩 또는 분석 오류]:::failure
    
    %% IO 유틸리티 테스트
    N[🔧 IO 유틸리티 테스트<br/>Excel 읽기/쓰기]:::test
    O{IO 테스트<br/>통과?}:::test
    P[✅ IO 테스트 성공<br/>파일 처리 확인]:::success
    Q[❌ IO 테스트 실패<br/>파일 접근 또는 형식 오류]:::failure
    
    %% 캐시 시스템 테스트
    R[💾 캐시 시스템 테스트<br/>저장/로드 기능]:::test
    S{캐시 테스트<br/>통과?}:::test
    T[✅ 캐시 테스트 성공<br/>캐시 동작 확인]:::success
    U[❌ 캐시 테스트 실패<br/>파일 시스템 오류]:::failure
    
    %% 통합 테스트 결과
    V[📊 전체 테스트 결과<br/>성공/실패 통계]:::test
    W[✅ Common 모듈 준비 완료<br/>SA/PA 모듈에서 사용 가능]:::startEnd
    
    A --> B
    B --> C
    C -->|성공| D
    C -->|실패| E
    D --> F
    E --> F
    
    F --> G
    G -->|성공| H
    G -->|실패| I
    H --> J
    I --> J
    
    J --> K
    K -->|성공| L
    K -->|실패| M
    L --> N
    M --> N
    
    N --> O
    O -->|성공| P
    O -->|실패| Q
    P --> R
    Q --> R
    
    R --> S
    S -->|성공| T
    S -->|실패| U
    T --> V
    U --> V
    V --> W
```

이제 Common 디렉토리의 모든 구성 요소와 상세한 워크플로우를 시각화했습니다. 이를 통해 SA와 PA 모듈이 어떻게 공통 라이브러리를 활용하는지 명확하게 이해할 수 있습니다!
