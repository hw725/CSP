# PA 디렉토리 상세 플로우차트

## PA (Paragraph Aligner) 모듈 구조 및 워크플로우 - 구문분석 기반 (2025-08-21 최신)

---

## PA 디렉토리 전체 아키텍처 플로우차트

```mermaid
flowchart TB
    %% 스타일 정의 (글자가 잘 보이도록)
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef mainFile fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,color:#000,font-weight:bold
    classDef processor fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef module fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef parser fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef config fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 PA 디렉토리 시작<br/>구문분석 기반 문단→문장 정렬]:::startEnd
    
    %% 메인 실행기
    B[📋 main.py<br/>CLI 인터페이스<br/>및 인수 파싱]:::mainFile
    
    %% 핵심 처리기
    C[🔧 processor.py<br/>메인 처리 로직<br/>전체 플로우 제어]:::processor
    
    %% 🆕 구문분석 모듈들
    D[🏮 sentence_splitter.py<br/>구문분석 기반 문장 분할<br/>SuPar-Kanbun + Stanza + BGE-M3]:::parser
    E[🎯 aligner.py<br/>BGE-M3 FlagModel<br/>정렬 알고리즘]:::module
    
    %% 데이터 파일들
    F[📊 input.xlsx<br/>입력 문단 데이터<br/>원문과 번역문]:::data
    G[📊 output.xlsx<br/>구문분석 처리 결과<br/>문단ID와 문장ID와 정렬]:::data
    
    %% 설정 파일
    H[⚙️ config_example.json<br/>설정 예시<br/>임계값, 모델 등]:::config
    
    %% 공통 모듈 (상위 디렉토리)
    I[🧠 ../common/embedders/bge.py<br/>BGE-M3 FlagModel<br/>안정화된 임베딩]:::module
    J[🔧 ../common/io_utils.py<br/>Excel 입출력<br/>유틸리티]:::module
    K[🏮 ../common/new_parsers.py<br/>🆕 SuPar-Kanbun + Stanza<br/>구문분석기 통합]:::parser
    
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    C --> G
    
    %% 의존성 관계
    B -.-> H
    C -.-> I
    C -.-> J
    D -.-> K
    E -.-> I
```

---

## PA main.py 실행 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef cli fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 python pa/main.py 실행]:::startEnd
    B[📋 CLI 인수 파싱<br/>argparse 설정]:::cli
    
    %% CLI 옵션들
    C[📄 필수 인수<br/>input_file<br/>output_file]:::cli
    D[⚙️ 선택 옵션<br/>--embedder bge/openai<br/>--max-length 180<br/>--threshold 0.7]:::cli
    E[🔑 OpenAI 옵션<br/>--openai-model<br/>--openai-api-key]:::cli
    F[📊 기타 옵션<br/>--verbose]:::cli
    
    %% 검증 단계
    G{인수 검증<br/>성공?}:::decision
    H[❌ 인수 오류<br/>도움말 출력]:::error
    
    %% 🆕 구문분석기 초기화
    HT[🏮 구문분석기 초기화<br/>SuPar-Kanbun 원문 + Stanza 번역문<br/>+ BGE-M3 FlagModel 임베딩]:::process
    HT2{구문분석기<br/>초기화 성공?}:::decision
    HT3[⚠️ 구문분석기 초기화 실패<br/>경고 메시지 출력]:::error
    
    %% 처리 시작
    I[🔧 processor.py import<br/>process_paragraph_file]:::process
    J[⏱️ 시작 시간 기록<br/>처리 시작 로그]:::process
    K[🎯 메인 처리 함수 호출<br/>모든 옵션 전달]:::process
    
    %% 결과 처리
    L[📊 처리 결과 받기<br/>DataFrame 반환]:::process
    M[⏱️ 처리 시간 계산<br/>완료 로그 출력]:::process
    N[✅ PA 처리 완료]:::startEnd
    
    %% 오류 처리
    O[❌ 예외 발생<br/>에러 메시지 출력]:::error
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G -->|실패| H
    G -->|성공| HT
    HT --> HT2
    HT2 -->|실패| HT3
    HT2 -->|성공| I
    HT3 --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    
    K -.->|예외 발생| O
    O -.-> H
```

---

## PA processor.py 메인 처리 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef splitter fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef aligner fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 process_paragraph_file 시작]:::startEnd
    
    %% 입력 파일 처리
    B[📂 Excel 파일 읽기<br/>pandas.read_excel]:::data
    C{파일 읽기<br/>성공?}:::decision
    D[❌ 파일 없음 오류<br/>FileNotFoundError]:::error
    
    %% 데이터 검증
    E[📊 데이터 구조 확인<br/>문단 개수 로그]:::process
    F[🔍 컬럼 검증<br/>원문/번역문 확인]:::process
    
    %% 구문분석 기반 분할
    G[🏮 sentence_splitter 호출<br/>구문분석 기반 문단 → 문장 분할<br/>SuPar-Kanbun + Stanza + BGE-M3]:::splitter
    H[📋 분할 결과 검증<br/>구문 구조 기반 문장 개수 확인]:::process
    
    %% BGE-M3 FlagModel 임베딩 및 정렬
    I[🎯 aligner 모듈 호출<br/>BGE-M3 FlagModel 초기화]:::aligner
    J{BGE-M3 FlagModel<br/>초기화 성공?}:::decision
    K[❌ BGE-M3 오류<br/>기능 비활성화]:::error
    
    L[🧠 문장 임베딩<br/>FlagModel 벡터 변환]:::aligner
    M[📐 유사도 계산<br/>코사인 유사도 매트릭스]:::aligner
    N[🎯 정렬 수행<br/>구문+의미 하이브리드 정렬]:::aligner
    
    %% 결과 처리
    O[📊 결과 DataFrame 생성<br/>문단ID와 문장ID와 정렬]:::data
    P[💾 Excel 파일 저장<br/>output_file 경로]:::data
    Q[✅ 처리 완료<br/>결과 반환]:::startEnd
    
    A --> B
    B --> C
    C -->|실패| D
    C -->|성공| E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J -->|실패| K
    J -->|성공| L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    
    %% 오류 경로
    D -.-> Q
    K -.-> O
```

---

## PA sentence_splitter.py 분할 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef parser fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef embedder fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef validation fill:#f1f8e9,stroke:#558b2f,stroke-width:2px,color:#000,font-weight:bold
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 split_target_sentences_advanced<br/>구문분석 기반 고급 분할]:::startEnd
    
    %% 🆕 구문분석기 초기화
    PA[🏮 구문분석기 초기화<br/>SuPar-Kanbun 원문 + Stanza 번역문]:::parser
    B[🤖 BGE-M3 FlagModel 로드<br/>의미적 매칭용 임베더]:::embedder
    C{구문분석기 로드<br/>성공?}:::decision
    D[❌ 구문분석기 없음<br/>설치 안내 출력]:::process
    
    %% 문단별 처리
    E[📋 문단 데이터 순회<br/>tqdm 진행률 표시]:::process
    F[📝 원문 문단 구문분석<br/>🆕 SuPar-Kanbun 한문 구문분석<br/>구문 구조 기반 정확 분할]:::parser
    G[📝 번역문 문단 구문분석<br/>🆕 Stanza 한국어 구문분석<br/>구문 구조 기반 정확 분할]:::parser
    
    %% BGE-M3 기반 의미적 정렬
    H[🧠 BGE-M3 FlagModel 임베딩<br/>구문 분할된 문장들의 의미 매칭]:::embedder
    I[🔍 분할 품질 검증<br/>구문+의미 이중 검증]:::validation
    J{구문분석+의미매칭<br/>품질 통과?}:::decision
    K[✂️ 보수적 재분할<br/>구문 구조 우선 재분석]:::validation
    
    %% 문장 검증
    L[🔍 문장 길이 검증<br/>max_length 기준]:::validation
    M{길이 기준<br/>통과?}:::decision
    N[✂️ 긴 문장 재분할<br/>구두점 기준]:::process
    
    %% 문장 정제
    O[🧹 문장 정제<br/>공백/특수문자 제거]:::process
    P[📊 빈 문장 필터링<br/>유효 문장만 유지]:::validation
    
    %% 결과 생성
    Q[📋 문장 리스트 생성<br/>구문분석 기반 정확한 문장쌍]:::output
    R[📊 분할 통계 출력<br/>구문 구조별 문장 개수]:::output
    S[✅ 구문분석 기반 분할 완료<br/>고품질 문장 리스트 반환]:::startEnd
    
    A --> PA
    PA --> B
    B --> C
    C -->|실패| D
    C -->|성공| E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J -->|실패| K
    J -->|성공| L
    K --> L
    L --> M
    M -->|실패| N
    M -->|성공| O
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    
    D -.-> S
```

---

## PA aligner.py 정렬 플로우차트

```mermaid
flowchart TD
    %% 스타일 정의
    classDef startEnd fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000,font-weight:bold
    classDef embedder fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000,font-weight:bold
    classDef similarity fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    
    A[🚀 improved_align_paragraphs]:::startEnd
    
    %% BGE-M3 FlagModel 초기화
    B[🧠 BGE-M3 FlagModel 로드<br/>FlagEmbedding 1.1.7 안정화<br/>transformers 4.36.0 호환]:::embedder
    C{BGE-M3 FlagModel<br/>로드 성공?}:::decision
    D[🤖 BGE-M3 FlagModel 준비<br/>GPU 가속 임베딩]:::embedder
    E[🌐 OpenAI API 백업<br/>text-embedding-3-large]:::embedder
    
    %% 임베딩 처리
    F[📊 문장 배치 처리<br/>구문분석된 원문/번역문 분리]:::process
    G[🔢 BGE-M3 FlagModel 임베딩<br/>문장 → 1024차원 벡터]:::embedder
    H[💾 임베딩 캐시<br/>중복 계산 방지]:::process
    
    %% 유사도 계산
    I[📐 유사도 매트릭스<br/>M×N 행렬 생성]:::similarity
    J[🎯 코사인 유사도<br/>FlagModel 벡터 내적 계산]:::similarity
    K[📊 임계값 필터링<br/>고품질 매칭만 유지]:::similarity
    
    %% 구문+의미 하이브리드 정렬
    L[🎯 하이브리드 매칭<br/>구문구조 + 의미유사도]:::process
    M[📋 매칭 쌍 생성<br/>구문분석_idx와 의미매칭_idx]:::process
    N[🔍 매칭 검증<br/>구문+의미 이중 검증]:::process
    
    %% 결과 생성
    O[📊 정렬 결과 구성<br/>문단ID와 문장ID와 구문+의미 정렬쌍]:::output
    P[📈 정렬 품질 평가<br/>구문정확도 + 의미유사도]:::output
    Q[✅ 구문+의미 하이브리드 정렬 완료<br/>고품질 결과 리스트 반환]:::startEnd
    
    A --> B
    B --> C
    C -->|BGE-M3| D
    C -->|OpenAI 백업| E
    D --> F
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    
    %% 캐시 연결
    R[💾 ../embeddings_cache_openai/<br/>OpenAI 캐시 디렉토리]:::process
    E -.-> R
    R -.-> H
```

---

## PA 디렉토리 데이터 플로우

```mermaid
flowchart LR
    %% 스타일 정의
    classDef input fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px,color:#000,font-weight:bold
    classDef intermediate fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000,font-weight:bold
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000,font-weight:bold
    classDef config fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:bold
    
    %% 입력 데이터
    A[📊 input.xlsx<br/>문단 원문과 문단 번역문]:::input
    
    %% 중간 처리 데이터
    B[🏮 구문분석된 문장<br/>SuPar-Kanbun원문 + Stanza번역문]:::intermediate
    C[🔢 BGE-M3 FlagModel 벡터<br/>v1, v2... 1024차원 배열]:::intermediate
    D[📐 구문+의미 유사도 매트릭스<br/>M×N 하이브리드 점수]:::intermediate
    E[🎯 하이브리드 매칭 쌍<br/>구문구조 + 의미유사도 기반]:::intermediate
    
    %% 출력 데이터
    F[📊 output.xlsx<br/>문단ID와 문장ID와 구문분석결과와 의미정렬]:::output
    
    %% 설정 파일
    G[⚙️ config_example.json<br/>구문분석 + 의미매칭 임계값<br/>BGE-M3 FlagModel 설정]:::config
    
    %% 데이터 플로우
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    %% 설정 참조
    G -.-> B
    G -.-> C
    G -.-> D
    
    %% 데이터 형태 예시
    H[📝 입력 예시<br/>원문: 子曰 學而時習之 不亦說乎<br/>번역: 공자가 말씀하셨다...]:::input
    I[📝 출력 예시<br/>1_1_子曰_공자가 말씀하셨다<br/>구문분석+의미매칭 0.87<br/>1_2_學而時習之_배우고 때때로 익히면<br/>구문분석+의미매칭 0.92...]:::output
    
    A -.-> H
    F -.-> I
```

이제 PA 디렉토리의 모든 구성 요소와 데이터 플로우를 상세히 시각화했습니다. 다음으로 SA 디렉토리 플로우차트를 작성하겠습니다.
