# XLSX Pipeline - Excel 기반 데이터 처리 파이프라인

## 개요

XML 대신 Excel 파일(구병렬, 문장병렬, 문단병렬)을 입력으로 사용하는 효율적인 데이터 처리 파이프라인입니다.

## 특징

- ✅ **Excel 기반**: XML 파싱 불필요, 즉시 데이터 활용
- ✅ **다중 레벨 지원**: 구병렬(단어), 문장병렬, 문단병렬
- ✅ **자동 발견**: xlsx 디렉토리에서 자동으로 책 탐색
- ✅ **배치 처리**: 여러 책 동시 처리
- ✅ **통계 분석**: 데이터 품질 및 현황 파악
- ✅ **PA/SA 통합**: 문단/문장 정렬 분석 지원
- ✅ **정확도 평가**: 자동 정확도 측정 및 리포트

## 디렉토리 구조

```
/workspace/tsv_output/          # Excel 파일 루트
├── 당송팔대가문초구양수1/
│   ├── 당송팔대가문초구양수1_구병렬.xlsx
│   ├── 당송팔대가문초구양수1_문장병렬.xlsx
│   └── 당송팔대가문초구양수1_문단병렬.xlsx
├── 당송팔대가문초구양수2/
│   └── ...
└── ... (총 43개 책)

xlsx_pipeline_results/          # 처리 결과 출력
├── statistics_*.json
├── batch_*.json
└── {책이름}_*.json
```

## Excel 파일 형식

### 1. 구병렬 (단어/구 단위)
| 문장식별자 | 구식별자 | 원문 | 번역문 |
|-----------|---------|------|--------|
| 1 | 1 | 子曰 | 공자께서 말씀하시기를 |
| 1 | 2 | 學而時習之 | 배우고 때때로 익히면 |

### 2. 문장병렬 (문장 단위)
| 문단식별자 | 문장식별자 | 원문 | 번역문 |
|-----------|-----------|------|--------|
| 1 | 1 | 子曰學而時習之 | 공자께서 말씀하시기를 배우고... |
| 1 | 2 | 不亦說乎 | 또한 기쁘지 아니한가 |

### 3. 문단병렬 (문단 단위)
| 문단식별자 | 원문 | 번역문 |
|-----------|------|--------|
| 1 | 子曰學而時習之不亦說乎... | 공자께서 말씀하시기를... |
| 2 | 有朋自遠方來... | 벗이 먼 곳에서 찾아오니... |

## 설치 및 실행

### Docker 환경 (권장)

```bash
# 컨테이너 시작
cd CSP
docker-compose up -d

# CLI 실행
docker exec -it csp-workspace python xlsx_pipeline/xlsx_pipeline_cli.py [command]
```

### 로컬 환경

```bash
cd xlsx_pipeline
python xlsx_pipeline_cli.py [command]
```

## 사용법

### 1. 책 자동 발견

```bash
# Excel 파일 디렉토리에서 모든 책 자동 탐색
python xlsx_pipeline_cli.py discover
```

**출력 예시:**
```
🔍 책 자동 발견 중...
✅ 43개 책 발견됨:
  - 당송팔대가문초구양수1
  - 당송팔대가문초구양수2
  ...
```

### 2. 책 목록 조회

```bash
# 모든 책 목록 및 파일 상태 확인
python xlsx_pipeline_cli.py list
```

**출력 예시:**
```
📚 책 목록
  1. 당송팔대가문초구양수1
     파일: 구병렬, 문장병렬, 문단병렬
  2. 당송팔대가문초구양수2
     파일: 구병렬, 문장병렬, 문단병렬
...
총 43개 책
```

### 3. 통계 정보 조회

```bash
# 전체 통계
python xlsx_pipeline_cli.py stats

# 특정 책 통계
python xlsx_pipeline_cli.py stats --book 당송팔대가문초구양수1
```

**출력 예시:**
```
📊 전체 통계
총 책 수: 43

파일 유형별:
  - 구병렬: 43개
  - 문장병렬: 43개
  - 문단병렬: 43개

총 데이터량:
  - 구병렬: 318,086행
  - 문장병렬: 87,269행
  - 문단병렬: 21,000+행
```

### 4. 단일 책 처리

```bash
# 기본 처리 (모든 레벨)
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1

# 특정 레벨만 처리
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --levels word sentence

# 통계만 수행
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --analysis statistics
# PA 분석 실행
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --analysis pa

# SA 분석 실행
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --analysis sa

# 전체 파이프라인 (PA + SA + Accuracy)
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --analysis full```

### 5. 배치 처리

```bash
# 전체 책 배치 처리
python xlsx_pipeline_cli.py batch

# 특정 책들만 처리
python xlsx_pipeline_cli.py batch --books 당송팔대가문초구양수1 당송팔대가문초구양수2

# 문장병렬과 문단병렬만 처리
python xlsx_pipeline_cli.py batch --levels sentence paragraph
```

## API 사용 (Python 스크립트)

### 기본 사용법

```python
from xlsx_pipeline.xlsx_pipeline_processor import XLSXPipelineProcessor

# 프로세서 초기화
processor = XLSXPipelineProcessor(
    xlsx_root_dir="/workspace/tsv_output",
    output_dir="xlsx_pipeline_results"
)

# 책 자동 발견
discovered = processor.discover_books()
print(f"발견된 책: {len(discovered)}개")

# 특정 책 가져오기
book = processor.get_book("당송팔대가문초구양수1")

# 데이터 로드
word_df = book.load_word_parallel()
sentence_df = book.load_sentence_parallel()
paragraph_df = book.load_paragraph_parallel()

print(f"구병렬: {len(word_df)}행")
print(f"문장병렬: {len(sentence_df)}행")
print(f"문단병렬: {len(paragraph_df)}행")
```

### 파이프라인 실행

```python
# 단일 책 처리
result = processor.process_book_pipeline(
    "당송팔대가문초구양수1",
    config={
        "levels": ["word", "sentence", "paragraph"],
        "analysis": ["statistics"]
    }
)

# 배치 처리
batch_results = processor.batch_process(
    book_ids=["당송팔대가문초구양수1", "당송팔대가문초구양수2"],
    config={"levels": ["sentence", "paragraph"]}
)
```

### 통계 정보 수집

```python
# 모든 책 통계
all_stats = processor.get_all_statistics()

# JSON으로 저장
processor.save_statistics("statistics.json")

# 특정 책 통계
book = processor.get_book("당송팔대가문초구양수1")
stats = book.get_statistics()
```

## 고급 기능

### PA (Paragraph Alignment) 분석
```python
from xlsx_pipeline.xlsx_pipeline_processor import XLSXPipelineProcessor

processor = XLSXPipelineProcessor()
book = processor.get_book("당송팔대가문초구양수1")

# PA 분석 실행
result = processor.process_book_pipeline(
    "당송팔대가문초구양수1",
    config={"analysis": ["pa"]}
)

# 결과: {book_id}_pa_output.xlsx
```

### SA (Sentence Alignment) 분석
```python
# SA 분석 실행
result = processor.process_book_pipeline(
    "당송팔대가문초구양수1",
    config={"analysis": ["sa"]}
)

# 결과: {book_id}_sa_output.xlsx
```

### 전체 파이프라인 (PA + SA + Accuracy)
```python
# PA, SA 실행 후 정확도 자동 평가
result = processor.process_book_pipeline(
    "당송팔대가문초구양수1",
    config={
        "analysis": ["full"],
        "project": "guanzi"  # 임계값 설정
    }
)

# 결과:
# - {book_id}_pa_output.xlsx
# - {book_id}_sa_output.xlsx
# - {book_id}_pa_accuracy.json
# - {book_id}_sa_accuracy.json
# - {book_id}_full_pipeline.json
```

### 품질 검사 (기본 제공)
- NaN 값 자동 탐지 및 로깅
- 데이터 통계 자동 수집
- 파일 존재 여부 확인

## 명령 옵션

### 공통 옵션

- `--xlsx-root PATH`: Excel 파일 루트 디렉토리 (기본값: `/workspace/tsv_output`)
- `--output PATH`: 결과 출력 디렉토리 (기본값: `xlsx_pipeline_results`)
- `--verbose, -v`: 상세 로그 출력

### 처리 옵션

- `--book BOOK_ID`: 처리할 책 ID
- `--books BOOK_ID [BOOK_ID ...]`: 배치 처리할 책 ID 리스트
- `--levels {word,sentence,paragraph} [...]`: 처리할 레벨 선택
- `--analysis {statistics,pa,sa,accuracy,full} [...]`: 수행할 분석 선택
- `--project PROJECT_NAME`: 프로젝트 이름 (정확도 평가용 임계값)

## 출력 파일

### 통계 JSON
```json
{
  "total_books": 43,
  "timestamp": "2024-12-14T...",
  "books": {
    "당송팔대가문초구양수1": {
      "word_count": 7321,
      "sentence_count": 2028,
      "paragraph_count": 404,
      "word_nan_count": {...},
      ...
    }
  }
}
```

### 처리 결과 JSON
```json
{
  "book_id": "당송팔대가문초구양수1",
  "timestamp": "2024-12-14T...",
  "config": {...},
  "data": {
    "word": {"rows": 7321, "columns": [...]},
    "sentence": {"rows": 2028, "columns": [...]},
    "paragraph": {"rows": 404, "columns": [...]}
  },
  "analysis": {
    "statistics": {...}
  }
}
```

## 성능 최적화

- **캐싱**: 데이터프레임 자동 캐싱으로 반복 로드 방지
- **병렬 처리**: 배치 처리 시 여러 책 동시 처리 (예정)
- **메모리 효율**: 필요한 레벨만 선택적 로드

## 문제 해결

### Excel 파일을 찾을 수 없음
```bash
# 파일 경로 확인
ls -la /workspace/tsv_output/

# xlsx-root 옵션으로 경로 지정
python xlsx_pipeline_cli.py list --xlsx-root /custom/path
```

### 특정 책 처리 실패
```bash
# 상세 로그로 원인 파악
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 -v

# 특정 레벨만 처리
python xlsx_pipeline_cli.py process --book 당송팔대가문초구양수1 --levels sentence
```

## XML Pipeline과 비교

| 항목 | XML Pipeline | XLSX Pipeline |
|------|-------------|---------------|
| 입력 | XML 파일 쌍 | Excel 파일 |
| 파싱 | XML 파싱 필요 | 즉시 로드 |
| 처리 속도 | 느림 | 빠름 |
| 메모리 | 높음 | 낮음 |
| 유지보수 | 복잡 | 간단 |
| 데이터 검증 | 어려움 | 쉬움 |

## 기능 구현 상태

1. ✅ 기본 프로세서 구현
2. ✅ CLI 도구 구현
3. ✅ PA/SA 통합 분석
4. ✅ 정확도 평가 (Accuracy)
5. ✅ 통계 및 품질 검사
6. ⬜ 유사도 시각화 도구
7. ⬜ 웹 인터페이스

## XML Pipeline 대비 개선사항

| 기능 | XML Pipeline | XLSX Pipeline |
|------|-------------|---------------|
| 데이터 로드 | XML 파싱 (느림) | Excel 직접 로드 (빠름) |
| PA 분석 | ✅ | ✅ |
| SA 분석 | ✅ | ✅ |
| 정확도 평가 | ✅ | ✅ |
| 배치 처리 | ✅ | ✅ |
| 통계 수집 | 부분적 | ✅ 완전 |
| 메모리 효율 | 낮음 | 높음 |
| 유지보수성 | 복잡 | 간단 |

## 라이선스

CSP 프로젝트와 동일
