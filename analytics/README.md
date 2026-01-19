# Analytics Module

코퍼스 분석 및 시각화 도구 모음.

## 주요 스크립트

| 파일 | 설명 |
|------|------|
| `corpus_analyzer.py` | 전체 서종 누적 분석기 |
| `cluster_visualizer.py` | 시각화 및 클러스터링 분석기 |
| `monitoring_dashboard.py` | 실시간 모니터링 대시보드 생성 |
| `book_metadata_extractor.py` | 서종 메타데이터 추출기 |

## 메타데이터 관리

| 파일 | 설명 |
|------|------|
| `book_metadata.json` | 서종별 메타데이터 (저자, 시대, 사부, 역자, 레이어 등) |
| `book_metadata_editor.html` | 메타데이터 GUI 편집기 (브라우저에서 열기) |

## 보조 스크립트

| 파일 | 설명 |
|------|------|
| `aggregate_batch_results.py` | 배치 결과 집계 |
| `analyze_sentence_eval.py` | 문장 평가 분석 |
| `extract_worst_sentences_report.py` | 저성능 문장 추출 |
| `row_compare_normalized_report.py` | 행별 정규화 비교 |
| `run_row_eval_normalized.py` | 행별 평가 실행 |

## 디렉토리

- `data/` - 분석 결과 데이터 (gitignore)
- `utils/` - 공통 유틸리티 (`text_normalizer.py` 등)
- `logs/` - 로그 파일 (gitignore)
- `visualization_results/` - 시각화 결과
