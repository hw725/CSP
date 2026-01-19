# Analytics Module

코퍼스 분석 및 시각화 도구 모음.

## 주요 파일

| 파일 | 설명 |
|------|------|
| `corpus_analyzer.py` | 전체 서종 누적 분석기 |
| `cluster_visualizer.py` | 시각화 및 클러스터링 분석기 |
| `monitoring_dashboard.py` | 실시간 모니터링 대시보드 생성 |
| `book_metadata_extractor.py` | 서종 메타데이터 추출기 |
| `book_metadata_editor.html` | 메타데이터 GUI 편집기 |

## 보조 스크립트

| 파일 | 설명 |
|------|------|
| `aggregate_batch_results.py` | 배치 결과 집계 |
| `analyze_sentence_eval.py` | 문장 평가 분석 |
| `extract_worst_sentences_report.py` | 저성능 문장 추출 |
| `row_compare_normalized_report.py` | 행별 정규화 비교 |
| `run_row_eval_normalized.py` | 행별 평가 실행 |

## 디렉토리

- `data/` - 분석 결과 데이터 (db, csv, xlsx, log)
- `utils/` - 유틸리티 스크립트
- `logs/` - 로그 파일
- `visualization_results/` - 시각화 결과

## 메타데이터 스키마

`book_metadata.json` 참조. GUI 편집: `book_metadata_editor.html` 사용.
