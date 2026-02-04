# Analytics Directory

이 디렉토리는 CSP(고전 산문 병렬화) 프로젝트의 분석 및 시각화 도구들을 포함합니다.

## 📁 디렉토리 구조

```
analytics/
├── README.md                                    # 이 파일
├── 통합_시각화_클러스터링_분석기.py                 # 통합 시각화 및 클러스터링 분석 도구
├── 통합_전체서종_누적분석기_v3.py                  # 전체 서종 누적 분석기
├── 실시간_모니터링_대시보드_v3.py                  # 실시간 모니터링 대시보드
├── monitoring_dashboard.html                    # 생성된 모니터링 대시보드 HTML
├── cumulative_analysis.db                       # 누적 분석 데이터베이스
├── cumulative_analysis.log                      # 분석 로그 파일
├── cumulative_analysis_report.json             # 분석 리포트 JSON
├── cumulative_analysis_results.csv             # 분석 결과 CSV
└── visualization_results/                       # 시각화 결과 파일들
    ├── correlation_analysis.png
    ├── quality_distribution.png
    ├── clustering_analysis.png
    ├── performance_comparison.png
    └── quality_grade_analysis.png
```

## 🔧 주요 도구들

### 1. 통합_시각화_클러스터링_분석기.py
- **목적**: PA/SA 정확도 데이터의 종합적인 시각화 및 클러스터링 분석
- **기능**:
  - 5가지 유형의 시각화 차트 생성
  - K-means, DBSCAN, Hierarchical 클러스터링 분석
  - 한글 폰트 자동 설정 및 검증
  - 다양한 클러스터 개수(K) 최적화 분석

### 2. 통합_전체서종_누적분석기_v3.py
- **목적**: 전체 서종에 대한 누적 분석 및 트렌드 추적
- **기능**:
  - 시계열 누적 분석
  - 성능 트렌드 모니터링
  - 데이터베이스 기반 분석 결과 저장

### 3. 실시간_모니터링_대시보드_v3.py
- **목적**: 실시간 분석 결과 모니터링 및 대시보드 생성
- **기능**:
  - 실시간 성능 지표 모니터링
  - 인터랙티브 HTML 대시보드 생성
  - 알람 및 임계값 모니터링

## 🚀 사용법

### Docker 환경에서 실행
```bash
# 한글 폰트가 포함된 컨테이너에서 실행
docker-compose exec csp python analytics/통합_시각화_클러스터링_분석기.py

# 전체 서종 누적 분석 실행
docker-compose exec csp python analytics/통합_전체서종_누적분석기_v3.py

# 실시간 모니터링 대시보드 생성
docker-compose exec csp python analytics/실시간_모니터링_대시보드_v3.py
```

### 로컬 환경에서 실행
```bash
# 필요한 패키지 설치
pip install matplotlib seaborn plotly scikit-learn

# 분석 도구 실행
python analytics/통합_시각화_클러스터링_분석기.py
```

## 📊 결과 파일들

- `visualization_results/`: 생성된 모든 시각화 차트들
- `cumulative_analysis.*`: 누적 분석 결과 및 로그
- `monitoring_dashboard.html`: 인터랙티브 모니터링 대시보드

## ⚙️ 설정

### 한글 폰트 설정
Docker 컨테이너에서는 다음 한글 폰트들이 자동으로 설정됩니다:
- NanumGothic, NanumMyeongjo
- NotoSansCJK, NotoSerifCJK
- UnDotum, Baekmuk 계열

### 데이터 경로 설정
분석 도구들은 다음 경로에서 데이터를 읽습니다:
- PA 분석: `p2s/output.xlsx`
- SA 분석: `s2p/output.xlsx`
- 정확도 평가: `accuracy/관자4_문단병렬.xlsx`

## 🔍 문제 해결

### 한글 폰트 문제
한글이 네모(□)로 표시되는 경우:
1. Docker 컨테이너 재빌드: `docker-compose build --no-cache`
2. 폰트 캐시 새로고침: `fc-cache -fv`
3. matplotlib 폰트 캐시 삭제 후 재설정

### 데이터 파일 없음
필요한 Excel 파일들이 없는 경우:
1. PA/SA 분석을 먼저 실행하여 output.xlsx 파일 생성
2. 정확도 평가 파일의 경로 및 파일명 확인

## 📝 로그 및 디버깅

분석 실행 시 생성되는 로그:
- `cumulative_analysis.log`: 상세 실행 로그
- 콘솔 출력: 실시간 진행 상황 및 오류 정보
