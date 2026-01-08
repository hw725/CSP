# 📚 작가/역자 정보 포함 분석 시스템 사용법

## 🔄 작업 흐름

### 1단계: 첫 번째 분석 실행
```bash
cd CSP/analytics
python 통합_전체서종_누적분석기_v3.py
```

### 2단계: 수동 편집용 CSV 템플릿 생성
```python
from 통합_전체서종_누적분석기_v3 import CumulativeBookAnalyzer

analyzer = CumulativeBookAnalyzer()
analyzer.create_manual_csv_template()
```

### 3단계: 수동 편집
- `cumulative_analysis_results_manual.csv` 파일을 Excel로 열기
- 작가/역자 정보 수정 (특히 "미상"인 항목들)
- UTF-8으로 저장

### 4단계: 다음 분석 (자동 보호)
```bash
python 통합_전체서종_누적분석기_v3.py
```
→ 수동 편집 파일이 보호되고, 새로운 자동 파일 생성됨

### 5단계: 필요 시 병합
```python
analyzer = CumulativeBookAnalyzer()
analyzer.merge_manual_csv()
```

## 🛡️ 보호 메커니즘

1. **자동 감지**: `cumulative_analysis_results_manual.csv` 파일 존재 시 자동으로 새 파일명 사용
2. **타임스탬프**: `cumulative_analysis_results_auto_20250926_093045.csv` 형태로 생성
3. **백업**: 병합 시 기존 수동 파일 자동 백업
4. **선택적 병합**: 새로운 책만 추가, 기존 수동 편집 내용 보존

## 📁 파일 구조

```
CSP/analytics/
├── cumulative_analysis_results.csv           # 기본 자동 생성 파일
├── cumulative_analysis_results_manual.csv    # 수동 편집용 (보호됨)
├── cumulative_analysis_results_auto_*.csv    # 보호 모드에서 생성되는 파일
├── cumulative_analysis.db                    # SQLite 데이터베이스
└── book_metadata_extractor.py               # 작가/역자 추출기
```

## 🎯 수동 편집 팁

### Excel에서 작업 시
1. **인코딩 주의**: UTF-8로 저장
2. **찾기/바꾸기**: `Ctrl+F`로 "미상" 검색하여 일괄 수정
3. **필터 활용**: 작가 컬럼으로 필터링하여 같은 작가 그룹 일괄 처리
4. **복수 역자 표시**: 세미콜론과 공백으로 구분 (예: `김철수; 이영희; 박민수`)

### 복수 역자 표시 방법
- **개인 역자들**: `김철수; 이영희; 박민수`
- **기관 + 개인**: `한국고전번역원 (김철수 주역)`
- **공동 번역**: `A대학 번역팀; B연구소`
- **감수자 포함**: `김철수 번역; 이영희 감수`

### 자주 수정할 항목들
- 작가: "미상" → 실제 작가명
- 역자: "한국고전번역원" → 구체적인 역자명
- 복수 역자: 세미콜론(;)으로 구분하여 나열

## 🔧 추출기 업데이트

자주 나오는 패턴은 `book_metadata_extractor.py`에 추가:

### 단일 역자
```python
# author_mappings 또는 text_name_mappings에 추가
'새로운_jti_코드': {'author': '작가명', 'translator': '역자명'},
```

### 복수 역자
```python
# 복수 역자 매핑 추가 (동적)
extractor = BookMetadataExtractor()
extractor.add_translator_mapping('jti_code', '작가명', ['역자1', '역자2', '역자3'])

# 또는 직접 문자열로
extractor.add_translator_mapping('jti_code', '작가명', '김철수; 이영희; 박민수')
```

### 특별한 경우
```python
# multiple_translator_mappings에 추가
self.multiple_translator_mappings = {
    'special_book_pattern': {
        'author': '작가명',
        'translator': '한국고전번역원 (김철수 주역; 이영희 감수)'
    },
}
```

## ⚠️ 주의사항

1. **파일명 변경 금지**: 수동 편집 파일은 정확히 `cumulative_analysis_results_manual.csv`로 유지
2. **컬럼 순서 유지**: CSV 헤더와 컬럼 순서 변경하지 않기
3. **백업 확인**: 중요한 수동 편집 후에는 별도 백업 권장

## 🚀 고급 사용법

### 특정 책만 재분석
```python
analyzer = CumulativeBookAnalyzer()
# 특정 결과 폴더만 처리하도록 커스터마이징 가능
```

### 대량 일괄 편집
```python
import pandas as pd

# CSV 읽기
df = pd.read_csv('cumulative_analysis_results_manual.csv', encoding='utf-8-sig')

# 일괄 수정 (예: 특정 패턴의 역자 변경)
df.loc[df['책명'].str.contains('특정패턴'), '역자'] = '새로운역자'

# 저장
df.to_csv('cumulative_analysis_results_manual.csv', index=False, encoding='utf-8-sig')
```