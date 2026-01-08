# 📚 복수 역자 처리 가이드

## 🎯 복수 역자 표시 방법

### 1. 기본 구분자
- **구분자**: 세미콜론 + 공백 (`;` + ` `)
- **예시**: `김철수; 이영희; 박민수`

### 2. 다양한 복수 역자 패턴

#### 👥 공동 번역
```
김철수; 이영희; 박민수
```

#### 🏢 기관 + 개인
```
한국고전번역원 (김철수 주역)
서울대학교 번역팀 (이영희 책임)
```

#### 📝 역할 구분
```
김철수 번역; 이영희 감수
A대학 번역; B연구소 감수
```

#### 🎓 학술 번역
```
한국고전학회 (김철수; 이영희; 박민수)
고전번역연구소 번역팀
```

## 💻 프로그래밍 방식 추가

### Python 코드로 복수 역자 추가
```python
from book_metadata_extractor import BookMetadataExtractor

# 추출기 생성
extractor = BookMetadataExtractor()

# 방법 1: 리스트로 추가
extractor.add_translator_mapping(
    jti_code='4c0999', 
    author='테스트작가', 
    translators=['김철수', '이영희', '박민수']
)

# 방법 2: 문자열로 직접 추가
extractor.add_translator_mapping(
    jti_code='4c0998', 
    author='테스트작가2', 
    translators='김철수; 이영희; 박민수'
)

# 방법 3: 특별한 케이스는 multiple_translator_mappings에 추가
extractor.multiple_translator_mappings['특별한패턴'] = {
    'author': '작가명',
    'translator': '한국고전번역원 (김철수 주역; 이영희 감수)'
}
```

## 📋 CSV에서 수동 편집 시

### Excel/LibreOffice에서
1. 역자 컬럼 선택
2. 기존 텍스트를 복수 역자로 교체
3. 구분자 통일성 확인: `;` + 공백

### 예시 변경
```
변경 전: 한국고전번역원
변경 후: 김철수; 이영희; 박민수

변경 전: 미상
변경 후: A대학 번역팀 (김철수 책임)
```

## 🔄 자동 처리 확인

### 복수 역자 감지 로그
```
INFO - 복수 역자 매핑 성공: 책명 -> {'author': '작가', 'translator': '김철수; 이영희'}
INFO - 복수 역자 감지: 2명
```

### 테스트 실행
```bash
cd CSP/analytics
python book_metadata_extractor.py
```

## ⚡ 대량 처리 스크립트

### 일괄 복수 역자 추가
```python
import pandas as pd
from book_metadata_extractor import BookMetadataExtractor

# CSV 읽기
df = pd.read_csv('cumulative_analysis_results_manual.csv', encoding='utf-8-sig')

# 특정 조건의 역자를 복수 역자로 변경
mask = df['책명'].str.contains('특정패턴')
df.loc[mask, '역자'] = '김철수; 이영희; 박민수'

# 기관명을 구체적인 역자로 변경
df['역자'] = df['역자'].replace(
    '한국고전번역원', 
    '김철수; 이영희'
)

# 저장
df.to_csv('cumulative_analysis_results_manual.csv', index=False, encoding='utf-8-sig')
print("복수 역자 일괄 처리 완료!")
```

## 📊 분석 시 고려사항

### 통계 분석
- 복수 역자는 각각을 개별 역자로 집계 가능
- 세미콜론 구분자로 split하여 개별 처리
- 기관명과 개인명 혼재 시 정규화 필요

### 검색 및 필터링
```python
# 특정 역자가 참여한 모든 책 찾기
def find_books_by_translator(df, translator_name):
    return df[df['역자'].str.contains(translator_name, na=False)]

# 복수 역자 책만 필터링
def find_collaborative_books(df):
    return df[df['역자'].str.contains(';', na=False)]
```

## ✅ 품질 관리

### 일관성 검사
- 구분자 통일성: 모두 `;` + 공백 사용
- 이름 표기 통일: 한글 성명 vs 영문 이니셜
- 기관명 표준화: 정식 명칭 사용

### 검증 스크립트
```python
def validate_translator_format(df):
    """역자 컬럼 형식 검증"""
    issues = []
    
    for idx, row in df.iterrows():
        translator = str(row['역자'])
        
        # 잘못된 구분자 검사
        if ';' in translator and '; ' not in translator:
            issues.append(f"행 {idx}: 구분자 오류 - '{translator}'")
        
        # 빈 역자명 검사
        if translator.strip() == '' or translator == 'nan':
            issues.append(f"행 {idx}: 빈 역자명")
    
    return issues
```

이제 **복수 역자도 완벽하게 처리할 수 있습니다!** 🎉