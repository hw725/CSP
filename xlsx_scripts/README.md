# XLSX Scripts

XML 파일 쌍을 Excel 형식으로 변환하는 스크립트 모음입니다.

## 스크립트 목록

### 1. `xml_to_tsv_converter.py`
**목적**: 구병렬(단어/구 단위) Excel 파일 생성
- **컬럼**: 문장식별자, 구식별자, 원문, 번역문
- **파일명**: `{책이름}_구병렬.xlsx`
- **출력**: 318,086개 행 (43개 파일)

**실행 방법**:
```bash
docker-compose run --rm csp python xlsx_scripts/xml_to_tsv_converter.py
```

---

### 2. `xml_to_sentence_parallel.py`
**목적**: 문장병렬(문장 단위) Excel 파일 생성
- **컬럼**: 문단식별자, 문장식별자, 원문, 번역문
- **파일명**: `{책이름}_문장병렬.xlsx`
- **출력**: 87,269개 행 (43개 파일)
- **특징**: ID:W1_T를 ID:W1로 정규화하여 원문/번역문 매칭

**실행 방법**:
```bash
docker-compose run --rm csp python xlsx_scripts/xml_to_sentence_parallel.py
```

---

### 3. `renumber_excel_indices.py`
**목적**: 문단식별자를 식별자 값 변화에 따라 누적 번호로 재설정
- **작동**: 같은 식별자(ID:W1, ID:W10 등)끼리 같은 번호, 바뀌면 다음 번호
- **입력**: 모든 `*_문장병렬.xlsx` 파일
- **출력**: 문단식별자 재설정된 파일들

**실행 방법**:
```bash
docker-compose run --rm csp python xlsx_scripts/renumber_excel_indices.py
```

**예시**:
```
ID:W10 → 1번 문단
ID:W10 → 1번 문단
ID:W1  → 2번 문단  (식별자 변화)
ID:W1  → 2번 문단
```

---

### 4. `log_nan_values.py`
**목적**: 모든 문장병렬 파일에서 NaN 값을 찾아 로그 기록
- **출력 파일**: `/workspace/nan_log.txt`
- **로그 내용**: NaN 발견 부분의 직전행, 현재행(NaN), 직후행 최소 3행
- **특징**: NaN 값 자체는 수정하지 않음 (보존)

**실행 방법**:
```bash
docker-compose run --rm csp python xlsx_scripts/log_nan_values.py
```

**로그 예시**:
```
파일: 당송팔대가문초구양수1_문장병렬.xlsx
▶ NaN 발견 (행 302):
  행301: 문단=37, 문장=302, 원문=..., 번역문=...
  행302: 문단=37, 문장=303, 원문=..., 번역문=[NaN] ← NaN
  행303: 문단=37, 문장=304, 원문=..., 번역문=...
```

---

### 5. `create_paragraph_parallel.py`
**목적**: 문장병렬 파일을 문단병렬로 변환
- **컬럼**: 문단식별자, 원문, 번역문
- **파일명**: `{책이름}_문단병렬.xlsx`
- **작동**: 같은 문단식별자끼리 문장들을 공백으로 연결
- **NaN 처리**: NaN 값을 제외하고 연결하되, 모두 NaN이면 NaN 보존

**실행 방법**:
```bash
docker-compose run --rm csp python xlsx_scripts/create_paragraph_parallel.py
```

**출력 예시**:
```
✓ 당송팔대가문초구양수1_문단병렬.xlsx - 404개 문단
✓ 당송팔대가문초구양수2_문단병렬.xlsx - 572개 문단
...
```

---

## 실행 순서

### 일반 서종 (당송팔대가문초 등)

1. **구병렬 생성**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/xml_to_tsv_converter.py
   ```

2. **문장병렬 생성**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/xml_to_sentence_parallel.py
   ```

### 특수 서종 (예기집설대전1,2 + 당시삼백수1~3)

3. **문장병렬 생성 (단락 ID 기반)**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/extract_yeogi.py
   ```

### 공통 (일반 + 특수)

4. **문단식별자 누적 번호 매기기**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/renumber_excel_indices.py
   ```

5. **NaN 값 로그 기록** (선택사항):
   ```bash
   docker-compose run --rm csp python xlsx_scripts/log_nan_values.py
   ```

6. **문단병렬 생성 (일반)**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/create_paragraph_parallel.py
   ```

7. **문단병렬 생성 (특수)**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/create_yeogi_paragraph.py
   ```

8. **구병렬에 문단식별자 추가**:
   ```bash
   docker-compose run --rm csp python xlsx_scripts/add_paragraph_id_to_gubyeollyeol.py
   ```

---

## 출력 디렉토리 구조

```
/workspace/tsv_output/
├── 당송팔대가문초구양수1/
│   ├── 당송팔대가문초구양수1_구병렬.xlsx
│   ├── 당송팔대가문초구양수1_문장병렬.xlsx
│   └── 당송팔대가문초구양수1_문단병렬.xlsx
├── 당송팔대가문초구양수2/
│   └── ...
└── ... (총 43개 책)
```

---

## 데이터 현황

- **원본 파일**: 43개 XML 쌍 (원문/번역문)
- **구병렬**: 318,086개 행
- **문장병렬**: 87,269개 행
- **문단병렬**: 약 21,000개 이상 문단
- **NaN 값**: 28개 (로그 기록됨)

---

## 경로 설정

모든 스크립트는 **절대경로** (`/workspace/...`)를 사용하므로, 디렉토리 위치와 관계없이 정상 작동합니다.

**도커 환경에서 실행**:
```bash
cd CSP
docker-compose run --rm csp python xlsx_scripts/xml_to_sentence_parallel.py
```

---

## 주의사항

- 모든 스크립트는 **도커 환경**에서 실행됩니다.
- NaN 값이 있는 행도 파일에 유지되므로, 데이터 검증이 필요합니다 (nan_log.txt 참조).
- 문단식별자 번호 매기기는 **식별자 값이 변할 때마다** 새 번호가 시작됩니다.
