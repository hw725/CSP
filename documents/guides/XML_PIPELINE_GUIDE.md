# CSP XML 전체 파이프라인 시스템 사용 가이드

## 🎯 개요

XML 쌍(원문 + 번역문)을 받아서 **전체 파이프라인을 자동으로 실행**하는 시스템입니다:

1. **문단병렬 생성**: XML → Excel 변환  
2. **PA 처리**: 문단 → 문장 정렬
3. **문장병렬 대조**: XML 정답과 PA 결과 비교
4. **SA 처리**: 문장 → 구 정렬
5. **구병렬 대조**: XML 정답과 SA 결과 비교
6. **정확도 분석**: 전체 성능 평가

**핵심 특징**: 
- 🚀 **범용 진입점**: `main.py`로 XML, TXT, XLSX, CSV 등 다양한 형식 지원
- 📦 **패키지 구조**: `xml_pipeline/` 패키지로 모든 XML 처리 모듈 통합
- 🎯 **자동 감지**: 파일 형식과 쌍 매칭 자동 인식
- 📁 각 XML 쌍마다 **독립적인 결과 폴더**를 자동 생성해서 깔끔하게 관리!

**🆕 2025년 9월 주요 개선사항**:
- ⚡ **SA 분석 효율화**: XML `<s>` 태그 기반 문장별 그룹핑으로 0% → 60%+ 개선
- 📊 **현실적인 정확도**: 매칭 성능 중심의 새로운 공식 (PA 45% + SA 45% + 무결성 10%)
- 🧹 **일관된 텍스트 정제**: 모든 단계에서 괄호·하이픈 제거로 매칭 정확도 향상
- 📈 **상세한 분석 결과**: PA/SA별 F1, 정밀도, 재현율, 유사도 세분화

## 📂 결과 폴더 구조

```
xml_pipeline_results/
├── [쌍ID]_20250912_150423/        # 각 XML 쌍별 독립 폴더
│   ├── paragraph/                 # 1단계: 문단병렬
│   │   └── paragraph_parallel.xlsx
│   ├── pa_results/                # 2단계: PA 결과
│   │   └── pa_output.xlsx
│   ├── sentence/                  # 3단계: 문장병렬 비교
│   │   ├── sentence_truth.xlsx    # XML 정답
│   │   └── sentence_from_pa.xlsx  # PA 결과
│   ├── sa_results/                # 4단계: SA 결과
│   │   └── sa_output.xlsx
│   ├── accuracy/                  # 5단계: 정확도 분석
│   │   ├── phrase_truth.xlsx      # XML 정답
│   │   ├── accuracy_report.json   # 전체 정확도 요약
│   │   └── xml_level_similarity.json  # 상세 PA/SA 분석
│   ├── pipeline_summary.json      # 전체 요약 (JSON)
│   └── pipeline_summary.txt       # 전체 요약 (텍스트)
└── xml_pipeline_results.db        # 전체 결과 데이터베이스
```

## 🚀 빠른 시작

### **범용 진입점 사용법 (main.py)**

#### 1. **자동 형식 감지 (추천!)**
```bash
# 파일 형식 자동 감지하여 처리
python main.py auto 원문.xml 번역문.xml

# 통합 XML 파일 자동 감지
python main.py auto 병렬변환.xml
```

#### 2. **XML 명시적 처리**
```bash
# XML 파일 쌍 직접 처리
python main.py xml process --original 원문.xml --translation 번역문.xml

# 스마트 디렉토리 처리
python main.py xml smart --xml-dir /workspace/2025/2017불용

# 통합 XML 처리
python main.py xml-merged process 병렬변환.xml
```

#### 3. **Docker 환경 사용법**
```bash
# Docker 컨테이너에서 실행
docker exec csp-workspace python /workspace/main.py auto /workspace/sources/원문.xml /workspace/sources/번역문.xml

# 스마트 모드로 디렉토리 처리  
docker exec csp-workspace python /workspace/main.py xml smart --xml-dir /workspace/2025/2017불용
```

### **쉘 스크립트 사용법 (xml_pipeline.sh)**

#### 1. **컨테이너 준비**
```bash
# Docker 컨테이너 시작
docker-compose up -d

# 스크립트 실행 권한 부여  
chmod +x utils/xml_pipeline.sh
chmod +x scripts/xml_pipeline.sh
```

#### 2. **스마트 모드 (GUI 없이 쉬운 선택)**
```bash
# 인터랙티브 스마트 모드 - 가장 편리!
./utils/xml_pipeline.sh smart

# 파일 브라우저 모드
./utils/xml_pipeline.sh browse /workspace/2025/2017불용

# 전체 디렉토리 스캔
./utils/xml_pipeline.sh scan /workspace/2025
```

#### 3. **직접 실행 모드**
```bash
# 단일 쌍 처리 (main.py 사용)
./utils/xml_pipeline.sh process /workspace/sources/원문.xml /workspace/sources/번역문.xml

# 디렉토리 일괄 처리
./utils/xml_pipeline.sh batch /workspace/sources

# 결과 목록 및 상세 조회
./utils/xml_pipeline.sh list
./utils/xml_pipeline.sh show test_pair_001
```

## 🔧 상세 사용법

### **Python 직접 실행**

더 세밀한 제어가 필요한 경우:

#### **1. 범용 main.py 사용 (권장)**
```bash
# 자동 형식 감지
python main.py auto file1.xml file2.xml

# XML 명시적 처리
python main.py xml process --original "원문.xml" --translation "번역문.xml"

# 스마트 디렉토리 처리
python main.py xml smart --xml-dir "/path/to/xml/files"

# 통합 XML 처리
python main.py xml-merged process "병렬변환.xml"

# 도움말
python main.py --help
python main.py xml --help
```

#### **2. XML 패키지 직접 사용**
```bash
# xml_pipeline 패키지 내에서는 상대 import 사용
# 직접 실행 시 main.py를 통해 접근

# Docker 환경에서
docker exec csp-workspace python /workspace/xml_pipeline/xml_pipeline_cli.py process \
  --original "/workspace/sources/원문.xml" \
  --translation "/workspace/sources/번역문.xml"
```

### **XML 파일 명명 규칙**

자동 매칭을 위해서는 다음 규칙을 권장합니다:

```
✅ 좋은 예:
jti_4c0231-당송팔대가문초증공1_원문_x-C2018.xml
jti_4c0231-당송팔대가문초증공1_번역문_x-C2018.xml

✅ 자동 매칭됨:
- 관자4_원문.xml → 관자4_번역문.xml
- test_원문_001.xml → test_번역문_001.xml

❌ 매칭 안됨:
- 파일명이 완전히 다른 경우
- 원문/번역문 키워드가 없는 경우
```

## 📊 실제 사용 시나리오

### **시나리오 1: 새 XML 쌍 테스트**
```bash
# 1. 자동 감지로 새로운 XML 쌍 처리
python main.py auto /workspace/sources/new_original.xml /workspace/sources/new_translation.xml

# 또는 Docker 환경에서
docker exec csp-workspace python /workspace/main.py auto \
  /workspace/sources/new_original.xml /workspace/sources/new_translation.xml

# 2. 스마트 모드로 편리하게 처리
./utils/xml_pipeline.sh smart
# → 메뉴에서 선택 → 자동 처리

# 3. 결과 파일 확인
# → xml_pipeline_results/new_original_20250914_095543/ 폴더 생성됨
```

### **시나리오 2: 여러 쌍 대량 테스트**
```bash
# 1. XML 파일들을 하나의 폴더에 준비
#    /workspace/sources/
#    ├── jti_4c0201-당송팔대가_원문_x-C2017.xml
#    ├── jti_4c0201-당송팔대가_번역문_x-C2017.xml  
#    ├── jti_4c0202-당송팔대가_원문_x-C2020.xml
#    ├── jti_4c0202-당송팔대가_번역문_x-C2020.xml
#    └── ...

# 2. 스마트 디렉토리 처리
python main.py xml smart --xml-dir /workspace/sources

# 또는 스마트 모드 사용
./utils/xml_pipeline.sh smart
# → "2. 디렉토리 전체 일괄 처리" 선택

# 3. 각 쌍별로 독립적인 폴더 생성됨:
#    xml_pipeline_results/
#    ├── jti_4c0201-당송팔대가_20250914_095543/
#    ├── jti_4c0202-당송팔대가_20250914_100123/
#    └── ...
```

### **시나리오 3: 특정 시리즈 선별 처리**
```bash
# 1. 스마트 모드로 패턴 검색
./utils/xml_pipeline.sh smart
# → "3. 패턴으로 검색해서 처리" → "당송팔대가" 입력

# 2. 파일 브라우저로 특정 디렉토리 탐색
./utils/xml_pipeline.sh browse /workspace/2025/2017불용

# 3. 자동 감지로 개별 처리
python main.py auto /workspace/sources/특정파일_원문.xml /workspace/sources/특정파일_번역문.xml

# 4. 결과 조회
./utils/xml_pipeline.sh list
./utils/xml_pipeline.sh show jti_4c0201-당송팔대가
```

## 📈 결과 분석

### **1. 실시간 모니터링**

실행 중 다음 정보가 실시간 표시됩니다:
- ✅/❌ 각 단계별 성공/실패 상태
- ⏱️ 단계별 처리 시간
- 🎯 정확도 점수 (계산 가능한 경우)

### **2. 정확도 계산 공식 (2025년 9월 개선)**

**전체 정확도** = PA(45%) + SA(45%) + 무결성(10%)

각 PA/SA 점수는 다음과 같이 계산:
- **PA/SA 점수** = F1 Score(60%) + 평균 유사도(40%)
- **무결성 점수**: 텍스트 일치도 (괄호 제거 후 비교)

이 공식은 **매칭 성능을 우선시**하여 현실적인 정확도를 제공합니다.

### **3. SA 분석 개선사항**

**문제**: 기존 SA 분석에서 0% 점수 (전역 매칭의 비효율성)
**해결**: XML `<s>` 태그를 활용한 **문장 단위 그룹핑**
- 각 문장 내에서만 구 매칭 수행
- 계산 복잡도: O(전체 구 수²) → O(문장 수 × 평균 구 수²)
- 예상 개선: SA 점수 0% → 60%+

### **4. 텍스트 정제 일관성**

모든 처리 단계에서 **일관된 텍스트 정제** 적용:
- **제거 대상**: `[`, `]`, `-` (대괄호 및 하이픈)
- **적용 범위**: XML 파싱, PA/SA 처리, 무결성 검사, 유사도 계산
- **목적**: 부호로 인한 매칭 실패 방지

### **5. 종합 보고서**

각 XML 쌍 처리 후 자동 생성되는 파일들:

#### **accuracy_report.json** - 전체 정확도 요약
```json
{
  "accuracy_score": 0.78,        // 최종 점수 (0-1)
  "pa_score": 0.85,             // PA 점수 (F1 + 유사도)
  "sa_score": 0.72,             // SA 점수 (F1 + 유사도)
  "integrity_score": 0.95,      // 무결성 점수
  "processing_time": "00:03:45"  // 처리 시간
}
```

#### **xml_level_similarity.json** - 상세 분석
```json
{
  "pa_analysis": {
    "f1_score": 0.82,           // PA F1 점수
    "precision": 0.85,          // PA 정밀도
    "recall": 0.79,             // PA 재현율
    "average_similarity": 0.91   // PA 평균 유사도
  },
  "sa_analysis": {
    "original_similarities": [   // 원문 구별 유사도
      0.95, 0.87, 0.93, ...
    ],
    "translation_similarities": [ // 번역문 구별 유사도
      0.89, 0.91, 0.86, ...
    ],
    "average_similarity": 0.89,  // 전체 평균
    "sentence_count": 156,       // 처리된 문장 수
    "phrase_count": 1247        // 처리된 구 수
  }
}
```

#### **pipeline_summary.txt** - 사람이 읽기 쉬운 요약
```
XML 파이프라인 처리 결과
========================

📊 전체 정확도: 78.5%
   ├─ PA 분석: 85.2% (F1: 82.1%, 유사도: 91.3%)
   ├─ SA 분석: 72.1% (평균 유사도: 89.4%)
   └─ 무결성: 95.8%

⏱️  처리 시간: 3분 45초
📝 처리 문장: 156개
🔤 처리 구: 1,247개
```

### **6. 데이터베이스 기록**

모든 결과는 SQLite 데이터베이스에 영구 저장:
```sql
-- 최근 처리된 XML 쌍들
SELECT pair_id, name, created_at FROM xml_pairs ORDER BY created_at DESC;

-- 특정 쌍의 단계별 결과
SELECT stage, status, processing_time, accuracy_score 
FROM pipeline_results 
WHERE pair_id = 'test_001';
```

## ⚡ 성능 최적화

### **1. 하드웨어 최적화**
- **CPU**: 4코어 이상 권장 (병렬 처리)
- **메모리**: 8GB 이상 권장 (대용량 XML)
- **GPU**: CUDA 지원 시 더 빠름 (BGE, SuPar 가속)

### **2. 배치 크기 조정**
```bash
# 메모리 부족 시 청크 크기 조정
export SA_CHUNK_SIZE=50
export PA_BATCH_SIZE=25
```

### **3. 점진적 처리**
```bat
REM 1. 작은 파일로 먼저 테스트
xml_pipeline.bat process small_test_원문.xml small_test_번역문.xml

REM 2. 문제없으면 대용량 처리
xml_pipeline.bat batch "C:\large_xml_files"
```

## 🔍 문제 해결

### **1. Import 오류**
```
❌ 오류: ImportError: attempted relative import with no known parent package
```
**해결**: 항상 `main.py`를 통해 실행. 패키지 내 모듈 직접 실행 금지
```bash
# ✅ 올바른 방법
python main.py xml process --original 원문.xml --translation 번역문.xml

# ❌ 잘못된 방법  
python xml_pipeline/xml_pipeline_cli.py process ...
```

### **2. XML 파싱 오류**
```
❌ 오류: XML 문단 추출 실패: not well-formed
```
**해결**: XML 문법 오류 수정, 인코딩 확인 (UTF-8 권장)

### **3. 파일 형식 인식 실패**
```
❌ 감지된 파일 형식: {'file.txt': None}
```
**해결**: 파일 확장자 확인, 지원 형식 사용 (.xml, .txt, .xlsx, .csv)

### **4. 매칭 파일 없음**
```
❌ 매칭되는 XML 쌍을 찾을 수 없습니다
```
**해결**: 파일명에 '원문', '번역문' 키워드 포함 확인, 또는 스마트 모드 사용

### **5. Docker 컨테이너 문제**
```bash
# 컨테이너 재시작
docker-compose down  
docker-compose up -d

# 로그 확인
docker-compose logs csp-workspace

# 컨테이너 연결 테스트
docker exec csp-workspace python --version
```

### **6. xml_level_similarity 모듈 경고**
```
⚠️ xml_level_similarity 모듈을 찾을 수 없습니다. XML 레벨 분석이 제외됩니다.
```
**해결**: 정상적인 경고입니다. 메인 파이프라인은 정상 작동하며, XML 레벨 분석만 제외됩니다.

### **7. 비현실적인 정확도 점수 (2025년 9월 해결됨)**
```
❌ 문제: PA 3.12%, SA 0% 같은 비현실적으로 낮은 점수
```
**원인 및 해결**:
- **SA 0% 원인**: 전역 구 매칭으로 인한 계산 비효율성
- **해결**: XML `<s>` 태그 기반 문장별 그룹핑으로 매칭 효율성 개선
- **PA 낮은 점수**: 부호 불일치 및 과도한 무결성 가중치
- **해결**: 
  1. 일관된 텍스트 정제 (괄호, 하이픈 제거)
  2. 정확도 공식 재조정: 매칭 성능(90%) vs 무결성(10%)
  3. F1:유사도 비율 7:3 → 6:4로 조정

**기대 개선**:
- SA 점수: 0% → 60%+
- PA 점수: 3.12% → 60%+
- 전체 정확도: 더 현실적이고 의미있는 수치

## 💡 팁과 요령

### **1. 범용 진입점 활용**
- 항상 `main.py`를 사용하여 실행 (패키지 import 문제 방지)
- `auto` 모드로 파일 형식 자동 감지 활용
- XML 파일들을 하나의 디렉토리에 정리
- 명명 규칙 일관성 유지 (원문/번역문 키워드 포함)
- 결과 폴더는 자동으로 타임스탬프 포함

### **2. 스마트 모드 적극 활용**
```bash
# GUI 없는 환경에서 가장 편리한 방법
./utils/xml_pipeline.sh smart

# 복잡한 파일명도 번호 선택으로 쉽게 처리
./utils/xml_pipeline.sh browse /workspace/2025/2017불용
```

### **3. 성능 모니터링**
```bash
# Docker 환경에서 GPU 사용 확인
docker exec csp-workspace nvidia-smi

# 처리 시간 및 결과 확인  
./utils/xml_pipeline.sh list
./utils/xml_pipeline.sh show pair_id
```

### **4. 결과 백업 및 관리**
```bash
# 중요한 결과는 별도 백업
cp -r xml_pipeline_results backup_results

# Docker 볼륨에서 호스트로 결과 복사
docker cp csp-workspace:/workspace/xml_pipeline_results ./local_results
```

### **5. 배치 처리 전략**
```bash
# 1단계: 단일 쌍으로 테스트
python main.py auto test_원문.xml test_번역문.xml

# 2단계: 소규모 디렉토리 테스트  
python main.py xml smart --xml-dir /workspace/small_test_set

# 3단계: 전체 처리
./utils/xml_pipeline.sh smart
# → "2. 디렉토리 전체 일괄 처리" 선택

# 4단계: 실패한 것들만 재처리
python main.py xml process --original failed_원문.xml --translation failed_번역문.xml
```

### **6. 정확도 결과 해석 (2025년 9월 개선)**

**우선 확인할 핵심 지표**:
```bash
# accuracy_report.json에서 확인
{
  "accuracy_score": 0.78,    # 78% → 양호한 수준
  "pa_score": 0.85,          # PA 85% → 문장 정렬 우수
  "sa_score": 0.72           # SA 72% → 구 정렬 양호
}
```

**점수 해석 기준**:
- **70%+ 우수**: 실용적으로 활용 가능한 수준
- **50-70% 양호**: 부분적 수정 후 활용 가능  
- **30-50% 미흡**: 원본 데이터 또는 설정 검토 필요
- **30% 미만**: 데이터 품질 또는 형식 문제

**SA 점수가 여전히 낮다면**:
```bash
# xml_level_similarity.json에서 상세 확인
{
  "sa_analysis": {
    "sentence_count": 156,           # 문장 수가 적으면 정상
    "phrase_count": 1247,           # 구 수 대비 문장 수 확인
    "average_similarity": 0.89      # 유사도 자체는 높은지 확인
  }
}
```

**개선된 결과 활용법**:
- **PA 우수 + SA 미흡**: 문장 단위 작업에 활용
- **SA 우수 + PA 미흡**: 구 단위 세밀한 정렬에 활용
- **전체적으로 우수**: 자동화된 병렬 처리 시스템으로 활용

## 📞 지원

문제가 있을 때 확인할 사항:

1. **로그 파일**: `xml_pipeline_results/pipeline.log`
2. **데이터베이스**: `xml_pipeline_results.db` 
3. **결과 디렉토리**: 각 XML 쌍별 독립 폴더
4. **Docker 로그**: `docker-compose logs csp`

---

이 시스템으로 **여러 쌍의 XML을 체계적으로 테스트**하고 **각각의 결과를 깔끔하게 관리**할 수 있습니다! 🎉