# WSL Ubuntu에서 MeCab 한국어 사전 빌드 매뉴얼

## 개요
본 매뉴얼은 WSL Ubuntu 환경에서 MeCab 한국어 사전을 빌드하는 전체 과정을 상세히 설명합니다. CSV 형태의 사용자 사전 데이터를 MeCab 바이너리 사전 파일(.dic)로 컴파일하는 과정을 포함합니다.

## 1. 환경 준비

### 1.1 WSL Ubuntu 업데이트
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 1.2 필수 패키지 설치
```bash
# 컴파일 도구 설치
sudo apt-get install -y build-essential autoconf automake libtool

# MeCab 소스 컴파일을 위한 추가 도구
sudo apt-get install -y wget curl tar gzip
```

## 2. MeCab 소스 다운로드 및 컴파일

### 2.1 MeCab 소스 다운로드
```bash
cd /tmp
curl -L https://github.com/taku910/mecab/archive/master.tar.gz -o mecab-master.tar.gz
tar -xzf mecab-master.tar.gz
cd mecab-master/mecab
```

### 2.2 MeCab 컴파일 및 설치
```bash
# configure 실행
./configure --prefix=/usr/local

# 컴파일 (멀티코어 사용)
make -j$(nproc)

# 시스템에 설치
sudo make install

# 라이브러리 캐시 업데이트
sudo ldconfig
```

### 2.3 MeCab 설치 확인
```bash
# MeCab 버전 확인
/usr/local/bin/mecab --version

# mecab-dict-index 위치 확인
find /usr/local -name "mecab-dict-index"
# 결과: /usr/local/libexec/mecab/mecab-dict-index
```

## 3. MeCab-ko 사전 다운로드 및 설정

### 3.1 MeCab-ko 사전 다운로드
```bash
cd /tmp
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar -xzf mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
```

### 3.2 MeCab-ko 사전 설정
```bash
# MeCab-ko 사전 configure
./configure --with-mecab-config=/usr/local/bin/mecab-config
```

## 4. CSV 데이터 준비

### 4.1 CSV 파일 형식 요구사항
MeCab-ko 사전의 CSV 형식은 다음과 같습니다:
```
표층형,Left-ID,Right-ID,Cost,품사태그,의미부류,종성유무,읽기,*,*,*,*,*
```

**예시:**
```csv
亦可居,1780,3533,2639,NNP,*,F,역가거,*,*,*,*,*
桐子,1780,3533,2639,NNG,*,F,동자,*,*,*,*,*
```

### 4.2 원본 CSV를 MeCab 형식으로 변환

#### 원본 형식 (헤더 포함):
```csv
표층형,품사,품사세분류1,품사세분류2,품사세분류3,활용형,원형,읽기,발음
亦可居,NNP,*,*,*,*,亦可居,역가거,역가거
```

#### 변환 스크립트:
```bash
# 작업 디렉토리 생성
mkdir -p /tmp/mecab_build
cd /tmp/mecab_build

# CSV 파일 복사 (Windows 파일 시스템에서)
cp /mnt/c/Users/[사용자명]/[경로]/user_handic_jongseong.csv .

# MeCab 형식으로 변환
tail -n +2 user_handic_jongseong.csv | awk -F',' '{
    surface = $1    # 표층형
    pos = $5        # 품사
    semantic = $6   # 의미부류
    ending = $7     # 종성유무
    reading = $8    # 읽기
    
    print surface ",1780,3533,2639," pos "," semantic "," ending "," reading ",*,*,*,*,*"
}' > user_handic_final.csv

# 결과 확인
head -5 user_handic_final.csv
wc -l user_handic_final.csv
```

## 5. 사전 컴파일 환경 구성

### 5.1 필수 정의 파일 복사
```bash
# MeCab-ko 사전의 필수 파일들을 작업 디렉토리로 복사
cp /tmp/mecab-ko-dic-2.1.1-20180720/char.def .
cp /tmp/mecab-ko-dic-2.1.1-20180720/feature.def .
cp /tmp/mecab-ko-dic-2.1.1-20180720/left-id.def .
cp /tmp/mecab-ko-dic-2.1.1-20180720/right-id.def .
cp /tmp/mecab-ko-dic-2.1.1-20180720/pos-id.def .
cp /tmp/mecab-ko-dic-2.1.1-20180720/matrix.def .
cp /tmp/mecab-ko-dic-2.1.1-20180720/dicrc .
cp /tmp/mecab-ko-dic-2.1.1-20180720/rewrite.def .

# 파일 복사 확인
ls -la *.def *.csv dicrc
```

### 5.2 필요한 파일 목록
- `char.def`: 문자 정의
- `feature.def`: 품사 특성 정의
- `left-id.def`: 왼쪽 연결 ID 정의
- `right-id.def`: 오른쪽 연결 ID 정의
- `pos-id.def`: 품사 ID 정의
- `matrix.def`: 연결 비용 매트릭스
- `dicrc`: 사전 설정 파일
- `rewrite.def`: 재작성 규칙

## 6. 사전 컴파일 실행

### 6.1 단일 사전 컴파일
```bash
# 첫 번째 사전 컴파일
/usr/local/libexec/mecab/mecab-dict-index \
    -d . \
    -u user_handic.dic \
    -f utf-8 \
    -t utf-8 \
    user_handic_final.csv

# 컴파일 성공 확인
ls -la *.dic
```

### 6.2 다중 사전 파일 처리

#### 두 번째 사전 파일 준비:
```bash
# 두 번째 CSV 파일 변환
tail -n +2 stdict_hanja_jongseong.csv | awk -F',' '{
    surface = $1
    pos = $5
    semantic = $6
    ending = $7
    reading = $8
    
    print surface ",1780,3533,2639," pos "," semantic "," ending "," reading ",*,*,*,*,*"
}' > stdict_hanja_final.csv

# CSV 정리 (빈 줄, 특수문자 제거)
awk -F',' '
    NF == 13 && $1 != "" && $1 !~ /^"/ && $1 !~ /"$/ {
        print $0
    }
' stdict_hanja_final.csv > stdict_hanja_cleaned.csv

# 두 번째 사전 컴파일
/usr/local/libexec/mecab/mecab-dict-index \
    -d . \
    -u stdict_hanja.dic \
    -f utf-8 \
    -t utf-8 \
    stdict_hanja_cleaned.csv
```

### 6.3 사전 파일 합치기 및 컴파일
```bash
# 두 파일 합치기
cat user_handic_final.csv stdict_hanja_cleaned.csv > combined_dict.csv

# 합친 사전 컴파일
/usr/local/libexec/mecab/mecab-dict-index \
    -d . \
    -u combined_dict.dic \
    -f utf-8 \
    -t utf-8 \
    combined_dict.csv

# 최종 결과 확인
ls -la *.dic
wc -l *.csv
```

## 7. 결과 파일 관리

### 7.1 생성된 파일들
```bash
# 컴파일된 사전 파일들
user_handic.dic          # 첫 번째 사전 (415,576 항목)
stdict_hanja.dic         # 두 번째 사전 (370,679 항목)
combined_dict.dic        # 합친 사전 (786,255 항목)

# CSV 소스 파일들
user_handic_final.csv    # 첫 번째 사전 소스
stdict_hanja_cleaned.csv # 두 번째 사전 소스
combined_dict.csv        # 합친 사전 소스
```

### 7.2 Windows로 파일 복사
```bash
# 컴파일된 사전 파일들을 Windows로 복사
cp *.dic /mnt/c/Users/[사용자명]/[프로젝트경로]/
cp *_final.csv /mnt/c/Users/[사용자명]/[프로젝트경로]/
cp *_cleaned.csv /mnt/c/Users/[사용자명]/[프로젝트경로]/
cp combined_dict.csv /mnt/c/Users/[사용자명]/[프로젝트경로]/

# 복사 확인
ls -la /mnt/c/Users/[사용자명]/[프로젝트경로]/*.dic
```

## 8. 문제 해결

### 8.1 일반적인 오류 및 해결방법

#### "no such file or directory: ./dicrc"
```bash
# 해결: dicrc 파일을 작업 디렉토리로 복사
cp /tmp/mecab-ko-dic-2.1.1-20180720/dicrc .
```

#### "no such file or directory: ./rewrite.def"
```bash
# 해결: rewrite.def 파일을 작업 디렉토리로 복사
cp /tmp/mecab-ko-dic-2.1.1-20180720/rewrite.def .
```

#### "empty word is found, discard this line"
```bash
# 해결: CSV 파일에서 빈 줄 제거
awk -F',' 'NF == 13 && $1 != "" { print $0 }' input.csv > output.csv
```

#### "format error" with quoted strings
```bash
# 해결: 쌍따옴표가 포함된 줄 제거
awk -F',' '$1 !~ /^"/ && $1 !~ /"$/ { print $0 }' input.csv > output.csv
```

### 8.2 CSV 형식 검증
```bash
# CSV 형식 확인 스크립트
awk -F',' '{
    if (NF != 13) {
        print "Line " NR ": Wrong field count (" NF " instead of 13)"
        print $0
    }
    if ($1 == "") {
        print "Line " NR ": Empty surface form"
    }
}' your_file.csv
```

## 9. 성능 최적화

### 9.1 컴파일 시간 단축
```bash
# CPU 코어 수 확인
nproc

# 멀티코어를 활용한 컴파일 (MeCab 자체 컴파일 시)
make -j$(nproc)
```

### 9.2 메모리 사용량 모니터링
```bash
# 컴파일 중 메모리 사용량 확인
free -h
htop  # 실시간 모니터링
```

## 10. 검증 및 테스트

### 10.1 사전 파일 무결성 확인
```bash
# 파일 크기 확인
ls -lh *.dic

# 파일 타입 확인
file *.dic
```

### 10.2 Poetry 환경에서 사전 사용 (Windows)
```bash
# Windows PowerShell에서
cd C:\Users\[사용자명]\[프로젝트경로]
poetry shell
python -c "
import MeCab
mecab = MeCab.Tagger('-u user_handic.dic')
print(mecab.parse('역가거'))
"
```

## 부록: 참고 정보

### A. MeCab 명령어 옵션
```
-d DIR    : 사전 디렉토리 지정
-u FILE   : 사용자 사전 파일 지정
-f FORMAT : 입력 문자 인코딩
-t FORMAT : 출력 문자 인코딩
```

### B. CSV 필드 상세 설명
1. **표층형**: 실제 나타나는 단어 형태
2. **Left-ID**: 왼쪽 연결 컨텍스트 ID (보통 1780)
3. **Right-ID**: 오른쪽 연결 컨텍스트 ID (보통 3533)
4. **Cost**: 비용 점수 (보통 2639)
5. **품사태그**: NNG, NNP, VV 등
6. **의미부류**: 의미 분류 (* = 없음)
7. **종성유무**: T(받침있음), F(받침없음)
8. **읽기**: 발음 정보
9-13. **고정값**: 모두 * 또는 빈값

### C. 프로젝트 디렉토리 구조
```
project/
├── user_handic.dic              # 컴파일된 사전 1
├── stdict_hanja.dic             # 컴파일된 사전 2
├── combined_dict.dic            # 합친 사전
├── user_handic_final.csv        # 소스 CSV 1
├── stdict_hanja_cleaned.csv     # 소스 CSV 2
└── combined_dict.csv            # 합친 소스 CSV
```

---

**작성일**: 2025년 8월 16일  
**테스트 환경**: WSL Ubuntu 24.04, MeCab 0.996  
**총 처리 항목 수**: 786,255개 사전 항목
