# Windows Poetry 환경에서 MeCab 사전 빌드 간소화 매뉴얼

## 개요
Poetry 가상환경에서 직접 MeCab 사전을 빌드하는 간소화된 방법입니다.

## 1. 환경 준비

### 1.1 Poetry 환경 활성화
```cmd
cd C:\Users\junto\Downloads\head-repo\CSP
poetry shell
```

### 1.2 MeCab 설치 확인
```cmd
poetry show | findstr mecab
pip list | findstr mecab
```

### 1.3 MeCab 패키지 확인 및 설치
```cmd
# 기존 mecab 패키지 확인
poetry show | findstr mecab

# ⚠️ 중요: mecab-python3과 mecab-ko는 충돌 가능
# 한국어 프로젝트에서는 mecab-ko만 사용 권장

# 필요시 정확한 패키지만 설치
poetry add mecab-ko mecab-ko-dic

# ❌ 설치하지 말 것:
# poetry add mecab-python3  # 일반 MeCab (일본어 기본)
```

## 2. 사전 파일 경로 확인

### 2.1 MeCab 설치 경로 찾기
```cmd
python -c "import mecab_ko_dic; print(mecab_ko_dic.dicdir)"
```
또는
```cmd
dir /s C:\Users\junto\Downloads\head-repo\CSP\.venv\Lib\site-packages\mecab_ko_dic\dicdir
```

### 2.2 필수 파일 확인
```cmd
cd C:\Users\junto\Downloads\head-repo\CSP\.venv\Lib\site-packages\mecab_ko_dic\dicdir
dir *.def
dir dicrc
dir matrix.def
```

## 3. CSV 파일 준비

### 3.1 작업 디렉토리 설정
```cmd
cd C:\Users\junto\Downloads\head-repo\CSP\mecab
```

### 3.2 CSV 형식 변환 (PowerShell)
```powershell
# 기존 CSV를 MeCab 형식으로 변환
Get-Content user_handic_jongseong.csv | Select-Object -Skip 1 | ForEach-Object {
    $fields = $_ -split ','
    $surface = $fields[0]
    $pos = $fields[4]
    $semantic = $fields[5]
    $ending = $fields[6]
    $reading = $fields[7]
    
    "$surface,1780,3533,2639,$pos,$semantic,$ending,$reading,*,*,*,*,*"
} | Out-File -Encoding UTF8 user_handic_final.csv

# 파일 확인
Get-Content user_handic_final.csv | Select-Object -First 5
```

## 4. 사전 컴파일 (간소화 방법)

### 4.1 mecab-dict-index 경로 확인
```cmd
where mecab-dict-index
```
또는
```cmd
C:\Users\junto\Downloads\head-repo\CSP\.venv\Scripts\mecab-dict-index --help
```

### 4.2 사전 컴파일 실행
```cmd
cd C:\Users\junto\Downloads\head-repo\CSP\mecab

# 단일 사전 컴파일
C:\Users\junto\Downloads\head-repo\CSP\.venv\Scripts\mecab-dict-index ^
    -d C:\Users\junto\Downloads\head-repo\CSP\.venv\Lib\site-packages\mecab_ko_dic\dicdir ^
    -u user_handic.dic ^
    -f utf-8 ^
    -t utf-8 ^
    user_handic_final.csv
```

### 4.3 다중 파일 처리
```cmd
# 두 번째 파일도 변환
powershell -Command "Get-Content stdict_hanja_jongseong.csv | Select-Object -Skip 1 | ForEach-Object { $fields = $_ -split ','; \"$($fields[0]),1780,3533,2639,$($fields[4]),$($fields[5]),$($fields[6]),$($fields[7]),*,*,*,*,*\" } | Out-File -Encoding UTF8 stdict_hanja_final.csv"

# 파일 합치기
copy user_handic_final.csv + stdict_hanja_final.csv combined_dict.csv

# 합친 사전 컴파일
C:\Users\junto\Downloads\head-repo\CSP\.venv\Scripts\mecab-dict-index ^
    -d C:\Users\junto\Downloads\head-repo\CSP\.venv\Lib\site-packages\mecab_ko_dic\dicdir ^
    -u combined_dict.dic ^
    -f utf-8 ^
    -t utf-8 ^
    combined_dict.csv
```

## 5. 컴파일 스크립트 자동화

### 5.1 배치 스크립트 생성
```cmd
echo off > build_mecab_dict.bat
echo cd C:\Users\junto\Downloads\head-repo\CSP\mecab >> build_mecab_dict.bat
echo. >> build_mecab_dict.bat
echo REM Poetry 환경 활성화 >> build_mecab_dict.bat
echo call C:\Users\junto\Downloads\head-repo\CSP\.venv\Scripts\activate.bat >> build_mecab_dict.bat
echo. >> build_mecab_dict.bat
echo REM 사전 컴파일 >> build_mecab_dict.bat
echo C:\Users\junto\Downloads\head-repo\CSP\.venv\Scripts\mecab-dict-index ^^ >> build_mecab_dict.bat
echo     -d C:\Users\junto\Downloads\head-repo\CSP\.venv\Lib\site-packages\mecab_ko_dic\dicdir ^^ >> build_mecab_dict.bat
echo     -u combined_dict.dic ^^ >> build_mecab_dict.bat
echo     -f utf-8 ^^ >> build_mecab_dict.bat
echo     -t utf-8 ^^ >> build_mecab_dict.bat
echo     combined_dict.csv >> build_mecab_dict.bat
echo. >> build_mecab_dict.bat
echo echo 사전 컴파일 완료! >> build_mecab_dict.bat
echo pause >> build_mecab_dict.bat
```

## 6. 테스트 및 검증

### 6.1 사전 파일 확인
```cmd
dir *.dic
echo. & echo 파일 크기:
for %f in (*.dic) do echo %f: %~zf bytes
```

### 6.2 Python에서 테스트
```cmd
python -c "
import MeCab
tagger = MeCab.Tagger('-u combined_dict.dic')
print('테스트 결과:')
print(tagger.parse('역가거'))
print(tagger.parse('동자'))
"
```

## 7. 문제 해결

### 7.1 경로 문제
```cmd
# MeCab 설치 경로 재확인
python -c "
import site
print('Site-packages:', site.getsitepackages())
import mecab_ko_dic
print('MeCab dicdir:', mecab_ko_dic.dicdir)
"
```

### 7.2 인코딩 문제
```cmd
# UTF-8 BOM 제거
powershell -Command "
$content = Get-Content combined_dict.csv -Raw
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText('combined_dict_nobom.csv', $content, $utf8NoBom)
"
```

### 7.3 프로세스 충돌
```cmd
# Python 프로세스 종료
taskkill /F /IM python.exe
taskkill /F /IM mecab.exe
```

## 8. 자동화 Python 스크립트

### 8.1 통합 빌드 스크립트
```python
import os
import subprocess
import sys
from pathlib import Path

def build_mecab_dict():
    """MeCab 사전 빌드 자동화"""
    
    # 경로 설정
    project_root = Path("C:/Users/junto/Downloads/head-repo/CSP")
    venv_path = project_root / ".venv"
    mecab_dir = project_root / "mecab"
    dicdir = venv_path / "Lib/site-packages/mecab_ko_dic/dicdir"
    mecab_dict_index = venv_path / "Scripts/mecab-dict-index.exe"
    
    # 작업 디렉토리로 이동
    os.chdir(mecab_dir)
    
    # CSV 파일 변환 및 합치기
    print("CSV 파일 변환 중...")
    # (여기에 CSV 변환 로직 추가)
    
    # 사전 컴파일
    cmd = [
        str(mecab_dict_index),
        "-d", str(dicdir),
        "-u", "combined_dict.dic",
        "-f", "utf-8",
        "-t", "utf-8",
        "combined_dict.csv"
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 사전 컴파일 성공!")
        print(f"생성된 파일: {mecab_dir}/combined_dict.dic")
    else:
        print("❌ 사전 컴파일 실패:")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    build_mecab_dict()
```

## 9. 주요 차이점 요약

### WSL vs Windows 방식:
- **WSL**: 소스 컴파일 → 복잡하지만 완전한 제어
- **Windows**: Poetry 패키지 사용 → 간단하지만 패키지 의존성

### 권장 방법:
1. **개발/테스트**: Windows Poetry 방식 (빠르고 간단)
2. **프로덕션**: WSL 방식 (완전한 제어와 커스터마이징)

현재 개발 환경에서는 **Windows Poetry 방식**이 더 효율적입니다.
