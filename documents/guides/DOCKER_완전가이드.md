# 🐳 CSP Docker 완전 가이드

> **Chinese Sentence Processing (CSP) 프로젝트의 Docker 기반 개발환경 완전 정복 가이드**

## 📋 목차
- [🚨 마이그레이션 배경](#-마이그레이션-배경)
- [⚙️ Docker 설치 및 환경 구성](#%EF%B8%8F-docker-설치-및-환경-구성)
- [🏗️ CSP Docker 환경 구조](#%EF%B8%8F-csp-docker-환경-구조)
- [🛡️ 보안 시스템](#%EF%B8%8F-보안-시스템)
- [🚀 일상 운영 가이드](#-일상-운영-가이드)
- [🎯 복잡한 XML 파일명 쉽게 처리하기](#-복잡한-xml-파일명-쉽게-처리하기)
- [🔧 문제 해결](#-문제-해결)

---

## 🚨 마이그레이션 배경

### Poetry 의존성 지옥 문제
- **문제**: Poetry가 OpenAI 설치 시 PyTorch를 CPU 버전으로 강제 변경
- **영향**: CUDA 기능 상실로 GPU 가속 불가능
- **해결**: Docker 기반 완전 격리 환경 구축

### 해결된 핵심 이슈들
1. ✅ **간접 의존성 공격 차단**: Poetry가 torch 버전을 변경하지 못함
2. ✅ **PyTorch CUDA 보존**: 2.6.0+cu124 완벽 유지
3. ✅ **AI 라이브러리 안정성**: 모든 패키지 시스템 레벨 고정
4. ✅ **개발 환경 일관성**: 모든 환경에서 동일한 결과 보장

---

## ⚙️ Docker 설치 및 환경 구성

### 🎯 시스템 요구사항

#### 하드웨어
- **CPU**: Intel i5 이상 또는 AMD Ryzen 5 이상
- **메모리**: 16GB RAM 이상 권장 (최소 8GB)
- **저장공간**: 50GB 이상 여유 공간
- **GPU**: NVIDIA GeForce GTX 1060 이상 (CUDA 지원)

#### 소프트웨어
- **OS**: Windows 10 Pro 2004 이상 / Windows 11
- **WSL2**: Windows Subsystem for Linux 2
- **Docker Desktop**: 최신 버전

### 🔧 설치 단계

#### 1. WSL2 활성화
```powershell
# PowerShell을 관리자 권한으로 실행
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 시스템 재부팅 후
wsl --set-default-version 2
```

#### 2. Docker Desktop 다운로드 및 설치
1. [Docker Desktop for Windows](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe) 다운로드
2. 설치 프로그램 실행
3. **"Use WSL 2 instead of Hyper-V"** 옵션 체크
4. 설치 완료 후 재부팅

#### 3. Docker Desktop 설정
```json
// Settings > Docker Engine
{
  "experimental": false,
  "features": {
    "buildkit": true
  }
}
```

### 🎮 GPU 설정

#### 1. NVIDIA Container Toolkit 설치
```bash
# WSL2 Ubuntu 터미널에서 실행
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 2. GPU 지원 확인
```bash
# Docker에서 GPU 인식 테스트
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 🏗️ CSP 환경 구축

#### 1. 프로젝트 복제 및 이동
```bash
# Git 복제 (또는 기존 폴더 사용)
cd c:\Users\junto\Downloads\head-repo\CSP
```

#### 2. 환경 변수 설정
```bash
# .env 파일 생성 (선택사항)
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

#### 3. Docker 이미지 빌드 및 실행
```bash
# 컨테이너 빌드 및 시작
docker-compose up -d

# 설치 확인
docker exec -it csp-workspace python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🏗️ CSP Docker 환경 구조

### 핵심 파일 구조
```
CSP/
├── Dockerfile              # PyTorch 2.6 CUDA 환경
├── docker-compose.yml      # GPU 패스스루 설정
├── constraints.txt         # 패키지 버전 고정
├── pyproject.toml          # Poetry 안전 설정
└── documents/
    └── DOCKER_완전가이드.md
```

### Docker 컨테이너 스펙
- **베이스 이미지**: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
- **Python**: 3.11.10
- **PyTorch**: 2.6.0+cu124 (CUDA 완전 지원)
- **GPU**: NVIDIA GeForce RTX 3070 Ti 패스스루

### 설치된 핵심 패키지 (시스템 레벨)
```bash
# AI 핵심 라이브러리
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
transformers==4.55.3
FlagEmbedding==1.2.11
sentence-transformers==5.1.0

# 구문 분석기
supar==1.1.4              # 한문 구문 분석
stanza==1.10.1            # 다언어 NLP 파이프라인

# API 라이브러리
openai==1.101.0

# 유틸리티 (Poetry 관리)
pandas, openpyxl, regex, tqdm, requests
jupyter, black, isort, pytest
```

---

## 🛡️ 보안 시스템

### 3단계 보호 체계

#### 1️⃣ constraints.txt 강제 고정
```bash
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
transformers==4.55.3
# ... 모든 AI 패키지 버전 락
```

#### 2️⃣ 환경변수 보호
```dockerfile
ENV PIP_CONSTRAINT=/workspace/constraints.txt
ENV PIP_EXISTS_ACTION=i
ENV PIP_FORCE_REINSTALL=false
```

#### 3️⃣ Poetry 완전 분리
```toml
[tool.poetry.dependencies]
python = "^3.10"
# 🚨 torch 관련 패키지는 Docker에서 시스템 레벨 설치
# 🚨 Poetry는 torch와 무관한 패키지만 관리
pandas = "*"
openpyxl = "*"
# ... AI 패키지 제외
```

---

## 🚀 일상 운영 가이드

### 프로젝트 시작
```bash
# 1. 컨테이너 시작
cd c:\Users\junto\Downloads\head-repo\CSP
docker-compose up -d

# 2. 컨테이너 접속
docker exec -it csp-workspace /bin/bash

# 3. 작업 확인
cd /workspace
```

### 스크립트 실행
```bash
# SA (Sentence Alignment) 실행
cd sa
python main.py input.xlsx output.xlsx --embedder openai

# PA (Paragraph Alignment) 실행  
cd ../pa
python main.py input.xlsx output.xlsx

# 정확도 평가
cd ../accuracy
python accuracy_evaluator.py 정답.xlsx 예측.xlsx
python row_pair_evaluator.py PA정답.xlsx PA예측.xlsx
```

### 컨테이너 관리
```bash
# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs

# 재시작
docker-compose restart

# 종료
docker-compose down
```

---

## 🎯 복잡한 XML 파일명 쉽게 처리하기

### 문제 상황
**문제**: Docker 환경에서는 GUI가 없어서 복잡한 파일명을 가진 XML 파일들을 선택하기 어려움

**해결**: 
- 🚀 **범용 진입점**: `main.py`의 자동 감지 기능으로 더 쉽게!
- 🎯 **스마트 도구**: 3가지 스마트 도구로 GUI 없이도 쉽게 파일 선택!
- 📦 **패키지 구조**: `xml_pipeline/` 통합으로 안정적인 실행!

### 🚀 추천 사용법 (가장 쉬운 방법부터)

#### **0️⃣ 범용 main.py 자동 감지 (NEW! 🌟)**

가장 간단하고 강력한 방법:

```bash
# Docker 컨테이너 시작
docker-compose up -d

# 파일 형식 자동 감지로 처리 - 가장 쉬운 방법!
docker exec csp-workspace python /workspace/main.py auto \
  /workspace/sources/복잡한파일명_원문.xml /workspace/sources/복잡한파일명_번역문.xml

# 또는 디렉토리 경로만 알면 자동으로 쌍 찾기
docker exec csp-workspace python /workspace/main.py xml smart \
  --xml-dir /workspace/sources
```

#### **1️⃣ 스마트 인터랙티브 모드 (GUI 대체! 🎯)**

복잡한 파일명을 번호로 쉽게 선택:

```bash
# 스마트 모드 실행 - 메뉴에서 모든 것을 선택
./utils/xml_pipeline.sh smart
```

**📱 메뉴 화면:**
```
🚀 XML 파이프라인 빠른 처리
=================================
1. 단일 XML 쌍 선택해서 처리     ← 가장 많이 사용!
2. 디렉토리 전체 일괄 처리
3. 패턴으로 검색해서 처리
4. 최근 결과 보기
5. 배치 스크립트 생성
q. 종료

선택: 1
```

#### **2️⃣ 파일 브라우저 모드**

특정 디렉토리의 파일들을 브라우징:

```bash
# sources 디렉토리 브라우징
./utils/xml_pipeline.sh browse /workspace/sources

# 2017불용 디렉토리 브라우징
./utils/xml_pipeline.sh browse /workspace/2025/2017불용

# 윤소장님 디렉토리 브라우징  
./utils/xml_pipeline.sh browse /workspace/2025/윤소장님
```

**🗂️ 브라우저 화면:**
```
💑 XML 쌍 목록 (1-5/23):
===============================================
  1. 📖 jti_3b0301-[역주]육도직해_
     📜 원문: jti_3b0301-[역주]육도직해_원문_x-C2017.xml
     📰 번역: jti_3b0301-[역주]육도직해_번역문_x-C2017.xml
     📂 위치: private725/2025/2017불용

  2. 📖 jti_3j0201-[역주]안씨가훈1_
     📜 원문: jti_3j0201-[역주]안씨가훈1_원문_x-C2017.xml
     📰 번역: jti_3j0201-[역주]안씨가훈1_번역문_x-C2017.xml
     📂 위치: private725/2025/2017불용

📋 명령어:
  번호 입력: 쌍 선택
  n: 다음 페이지
  p: 이전 페이지  
  q: 취소

선택: 1  ← 번호만 입력하면 자동 처리!
```

#### **3️⃣ 스캔 및 검색 모드**

전체 디렉토리를 스캔해서 XML 쌍들을 찾기:

```bash
# 전체 워크스페이스 스캔
./utils/xml_pipeline.sh scan /workspace

# 특정 디렉토리만 스캔
./utils/xml_pipeline.sh scan /workspace/2025/2017불용

# 패턴으로 검색
./utils/xml_pipeline.sh search "육도직해"
```

### 💡 실무 팁

#### 자주 사용하는 명령어들
```bash
# 최근 처리한 결과 확인
./utils/xml_pipeline.sh recent

# 처리 이력 보기
./utils/xml_pipeline.sh history

# 배치 스크립트 생성 (반복 작업용)
./utils/xml_pipeline.sh batch
```

#### 파일 경로 복사하기
```bash
# 파일 경로를 클립보드로 복사 (Windows)
./utils/xml_pipeline.sh copy-path

# 상대 경로로 변환해서 복사
./utils/xml_pipeline.sh relative-path
```

---

## 🔧 문제 해결

### 자주 발생하는 문제들

#### 1. Docker 컨테이너가 시작되지 않을 때
```bash
# 로그 확인
docker-compose logs

# 컨테이너 상태 확인
docker-compose ps

# 강제 재시작
docker-compose down
docker-compose up -d
```

#### 2. GPU가 인식되지 않을 때
```bash
# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# WSL2에서 GPU 드라이버 확인
nvidia-smi
```

#### 3. PyTorch CUDA 문제
```bash
# 컨테이너 내에서 CUDA 버전 확인
docker exec -it csp-workspace python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### 4. 메모리 부족 오류
```bash
# Docker Desktop 메모리 할당량 증가 (Settings > Resources)
# 또는 처리 배치 크기 조정
--batch-size 32  # 기본값에서 줄이기
```

### 성능 최적화

#### 1. 캐시 활용
```bash
# 임베딩 캐시 활성화
export EMBEDDING_CACHE_DIR=/workspace/embeddings_cache_openai
```

#### 2. GPU 메모리 관리
```bash
# GPU 메모리 사용량 모니터링
watch -n 1 nvidia-smi
```

#### 3. 병렬 처리 최적화
```bash
# CPU 코어 수에 맞게 워커 수 조정
--num-workers 8  # CPU 코어 수에 맞게 설정
```

---

## 📞 지원 및 문의

### 문서 위치
- **완전 가이드**: `CSP/documents/DOCKER_완전가이드.md` (현재 문서)
- **튜닝 가이드**: `CSP/documents/튜닝_가이드.md`
- **워크플로우**: `CSP/documents/CSP_전체_워크플로우.md`

### 추가 참고 자료
- [Docker 공식 문서](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch CUDA 설치 가이드](https://pytorch.org/get-started/locally/)

---

**📝 업데이트**: 2025년 1월 - Docker 설치부터 XML 처리까지 통합 가이드 완성