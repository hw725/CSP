# 📚 CSP 프로젝트 문서 디렉토리

> **Chinese Sentence Processing (CSP) 프로젝트의 체계적인 문서 관리 시스템**

## 📂 디렉토리 구조

```
CSP/documents/
├── 📊 architecture/           # 시스템 아키텍처 및 워크플로우
├── 📈 flowcharts/            # 상세 플로우차트 및 프로세스 다이어그램
├── 📖 guides/                # 사용자 가이드 및 튜토리얼
└── constraints.txt           # Docker 패키지 제약 조건
```

---

## 🏗️ Architecture (시스템 아키텍처)

시스템 전체 설계와 각 모듈 간의 관계를 설명합니다.

- **`CSP_전체_워크플로우.md`** - CSP 프로젝트 전체 워크플로우와 주요 구성요소
- **`Accuracy_디렉토리_플로우차트.md`** - 정확도 평가 모듈(accuracy/) 아키텍처
- **`Common_디렉토리_플로우차트.md`** - 공통 유틸리티 모듈(common/) 아키텍처  
- **`PA_디렉토리_플로우차트.md`** - 문단 정렬 모듈(pa/) 아키텉처
- **`SA_디렉토리_플로우차트.md`** - 문장 정렬 모듈(sa/) 아키텍처
- **`README_최적화시스템.md`** - 성능 최적화 시스템 개요

---

## 📈 Flowcharts (플로우차트)

각 기능과 프로세스의 상세한 실행 흐름을 Mermaid 다이어그램으로 제공합니다.

### 📋 플로우차트 카탈로그
- **`XML_Pipeline_플로우차트_README.md`** - 모든 플로우차트 인덱스 및 사용법

### 🏗️ 시스템 아키텍처 플로우차트
- **`XML_Pipeline_전체_아키텍처.mmd`** - 전체 시스템 아키텍처
- **`XML_Pipeline_데이터플로우.mmd`** - 데이터 흐름 시퀀스 다이어그램

### 🖥️ 사용자 인터페이스 플로우차트  
- **`XML_Pipeline_CLI_플로우.mmd`** - CLI 인터페이스 프로세스
- **`XML_File_Browser_플로우.mmd`** - 파일 브라우저 기능

### ⚙️ 핵심 처리 플로우차트
- **`XML_Pipeline_Processor_플로우.mmd`** - 메인 프로세서 로직
- **`XML_Unit_Parser_플로우.mmd`** - XML 파싱 프로세스
- **`XML_Advanced_Accuracy_플로우.mmd`** - 고급 정확도 분석
- **`XML_Level_Similarity_플로우.mmd`** - 계층적 유사도 분석

### 🔧 운영 및 최적화 플로우차트
- **`Performance_Optimizer_플로우.mmd`** - 성능 최적화 프로세스
- **`Docker_XML_Smart_플로우.mmd`** - Docker 스마트 실행 플로우
- **`Docker_환경_설정_플로우.mmd`** - Docker 환경 구축 프로세스

### 🧠 AI 처리 플로우차트
- **`SA_고급문장분할_플로우.mmd`** - 고급 문장 분할 알고리즘
- **`SA_순차정렬_플로우.mmd`** - 순차적 문장 정렬 프로세스

---

## 📖 Guides (가이드)

실무에서 바로 활용할 수 있는 상세한 사용법과 튜토리얼을 제공합니다.

- **`DOCKER_완전가이드.md`** - Docker 설치부터 XML 처리까지 통합 가이드
  - Docker 설치 및 GPU 설정
  - CSP 환경 구축 방법  
  - 복잡한 XML 파일명 처리 팁
  - 문제해결 및 성능 최적화

- **`튜닝_가이드.md`** - 성능 튜닝 및 최적화 가이드
  - 임베딩 모델 선택 및 튜닝
  - GPU 메모리 최적화
  - 배치 처리 최적화

- **`XML_PIPELINE_GUIDE.md`** - XML 파이프라인 사용법
  - 기본 사용법부터 고급 기능까지
  - 배치 처리 및 자동화

---

## 🚀 빠른 시작

### 새로 시작하는 사용자
1. **`guides/DOCKER_완전가이드.md`** - Docker 환경 설정
2. **`guides/XML_PIPELINE_GUIDE.md`** - 기본 사용법 학습
3. **`flowcharts/XML_Pipeline_플로우차트_README.md`** - 플로우차트 활용법

### 시스템 이해가 필요한 개발자  
1. **`architecture/CSP_전체_워크플로우.md`** - 전체 시스템 이해
2. **`flowcharts/XML_Pipeline_전체_아키텍처.mmd`** - 아키텍처 시각화
3. **`architecture/`** 각 모듈별 상세 아키텍처 문서

### 성능 최적화가 필요한 사용자
1. **`guides/튜닝_가이드.md`** - 성능 최적화 방법
2. **`flowcharts/Performance_Optimizer_플로우.mmd`** - 최적화 프로세스
3. **`architecture/README_최적화시스템.md`** - 최적화 시스템 개요

---

## 🔄 문서 업데이트 이력

- **2025-01-23**: 디렉토리 구조 대규모 정리
  - 중복 파일 제거 및 통합
  - Docker 가이드 3개 → 1개 통합
  - 체계적인 3단계 디렉토리 구조 구축
  - 플로우차트 명명 규칙 표준화

- **이전**: XML Pipeline 플로우차트 10개 추가, Mermaid 구문 오류 수정

---

## 📞 문서 관련 문의

- **기술 문의**: CSP 프로젝트 메인테이너
- **문서 업데이트 요청**: GitHub Issues를 통해 제출
- **플로우차트 수정**: Mermaid 구문을 따라 작성

---

**📝 마지막 업데이트**: 2025년 1월 23일 - 디렉토리 구조 완전 정리 완료