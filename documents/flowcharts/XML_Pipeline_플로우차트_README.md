# XML Pipeline 플로우차트 문서 인덱스

이 디렉토리에는 XML Pipeline의 전체 아키텍처와 각 모듈별 상세 플로우차트가 포함되어 있습니다.

## 📋 파일 목록

### 🏗️ 전체 아키텍처
- **[XML_Pipeline_전체_아키텍처.mmd](XML_Pipeline_전체_아키텍처.mmd)**: 전체 시스템 아키텍처 개요
- **[XML_Pipeline_데이터플로우.mmd](XML_Pipeline_데이터플로우.mmd)**: 모듈 간 데이터 흐름 시퀀스

### 🔧 핵심 처리 모듈
- **[XML_Pipeline_CLI_플로우.mmd](XML_Pipeline_CLI_플로우.mmd)**: CLI 인터페이스 처리 흐름
- **[XML_Pipeline_Processor_플로우.mmd](XML_Pipeline_Processor_플로우.mmd)**: 핵심 처리 엔진 흐름
- **[XML_Unit_Parser_플로우.mmd](XML_Unit_Parser_플로우.mmd)**: XML 구문 분석 프로세스

### 📊 분석 모듈
- **[XML_Advanced_Accuracy_플로우.mmd](XML_Advanced_Accuracy_플로우.mmd)**: 정확도 분석 프로세스
- **[XML_Level_Similarity_플로우.mmd](XML_Level_Similarity_플로우.mmd)**: 유사도 분석 프로세스

### ⚡ 최적화 및 지원 모듈
- **[Performance_Optimizer_플로우.mmd](Performance_Optimizer_플로우.mmd)**: 성능 최적화 프로세스
- **[Docker_XML_Smart_플로우.mmd](Docker_XML_Smart_플로우.mmd)**: Docker 스마트 실행 프로세스
- **[XML_File_Browser_플로우.mmd](XML_File_Browser_플로우.mmd)**: 파일 브라우저 관리 프로세스

## 🎯 사용 방법

각 `.mmd` 파일은 Mermaid 다이어그램 형식으로 작성되어 있습니다. 다음과 같은 도구에서 렌더링할 수 있습니다:

### VS Code에서 보기
1. Mermaid Preview 확장 설치
2. `.mmd` 파일 열기
3. `Ctrl+Shift+P` → "Mermaid: Preview"

### 온라인에서 보기
1. [Mermaid Live Editor](https://mermaid.live/)에 접속
2. 파일 내용 복사 후 붙여넣기

### GitHub에서 보기
- GitHub은 `.mmd` 파일을 자동으로 렌더링합니다

## 🔍 다이어그램 설명

### 색상 코드
- 🟢 **초록색**: 시작점/진입점
- 🔴 **빨간색**: 종료점/에러 상태
- 🟠 **주황색**: 분기점/조건 판단
- 🔵 **파란색**: 주요 처리 단계
- 🟣 **보라색**: 집계/통합 단계

### 주요 패턴
- **순차 처리**: 직선 화살표로 연결
- **조건 분기**: 다이아몬드 모양의 판단 노드
- **루프 처리**: 순환 화살표로 표시
- **서브프로세스**: 점선으로 둘러싸인 그룹

## 📈 업데이트 이력

- **2025-09-22**: 초기 플로우차트 문서 생성
  - 전체 아키텍처 다이어그램
  - 8개 주요 모듈별 상세 플로우
  - 데이터 플로우 시퀀스 다이어그램

---

**참고**: 이 문서들은 XML Pipeline 코드베이스의 현재 상태를 반영하며, 코드 변경 시 함께 업데이트해야 합니다.