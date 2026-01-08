#!/bin/bash
# CSP XML 전체 파이프라인 실행 스크립트 (Docker/Linux)
# XML 쌍(원문+번역문) → 문단병렬 → PA → 문장병렬 → SA → 구병렬 → 정확도 분석

echo "========================================"
echo "CSP XML 전체 파이프라인 시스템"  
echo "🐳 Docker 스마트 모드 지원"
echo "========================================"

CONTAINER_NAME="csp-workspace"

# 함수: Docker 컨테이너에서 명령 실행
run_in_container() {
    docker exec -it $CONTAINER_NAME /bin/bash -c "cd /workspace && $1"
}

# 함수: Docker 컨테이너에서 Python 스크립트 실행 (인터랙티브)
run_python_interactive() {
    docker exec -it $CONTAINER_NAME /bin/bash -c "cd /workspace && python -u $1"
}

# 함수: 사용법 출력  
show_usage() {
    echo ""
    echo "🐳 Docker XML 파이프라인 스마트 실행 도구"
    echo "복잡한 파일명도 쉽게! GUI 없이도 편리하게!"
    echo ""
    echo "사용법:"
    echo "  ./xml_pipeline.sh [명령어] [옵션들]"
    echo ""
    echo "🎯 추천 명령어 (복잡한 파일명 처리용):"
    echo "  smart    - 📱 스마트 인터랙티브 메뉴 (가장 쉬움!)"
    echo "  browse   - 🗂️  파일 브라우저로 선택"
    echo "  scan     - 🔍 XML 파일 스캔 및 쌍 찾기"
    echo ""
    echo "📋 기본 명령어:"
    echo "  process  - 단일 XML 쌍 처리"
    echo "  batch    - 디렉토리 일괄 처리" 
    echo "  list     - 최근 처리 결과 목록"
    echo "  show     - 특정 쌍 상세 결과"
    echo "  cleanup  - 오래된 결과 정리"
    echo ""
    echo "💡 복잡한 파일명 처리 예제:"
    echo "  ./xml_pipeline.sh smart                           # 🎯 추천! 메뉴에서 선택"
    echo "  ./xml_pipeline.sh browse /workspace/private725    # 파일 브라우저"
    echo "  ./xml_pipeline.sh scan /workspace/private725      # 전체 스캔"
    echo ""
    echo "📝 일반 예시:"
    echo "  # 단일 쌍 처리"
    echo "  ./xml_pipeline.sh process 원문.xml 번역문.xml"
    echo ""
    echo "  # 디렉토리 일괄 처리"
    echo "  ./xml_pipeline.sh batch /workspace/xml_files"
    echo ""
    echo "  # 최근 결과 목록"
    echo "  ./xml_pipeline.sh list"
    echo ""
    echo "  # 상세 결과 조회"
    echo "  ./xml_pipeline.sh show 쌍ID"
    echo ""
}

# 인자 확인
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

# Docker 컨테이너 상태 확인
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "❌ 오류: Docker 컨테이너 '$CONTAINER_NAME'가 실행 중이 아닙니다."
    echo "다음 명령으로 컨테이너를 시작하세요:"
    echo "  docker-compose up -d"
    exit 1
fi

echo "✅ Docker 컨테이너 '$CONTAINER_NAME' 연결 확인"

# 명령어별 처리
case "$1" in
    "smart")
        echo ""
        echo "=== 🐳 Docker XML Smart Pipeline ==="
        echo "복잡한 파일명을 번호로 쉽게 선택하세요!"
        echo ""
        
        run_python_interactive "xml_pipeline/docker_xml_smart.py --mode menu"
        ;;
        
    "browse")
        echo ""
        echo "=== 📂 XML 파일 브라우저 ==="
        
        BROWSE_DIR="$2"
        if [ -z "$BROWSE_DIR" ]; then
            BROWSE_DIR="sources"
        fi
        
        echo "탐색 디렉토리: $BROWSE_DIR"
        echo ""
        
        run_python_interactive "xml_pipeline/xml_file_browser.py --dir '$BROWSE_DIR' --interactive"
        ;;
        
    "scan")
        echo ""
        echo "=== 🔍 XML 파일 스캔 ==="
        
        SCAN_DIR="$2"
        SCAN_PATTERN="$3"
        
        if [ -z "$SCAN_DIR" ]; then
            SCAN_DIR="sources"
        fi
        
        echo "스캔 디렉토리: $SCAN_DIR"
        [ ! -z "$SCAN_PATTERN" ] && echo "패턴: $SCAN_PATTERN"
        echo ""
        
        if [ ! -z "$SCAN_PATTERN" ]; then
            run_python_interactive "xml_pipeline/xml_file_browser.py --dir '$SCAN_DIR' --pattern '$SCAN_PATTERN'"
        else
            run_python_interactive "xml_pipeline/xml_file_browser.py --dir '$SCAN_DIR' --interactive"
        fi
        ;;
        
    "quick")
        echo ""
        echo "=== ⚡ 빠른 패턴 처리 ==="
        
        QUICK_PATTERN="$2"
        
        if [ -z "$QUICK_PATTERN" ]; then
            echo "❌ 오류: 검색 패턴이 필요합니다."
            echo "사용법: ./xml_pipeline.sh quick [패턴]"
            echo "예시: ./xml_pipeline.sh quick 한유"
            exit 1
        fi
        
        echo "패턴: $QUICK_PATTERN"
        echo ""
        
        run_python_interactive "xml_pipeline/docker_xml_smart.py --mode pattern --pattern '$QUICK_PATTERN'"
        ;;
        
    "process")
        echo ""
        echo "=== 단일 XML 쌍 처리 ==="
        
        ORIGINAL_XML="$2"
        TRANSLATION_XML="$3"
        PAIR_ID="$4"
        
        if [ -z "$ORIGINAL_XML" ]; then
            echo "❌ 오류: 원문 XML 파일이 필요합니다."
            echo "사용법: ./xml_pipeline.sh process [원문.xml] [번역문.xml] [쌍ID]"
            exit 1
        fi
        
        if [ -z "$TRANSLATION_XML" ]; then
            echo "❌ 오류: 번역문 XML 파일이 필요합니다."
            echo "사용법: ./xml_pipeline.sh process [원문.xml] [번역문.xml] [쌍ID]"
            exit 1
        fi
        
        echo "원문 XML: $ORIGINAL_XML"
        echo "번역문 XML: $TRANSLATION_XML"
        [ ! -z "$PAIR_ID" ] && echo "쌍 ID: $PAIR_ID"
        echo ""
        
        if [ ! -z "$PAIR_ID" ]; then
            CMD="python main.py process --original '$ORIGINAL_XML' --translation '$TRANSLATION_XML' --pair-id '$PAIR_ID'"
        else
            CMD="python main.py process --original '$ORIGINAL_XML' --translation '$TRANSLATION_XML'"
        fi
        
        run_in_container "$CMD"
        ;;
        
    "batch")
        echo ""
        echo "=== 디렉토리 일괄 처리 ==="
        
        XML_DIR="$2"
        PATTERN="$3"
        
        if [ -z "$XML_DIR" ]; then
            echo "❌ 오류: XML 디렉토리가 필요합니다."
            echo "사용법: ./xml_pipeline.sh batch [XML_디렉토리] [패턴]"
            exit 1
        fi
        
        echo "XML 디렉토리: $XML_DIR"
        
        if [ ! -z "$PATTERN" ]; then
            echo "파일 패턴: $PATTERN"
            CMD="python main.py batch --xml-dir '$XML_DIR' --pattern '$PATTERN'"
        else
            echo "파일 패턴: *원문*.xml (기본값)"
            CMD="python main.py batch --xml-dir '$XML_DIR'"
        fi
        
        run_in_container "$CMD"
        ;;
        
    "list")
        echo ""
        echo "=== 최근 처리 결과 목록 ==="
        
        LIMIT="$2"
        
        if [ ! -z "$LIMIT" ]; then
            echo "조회 개수: $LIMIT"
            CMD="python main.py list --limit $LIMIT"
        else
            echo "조회 개수: 10 (기본값)"
            CMD="python main.py list"
        fi
        
        run_in_container "$CMD"
        ;;
        
    "show")
        echo ""
        echo "=== 상세 결과 조회 ==="
        
        PAIR_ID="$2"
        
        if [ -z "$PAIR_ID" ]; then
            echo "❌ 오류: XML 쌍 ID가 필요합니다."
            echo "사용법: ./xml_pipeline.sh show [쌍ID]"
            exit 1
        fi
        
        echo "쌍 ID: $PAIR_ID"
        echo ""
        
        CMD="python main.py show '$PAIR_ID'"
        run_in_container "$CMD"
        ;;
        
    "cleanup")
        echo ""
        echo "=== 오래된 결과 정리 ==="
        
        DAYS="$2"
        
        if [ ! -z "$DAYS" ]; then
            echo "보관 기간: ${DAYS}일"
            CMD="python main.py cleanup --days $DAYS"
        else
            echo "보관 기간: 7일 (기본값)"
            CMD="python main.py cleanup"
        fi
        
        run_in_container "$CMD"
        ;;
    
    "smart")
        echo ""
        echo "=== 🚀 스마트 인터랙티브 모드 ==="
        echo "복잡한 파일명도 쉽게 처리합니다!"
        
        CMD="python xml_pipeline/docker_xml_smart.py menu"
        run_in_container "$CMD"
        ;;
        
    "browse")
        echo ""
        echo "=== 🗂️ XML 파일 브라우저 ==="
        
        BROWSE_DIR="$2"
        if [ -z "$BROWSE_DIR" ]; then
            BROWSE_DIR="/workspace"
        fi
        
        echo "디렉토리: $BROWSE_DIR"
        CMD="python xml_pipeline/xml_file_browser.py select-pair"
        run_in_container "cd '$BROWSE_DIR' && $CMD"
        ;;
        
    "scan")
        echo ""
        echo "=== 🔍 XML 파일 스캔 및 쌍 찾기 ==="
        
        SCAN_DIR="$2"
        if [ -z "$SCAN_DIR" ]; then
            SCAN_DIR="/workspace"
        fi
        
        echo "스캔 디렉토리: $SCAN_DIR"
        CMD="python xml_pipeline/xml_file_browser.py pair"
        run_in_container "cd '$SCAN_DIR' && $CMD"
        ;;
        
    *)
        echo "❌ 알 수 없는 명령어: $1"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "✅ 작업 완료"