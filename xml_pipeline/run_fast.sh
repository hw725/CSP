#!/bin/bash
# XML 파이프라인 고속 실행 스크립트
# High-performance XML pipeline runner

# 성능 모드 활성화
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 메모리 최적화
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_TOP_PAD_=100000

echo "🚀 고성능 XML 파이프라인 시작"
echo "   성능 최적화 설정 활성화"
echo "   배치 크기: 자동 최적화"
echo "   병렬 처리: 최대 성능"

# 단일 파일 처리
if [ "$1" == "single" ] && [ "$#" == 3 ]; then
    echo "📖 단일 XML 쌍 고속 처리: $2, $3"
    python xml_pipeline_cli.py single "$2" "$3" --performance-mode --max-workers 4 --batch-size 100 --enable-cache
    
# 디렉토리 배치 처리    
elif [ "$1" == "directory" ] && [ "$#" == 2 ]; then
    echo "📚 디렉토리 배치 고속 처리: $2"
    python xml_pipeline_cli.py directory "$2" --performance-mode --max-workers 6 --batch-size 150 --enable-cache --streaming
    
# 대용량 텍스트 처리 (당송팔대가문초, 관자 등)
elif [ "$1" == "large" ] && [ "$#" == 3 ]; then
    echo "📊 대용량 텍스트 고속 처리: $2, $3"
    python xml_pipeline_cli.py single "$2" "$3" --performance-mode --max-workers 6 --batch-size 200 --enable-cache --streaming --chunk-size 2000
    
else
    echo "사용법:"
    echo "  ./run_fast.sh single 원문.xml 번역문.xml     # 단일 파일 고속 처리"
    echo "  ./run_fast.sh directory /path/to/xmls      # 디렉토리 고속 처리"
    echo "  ./run_fast.sh large 대용량원문.xml 대용량번역문.xml  # 대용량 고속 처리"
    exit 1
fi

echo "✅ 고성능 XML 파이프라인 완료"