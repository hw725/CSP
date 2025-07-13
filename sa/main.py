"""SA 정렬 시스템 메인 실행기 - 간소화"""

import argparse
import logging
import time
import sys
import os
from pathlib import Path

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sa_processing.log', encoding='utf-8')
        ]
    )

def process_single_file(
    input_file: str,
    output_file: str,
    embedder_name: str = 'bge',
    parallel: bool = True,
    workers: int = 4,
    openai_api_key: str = None,
    verbose: bool = False
) -> bool:
    """단일 파일 처리 - 간소화"""
    
    start_time = time.time()
    print(f"🚀 SA 파일 처리 시작: {input_file}")
    print(f"⚙️  설정: 임베더={embedder_name}, 병렬={parallel}, 워커={workers}")
    print()
    
    try:
        from io_manager import process_file
        
        results_df = process_file(
            input_file,
            output_file,
            parallel=parallel,
            workers=workers,
            embedder_name=embedder_name
        )
        
        if results_df is not None:
            end_time = time.time()
            print(f"\n🎉 처리 완료!")
            print(f"⏱️  처리 시간: {end_time - start_time:.2f}초")
            print(f"📊 결과: {len(results_df)}개 구")
            print(f"📁 출력: {output_file}")
            return True
        else:
            print(f"\n❌ 처리 실패")
            return False
            
    except Exception as e:
        print(f"❌ 처리 오류: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def main():
    """메인 함수 - 간소화"""
    
    parser = argparse.ArgumentParser(
        description='SA: 한문-한국어 문장/구 정렬 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py input.xlsx output.xlsx
  python main.py input.xlsx output.xlsx --embedder openai --openai-api-key your-key
  python main.py input.xlsx output.xlsx --no-parallel
        """
    )
    
    # 필수 인자
    parser.add_argument('input_file', help='입력 Excel 파일')
    parser.add_argument('output_file', help='출력 Excel 파일')
    
    # 선택 인자
    parser.add_argument('--embedder', '-e', default='bge',
                       choices=['bge', 'openai'], 
                       help='임베더 선택 (기본: bge)')
    
    parser.add_argument('--no-parallel', action='store_true',
                       help='병렬 처리 비활성화')
    
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='워커 프로세스 수 (기본: 4)')
    
    parser.add_argument('--openai-api-key', 
                       help='OpenAI API 키')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input_file}")
        sys.exit(1)
    
    # OpenAI 설정 확인
    if args.embedder == 'openai':
        if not args.openai_api_key and not os.getenv('OPENAI_API_KEY'):
            print("❌ OpenAI 임베더 사용 시 API 키가 필요합니다")
            sys.exit(1)
        
        if args.openai_api_key:
            os.environ['OPENAI_API_KEY'] = args.openai_api_key
    
    # 처리 실행
    success = process_single_file(
        input_file=args.input_file,
        output_file=args.output_file,
        embedder_name=args.embedder,
        parallel=not args.no_parallel,
        workers=args.workers,
        openai_api_key=args.openai_api_key,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()