"""SA 정렬 시스템 메인 실행기"""

import argparse
import logging
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from typing import Optional
from io_manager import process_file as process_file_parallel

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

def get_tokenizer_module(tokenizer_name: str):
    """토크나이저 모듈 동적 로드 - jieba(원문), mecab(번역문)만 지원"""
    tokenizer_map = {
        'jieba': 'sa_tokenizers.jieba_mecab',
        'mecab': 'sa_tokenizers.jieba_mecab',
    }
    if tokenizer_name not in tokenizer_map:
        raise ValueError(f"지원하지 않는 토크나이저: {tokenizer_name}. 지원: jieba(원문), mecab(번역문)")
    module_name = tokenizer_map[tokenizer_name]
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"토크나이저 모듈 로드 실패: {module_name} ({e})")

def get_embedder_module(embedder_name: str):
    """임베더 모듈 동적 로드"""
    embedder_map = {
        'openai': 'sa_embedders.openai',
        'bge': 'sa_embedders.bge',
    }
    
    if embedder_name not in embedder_map:
        raise ValueError(f"지원하지 않는 임베더: {embedder_name}. 지원: openai, bge")
    
    module_name = embedder_map[embedder_name]
    
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"임베더 모듈 로드 실패: {module_name} ({e})")

def process_single_file(
    input_file: str,
    output_file: str,
    tokenizer_name: str = 'jieba',
    embedder_name: str = 'bge',
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    parallel: bool = False,
    openai_model: str = "text-embedding-3-large",
    openai_api_key: str = None,
    progress_callback=None,
    stop_flag=None,
    **kwargs
) -> bool:
    """단일 파일 처리 (병렬 옵션 지원)"""
    start_time = time.time()
    print(f"🚀 파일 처리 시작: {input_file}")
    print(f"📊 설정:")
    print(f"   토크나이저: {tokenizer_name}")
    print(f"   임베더: {embedder_name}")
    print(f"   의미 매칭: {use_semantic}")
    print(f"   병렬 처리: {parallel}")
    print(f"   토큰 범위: {min_tokens}-{max_tokens}")
    
    try:
        if parallel:
            print("⚡ 병렬 처리 모드로 실행합니다.")
            from io_manager import process_file as io_process_file
            
            # 임베더별 파라미터 설정
            if embedder_name == 'bge':
                results_df = io_process_file(
                    input_file,
                    output_file,
                    parallel=True,
                    workers=4,
                    verbose=True,  # verbose 모드 활성화
                    embedder_name=embedder_name,
                    openai_model=None,
                    openai_api_key=None
                )
            else:  # openai
                results_df = io_process_file(
                    input_file,
                    output_file,
                    parallel=True,
                    workers=4,
                    verbose=True,  # verbose 모드 활성화
                    embedder_name=embedder_name,
                    openai_model=openai_model,
                    openai_api_key=openai_api_key
                )
            
            end_time = time.time()
            
            if results_df is not None:
                print(f"🎉 병렬 처리 완료!")
                print(f"⏱️  처리 시간: {end_time - start_time:.2f}초")
                print(f"📊 결과: {len(results_df)}개 구")
                return True
            else:
                print(f"❌ 병렬 처리 실패")
                return False
        
        # 항상 동적 모듈 로딩 경로 사용
        print("✅ 동적 모듈 로딩...")
        tokenizer_module = get_tokenizer_module(tokenizer_name)
        embedder_module = get_embedder_module(embedder_name)
        print(f"✅ 모듈 로드 완료")
        from processor import process_file_with_modules
        results = process_file_with_modules(
            input_file, output_file,
            tokenizer_module, embedder_module,
            embedder_name,  # 추가!
            use_semantic, min_tokens, max_tokens,
            openai_model=openai_model,
            openai_api_key=openai_api_key
        )

        end_time = time.time()  # ⏱️ 처리 종료 시간 기록

        if results is not None:
            print(f"🎉 처리 완료!")
            print(f"⏱️  처리 시간: {end_time - start_time:.2f}초")
            print(f"📊 처리 결과: {len(results)}개 문장")
            print(f"📁 출력 파일: {output_file}")

            # 기본 형식 저장
            output_file_basic = output_file

            # 구 단위 형식 저장
            output_file_phrase = output_file.replace('.xlsx', '_phrase.xlsx')

            from io_utils import save_phrase_format_results
            save_phrase_format_results(results, output_file_phrase)

            print(f"📁 기본 출력: {output_file_basic}")
            print(f"📁 구 단위 출력: {output_file_phrase}")

            return True
        else:
            print(f"❌ 처리 실패")
            return False

    except Exception as e:
        print(f"💥 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(progress_callback=None, stop_flag=None):
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='SA 정렬 시스템 - 문장 단위 토큰 정렬',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 처리
  python main.py input.xlsx output.xlsx
  
  # 토크나이저/임베더 지정
  python main.py input.xlsx output.xlsx --tokenizer jieba --embedder openai
  
  # 병렬 처리 (워커 수 조정)
  python main.py input.xlsx output.xlsx --parallel --workers 2
  
  # 상세 설정
  python main.py input.xlsx output.xlsx --tokenizer jieba --embedder bge --min-tokens 2 --max-tokens 15 --no-semantic --workers 4
  
지원 토크나이저: jieba, mecab
지원 임베더: openai, bge
        """
    )
    
    parser.add_argument("input_file", help="입력 Excel 파일")
    parser.add_argument("output_file", help="출력 Excel 파일")
    
    # 토크나이저 옵션
    parser.add_argument("--tokenizer", default="jieba", choices=["jieba", "mecab"], 
                       help="토크나이저 선택 (기본값: jieba)")
    
    # 임베더 옵션
    parser.add_argument("--embedder", default="bge", choices=["bge", "openai"], 
                       help="임베더 선택 (기본값: bge)")
    
    # 처리 방식 옵션
    parser.add_argument("--parallel", action="store_true", 
                       help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=4, 
                       help="병렬 처리 워커 수 (기본값: 4)")
    
    # 의미적 매칭 옵션
    parser.add_argument("--no-semantic", action="store_true", 
                       help="의미적 매칭 비활성화")
    
    # 토큰 범위 옵션
    parser.add_argument("--min-tokens", type=int, default=1, 
                       help="최소 토큰 수 (기본값: 1)")
    parser.add_argument("--max-tokens", type=int, default=10, 
                       help="최대 토큰 수 (기본값: 10)")
    
    # OpenAI 관련 옵션
    parser.add_argument("--openai-model", default="text-embedding-3-large", 
                       help="OpenAI 임베딩 모델 (기본값: text-embedding-3-large)")
    parser.add_argument("--openai-api-key", 
                       help="OpenAI API 키 (환경변수 OPENAI_API_KEY 사용 가능)")
    
    # 기타 옵션
    parser.add_argument("--verbose", action="store_true", 
                       help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    print("🚀 SA (Sentence Aligner) 시작")
    print(f"📂 입력 파일: {args.input_file}")
    print(f"📁 출력 파일: {args.output_file}")
    print(f"🔧 설정:")
    print(f"   토크나이저: {args.tokenizer}")
    print(f"   임베더: {args.embedder}")
    print(f"   병렬 처리: {'Yes' if args.parallel else 'No'}")
    if args.parallel:
        print(f"   워커 수: {args.workers}")
    print(f"   의미적 매칭: {'No' if args.no_semantic else 'Yes'}")
    print(f"   토큰 범위: {args.min_tokens}-{args.max_tokens}")
    print()
    
    try:
        success = process_single_file(
            input_file=args.input_file,
            output_file=args.output_file,
            tokenizer_name=args.tokenizer,
            embedder_name=args.embedder,
            use_semantic=not args.no_semantic,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            parallel=args.parallel,
            workers=args.workers,  # 추가
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
            progress_callback=progress_callback,
            stop_flag=stop_flag
        )
        
        if success:
            print("🎉 처리 성공!")
            return True
        else:
            print("❌ 처리 실패!")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"💥 예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()