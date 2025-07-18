"""SA (Sentence Aligner) 메인 실행 파일"""

import argparse
import time
import logging
import traceback
from pathlib import Path

def setup_logging(verbose: bool = False):
    """로깅 설정"""
    if verbose:
        level = logging.DEBUG
        format_str = '%(asctime)s - %(levelname)s:%(name)s:%(message)s'
    else:
        level = logging.WARNING  # 🔧 기본 모드에서는 WARNING 이상만
        format_str = '%(levelname)s: %(message)s'  # 🔧 간단한 형식
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 🔧 특정 모듈들의 로깅 레벨 조정 (non-verbose 모드에서)
    if not verbose:
        # 🔧 환경 변수로 출력 제어
        import os
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['DATASETS_VERBOSITY'] = 'error'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # 🔧 huggingface 다운로드 진행률 숨김
        
        # 외부 라이브러리들 조용히 하기
        logging.getLogger('datasets').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('FlagEmbedding').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('punctuation').setLevel(logging.ERROR)  # 🔧 무결성 경고 숨기기
        logging.getLogger('io_manager').setLevel(logging.ERROR)
        logging.getLogger('sa_tokenizers').setLevel(logging.ERROR)
        
        # 🔧 모든 경고 메시지 완전 숨김
        import warnings
        warnings.filterwarnings("ignore")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='SA: 한문-한국어 문장 분할 도구')
    
    # 필수 인수 (기본값 제공)
    parser.add_argument('input_file', nargs='?', default='input.xlsx', help='입력 엑셀 파일 경로 (기본: input.xlsx)')
    parser.add_argument('output_file', nargs='?', default='output.xlsx', help='출력 엑셀 파일 경로 (기본: output.xlsx)')
    
    # 선택적 인수
    parser.add_argument('--embedder', choices=['bge', 'openai'], default='bge',
                       help='임베더 선택 (기본: bge)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='최대 워커 수 (기본: 4)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='청크 크기 (기본: 100)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='병렬 처리 비활성화')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
    # 토크나이저 옵션
    parser.add_argument('--min-src-tokens', type=int, default=1,
                       help='원문 최소 토큰 수 (기본: 1)')
    parser.add_argument('--max-src-tokens', type=int, default=20,
                       help='원문 최대 토큰 수 (기본: 20)')
    parser.add_argument('--min-tgt-tokens', type=int, default=1,
                       help='번역문 최소 토큰 수 (기본: 1)')
    parser.add_argument('--max-tgt-tokens', type=int, default=30,
                       help='번역문 최대 토큰 수 (기본: 30)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # use_parallel 계산 (기존 코드와 호환)
    use_parallel = not args.no_parallel
    
    if args.verbose:
        print("🚀 SA 파일 처리 시작:", args.input_file)
        print(f"⚙️  설정: 임베더={args.embedder}, 병렬={use_parallel}, 워커={args.max_workers}")
        print()
    # 🔧 기본 모드에서는 시작 메시지 제거 (io_manager에서 처리)
    
    start_time = time.time()
    
    try:
        # io_manager의 process_file 함수 호출
        from io_manager import process_file
        
        success = process_file(
            input_file=args.input_file,
            output_file=args.output_file,
            embedder_name=args.embedder,
            max_workers=args.max_workers,
            chunk_size=args.chunk_size,
            use_parallel=use_parallel,  # 계산된 값 사용
            min_src_tokens=args.min_src_tokens,
            max_src_tokens=args.max_src_tokens,
            min_tgt_tokens=args.min_tgt_tokens,
            max_tgt_tokens=args.max_tgt_tokens,
            verbose=args.verbose
        )
        
        elapsed_time = time.time() - start_time
        
        if not args.verbose:
            # 🔧 기본 모드에서는 간단한 완료 메시지만
            if success:
                try:
                    import pandas as pd
                    result_df = pd.read_excel(args.output_file)
                    print(f"✅ 완료: {len(result_df):,}개 구문 ({elapsed_time:.1f}초)")
                except:
                    print(f"✅ 완료 ({elapsed_time:.1f}초)")
            else:
                print("❌ 처리 실패")
        else:
            # verbose 모드에서는 상세 정보 출력
            print()
            print("🎉 처리 완료!")
            print(f"⏱️  처리 시간: {elapsed_time:.2f}초")
            
            if success:
                print(f"✅ 결과 파일: {args.output_file}")
                
                # 결과 파일 통계 출력
                try:
                    import pandas as pd
                    result_df = pd.read_excel(args.output_file)
                    print(f"📊 처리 결과: {len(result_df)}개 문장")
                    
                    # 분할 방법별 통계
                    if '분할방법' in result_df.columns:
                        method_counts = result_df['분할방법'].value_counts()
                        print("📈 분할 방법별 통계:")
                        for method, count in method_counts.items():
                            print(f"   {method}: {count}개")
                    
                except Exception as stats_error:
                    print(f"📊 통계 계산 오류: {stats_error}")
            else:
                print("❌ 처리 실패")
                return 1
            
    except Exception as e:
        print(f"❌ 처리 오류: {e}")
        if args.verbose:
            print("\n상세 오류 정보:")
            print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())