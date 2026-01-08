"""SA (Sentence Aligner) 메인 실행 파일"""

import argparse
import time
import logging
import traceback
import warnings
from pathlib import Path

# torch.load 보안 경고 전역 무시 (PyTorch 2.6 호환성)
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*vulnerability.*")
warnings.filterwarnings("ignore", message=".*CVE-2025-32434.*")

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
        logging.getLogger('common.tokenizers').setLevel(logging.ERROR)  # 🔧 토크나이저 로깅 숨기기
        
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
    parser.add_argument('--embedder', choices=['bge', 'openai', 'none'], default='bge',
                       help='임베더 선택 (기본: bge, OpenAI: --embedder openai, 순차분할: --embedder none)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='최대 워커 수 (기본: 4, OpenAI 병렬 처리 지원)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='청크 크기 (기본: 100, OpenAI 병렬 최적화)')
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
    parser.add_argument('--max-tgt-tokens', type=int, default=40,
                       help='번역문 최대 토큰 수 (기본: 40)')
    # 가중치/스코어링 옵션 (실험적)
    parser.add_argument('--dp-window', type=int, default=2,
                       help='DP 예상 위치 허용 창 크기 (기본: 2)')
    parser.add_argument('--distance-decay', type=float, default=0.9,
                       help='위치 거리 감쇠 알파 (기본: 0.9)')
    parser.add_argument('--boundary-bonus', type=float, default=0.3,
                       help='문장 경계 보너스 (기본: 0.3)')
    parser.add_argument('--particle-bonus', type=float, default=0.1,
                       help='토씨 경계 보너스 (기본: 0.2)')
    parser.add_argument('--length-penalty', type=float, default=0.05,
                       help='세그먼트 길이 패널티 알파 (기본: 0.05)')
    parser.add_argument('--sim-gamma', type=float, default=1.5,
                       help='유사도 샤프닝 지수 (기본: 1.5)')
    # 문장 내부 경계 힌트(옵션)
    parser.add_argument('--syntax-hints', choices=['none', 'ko', 'zh', 'both'], default='both',
                       help='구문 파서 힌트 사용 (기본: both)')
    parser.add_argument('--comma-bonus', type=float, default=0.0,
                       help='콤마(,) 경계 보너스 (기본: 0.0, soft 모드)')
    parser.add_argument('--comma-mode', choices=['soft', 'strict'], default='soft',
                       help='콤마 경계 모드: soft(나열 제외) | strict(전부 적용)')
    parser.add_argument('--syntax-when', choices=['ambiguous', 'always'], default='always',
                       help='구문 힌트 실행 시점: ambiguous(애매할 때만) | always(항상, 기본)')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)

    # SuPar 안전 로딩 준비 (torch 2.6 weights_only 대응)
    try:
        from sa_aligner import _prepare_supar_safe_loading  # 내부 유틸
        _prepare_supar_safe_loading()
    except Exception:
        pass
    
    # use_parallel 계산 (기존 코드와 호환)
    use_parallel = not args.no_parallel
    
    if args.verbose:
        print("🚀 SA 파일 처리 시작:", args.input_file)
        print(f"⚙️  설정: 임베더={args.embedder}, 병렬={use_parallel}, 워커={args.max_workers}")
        if args.embedder == 'openai':
            print("🔥 OpenAI 병렬 처리 활성화")
        elif args.embedder == 'none':
            print("⚡ 순차 분할 모드 (임베더 미사용)")
        print()
    else:
        print("🚀 SA (Sentence Aligner) 시작")
        print(f"⚙️ 설정: 임베더={args.embedder}, 워커={args.max_workers}, 청크={args.chunk_size}")
        if args.embedder == 'openai':
            print("🔥 OpenAI 병렬 처리 활성화")
        elif args.embedder == 'none':
            print("⚡ 순차 분할 모드 (임베더 미사용, 빠른 처리)")
        else:
            print("📊 BGE 임베더 사용 (기본)")
    # 🔧 기본 모드에서는 시작 메시지 제거 (io_manager에서 처리)
    
    start_time = time.time()
    
    # 🚀 하이브리드 토크나이저 초기화
    try:
        from common.tokenizers import get_siku_tokenizer, get_hybrid_korean_tokenizer
        # 하이브리드 토크나이저 초기화 (중국어: SikuBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)
        if args.verbose:
            print("🏮 SA: 하이브리드 토크나이저 초기화 중...")
        
        # 토크나이저들 미리 로드 (지연 초기화)
        get_siku_tokenizer()  # SikuBERT 초기화
        get_hybrid_korean_tokenizer()  # RoBERTa-Hanja+Kiwipiepy 초기화
        
        if args.verbose:
            print("✅ SA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)")
        else:
            print("SA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)")
    except Exception as e:
        if args.verbose:
            print(f"⚠️ SA: 하이브리드 토크나이저 초기화 실패: {e}")
        else:
            print(f"⚠️ SA: 하이브리드 토크나이저 초기화 실패: {e}")
    
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
            dp_window=args.dp_window,
            distance_decay=args.distance_decay,
            boundary_bonus=args.boundary_bonus,
            particle_bonus=args.particle_bonus,
            length_penalty=args.length_penalty,
            sim_gamma=args.sim_gamma,
            syntax_hints=args.syntax_hints,
            comma_bonus=args.comma_bonus,
            comma_mode=args.comma_mode,
            syntax_when=args.syntax_when,
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
                
                # 🆕 자동 의미 기반 재정렬 후처리
                try:
                    from common.auto_semantic_reorderer import get_auto_semantic_reorderer
                    print("🔄 자동 의미 기반 재정렬 후처리 시작...")
                    
                    reorderer = get_auto_semantic_reorderer()
                    
                    # 현재 결과 파일 읽기
                    result_df = pd.read_excel(args.output_file)
                    
                    # 자동 재정렬 적용 (임계값 조절 가능)
                    print("🧠 임베딩 기반 어순 최적화 중...")
                    result_df['번역문_재정렬'] = result_df.apply(
                        lambda row: reorderer.reorder_translation(
                            str(row['원문']) if '원문' in result_df.columns else str(row.iloc[1]),
                            str(row['번역문']) if '번역문' in result_df.columns else str(row.iloc[2]),
                            threshold=0.25  # 낮은 임계값으로 더 많은 재정렬 허용
                        ), axis=1
                    )
                    
                    # 재정렬 결과를 포함한 파일 저장
                    output_with_reorder = args.output_file.replace('.xlsx', '_재정렬.xlsx')
                    result_df.to_excel(output_with_reorder, index=False)
                    
                    print(f"✨ 자동 의미 재정렬 완료: {output_with_reorder}")
                    
                except Exception as reorder_error:
                    print(f"⚠️ 자동 재정렬 오류 (원본 결과는 유지됨): {reorder_error}")
                
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