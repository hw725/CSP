"""SA (Sentence Aligner) 최적화 버전 메인 실행 파일"""

import argparse
import time
import logging
import traceback
import warnings
import os
import sys
from pathlib import Path

# 최적화 설정 로더 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.text_optimization_configs import get_sa_config_by_jti

# torch.load 보안 경고 전역 무시 (PyTorch 2.6 호환성)
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*vulnerability.*")
warnings.filterwarnings("ignore", message=".*CVE-2025-32434.*")

def extract_jti_code_from_filename(input_file: str) -> str:
    """파일명에서 JTI 코드 추출"""
    try:
        filename = Path(input_file).stem.lower()
        
        # JTI 코드 패턴들
        import re
        patterns = [
            r'(jti_[0-9a-z]+)',  # jti_4c0201 형식
            r'([0-9a-z]+-.*?번역문)',  # 4c0201-당송팔대가문초한유1_번역문 형식  
            r'([0-9][a-z][0-9]+)',  # 4c0201 형식 (단독)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                code = match.group(1)
                if code.startswith('jti_'):
                    return code[4:]  # 'jti_' 제거
                return code
                
        # 알려진 텍스트 이름으로 매핑
        text_mappings = {
            '관자': '3a0101',
            '논어': '4j0201', 
            '대학': '4j0202',
            '맹자': '4j0201',
            '사기': '2k0101',
            '한서': '2k0201',
            '후한서': '2k0301', 
            '삼국지': '2k0401',
            '진서': '2k0501',
            '당서': '2k0601',
            '당송팔대가': '4c0201',
            '한유': '4c0201',
            '육도직해': '3b0301',
            '안씨가훈': '3j0201',
            '양자법언': '3j0301'
        }
        
        for name, code in text_mappings.items():
            if name in filename:
                return code
                
        return None
        
    except Exception:
        return None

def setup_logging(verbose: bool = False):
    """로깅 설정"""
    if verbose:
        level = logging.DEBUG
        format_str = '%(asctime)s - %(levelname)s:%(name)s:%(message)s'
    else:
        level = logging.WARNING
        format_str = '%(levelname)s: %(message)s'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler()]
    )
    
    if not verbose:
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['DATASETS_VERBOSITY'] = 'error'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
        logging.getLogger('datasets').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('FlagEmbedding').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('punctuation').setLevel(logging.ERROR)
        logging.getLogger('io_manager').setLevel(logging.ERROR)
        logging.getLogger('common.tokenizers').setLevel(logging.ERROR)
        
        import warnings
        warnings.filterwarnings("ignore")

def main():
    """메인 실행 함수 - 최적화 설정 통합"""
    parser = argparse.ArgumentParser(description='SA: 한문-한국어 문장 분할 도구 (최적화 버전)')
    
    # 필수 인수
    parser.add_argument('input_file', nargs='?', default='input.xlsx', help='입력 엑셀 파일 경로')
    parser.add_argument('output_file', nargs='?', default='output.xlsx', help='출력 엑셀 파일 경로')
    
    # 최적화 옵션
    parser.add_argument('--optimize', action='store_true',
                       help='텍스트별 최적화 설정 자동 적용')
    parser.add_argument('--jti-code', type=str,
                       help='JTI 코드 직접 지정 (자동 감지 무시)')
    
    # 기존 옵션들 (최적화 설정으로 덮어씌워질 수 있음)
    parser.add_argument('--embedder', choices=['bge', 'openai', 'none'], default='bge')
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--no-parallel', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    # 토크나이저 옵션
    parser.add_argument('--min-src-tokens', type=int, default=1)
    parser.add_argument('--max-src-tokens', type=int, default=20)
    parser.add_argument('--min-tgt-tokens', type=int, default=1)
    parser.add_argument('--max-tgt-tokens', type=int, default=40)
    
    # SA 하이퍼파라미터들
    parser.add_argument('--dp-window', type=int, default=2)
    parser.add_argument('--distance-decay', type=float, default=0.9)
    parser.add_argument('--boundary-bonus', type=float, default=0.3)
    parser.add_argument('--particle-bonus', type=float, default=0.1)
    parser.add_argument('--length-penalty', type=float, default=0.05)
    parser.add_argument('--sim-gamma', type=float, default=1.5)
    parser.add_argument('--syntax-hints', choices=['none', 'ko', 'zh', 'both'], default='both')
    parser.add_argument('--comma-bonus', type=float, default=0.0)
    parser.add_argument('--comma-mode', choices=['soft', 'strict'], default='soft')
    parser.add_argument('--syntax-when', choices=['ambiguous', 'always'], default='always')
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # JTI 코드 추출 및 최적화 설정 적용
    if args.optimize:
        jti_code = args.jti_code or extract_jti_code_from_filename(args.input_file)
        
        if jti_code:
            config = get_sa_config_by_jti(jti_code)
            if config:
                print(f"🎯 텍스트 최적화: JTI {jti_code} → {config.get('text_name', '알 수 없음')}")
                print(f"   성능 등급: {config.get('performance_grade', 'N/A')}")
                print(f"   목표 F1: {config.get('target_f1', 'N/A')}")
                
                # SA 설정 적용
                sa_config = config.get('sa_config', {})
                if sa_config:
                    # 기존 인수값을 최적화 설정으로 덮어씀
                    for key, value in sa_config.items():
                        if key == 'min_src_tokens':
                            args.min_src_tokens = value
                        elif key == 'max_src_tokens':
                            args.max_src_tokens = value
                        elif key == 'min_tgt_tokens':
                            args.min_tgt_tokens = value
                        elif key == 'max_tgt_tokens':
                            args.max_tgt_tokens = value
                        elif key == 'dp_window':
                            args.dp_window = value
                        elif key == 'distance_decay':
                            args.distance_decay = value
                        elif key == 'boundary_bonus':
                            args.boundary_bonus = value
                        elif key == 'particle_bonus':
                            args.particle_bonus = value
                        elif key == 'length_penalty':
                            args.length_penalty = value
                        elif key == 'sim_gamma':
                            args.sim_gamma = value
                        elif key == 'syntax_hints':
                            args.syntax_hints = value
                        elif key == 'comma_bonus':
                            args.comma_bonus = value
                        elif key == 'comma_mode':
                            args.comma_mode = value
                        elif key == 'syntax_when':
                            args.syntax_when = value
                    
                    print(f"   SA 최적화 파라미터 {len(sa_config)}개 적용됨")
                else:
                    print("   SA 최적화 설정 없음 (기본값 사용)")
            else:
                print(f"⚠️ JTI {jti_code}에 대한 최적화 설정을 찾을 수 없음")
        else:
            print("⚠️ 파일명에서 JTI 코드를 추출할 수 없어 기본 설정 사용")
    
    # SuPar 안전 로딩
    try:
        from sa_aligner import _prepare_supar_safe_loading
        _prepare_supar_safe_loading()
    except Exception:
        pass
    
    use_parallel = not args.no_parallel
    
    if args.verbose:
        print("🚀 SA 파일 처리 시작:", args.input_file)
        print(f"⚙️  설정: 임베더={args.embedder}, 병렬={use_parallel}, 워커={args.max_workers}")
    else:
        print("🚀 SA (Sentence Aligner) 최적화 버전 시작")
        print(f"⚙️ 설정: 임베더={args.embedder}, 워커={args.max_workers}")
    
    start_time = time.time()
    
    # 하이브리드 토크나이저 초기화
    try:
        from common.tokenizers import get_siku_tokenizer, get_hybrid_korean_tokenizer
        if args.verbose:
            print("🏮 SA: 하이브리드 토크나이저 초기화 중...")
        
        get_siku_tokenizer()
        get_hybrid_korean_tokenizer()
        
        if args.verbose:
            print("✅ SA: 하이브리드 토크나이저 초기화 완료")
        else:
            print("SA: 하이브리드 토크나이저 초기화 완료")
    except Exception as e:
        print(f"⚠️ SA: 하이브리드 토크나이저 초기화 실패: {e}")
    
    try:
        from io_manager import process_file
        
        success = process_file(
            input_file=args.input_file,
            output_file=args.output_file,
            embedder_name=args.embedder,
            max_workers=args.max_workers,
            chunk_size=args.chunk_size,
            use_parallel=use_parallel,
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
            print()
            print("🎉 처리 완료!")
            print(f"⏱️  처리 시간: {elapsed_time:.2f}초")
            
            if success:
                print(f"✅ 결과 파일: {args.output_file}")
                
                # 자동 의미 기반 재정렬 후처리
                try:
                    from common.auto_semantic_reorderer import get_auto_semantic_reorderer
                    print("🔄 자동 의미 기반 재정렬 후처리 시작...")
                    
                    reorderer = get_auto_semantic_reorderer()
                    
                    import pandas as pd
                    result_df = pd.read_excel(args.output_file)
                    
                    print("🧠 임베딩 기반 어순 최적화 중...")
                    result_df['번역문_재정렬'] = result_df.apply(
                        lambda row: reorderer.reorder_translation(
                            str(row['원문']) if '원문' in result_df.columns else str(row.iloc[1]),
                            str(row['번역문']) if '번역문' in result_df.columns else str(row.iloc[2]),
                            threshold=0.25
                        ), axis=1
                    )
                    
                    output_with_reorder = args.output_file.replace('.xlsx', '_재정렬.xlsx')
                    result_df.to_excel(output_with_reorder, index=False)
                    
                    print(f"✨ 자동 의미 재정렬 완료: {output_with_reorder}")
                    
                except Exception as reorder_error:
                    print(f"⚠️ 자동 재정렬 오류: {reorder_error}")
                
                # 결과 통계
                try:
                    import pandas as pd
                    result_df = pd.read_excel(args.output_file)
                    print(f"📊 처리 결과: {len(result_df)}개 문장")
                    
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