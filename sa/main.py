"""SA (Sentence Aligner) 메인 실행 파일"""

import argparse
import time
import logging
import traceback
import warnings
import sys
import os
from pathlib import Path

# 🔧 common 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# torch.load 보안 경고 전역 무시 (PyTorch 2.6 호환성)
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*vulnerability.*")
warnings.filterwarnings("ignore", message=".*CVE-2025-32434.*")
# BGE-M3 구버전 API 경고 억제
warnings.filterwarnings("ignore", message=".*BGE-M3 구버전 API 감지.*")

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

def _preload_models(use_boundary_model: bool = False, device: str = 'cuda', verbose: bool = False):
    """모델 사전 로드 및 워밍업"""
    try:
        # 1. BGE 임베더 로드
        if verbose: print("  - BGE 임베더 로드 중...")
        from common.embedders.bge import get_embedding_manager
        manager = get_embedding_manager()
        manager._load_model()
        
        # 워밍업 (더미 데이터로 첫 실행 지연 제거)
        manager.compute_embeddings_with_cache(["워밍업"])
        
        # 2. SA 처리 함수 캐싱
        if verbose: print("  - SA Aligner 모듈 로드 중...")
        from sa.sa_aligner import process_single_row
        from sa.io_manager import safe_process_sa_row
        safe_process_sa_row._process_func = process_single_row
        
        # 3. 경계 모델 로드 (옵션)
        if use_boundary_model:
            if verbose: print("  - Cross-Attention 경계 모델 로드 중...")
            from common.sa_crossattn_boundary_loader import get_crossattn_boundary_tagger
            safe_process_sa_row._boundary_model = get_crossattn_boundary_tagger(device=device)
            # 워밍업
            try:
                safe_process_sa_row._boundary_model.segment_text("원문", "번역문")
            except:
                pass
            
            # Alignment 모델
            from common.alignment_model_loader import AlignmentMatcher
            alignment_model_path = Path("models/dual_encoder_alignment_sa.pt")
            if alignment_model_path.exists():
                if verbose: print("  - Alignment 모델 로드 중...")
                safe_process_sa_row._alignment_model = AlignmentMatcher(model_path=alignment_model_path, device=device)
                
    except Exception as e:
        print(f"⚠️ 모델 사전 로드 실패 (무시됨): {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='SA: 한문-한국어 문장 분할 도구')
    
    # 필수 인수 (기본값 제공)
    parser.add_argument('input_file', nargs='?', default='input.xlsx', help='입력 엑셀 파일 경로 (기본: input.xlsx)')
    parser.add_argument('output_file', nargs='?', default='output.xlsx', help='출력 엑셀 파일 경로 (기본: output.xlsx)')
    
    # 선택적 인수
    parser.add_argument('--embedder', choices=['bge', 'none'], default='bge',
                       help='임베더 선택 (기본: bge, 순차분할: --embedder none)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='최대 워커 수 (기본: 4)')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='청크 크기 (기본: 200, 대용량 처리 시 300 권장)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='병렬 처리 비활성화')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    # 기본 출력 디렉터리 자동 설정 옵션
    parser.add_argument('--default-output-dir', type=str, default=None,
                       help='출력 디렉터리를 지정하면 입력 파일명에서 서종명을 추출해 <dir>/<book>_output.xlsx로 자동 저장')
    
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
    parser.add_argument('--dp-window', type=int, default=3,
                       help='DP 예상 위치 허용 창 크기 (기본: 3, 범위 넓어짐)')
    parser.add_argument('--distance-decay', type=float, default=0.03,
                       help='위치 거리 감쇠 알파 (기본: 0.03, 페널티 완화)')
    parser.add_argument('--boundary-bonus', type=float, default=0.2,
                       help='문장 경계 보너스 (기본: 0.2)')
    parser.add_argument('--particle-bonus', type=float, default=0.3,
                       help='토씨 경계 보너스 (기본: 0.3, 한글 강화)')
    parser.add_argument('--length-penalty', type=float, default=0.08,
                       help='세그먼트 길이 패널티 알파 (기본: 0.08, 페널티 완화)')
    parser.add_argument('--sim-gamma', type=float, default=1.0,
                       help='유사도 샤프닝 지수 (기본: 1.0, 선형)')
    # 문장 내부 경계 힌트(옵션)
    parser.add_argument('--syntax-hints', choices=['none', 'ko', 'zh', 'both'], default='ko',
                       help='구문 파서 힌트 사용 (기본: ko, 한국어 강화)')
    parser.add_argument('--comma-bonus', type=float, default=0.2,
                       help='콤마(,) 경계 보너스 (기본: 0.2, 강화됨)')
    parser.add_argument('--comma-mode', choices=['soft', 'strict'], default='soft',
                       help='콤마 경계 모드: soft(나열 제외) | strict(전부 적용)')
    parser.add_argument('--syntax-when', choices=['ambiguous', 'always'], default='always',
                       help='구문 힌트 실행 시점: ambiguous(애매할 때만) | always(항상, 기본)')
    
    # 하이브리드 임베딩 옵션 (기본: 활성화)
    parser.add_argument('--no-hybrid-embed', action='store_true',
                       help='하이브리드 임베딩 비활성화 (기본: 활성화, 한자/한글 세분화)')
    
    # 새로운 모델 옵션
    parser.add_argument('--use-boundary-model', action='store_true',
                       help='새로운 boundary_multitask + alignment 모델 사용')
    parser.add_argument('--boundary-threshold', type=float, default=0.55,
                       help='경계 모델 threshold (기본: 0.55, 범위: 0.0-1.0)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='디바이스 (기본: cuda, GPU 미지원시 자동 cpu)')
    parser.add_argument('--preload-models', action='store_true',
                       help='🆕 모델 사전 로드 (첫 문장 처리 전 로드, 대용량 처리 시 권장)')
    
    args = parser.parse_args()

    # 입력 파일명에서 서종명 추출 유틸리티
    def _derive_book_name_from_input(path_str: str) -> str:
        p = Path(path_str)
        stem = p.stem  # 예: 당송팔대가문초한유3_문장병렬
        # 알려진 접미사 제거
        for suffix in ['_문장병렬', '_문단병렬', '_구병렬', '_para_output', '_output']:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        # 파일명에 서종명이 없으면 상위 폴더명 사용
        if not stem or stem == p.stem:
            try:
                parent_name = p.parent.name
                if parent_name:
                    return parent_name
            except Exception:
                pass
        return stem

    # 기본 출력 디렉터리 지정 시 출력 경로 자동 설정
    if args.default_output_dir:
        book_name = _derive_book_name_from_input(args.input_file)
        auto_out = Path(args.default_output_dir) / f"{book_name}_output.xlsx"
        args.output_file = str(auto_out)
    
    # 로깅 설정
    setup_logging(args.verbose)

    # SuPar 안전 로딩 준비 (torch 2.6 weights_only 대응)
    try:
        from sa.sa_aligner import _prepare_supar_safe_loading  # 패키지 경로로 수정
        _prepare_supar_safe_loading()
    except Exception:
        pass
    
    # use_parallel 계산 (기존 코드와 호환)
    use_parallel = not args.no_parallel
    
    if args.verbose:
        print("🚀 SA 파일 처리 시작:", args.input_file)
        print(f"⚙️  설정: 임베더={args.embedder}, 병렬={use_parallel}, 워커={args.max_workers}")
        if args.embedder == 'none':
            print("⚡ 순차 분할 모드 (임베더 미사용)")
        print()
    else:
        print("🚀 SA (Sentence Aligner) 시작")
        print(f"⚙️ 설정: 임베더={args.embedder}, 워커={args.max_workers}, 청크={args.chunk_size}")
        if args.embedder == 'none':
            print("⚡ 순차 분할 모드 (임베더 미사용, 빠른 처리)")
        else:
            print("📊 BGE 임베더 사용 (기본)")
    # 🔧 기본 모드에서는 시작 메시지 제거 (io_manager에서 처리)
    
    start_time = time.time()
    
    # 🆕 모델 사전 로드 (--preload-models 옵션)
    if args.preload_models:
        print("🔄 모델 사전 로드 중...")
        _preload_models(args.use_boundary_model, args.device, args.verbose)
        print("✅ 모델 로드 완료")
    
    try:
        # io_manager의 process_file 함수 호출
        from sa.io_manager import process_file
        
        # 출력 디렉터리 생성 보장
        try:
            out_parent = Path(args.output_file).parent
            if str(out_parent):
                os.makedirs(out_parent, exist_ok=True)
        except Exception:
            pass

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
            hybrid_embed=not args.no_hybrid_embed,
            verbose=args.verbose,
            use_boundary_model=args.use_boundary_model,
            boundary_threshold=args.boundary_threshold,
            device=args.device
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