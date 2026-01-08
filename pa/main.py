"""PA (Paragraph Aligner) 메인 실행기"""

import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import os
import argparse
import time
import warnings
import random

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# torch은 Docker 환경에서만 필수입니다. 로컬(Windows)에서는 없을 수 있으므로 안전하게 처리합니다.
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class _TorchShim:
        pass
    torch = _TorchShim()  # type: ignore

# PyTorch 보안 경고 완전 비활성화
os.environ['TORCH_FORCE_WEIGHTS_ONLY'] = 'False'
os.environ['HF_HUB_DISABLE_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 모든 경고 필터링
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# PyTorch 2.6 torch.load 호환성 설정 (torch가 있을 때만)
if TORCH_AVAILABLE and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    try:
        # SuPar 모델을 위한 안전한 글로벌 추가
        from supar.utils.config import Config
        torch.serialization.add_safe_globals([Config])
    except Exception:
        pass

# torch.load에 대한 추가 보안 경고 억제
def suppress_torch_warnings():
    """PyTorch 보안 경고를 완전히 억제 (torch가 있을 때만)"""
    if not TORCH_AVAILABLE:
        return
    import logging
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # torch.load monkey patching
    original_load = torch.load
    def safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load

# torch가 있을 때만 경고 억제 활성화
suppress_torch_warnings()

# 프로젝트 루트와 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(current_dir))

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='PA: 한문-한국어 문단 정렬 도구')
    
    # 위치 인수
    parser.add_argument('input_file', help='입력 Excel 파일 경로')
    parser.add_argument('output_file', help='출력 Excel 파일 경로')
    
    # 선택적 인수들
    parser.add_argument('--embedder', default='bge', choices=['bge', 'openai', 'none'],
                       help='임베더 선택 (기본값: bge, OpenAI: --embedder openai, 순차분할: --embedder none)')
    parser.add_argument('--max-length', type=int, default=180,
                       help='최대 문장 길이 (기본값: 180)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='유사도 임계값 (기본값: 0.7)')
    parser.add_argument('--openai-model', default='text-embedding-3-large',
                       help='OpenAI 모델명')
    parser.add_argument('--openai-api-key', 
                       help='OpenAI API 키')
    
    # 🚀 병렬 처리 옵션 추가
    parser.add_argument('--max-workers', type=int, default=16,
                       help='병렬 워커 수 (기본: 16, BGE/OpenAI 모두 지원)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='배치 크기 (기본: 256, BGE/OpenAI 모두 지원)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='디바이스 (기본: cuda, GPU 미지원시 자동 cpu)')
    
    parser.add_argument('--use-boundary-model', action='store_true',
                       help='새로운 boundary_multitask + alignment 모델 사용')
    
    parser.add_argument('--boundary-threshold', type=float, default=0.70,
                       help='경계 모델 threshold (기본: 0.70, 범위: 0.0-1.0)')

    parser.add_argument(
        '--boundary-min-len',
        type=int,
        default=None,
        help='경계 모델 디코딩 min_len 오버라이드(task=pa, 기본 20). 지정 시 경계 후보 밀도에 영향',
    )

    parser.add_argument(
        '--enable-refine',
        action='store_true',
        help='(실험용) use-boundary-model에서 인접/DP refine의 이동 폭을 확장합니다(기본 1토큰 → 4토큰).',
    )

    parser.add_argument(
        '--disable-adjacent-boundary-refine',
        action='store_true',
        help='(실험용) boundary/supar 선택 시 수행되는 인접 경계 로컬 교정(_refine_adjacent_boundaries)을 비활성화.',
    )

    parser.add_argument(
        '--enable-src-marker-boundary-bonus',
        action='store_true',
        help=(
            '(실험용) 원문 내 현토(한글 marker) 패턴을 경계 선택 tie-break에 보너스로 반영합니다. '
            '원문 토큰 끝의 한글 marker(예: “也에”, “之者가”)를 이용해 경계 후보를 약하게 선호합니다.'
        ),
    )

    parser.add_argument(
        '--enable-src-marker-whitespace-dp-bonus',
        action='store_true',
        help=(
            '(실험용) whitespace_dp(어절 경계 DP 분할)에서도 원문 내 현토(한글 marker) 패턴을 '
            '후보 컷/DP 점수에 약하게 반영합니다. '
            'A/B 실험용으로, boundary-bonus는 동일하게 ON인 상태에서 이 옵션만 ON/OFF 하세요.'
        ),
    )

    parser.add_argument(
        '--trace-stages-jsonl',
        default=None,
        help=(
            'PA 단계별(경계 후보/매칭/후처리/restore 등) 중간 결과를 JSONL로 저장합니다. '
            '드리프트가 시작되는 단계를 찾기 위한 진단용 옵션입니다. '
            '미지정 시 환경변수 CSP_PA_TRACE_STAGES_JSONL을 사용합니다.'
        ),
    )
    
    parser.add_argument('--verbose', action='store_true',
                       help='상세 로그 출력')

    # 실험 재현성 옵션 (기본: 기존 동작 유지)
    parser.add_argument('--seed', type=int, default=None,
                       help='재현성 seed (지정 시 random/torch/numpy seed 고정)')
    parser.add_argument('--deterministic', action='store_true',
                       help='가능한 범위에서 deterministic 모드 활성화(속도 저하 가능)')
    
    args = parser.parse_args()

    # deterministic 설정은 CUDA 초기화(예: torch.cuda.is_available)보다 먼저 적용해야 효과가 있습니다.
    if args.deterministic and TORCH_AVAILABLE:
        # cuBLAS 결정성(가능한 경우) - 이미 설정돼 있으면 유지
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
        try:
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except TypeError:
                    torch.use_deterministic_algorithms(True)
        except Exception:
            # 일부 연산이 결정성을 지원하지 않으면 예외가 날 수 있으므로 조용히 폴백
            pass
        try:
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(1)
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(1)
        except Exception:
            pass

    # 재현성 seed 설정 (옵션)
    if args.seed is not None:
        random.seed(args.seed)
        if NUMPY_AVAILABLE:
            try:
                np.random.seed(args.seed)
            except Exception:
                pass
        if TORCH_AVAILABLE:
            try:
                torch.manual_seed(args.seed)
                if args.device == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            except Exception:
                pass
    
    # SA와 동일한 경고 숨김 설정
    if not args.verbose:
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        warnings.filterwarnings("ignore")
    
    print("🚀 PA (Paragraph Aligner) 시작")
    print(f"⚙️ 설정: 임베더={args.embedder}, 병렬 워커={args.max_workers}, 배치 크기={args.batch_size}")
    if args.use_boundary_model:
        extra = ""
        if args.boundary_min_len is not None:
            extra = f", min_len={args.boundary_min_len}"
        print(f"🔬 모델 모드: boundary_multitask + alignment (threshold={args.boundary_threshold}{extra})")
    if args.embedder == 'openai':
        print("🔥 OpenAI 병렬 처리 활성화")
    elif args.embedder == 'none':
        print("⚡ 순차 분할 모드 (임베더 미사용, 빠른 처리)")
    else:
        print("📊 BGE 임베더 사용 (기본)")
    print()

    # 사용자 요구사항: 임베더는 항상 bge
    if args.embedder != 'bge':
        raise SystemExit(f"PA는 --embedder bge만 허용합니다. 현재: {args.embedder}")
    
    # 하이브리드 토크나이저 초기화
    tokenizer_init_ok = False
    try:
        from common.tokenizers import get_siku_tokenizer, get_hybrid_korean_tokenizer
        
        if args.verbose:
            print("🏮 PA: 하이브리드 토크나이저 초기화 중...")
        
        # SikuBERT 초기화
        get_siku_tokenizer()
        # 한국어 토크나이저 초기화
        get_hybrid_korean_tokenizer()

        tokenizer_init_ok = True
        
        if args.verbose:
            print("✅ PA: 하이브리드 토크나이저 초기화 완료 (원문: SikuBERT+Kiwipiepy, 번역문: RoBERTa-Hanja+Kiwipiepy)")
        else:
            print("PA: 하이브리드 토크나이저 초기화 완료 (원문: SikuBERT+Kiwipiepy, 번역문: RoBERTa-Hanja+Kiwipiepy)")
    except Exception as e:
        if args.verbose:
            print(f"⚠️ PA: 하이브리드 토크나이저 초기화 실패: {e}")
        else:
            print(f"⚠️ PA: 하이브리드 토크나이저 초기화 실패: {e}")
    
    try:
        # 현재 디렉토리에서 processor 직접 import
        from processor import process_paragraph_file
        
        start_time = time.time()
        
        # 파일 처리 실행
        result_df = process_paragraph_file(
            input_file=args.input_file,
            output_file=args.output_file,
            embedder_name=args.embedder,
            max_length=args.max_length,
            similarity_threshold=args.threshold,
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            verbose=args.verbose,
            device=args.device,
            use_boundary_model=args.use_boundary_model,
            boundary_threshold=args.boundary_threshold,
            boundary_min_len=args.boundary_min_len,
            enable_refine=args.enable_refine,
            enable_adjacent_boundary_refine=(not args.disable_adjacent_boundary_refine),
            enable_src_marker_boundary_bonus=args.enable_src_marker_boundary_bonus,
            enable_src_marker_whitespace_dp_bonus=args.enable_src_marker_whitespace_dp_bonus,
            trace_stages_path=args.trace_stages_jsonl,
            seed=args.seed,
            tokenizer_init_ok=tokenizer_init_ok,
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result_df is not None:
            print(f"\n🎉 PA 처리 완료!")
            print(f"⏱️  총 처리 시간: {processing_time:.2f}초")
            print(f"📁 출력 파일: {args.output_file}")
            print(f"📊 생성된 문장 쌍: {len(result_df)}개")
            return True
        else:
            print(f"\n❌ PA 처리 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 실행 중 오류: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)