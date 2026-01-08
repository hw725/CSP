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
    parser.add_argument('--max-workers', type=int, default=4,
                       help='OpenAI API 병렬 워커 수 (기본: 4, OpenAI 전용)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='OpenAI API 배치 크기 (기본: 50, OpenAI 전용)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='디바이스 (기본: cuda, GPU 미지원시 자동 cpu)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # SA와 동일한 경고 숨김 설정
    if not args.verbose:
        import os
        import warnings
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        warnings.filterwarnings("ignore")
    
    print("🚀 PA (Paragraph Aligner) 시작")
    print(f"⚙️ 설정: 임베더={args.embedder}, 병렬 워커={args.max_workers}, 배치 크기={args.batch_size}")
    if args.embedder == 'openai':
        print("🔥 OpenAI 병렬 처리 활성화")
    elif args.embedder == 'none':
        print("⚡ 순차 분할 모드 (임베더 미사용, 빠른 처리)")
    else:
        print("📊 BGE 임베더 사용 (기본)")
    print()
    
    # 하이브리드 토크나이저 초기화
    try:
        from common.tokenizers import get_siku_tokenizer, get_hybrid_korean_tokenizer
        
        if args.verbose:
            print("🏮 PA: 하이브리드 토크나이저 초기화 중...")
        
        # SikuBERT 초기화
        get_siku_tokenizer()
        # 한국어 토크나이저 초기화
        get_hybrid_korean_tokenizer()
        
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
            max_workers=args.max_workers,  # 🚀 병렬 워커 수 전달
            batch_size=args.batch_size,    # 🚀 배치 크기 전달
            verbose=args.verbose,
            device=args.device             # 🚀 device 전달
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