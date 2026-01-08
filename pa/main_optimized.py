"""PA (Paragraph Aligner) 최적화 버전 메인 실행기"""

import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import os
import argparse
import time
import warnings

# 최적화 설정 로더 추가
from config.text_optimization_configs import get_pa_config_by_jti

# torch 안전 처리
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class _TorchShim:
        pass
    torch = _TorchShim()

# PyTorch 보안 경고 완전 비활성화
os.environ['TORCH_FORCE_WEIGHTS_ONLY'] = 'False'
os.environ['HF_HUB_DISABLE_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# PyTorch 2.6 호환성
if TORCH_AVAILABLE and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from supar.utils.config import Config
        torch.serialization.add_safe_globals([Config])
    except Exception:
        pass

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

def suppress_torch_warnings():
    """PyTorch 보안 경고 완전 억제"""
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

suppress_torch_warnings()
sys.path.insert(0, str(current_dir))

def main():
    """메인 실행 함수 - 최적화 설정 통합"""
    parser = argparse.ArgumentParser(description='PA: 한문-한국어 문단 정렬 도구 (최적화 버전)')
    
    # 위치 인수
    parser.add_argument('input_file', help='입력 Excel 파일 경로')
    parser.add_argument('output_file', help='출력 Excel 파일 경로')
    
    # 최적화 옵션
    parser.add_argument('--optimize', action='store_true',
                       help='텍스트별 최적화 설정 자동 적용')
    parser.add_argument('--jti-code', type=str,
                       help='JTI 코드 직접 지정 (자동 감지 무시)')
    
    # 기존 옵션들
    parser.add_argument('--embedder', default='bge', choices=['bge', 'openai', 'none'])
    parser.add_argument('--max-length', type=int, default=180)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--openai-model', default='text-embedding-3-large')
    parser.add_argument('--openai-api-key')
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # JTI 코드 추출 및 최적화 설정 적용
    if args.optimize:
        jti_code = args.jti_code or extract_jti_code_from_filename(args.input_file)
        
        if jti_code:
            config = get_pa_config_by_jti(jti_code)
            if config:
                print(f"🎯 텍스트 최적화: JTI {jti_code} → {config.get('text_name', '알 수 없음')}")
                print(f"   성능 등급: {config.get('performance_grade', 'N/A')}")
                print(f"   목표 F1: {config.get('target_f1', 'N/A')}")
                
                # PA 설정 적용
                pa_config = config.get('pa_config', {})
                if pa_config:
                    # 기존 인수값을 최적화 설정으로 덮어씀
                    for key, value in pa_config.items():
                        if key == 'max_length':
                            args.max_length = value
                        elif key == 'threshold':
                            args.threshold = value
                        elif key == 'max_workers':
                            args.max_workers = value
                        elif key == 'batch_size':
                            args.batch_size = value
                    
                    print(f"   PA 최적화 파라미터 {len(pa_config)}개 적용됨")
                else:
                    print("   PA 최적화 설정 없음 (기본값 사용)")
            else:
                print(f"⚠️ JTI {jti_code}에 대한 최적화 설정을 찾을 수 없음")
        else:
            print("⚠️ 파일명에서 JTI 코드를 추출할 수 없어 기본 설정 사용")
    
    # 경고 숨김 설정
    if not args.verbose:
        import os
        import warnings
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        warnings.filterwarnings("ignore")
    
    print("🚀 PA (Paragraph Aligner) 최적화 버전 시작")
    print(f"⚙️ 설정: 임베더={args.embedder}, 워커={args.max_workers}, 배치={args.batch_size}")
    print(f"   길이 제한: {args.max_length}, 임계값: {args.threshold}")
    
    if args.embedder == 'openai':
        print("🔥 OpenAI 병렬 처리 활성화")
    elif args.embedder == 'none':
        print("⚡ 순차 분할 모드 (임베더 미사용)")
    else:
        print("📊 BGE 임베더 사용 (기본)")
    print()
    
    # 하이브리드 토크나이저 초기화
    try:
        from common.tokenizers import get_siku_tokenizer, get_hybrid_korean_tokenizer
        
        try:
            get_siku_tokenizer()
            siku_ok = True
        except Exception as siku_error:
            print(f"⚠️ SikuBERT 로딩 실패: {str(siku_error)[:100]}...")
            print("   → SikuBERT 없이 계속 진행 (BGE-M3로 대체)")
            siku_ok = False
        
        get_hybrid_korean_tokenizer()
        
        if siku_ok:
            print("✅ PA: 하이브리드 토크나이저 초기화 완료")
        else:
            print("✅ PA: 부분 토크나이저 초기화 완료 (BGE-M3 대체)")
            
    except Exception as e:
        print(f"⚠️ PA: 토크나이저 초기화 실패: {e}")
        print("   → 기본 임베딩으로 대체하여 계속 진행")
    
    try:
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
            verbose=args.verbose
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result_df is not None:
            print(f"\n🎉 PA 처리 완료!")
            print(f"⏱️  총 처리 시간: {processing_time:.2f}초")
            print(f"📁 출력 파일: {args.output_file}")
            print(f"📊 생성된 문장 쌍: {len(result_df)}개")
            
            # 최적화 성능 검증
            if args.optimize and jti_code:
                try:
                    # 간단한 성능 추정 (실제 평가는 별도 필요)
                    import pandas as pd
                    avg_score = result_df.get('유사도', pd.Series([0])).mean() if '유사도' in result_df.columns else 0
                    print(f"📈 평균 유사도: {avg_score:.4f}")
                    
                    expected_f1 = config.get('target_f1', 'N/A')
                    if expected_f1 != 'N/A':
                        print(f"🎯 목표 F1: {expected_f1}")
                        
                except Exception:
                    pass
            
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