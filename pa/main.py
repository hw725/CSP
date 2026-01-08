"""PA (Paragraph Aligner) 메인 실행기"""

import os
import sys
import argparse
import time
from pathlib import Path

# 프로젝트 루트와 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='PA: 한문-한국어 문단 정렬 도구')
    
    # 위치 인수
    parser.add_argument('input_file', help='입력 Excel 파일 경로')
    parser.add_argument('output_file', help='출력 Excel 파일 경로')
    
    # 선택적 인수들
    parser.add_argument('--embedder', default='bge', choices=['bge', 'openai'],
                       help='임베더 선택 (기본값: bge)')
    parser.add_argument('--max-length', type=int, default=180,
                       help='최대 문장 길이 (기본값: 180)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='유사도 임계값 (기본값: 0.7)')
    parser.add_argument('--openai-model', default='text-embedding-3-large',
                       help='OpenAI 모델명')
    parser.add_argument('--openai-api-key', 
                       help='OpenAI API 키')
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
    print()
    
    # 🆕 하이브리드 토크나이저 초기화
    try:
        from common.tokenizers import get_siku_tokenizer, get_anchi_tokenizer, get_hybrid_korean_tokenizer, detect_chinese_period
        
        # 토크나이저들 미리 로드 (지연 초기화)
        get_siku_tokenizer()  # SikuBERT 초기화
        get_anchi_tokenizer()  # AnchiBERT 초기화
        get_hybrid_korean_tokenizer()  # RoBERTa-Hanja+Kiwipiepy 초기화
        
        print("PA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)")
    except Exception as e:
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
            verbose=args.verbose
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