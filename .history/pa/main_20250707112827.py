<<<<<<< HEAD
"""PA 메인 실행기 - 통합 아키텍처 버전"""

import sys
import os
import argparse
import json
import pandas as pd
from typing import Optional
=======
import sys
import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pa_processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
>>>>>>> 6faafed ("Refactor")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
sys.stdout.reconfigure(encoding='utf-8')

def check_dependencies():
    """의존성 및 환경 점검"""
    missing = []
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    
    # 선택적 패키지
    optional_missing = []
    try:
        import spacy
    except ImportError:
        optional_missing.append("spacy")
    try:
        import torch
    except ImportError:
        optional_missing.append("torch")
    
    if missing:
<<<<<<< HEAD
        print(f"❌ 필수 패키지 누락: {', '.join(missing)}")
        print("설치 명령: pip install " + " ".join(missing))
=======
        logger.error(f"필수 패키지 누락: {', '.join(missing)}")
        logger.error("설치 명령: pip install " + " ".join(missing))
>>>>>>> 6faafed ("Refactor")
        return False
    
    if optional_missing:
        print(f"⚠️ 선택적 패키지 누락: {', '.join(optional_missing)}")
        print("일부 기능이 제한될 수 있습니다.")
    
    return True

def main(progress_callback=None, stop_flag=None):
<<<<<<< HEAD
    print("PA (Paragraph Aligner) v2.0 시작")
=======
    logger.info("PA (Paragraph Aligner) 시작")
>>>>>>> 6faafed ("Refactor")
    
    # 의존성 확인
    if not check_dependencies():
        return
    
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="PA: Paragraph Aligner v2.0")
    parser.add_argument("input_file", help="입력 파일 (Excel) - 컬럼: source, target")
    parser.add_argument("output_file", help="출력 파일 (Excel)")
    
    # 정렬 모드 설정
    parser.add_argument("--mode", default="hybrid", 
                       choices=["sequential", "semantic", "hybrid"],
                       help="정렬 모드 (기본값: hybrid)")
    
    # 임베더 설정
    parser.add_argument("--embedder", default="bge", 
                       choices=["bge", "openai"],
                       help="임베더 종류 (기본값: bge)")
    parser.add_argument("--device", default="cpu", 
                       choices=["cpu", "cuda"],
                       help="임베더 연산 디바이스 (기본값: cpu)")
    parser.add_argument("--openai-model", default="text-embedding-3-large",
                       help="OpenAI 모델명 (기본값: text-embedding-3-large)")
    parser.add_argument("--openai-api-key", help="OpenAI API 키")
    
    # 품질 임계값
    parser.add_argument("--similarity-threshold", type=float, default=0.3,
                       help="유사도 임계값 (기본값: 0.3)")
    parser.add_argument("--hybrid-threshold", type=float, default=0.8,
                       help="하이브리드 품질 임계값 (기본값: 0.8)")
    
    # 가중치
    parser.add_argument("--sequential-weight", type=float, default=0.4,
                       help="순차적 정렬 가중치 (기본값: 0.4)")
    parser.add_argument("--semantic-weight", type=float, default=0.6,
                       help="의미적 정렬 가중치 (기본값: 0.6)")
    
    # 기타 옵션
    parser.add_argument("--no-integrity-check", action="store_true",
                       help="무결성 검증 비활성화")
    parser.add_argument("--no-quality-stats", action="store_true",
                       help="품질 통계 로깅 비활성화")
    parser.add_argument("--no-progress", action="store_true",
                       help="진행바 비활성화")
    parser.add_argument("--config", help="설정 파일 경로 (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="자세한 로그 출력")
=======
    parser = argparse.ArgumentParser(description="PA: Paragraph Aligner")
    parser.add_argument("input_file", help="입력 파일 (Excel) - 컬럼: 원문, 번역문")
    parser.add_argument("output_file", help="출력 파일 (Excel) - 컬럼: 문단식별자, 원문, 번역문")
    parser.add_argument("--embedder", default="bge", choices=["bge", "openai"])
    parser.add_argument("--threshold", type=float, default=0.3, help="유사도 임계값")
    parser.add_argument("--max-length", type=int, default=150, help="최대 문장 길이")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cpu", help="임베더 연산 디바이스 (cuda/gpu/cpu, 기본값: cpu)")
>>>>>>> 6faafed ("Refactor")

    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.input_file):
<<<<<<< HEAD
        print(f"입력 파일이 없습니다: {args.input_file}")
=======
        logger.error(f"입력 파일이 없습니다: {args.input_file}")
>>>>>>> 6faafed ("Refactor")
        return
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
<<<<<<< HEAD
        print(f"출력 디렉토리 생성: {output_dir}")
    
    try:
        from core.aligner import AlignmentConfig, process_paragraph_file
        from core.io_utils import IOManager
        
        io_manager = IOManager()
        
        # 설정 생성
        if args.config and os.path.exists(args.config):
            print(f"설정 파일 로드: {args.config}")
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = AlignmentConfig(**config_dict)
        else:
            config = AlignmentConfig(
                mode=args.mode,
                embedder_name=args.embedder,
                device=args.device,
                openai_model=args.openai_model,
                openai_api_key=args.openai_api_key,
                similarity_threshold=args.similarity_threshold,
                hybrid_threshold=args.hybrid_threshold,
                sequential_weight=args.sequential_weight,
                semantic_weight=args.semantic_weight,
                verify_integrity=not args.no_integrity_check,
                log_quality_stats=not args.no_quality_stats,
                progress_bar=not args.no_progress
            )
        
        print(f"설정:")
        print(f"  모드: {config.mode}")
        print(f"  임베더: {config.embedder_name}")
        print(f"  디바이스: {config.device}")
        print(f"  유사도 임계값: {config.similarity_threshold}")
        print(f"  하이브리드 임계값: {config.hybrid_threshold}")
        print()

        # 처리 실행
        success = process_paragraph_file(
            args.input_file,
            args.output_file,
            config
        )
        
        if success:
            print(f"PA 처리 완료!")
            print(f"입력: {args.input_file}")
            print(f"출력: {args.output_file}")
        else:
            print("PA 처리 실패!")
            
    except Exception as e:
        print(f"실행 중 오류: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
=======
        logger.info(f"출력 디렉토리 생성: {output_dir}")
    
    try:
        from core.aligner import AlignmentConfig, ParagraphAligner, process_paragraph_file
        import pandas as pd # pandas 임포트 추가

        # AlignmentConfig 객체 생성
        config = AlignmentConfig(
            mode="hybrid", # PA는 하이브리드 모드를 기본으로 사용
            embedder_name=args.embedder,
            device=args.device,
            similarity_threshold=args.threshold,
            openai_model="text-embedding-3-large", # 필요시 args로 받도록 수정
            openai_api_key=os.environ.get("OPENAI_API_KEY"), # 환경변수에서 가져오도록 수정
            log_quality_stats=True,
            progress_bar=True
        )

        result_df = process_paragraph_file(
            input_file=args.input_file,
            output_file=args.output_file,
            config=config
        )
        
        if result_df is not None and isinstance(result_df, pd.DataFrame):
            logger.info(f"\nPA 처리 완료!")
            logger.info(f"입력: {args.input_file}")
            logger.info(f"출력: {args.output_file}")
            logger.info(f"결과: {len(result_df)}개 문장 쌍")
        else:
            logger.error("\nPA 처리 실패!")
            
    except Exception as e:
        logger.error(f"\n실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
>>>>>>> 6faafed ("Refactor")


if __name__ == "__main__":
    main()