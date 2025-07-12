"""PA 메인 실행기 - 완전 버전"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(progress_callback=None, stop_flag=None):
    print("🚀 PA (Paragraph Aligner) 시작")
    
    parser = argparse.ArgumentParser(description="PA: Paragraph Aligner")
    parser.add_argument("input_file", help="입력 파일 (Excel) - 컬럼: 원문, 번역문")
    parser.add_argument("output_file", help="출력 파일 (Excel) - 컬럼: 문단식별자, 원문, 번역문")
    parser.add_argument("--embedder", default="bge", choices=["bge", "openai"])
    parser.add_argument("--threshold", type=float, default=0.3, help="유사도 임계값")
    parser.add_argument("--max-length", type=int, default=150, help="최대 문장 길이")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cuda", help="임베더 연산 디바이스 (cuda/gpu/cpu, 기본값: cuda)")
    parser.add_argument("--splitter", default="spacy", choices=["spacy", "stanza"], help="문장 분할기 선택")
    parser.add_argument("--openai-model", default="text-embedding-3-large", help="OpenAI 임베딩 모델명")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API 키 (미입력시 환경변수 사용)")

    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ 입력 파일이 없습니다: {args.input_file}")
        return
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 출력 디렉토리 생성: {output_dir}")
    
    try:
        from core.aligner import process_paragraph_file, AlignmentConfig

        config = AlignmentConfig(
            mode="semantic",
            embedder_name=args.embedder,
            similarity_threshold=args.threshold,
            device=args.device,
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key
        )

        result_df = process_paragraph_file(
            input_file=args.input_file,
            output_file=args.output_file,
            config=config
        )
        
        if result_df is not None:
            print(f"\n✅ PA 처리 완료!")
            print(f"입력: {args.input_file}")
            print(f"출력: {args.output_file}")
            print(f"결과: {len(result_df)}개 문장 쌍")
        else:
            print("\n❌ PA 처리 실패!")
            
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()