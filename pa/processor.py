import sys, os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""PA 메인 프로세서 - 의미적 병합만 사용 (단일 프로세스 버전)"""
from typing import List, Dict
from sentence_splitter import split_target_sentences_advanced
try:
    from aligner import get_embedder_function, improved_align_paragraphs
except ImportError as e:
    print(f"\u274c aligner import 실패: {e}")
    def get_embedder_function(*args, **kwargs):
        print("\u274c 임베더 기능을 사용할 수 없습니다.")
        return None
    def improved_align_paragraphs(*args, **kwargs):
        print("\u274c 의미적 병합 기능을 사용할 수 없습니다.")
        return []
from tqdm import tqdm

def process_paragraph_file(
    input_file, 
    output_file, 
    embedder_name="bge", 
    max_length=150, 
    similarity_threshold=0.3, 
    device="cuda"
):
    """
    입력 엑셀 파일을 읽어 문단 단위로 정렬하고, 결과를 출력 파일로 저장합니다.
    의미적 병합만 지원.
    """
    print(f"📂 PA 파일 처리 시작: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
    except FileNotFoundError:
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return None
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None
    if '원문' not in df.columns or '번역문' not in df.columns:
        print(f"❌ 입력 파일에 '원문', '번역문' 컬럼이 없습니다.")
        return None
    all_results = []
    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc="전체 진행률"):
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        if src_paragraph and tgt_paragraph:
            tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="spacy")
            embed_func = get_embedder_function(embedder_name, device=device)
            alignments = improved_align_paragraphs(
                tgt_sentences,
                src_paragraph,
                embed_func,
                similarity_threshold
            )
            for a in alignments:
                a['문단식별자'] = idx + 1
            all_results.extend(alignments)
    if not all_results:
        print("❌ 결과가 없습니다.")
        return None
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]
    result_df.to_excel(output_file, index=False)
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    return result_df


