"""PA 메인 프로세서 - import 문제 해결"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 로컬 모듈 import
from sentence_splitter import split_target_sentences_advanced

try:
    from aligner import get_embedder_function, improved_align_paragraphs
except ImportError as e:
    print(f"❌ aligner import 실패: {e}")
    
    def get_embedder_function(*args, **kwargs):
        print("❌ 임베더 기능을 사용할 수 없습니다.")
        return None
    
    def improved_align_paragraphs(*args, **kwargs):
        print("❌ 의미적 병합 기능을 사용할 수 없습니다.")
        return []

def process_paragraph_file(
    input_file, 
    output_file, 
    embedder_name="bge", 
    max_length=150, 
    similarity_threshold=0.3, 
    openai_model=None,
    openai_api_key=None,
    verbose=False,
    device="cpu"
):
    """입력 엑셀 파일을 읽어 문단 단위로 정렬하고, 결과를 출력 파일로 저장"""
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
    
    # 필수 컬럼 확인
    required_columns = ['원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ 입력 파일에 필수 컬럼이 없습니다: {missing_columns}")
        print(f"📋 현재 컬럼: {list(df.columns)}")
        return None
    
    all_results = []
    total = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="📊 문단 처리 진행률"):
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        
        if src_paragraph.strip() and tgt_paragraph.strip():
            # 번역문 분할
            tgt_sentences = split_target_sentences_advanced(
                tgt_paragraph, 
                max_length, 
                splitter="punctuation"
            )
            
            # 임베더 함수 가져오기
            embed_func = get_embedder_function(
                embedder_name, 
                device=device,
                openai_model=openai_model,
                openai_api_key=openai_api_key
            )
            
            # 정렬 실행
            alignments = improved_align_paragraphs(
                tgt_sentences,
                src_paragraph,
                embed_func,
                similarity_threshold
            )
            
            # 문단식별자 추가
            for a in alignments:
                a['문단식별자'] = idx + 1
            
            all_results.extend(alignments)
        
        elif verbose:
            print(f"⚠️ 문단 {idx + 1}: 빈 원문 또는 번역문 건너뜀")
    
    if not all_results:
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame(all_results)
    
    # 컬럼 순서 정리
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    available_columns = [col for col in final_columns if col in result_df.columns]
    result_df = result_df[available_columns]
    
    # 결과 저장
    try:
        result_df.to_excel(output_file, index=False)
        print(f"💾 결과 저장: {output_file}")
        print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
        
        # 간단한 통계
        if verbose:
            analyze_alignment_results(result_df)
        
        return result_df
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return None

def analyze_alignment_results(result_df: pd.DataFrame):
    """정렬 결과 분석"""
    print("\n📊 정렬 결과 분석:")
    
    # 전체 유사도 분포
    if 'similarity' in result_df.columns:
        print(f"🎯 전체 유사도:")
        print(f"   평균: {result_df['similarity'].mean():.3f}")
        print(f"   최고: {result_df['similarity'].max():.3f}")
        print(f"   최저: {result_df['similarity'].min():.3f}")
        
        # 고품질 매칭 비율
        high_quality = sum(1 for x in result_df['similarity'] if x > 0.7)
        medium_quality = sum(1 for x in result_df['similarity'] if 0.5 <= x <= 0.7)
        low_quality = sum(1 for x in result_df['similarity'] if x < 0.5)
        total = len(result_df)
        
        print(f"📊 품질별 매칭:")
        print(f"   고품질 (>0.7): {high_quality}/{total} ({high_quality/total*100:.1f}%)")
        print(f"   중품질 (0.5-0.7): {medium_quality}/{total} ({medium_quality/total*100:.1f}%)")
        print(f"   저품질 (<0.5): {low_quality}/{total} ({low_quality/total*100:.1f}%)")
    
    # 빈 매칭 확인
    empty_source = sum(1 for x in result_df['원문'] if not str(x).strip())
    empty_target = sum(1 for x in result_df['번역문'] if not str(x).strip())
    
    if empty_source > 0:
        print(f"⚠️ 빈 원문: {empty_source}개")
    if empty_target > 0:
        print(f"⚠️ 빈 번역문: {empty_target}개")
