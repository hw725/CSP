"""PA 전용 정렬기 - spaCy 순차적 분할 정렬만 사용 (SA 연동 완전 제거, circular import 완전 제거)"""
import sys
import os
import importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict
from sentence_splitter import split_target_sentences_advanced

# 패키지 import 방식으로 복원
from sa.sa_embedders import get_embedder

try:
    import torch
except ImportError:
    torch = None

<<<<<<< HEAD
# ✅ SA 임베더 직접 import
def get_embedder_function(embedder_name: str, device: str = "cpu"):
    """SA 임베더 함수 직접 로드 (GPU 지원)"""
    if embedder_name == 'bge':
        from sa_embedders.bge import compute_embeddings_with_cache
        def embed_func(texts):
            return compute_embeddings_with_cache(texts)  # device 인자 제거!
        return embed_func
    elif embedder_name == 'st':
        try:
            from sa_embedders.sentence_transformer import compute_embeddings_with_cache
            def embed_func(texts):
                return compute_embeddings_with_cache(texts, device=device)
            return embed_func
        except ImportError:
            print("❌ SentenceTransformer 임베더 import 실패")
            return fallback_embedder_bge(device)
    elif embedder_name == 'openai':
        try:
            from sa_embedders.openai import compute_embeddings_with_cache
            def embed_func(texts):
                try:
                    return compute_embeddings_with_cache(texts)
                except Exception as e:
                    print(f"⚠️ OpenAI 임베더 실패: {e}")
                    print("➡️ BGE-m3 fallback")
                    return fallback_embedder_bge(device)(texts)
            return embed_func
        except ImportError:
            print("❌ OpenAI 임베더 import 실패")
            return fallback_embedder_bge(device)
    return fallback_embedder_bge(device)
=======
def get_embedder_function(embedder_name: str, device: str = "cpu", openai_model: str = None, openai_api_key: str = None):
    # Robust device selection: if device=="cuda" but not available, fallback to cpu
    if device == "cuda":
        if torch is None or not torch.cuda.is_available():
            print("⚠️ torch 미설치 또는 CUDA 미지원: CPU로 전환합니다.")
            device = "cpu"
    if embedder_name == 'bge':
        return get_embedder("bge", device_id=device)
    elif embedder_name == 'openai':
        sa_openai = importlib.import_module('sa.sa_embedders.openai')
        compute_embeddings_with_cache = sa_openai.compute_embeddings_with_cache
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        def embed_func(texts):
            return compute_embeddings_with_cache(
                texts, 
                model=openai_model if openai_model else "text-embedding-3-large"
            )
        return embed_func
    else:
        raise ValueError(f"지원하지 않는 임베더: {embedder_name}. 지원: openai, bge")
>>>>>>> e1b9f69 (pa 책단위 테스트 완료)

# improved_align_paragraphs 직접 포함 (circular import 제거)
def improved_align_paragraphs(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    if not tgt_sentences:
        return []
<<<<<<< HEAD
    
    print(f"🎯 PA 정렬 시작 (SA DP): {len(tgt_sentences)}개 번역문 → {len(src_chunks)}개 원문")
    
    try:
        # ✅ SA 함수 직접 import (확인된 함수명)
        import importlib.util
        sa_path = os.path.abspath(os.path.join('..', 'sa'))
        aligner_path = os.path.join(sa_path, 'aligner.py')
        
        spec = importlib.util.spec_from_file_location("sa_aligner_module", aligner_path)
        sa_aligner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa_aligner)
        
        # ✅ 확인된 함수 사용
        sa_align_func = sa_aligner.align_tokens_with_embeddings
        
        print("🔗 SA DP 함수 연동 성공: align_tokens_with_embeddings")
        
        # ✅ 정확한 매개변수로 호출
        sa_alignments = sa_align_func(
            src_units=tgt_sentences,          # PA: 번역문이 기준 (src_units)
            tgt_units=src_chunks,             # PA: 원문이 정렬 대상 (tgt_units)
            embed_func=embed_func,
            similarity_threshold=similarity_threshold
        )
        
        print(f"📊 SA DP 결과: {len(sa_alignments) if sa_alignments else 0}개 정렬")
        
        # ✅ 결과 변환 (SA → PA 형식)
        pa_alignments = []
        
        if sa_alignments:
            for align in sa_alignments:
                # SA 결과 형식에 따른 처리
                if isinstance(align, dict):
                    # 딕셔너리 형식
                    src_text = align.get('src', '')      # SA의 src = PA의 번역문
                    tgt_text = align.get('tgt', '')      # SA의 tgt = PA의 원문
                    score = align.get('score', 0.0)
                elif isinstance(align, (list, tuple)) and len(align) >= 2:
                    # 리스트/튜플 형식
                    src_text = str(align[0])  # SA src → PA 번역문
                    tgt_text = str(align[1])  # SA tgt → PA 원문
                    score = float(align[2]) if len(align) > 2 else 0.0
                else:
                    print(f"⚠️ 알 수 없는 SA 결과 형식: {type(align)}")
                    continue
                
                pa_alignments.append({
                    '문단식별자': 1,
                    '원문': tgt_text,        # PA 원문 = SA tgt
                    '번역문': src_text,      # PA 번역문 = SA src
                    'similarity': score,
                    'split_method': 'spacy_lg',
                    'align_method': 'sa_dp_align_tokens_with_embeddings'
                })
        
        if pa_alignments:
            print(f"✅ SA DP 정렬 성공: {len(pa_alignments)}개 항목")
            return pa_alignments
        else:
            print("⚠️ SA에서 빈 결과 반환")
            
    except Exception as e:
        print(f"⚠️ SA DP 연동 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # SA 연동 실패시 고품질 대체 정렬
    print("🔄 고품질 대체 정렬 방식 사용...")
    return advanced_align_paragraphs(tgt_sentences, src_chunks, embed_func, similarity_threshold)

def advanced_align_paragraphs(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """고품질 대체 정렬 (DP 스타일)"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    tgt_embeddings = embed_func(tgt_sentences)
    src_embeddings = embed_func(src_chunks)

    # 임베딩 차원 체크
    if tgt_embeddings.shape[1] != src_embeddings.shape[1]:
        print(f"❌ 임베딩 차원 불일치: tgt={tgt_embeddings.shape}, src={src_embeddings.shape}")
        return []

    # ✅ DP 스타일 정렬 (순서 보존 + 무결성 보장)
    alignments = []
    used_src_indices = set()
    
    # 번역문 순서 기준으로 처리
    for tgt_idx, tgt_sent in enumerate(tgt_sentences):
        # 각 번역문에 대해 사용되지 않은 원문 중 최적 매칭
        similarities = sim_matrix[tgt_idx]
        
        best_score = -1.0
        best_src_idx = -1
        
        for src_idx in range(len(src_chunks)):
            if src_idx not in used_src_indices:
                if similarities[src_idx] > best_score:
                    best_score = similarities[src_idx]
                    best_src_idx = src_idx
        
        if best_src_idx != -1:
            used_src_indices.add(best_src_idx)
            src_text = src_chunks[best_src_idx]
        else:
            src_text = ""
        
        alignments.append({
            '문단식별자': 1,
            '원문': src_text,
=======
    if isinstance(src_chunks, str):
        source_text = src_chunks
    elif isinstance(src_chunks, list) and len(src_chunks) == 1:
        source_text = src_chunks[0]
    elif isinstance(src_chunks, list):
        source_text = ' '.join(str(chunk) for chunk in src_chunks)
    else:
        source_text = str(src_chunks) if src_chunks else ""
    if not source_text.strip():
        return [{
            '원문': '',
>>>>>>> e1b9f69 (pa 책단위 테스트 완료)
            '번역문': tgt_sent,
            'similarity': 0.0,
            'split_method': 'whitespace',
            'align_method': 'no_source'
        } for tgt_sent in tgt_sentences]
    print(f"🔄 의미적 병합 정렬 시작: {len(tgt_sentences)}개 번역문")
    from sentence_splitter import split_source_by_whitespace_and_align
    aligned_src_chunks = split_source_by_whitespace_and_align(source_text, tgt_sentences, embed_func, similarity_threshold)
    # 임베딩 유사도 계산 및 결과 생성
    from sklearn.metrics.pairwise import cosine_similarity
    def safe_embed(texts):
        try:
            return np.array(embed_func(texts))
        except Exception as e:
            print(f"임베딩 오류: {e}")
            return np.zeros((len(texts), 768))
    tgt_embeddings = safe_embed(tgt_sentences)
    src_embeddings = safe_embed(aligned_src_chunks)
    if tgt_embeddings.shape[1] != src_embeddings.shape[1]:
        min_dim = min(tgt_embeddings.shape[1], src_embeddings.shape[1])
        tgt_embeddings = tgt_embeddings[:, :min_dim]
        src_embeddings = src_embeddings[:, :min_dim]
    sim_matrix = cosine_similarity(tgt_embeddings, src_embeddings)
    alignments = []
    for i in range(len(tgt_sentences)):
        alignments.append({
            '원문': aligned_src_chunks[i] if i < len(aligned_src_chunks) else '',
            '번역문': tgt_sentences[i],
            'similarity': sim_matrix[i, i] if i < sim_matrix.shape[0] and i < sim_matrix.shape[1] else 0.0,
            'split_method': 'whitespace',
            'align_method': 'semantic_merge'
        })
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'whitespace',
            'align_method': 'semantic_merge_unmatched_src'
        })
    print(f"✅ 의미적 병합 정렬 완료: {len(alignments)}개 항목")
    return alignments

def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu"
):
    """PA 처리 (공백/구두점 기반 순차적 분할만 사용)"""
    print(f"🔄 PA 처리 시작 (공백/구두점 순차적 분할)")
    tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="spacy")
    # 원문 분할: spaCy 완전 배제, 공백/구두점 기준 분할만 사용
    src_chunks = src_paragraph  # improved_align_paragraphs에서 직접 분할
    print(f"   번역문: {len(tgt_sentences)}개 문장")
    print(f"   원문: {len(src_paragraph)}개 토큰")
    embed_func = get_embedder_function(embedder_name, device=device)
    alignments = improved_align_paragraphs(
        tgt_sentences, 
        src_chunks, 
        embed_func, 
        similarity_threshold
    )
    # 문단식별자 부여
    for a in alignments:
        a['문단식별자'] = 1
    return alignments


def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu"
):
    """파일 단위 처리 - spaCy 순차적 분할 정렬만 사용"""
    print(f"📂 파일 처리 시작: {input_file}")
    df = pd.read_excel(input_file)
    all_results = []
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        if src_paragraph and tgt_paragraph:
            alignments = process_paragraph_alignment(
                src_paragraph,
                tgt_paragraph,
                embedder_name=embedder_name,
                max_length=max_length,
                similarity_threshold=similarity_threshold,
                device=device
            )
            all_results.extend(alignments)
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]
    result_df.to_excel(output_file, index=False)
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    return result_df