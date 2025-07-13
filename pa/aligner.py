"""PA 전용 정렬기 - SA의 Vice Versa 방식 (기존 분할 방식 유지)"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

# 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

# 로컬 모듈 import (기존과 동일)
from sentence_splitter import split_target_sentences_advanced, split_source_by_whitespace_and_align

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ===== 새로운 spaCy 지원 함수들 추가 =====
def get_spacy_nlp():
    """spaCy 모델 안전하게 로드"""
    try:
        import spacy
        # 한국어 모델 시도
        try:
            nlp = spacy.load("ko_core_news_sm")
            print("✅ spaCy 한국어 모델 로드 성공")
            return nlp
        except OSError:
            # 영어 모델 폴백
            try:
                nlp = spacy.load("en_core_web_sm")
                print("⚠️ 한국어 모델 없음, 영어 모델 사용")
                return nlp
            except OSError:
                print("❌ spaCy 모델 없음")
                return None
    except ImportError:
        print("❌ spaCy 설치되지 않음")
        return None

def split_target_sentences_spacy_tokenizer(
    text: str, 
    max_length: int = 150,
    tokenizer_func=None,
    nlp=None
) -> List[str]:
    """
    spaCy + 토크나이저 융합 문장 분할
    기존 split_target_sentences_advanced의 대체 함수
    """
    if not text.strip():
        return []
    
    sentences = []
    
    # 1단계: spaCy로 문장 경계 감지
    if nlp:
        try:
            doc = nlp(text)
            spacy_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if spacy_sentences:
                print(f"🔍 spaCy 분할: {len(spacy_sentences)}개 문장")
                sentences = spacy_sentences
            else:
                # spaCy 분할 실패시 기본 분할
                sentences = [text]
        except Exception as e:
            print(f"⚠️ spaCy 분할 실패: {e}")
            sentences = [text]
    else:
        # spaCy 없으면 기존 방식 사용
        sentences = split_target_sentences_advanced(text, max_length, splitter="punctuation")
    
    # 2단계: 토크나이저로 긴 문장 세분화
    if tokenizer_func and sentences:
        refined_sentences = []
        
        for sentence in sentences:
            if len(sentence) > max_length:
                # 긴 문장을 토크나이저로 세분화
                refined_parts = split_long_sentence_with_tokenizer(
                    sentence, max_length, tokenizer_func
                )
                refined_sentences.extend(refined_parts)
            else:
                refined_sentences.append(sentence)
        
        print(f"🔧 토크나이저 조정: {len(sentences)} → {len(refined_sentences)}개 문장")
        sentences = refined_sentences
    
    return sentences if sentences else [text]

def split_long_sentence_with_tokenizer(
    sentence: str, 
    max_length: int, 
    tokenizer_func
) -> List[str]:
    """토크나이저를 사용하여 긴 문장을 의미 단위로 분할"""
    try:
        tokens = tokenizer_func(sentence)
        if not tokens:
            return [sentence]
        
        parts = []
        current_part = []
        current_length = 0
        
        for token in tokens:
            token_length = len(token)
            
            # 현재 파트가 최대 길이를 초과하면 새 파트 시작
            if current_length + token_length > max_length and current_part:
                parts.append(''.join(current_part))
                current_part = [token]
                current_length = token_length
            else:
                current_part.append(token)
                current_length += token_length
        
        # 마지막 파트 추가
        if current_part:
            parts.append(''.join(current_part))
        
        return parts if parts else [sentence]
        
    except Exception as e:
        print(f"⚠️ 토크나이저 분할 실패: {e}")
        return [sentence]

def split_src_by_tgt_units_spacy_tokenizer(
    tgt_sentences: List[str], 
    src_text: str, 
    tokenizer_func=None,
    nlp=None
) -> List[str]:
    """
    spaCy + 토크나이저를 활용한 Vice Versa 원문 분할
    기존 split_src_by_tgt_units_vice_versa의 개선 버전
    """
    if not tgt_sentences or not src_text.strip():
        return []
    
    # spaCy로 원문 구조 분석
    structure_info = analyze_source_structure_with_spacy(src_text, nlp)
    
    # 토크나이저로 원문 토큰화
    if tokenizer_func:
        try:
            src_tokens = tokenizer_func(src_text)
            if not src_tokens:
                src_tokens = list(src_text)
        except Exception as e:
            print(f"⚠️ 토크나이저 실패: {e}")
            src_tokens = list(src_text)
    else:
        src_tokens = list(src_text)
    
    if not src_tokens:
        return ['' for _ in tgt_sentences]
    
    num_tgt = len(tgt_sentences)
    if num_tgt == 1:
        return [''.join(src_tokens)]
    
    # spaCy 구조 정보를 활용한 스마트 분할
    if structure_info['entities'] or structure_info['noun_chunks']:
        return smart_split_with_spacy_structure(src_tokens, tgt_sentences, structure_info)
    else:
        # 기본 균등 분할
        return simple_equal_split_tokens(src_tokens, num_tgt)

def analyze_source_structure_with_spacy(src_text: str, nlp) -> Dict:
    """spaCy로 원문 구조 분석"""
    structure_info = {
        'entities': [],
        'noun_chunks': [],
        'pos_patterns': [],
        'sentence_count': 1
    }
    
    if not nlp:
        return structure_info
    
    try:
        doc = nlp(src_text)
        
        # 개체명 추출
        structure_info['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
        
        # 명사구 추출
        structure_info['noun_chunks'] = [chunk.text for chunk in doc.noun_chunks]
        
        # 품사 패턴 추출
        structure_info['pos_patterns'] = [token.pos_ for token in doc]
        
        # 문장 수
        structure_info['sentence_count'] = len(list(doc.sents))
        
        if structure_info['entities'] or structure_info['noun_chunks']:
            print(f"📊 spaCy 구조 분석: 개체명 {len(structure_info['entities'])}개, 명사구 {len(structure_info['noun_chunks'])}개")
        
    except Exception as e:
        print(f"⚠️ spaCy 구조 분석 실패: {e}")
    
    return structure_info

def smart_split_with_spacy_structure(
    src_tokens: List[str], 
    tgt_sentences: List[str], 
    structure_info: Dict
) -> List[str]:
    """spaCy 구조 정보를 활용한 스마트 분할"""
    
    src_text = ''.join(src_tokens)
    num_tgt = len(tgt_sentences)
    
    # 개체명이나 명사구 위치를 분할 경계로 활용
    split_points = []
    
    # 개체명 끝 위치들을 분할 후보로 추가
    for entity_text, _ in structure_info['entities']:
        pos = src_text.find(entity_text)
        if pos != -1:
            split_points.append(pos + len(entity_text))
    
    # 명사구 끝 위치들을 분할 후보로 추가
    for chunk_text in structure_info['noun_chunks']:
        pos = src_text.find(chunk_text)
        if pos != -1:
            split_points.append(pos + len(chunk_text))
    
    # 중복 제거 및 정렬
    split_points = sorted(set(split_points))
    
    if len(split_points) >= num_tgt - 1:
        # 충분한 분할점이 있으면 활용
        selected_points = split_points[:num_tgt-1]
        
        chunks = []
        start = 0
        for point in selected_points:
            chunks.append(src_text[start:point])
            start = point
        chunks.append(src_text[start:])  # 마지막 청크
        
        print(f"🎯 spaCy 구조 기반 분할: {len(chunks)}개 청크")
        return chunks
    else:
        # 분할점이 부족하면 기본 분할
        return simple_equal_split_tokens(src_tokens, num_tgt)

def simple_equal_split_tokens(src_tokens: List[str], num_chunks: int) -> List[str]:
    """기본 균등 분할 (토큰 기반)"""
    tokens_per_chunk = len(src_tokens) // num_chunks
    remainder = len(src_tokens) % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        current_size = tokens_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        
        if end_idx > len(src_tokens):
            end_idx = len(src_tokens)
        
        if start_idx < len(src_tokens):
            chunk = ''.join(src_tokens[start_idx:end_idx])
            chunks.append(chunk)
        else:
            chunks.append('')
        
        start_idx = end_idx
    
    return chunks
# ===== 새로운 spaCy 지원 함수들 끝 =====

def get_tokenizer_function(tokenizer_name: str = "jieba"):
    """토크나이저 함수 반환 - SA 재사용"""
    try:
        if tokenizer_name == "jieba":
            sys.path.insert(0, str(project_root / 'sa' / 'sa_tokenizers'))
            from jieba_mecab import tokenize_chinese_text
            print("✅ jieba 토크나이저 로드 성공")
            return tokenize_chinese_text
        elif tokenizer_name == "mecab":
            sys.path.insert(0, str(project_root / 'sa' / 'sa_tokenizers'))
            from jieba_mecab import tokenize_korean_text
            print("✅ mecab 토크나이저 로드 성공")
            return tokenize_korean_text
        else:
            print(f"⚠️ 기본 분할 사용: {tokenizer_name}")
            return lambda text: list(text)
    except ImportError as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}, 기본 분할 사용")
        return lambda text: list(text)

def get_embedder_function(embedder_name: str, device: str = "cpu", openai_model: str = None, openai_api_key: str = None):
    """임베더 함수 반환 - 기존과 동일"""
    
    # 디바이스 확인
    if device == "cuda":
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("⚠️ CUDA 미지원: CPU로 전환합니다.")
            device = "cpu"
    
    if embedder_name == 'bge':
        try:
            # common 모듈에서 BGE 임베더 가져오기
            sys.path.insert(0, str(project_root / 'common' / 'embedders'))
            from bge import get_embed_func
            embed_func = get_embed_func(device_id=0 if device == "cuda" else None)
            if embed_func is None:
                print("❌ BGE 임베더 초기화 실패")
                return None
            print("✅ BGE 임베더 초기화 성공")
            return embed_func
        except ImportError as e:
            print(f"❌ BGE 임베더 로드 실패: {e}")
            return None
            
    elif embedder_name == 'openai':
        try:
            # 환경변수 설정
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            
            # common 모듈에서 OpenAI 임베더 가져오기
            sys.path.insert(0, str(project_root / 'common' / 'embedders'))
            from openai import compute_embeddings_with_cache
            
            def embed_func(texts):
                return compute_embeddings_with_cache(
                    texts, 
                    model=openai_model if openai_model else "text-embedding-3-large"
                )
            print("✅ OpenAI 임베더 초기화 성공")
            return embed_func
        except ImportError as e:
            print(f"❌ OpenAI 임베더 로드 실패: {e}")
            return None
    else:
        print(f"❌ 지원하지 않는 임베더: {embedder_name}")
        return None

def split_src_by_tgt_units_vice_versa(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    similarity_threshold: float = 0.3
) -> List[str]:
    """
    SA의 Vice Versa: 번역문 문장들을 기준으로 원문을 분할
    """
    if not tgt_sentences or not src_text.strip():
        return []
    
    # 원문을 토큰으로 분할
    if tokenizer_func:
        try:
            src_tokens = tokenizer_func(src_text)
            if not src_tokens:
                src_tokens = list(src_text)
        except Exception as e:
            print(f"⚠️ 토크나이저 실패: {e}")
            src_tokens = list(src_text)
    else:
        src_tokens = list(src_text)
    
    if not src_tokens:
        return ['' for _ in tgt_sentences]
    
    num_tgt_sentences = len(tgt_sentences)
    
    if num_tgt_sentences == 1:
        return [''.join(src_tokens)]
    
    # 기본 균등 분할
    tokens_per_chunk = len(src_tokens) // num_tgt_sentences
    remainder = len(src_tokens) % num_tgt_sentences
    
    src_chunks = []
    start_idx = 0
    
    for i in range(num_tgt_sentences):
        current_size = tokens_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        
        if end_idx > len(src_tokens):
            end_idx = len(src_tokens)
        
        if start_idx < len(src_tokens):
            chunk_tokens = src_tokens[start_idx:end_idx]
            chunk_text = ''.join(chunk_tokens)
            src_chunks.append(chunk_text)
        else:
            src_chunks.append('')
        
        start_idx = end_idx
    
    # 임베더가 있으면 의미적 최적화 시도
    if embed_func:
        try:
            optimized_chunks = optimize_alignment_with_embedder(
                src_chunks, tgt_sentences, embed_func, similarity_threshold
            )
            src_chunks = optimized_chunks
        except Exception as e:
            print(f"⚠️ 의미적 최적화 실패: {e}")
    
    # 결과 보정
    while len(src_chunks) < len(tgt_sentences):
        src_chunks.append('')
    
    return src_chunks[:len(tgt_sentences)]

def optimize_alignment_with_embedder(
    src_chunks: List[str], 
    tgt_sentences: List[str], 
    embed_func,
    similarity_threshold: float
) -> List[str]:
    """임베더를 사용한 정렬 최적화"""
    optimized_chunks = []
    
    for i, (src_chunk, tgt_sentence) in enumerate(zip(src_chunks, tgt_sentences)):
        if not src_chunk.strip() or not tgt_sentence.strip():
            optimized_chunks.append(src_chunk)
            continue
        
        # 현재 매칭의 유사도 계산
        current_similarity = compute_similarity(src_chunk, tgt_sentence, embed_func)
        
        best_chunk = src_chunk
        best_similarity = current_similarity
        
        # 이전 청크와 합치기 시도
        if i > 0 and optimized_chunks:
            extended_chunk = optimized_chunks[-1] + src_chunk
            extended_similarity = compute_similarity(extended_chunk, tgt_sentence, embed_func)
            
            if extended_similarity > best_similarity + 0.1:  # 임계값
                # 이전 청크를 비우고 현재 청크를 확장
                optimized_chunks[-1] = ''
                best_chunk = extended_chunk
                best_similarity = extended_similarity
        
        optimized_chunks.append(best_chunk)
    
    return optimized_chunks

def compute_similarity(text1: str, text2: str, embed_func) -> float:
    """유사도 계산"""
    try:
        if not text1.strip() or not text2.strip():
            return 0.0
        
        embeddings = embed_func([text1, text2])
        if len(embeddings) != 2:
            return 0.0
        
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    except Exception as e:
        print(f"⚠️ 유사도 계산 실패: {e}")
        return compute_similarity_simple(text1, text2)

def compute_similarity_simple(text1: str, text2: str) -> float:
    """간단한 길이 기반 유사도"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    ratio = min(len1, len2) / max(len1, len2)
    return 0.5 + (ratio * 0.5)

def improved_align_paragraphs_vice_versa(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    Vice Versa 방식: 번역문 문장들을 기준으로 원문을 분할하여 정렬
    (기존 순차적 정렬과 동일한 인터페이스)
    """
    if not tgt_sentences:
        return []
    
    # Vice Versa: 번역문 문장들을 기준으로 원문을 분할
    aligned_src_chunks = split_src_by_tgt_units_vice_versa(
        tgt_sentences, 
        src_text, 
        embed_func,
        tokenizer_func,
        similarity_threshold
    )
    
    alignments = []
    for i in range(len(tgt_sentences)):
        src_chunk = aligned_src_chunks[i] if i < len(aligned_src_chunks) else ''
        tgt_sentence = tgt_sentences[i]
        
        # 유사도 계산
        if embed_func:
            try:
                similarity = compute_similarity(src_chunk, tgt_sentence, embed_func)
            except:
                similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        else:
            similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        
        alignments.append({
            '원문': src_chunk,
            '번역문': tgt_sentence,
            'similarity': similarity,
            'split_method': 'vice_versa_tokenized',
            'align_method': 'tgt_based_src_split'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'vice_versa_tokenized',
            'align_method': 'sequential_unmatched_src'
        })
    
    return alignments

# ===== 새로운 spaCy+토크나이저 융합 정렬 함수 추가 =====
def improved_align_paragraphs_spacy_tokenizer(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    tokenizer_func=None,
    nlp=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    spaCy + 토크나이저 융합 정렬
    새로운 4번째 방식
    """
    if not tgt_sentences:
        return []
    
    # spaCy + 토크나이저를 활용한 원문 분할
    aligned_src_chunks = split_src_by_tgt_units_spacy_tokenizer(
        tgt_sentences, 
        src_text, 
        tokenizer_func,
        nlp
    )
    
    alignments = []
    for i in range(len(tgt_sentences)):
        src_chunk = aligned_src_chunks[i] if i < len(aligned_src_chunks) else ''
        tgt_sentence = tgt_sentences[i]
        
        # 유사도 계산
        if embed_func:
            try:
                similarity = compute_similarity(src_chunk, tgt_sentence, embed_func)
            except:
                similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        else:
            similarity = compute_similarity_simple(src_chunk, tgt_sentence)
        
        alignments.append({
            '원문': src_chunk,
            '번역문': tgt_sentence,
            'similarity': similarity,
            'split_method': 'spacy_tokenizer_fusion',
            'align_method': 'spacy_tokenizer_based_split'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'spacy_tokenizer_fusion',
            'align_method': 'sequential_unmatched_src'
        })
    
    return alignments
# ===== 새로운 spaCy+토크나이저 융합 정렬 함수 끝 =====

def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    tokenizer_name: str = 'jieba',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8,
    use_spacy_tokenizer: bool = False  # 새로운 옵션 추가
):
    """
    PA 처리: 기존 방식 + Vice Versa 토크나이저 정렬 + spaCy 토크나이저 융합 병합
    """
    print(f"🔄 PA 처리 시작 (기존 + Vice Versa + spaCy 토크나이저)")
    
    # 1. 기존 순차적 정렬 (punctuation)
    tgt_sentences_seq = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="punctuation")
    alignments_seq = improved_align_paragraphs(
        tgt_sentences_seq, 
        src_paragraph
    )
    
    # 2. 기존 의미적 정렬 (spacy)
    tgt_sentences_sem = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="spacy")
    embed_func = get_embedder_function(embedder_name, device=device)
    alignments_sem = improved_align_paragraphs(
        tgt_sentences_sem,
        src_paragraph,
        embed_func,
        similarity_threshold
    )
    
    # 3. 기존 Vice Versa 토크나이저 정렬
    tokenizer_func = get_tokenizer_function(tokenizer_name)
    tgt_sentences_tok = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="punctuation")
    alignments_tok = improved_align_paragraphs_vice_versa(
        tgt_sentences_tok,
        src_paragraph,
        embed_func,
        tokenizer_func,
        similarity_threshold
    )
    
    # 4. 새로운 spaCy + 토크나이저 융합 정렬
    alignments_spacy_tok = []
    if use_spacy_tokenizer:
        nlp = get_spacy_nlp()
        # spaCy + 토크나이저로 문장 분할
        tgt_sentences_spacy_tok = split_target_sentences_spacy_tokenizer(
            tgt_paragraph, max_length, tokenizer_func, nlp
        )
        alignments_spacy_tok = improved_align_paragraphs_spacy_tokenizer(
            tgt_sentences_spacy_tok,
            src_paragraph,
            embed_func,
            tokenizer_func,
            nlp,
            similarity_threshold
        )
    
    # 4가지 방식 중 최적 선택
    all_alignments = [alignments_seq, alignments_sem, alignments_tok, alignments_spacy_tok]
    max_len = max(len(alignments) for alignments in all_alignments if alignments)
    
    results = []
    for i in range(max_len):
        seq = alignments_seq[i] if i < len(alignments_seq) else {'원문':'','번역문':'','similarity':0.0,'split_method':'punctuation','align_method':'sequential'}
        sem = alignments_sem[i] if i < len(alignments_sem) else {'원문':'','번역문':'','similarity':0.0,'split_method':'spacy','align_method':'semantic'}
        tok = alignments_tok[i] if i < len(alignments_tok) else {'원문':'','번역문':'','similarity':0.0,'split_method':'vice_versa_tokenized','align_method':'tgt_based_src_split'}
        spacy_tok = alignments_spacy_tok[i] if i < len(alignments_spacy_tok) else {'원문':'','번역문':'','similarity':0.0,'split_method':'spacy_tokenizer_fusion','align_method':'spacy_tokenizer_based_split'}
        
        if use_spacy_tokenizer and alignments_spacy_tok:
            # spaCy + 토크나이저 융합 사용시 가중치 조정 (기존:순차0.2+의미0.3+토크나이저0.2+spaCy토크나이저0.3)
            weighted_sim = seq['similarity']*0.2 + sem['similarity']*0.3 + tok['similarity']*0.2 + spacy_tok['similarity']*0.3
            
            if weighted_sim >= quality_threshold:
                # spaCy 토크나이저 결과 우선
                result = {
                    '원문': spacy_tok['원문'] if spacy_tok['원문'] else (tok['원문'] if tok['원문'] else (sem['원문'] if sem['원문'] else seq['원문'])),
                    '번역문': spacy_tok['번역문'] if spacy_tok['번역문'] else (tok['번역문'] if tok['번역문'] else (sem['번역문'] if sem['번역문'] else seq['번역문'])),
                    'similarity': weighted_sim,
                    'split_method': f"seq+sem+tok+spacy_tok",
                    'align_method': 'hybrid_with_spacy_tokenizer'
                }
            else:
                # spaCy 토크나이저 결과만 채택
                result = spacy_tok.copy()
                result['align_method'] = 'spacy_tokenizer_fusion_only'
        else:
            # 기존 3가지 방식 사용
            weighted_sim = seq['similarity']*0.3 + sem['similarity']*0.4 + tok['similarity']*0.3
            
            if weighted_sim >= quality_threshold:
                result = {
                    '원문': tok['원문'] if tok['원문'] else (sem['원문'] if sem['원문'] else seq['원문']),
                    '번역문': tok['번역문'] if tok['번역문'] else (sem['번역문'] if sem['번역문'] else seq['번역문']),
                    'similarity': weighted_sim,
                    'split_method': f"seq+sem+tok",
                    'align_method': 'hybrid_with_tokenizer'
                }
            else:
                result = tok.copy()
                result['align_method'] = 'tokenizer_vice_versa_only'
        
        results.append(result)
    
    return results

def improved_align_paragraphs(
    tgt_sentences: List[str], 
    src_text: str, 
    embed_func=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    기존 순차적 1:1 정렬 (공백/포맷 100% 보존) - 그대로 유지
    """
    if not tgt_sentences:
        return []
    
    # 원문을 번역문 개수에 맞춰 순차적으로 분할
    aligned_src_chunks = split_source_by_whitespace_and_align(src_text, len(tgt_sentences))
    
    alignments = []
    for i in range(len(tgt_sentences)):
        alignments.append({
            '원문': aligned_src_chunks[i] if i < len(aligned_src_chunks) else '',
            '번역문': tgt_sentences[i],
            'similarity': 1.0,  # 순차적 정렬이므로 유사도는 1.0
            'split_method': 'punctuation',
            'align_method': 'sequential'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'punctuation',
            'align_method': 'sequential_unmatched_src'
        })
    
    return alignments

def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    tokenizer_name: str = 'jieba',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8,
    use_spacy_tokenizer: bool = False,  # 새로운 옵션 추가
    verbose: bool = False,
    **kwargs
):
    """파일 단위 처리 - 기존 방식 + Vice Versa 토크나이저 + spaCy 토크나이저 융합 추가"""
    print(f"📂 PA 파일 처리 시작: {input_file}")
    if use_spacy_tokenizer:
        print(f"🔗 기존 방식 + Vice Versa 토크나이저 + spaCy 토크나이저 융합")
    else:
        print(f"🔄 기존 방식 + Vice Versa 토크나이저 통합")
    print(f"⚙️  토크나이저: {tokenizer_name}")
    print(f"⚙️  임베더: {embedder_name}")
    print(f"🔗  spaCy 융합: {use_spacy_tokenizer}")
    
    try:
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None
    
    all_results = []
    total = len(df)
    
    for idx, row in tqdm(df.iterrows(), total=total, desc="📊 문단 처리"):
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        
        if src_paragraph.strip() and tgt_paragraph.strip():
            try:
                alignments = process_paragraph_alignment(
                    src_paragraph,
                    tgt_paragraph,
                    embedder_name=embedder_name,
                    tokenizer_name=tokenizer_name,
                    max_length=max_length,
                    similarity_threshold=similarity_threshold,
                    device=device,
                    quality_threshold=quality_threshold,
                    use_spacy_tokenizer=use_spacy_tokenizer  # 새로운 옵션 전달
                )
                
                # 문단식별자 부여
                for a in alignments:
                    a['문단식별자'] = idx + 1
                
                all_results.extend(alignments)
                
            except Exception as e:
                print(f"❌ 문단 {idx + 1} 처리 실패: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
    
    if not all_results:
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]

    # === 무결성 검증 및 보완 (기존과 동일) ===
    input_src_all = ''.join([str(row.get('원문','')) for _, row in df.iterrows()])
    input_tgt_all = ''.join([str(row.get('번역문','')) for _, row in df.iterrows()])
    output_src_all = ''.join(result_df['원문'].fillna(''))
    output_tgt_all = ''.join(result_df['번역문'].fillna(''))
    
    # 원문 보완
    if input_src_all != output_src_all:
        print('⚠️ 원문 무결성 불일치: 누락/중복 보정 시도')
        sm = SequenceMatcher(None, output_src_all, input_src_all)
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                if len(result_df) > 0:
                    result_df.at[result_df.index[-1], '원문'] += input_src_all[j1:j2]
                else:
                    result_df.loc[len(result_df)] = [df.shape[0], input_src_all[j1:j2], '', 1.0, 'integrity', 'src_patch']
            elif tag == 'delete':
                if len(result_df) > 0:
                    last = result_df.at[result_df.index[-1], '원문']
                    result_df.at[result_df.index[-1], '원문'] = last.replace(output_src_all[i1:i2], '', 1)
    
    # 번역문 보완
    if input_tgt_all != output_tgt_all:
        print('⚠️ 번역문 무결성 불일치: 누락/중복 보정 시도')
        sm = SequenceMatcher(None, output_tgt_all, input_tgt_all)
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                if len(result_df) > 0:
                    result_df.at[result_df.index[-1], '번역문'] += input_tgt_all[j1:j2]
                else:
                    result_df.loc[len(result_df)] = [df.shape[0], '', input_tgt_all[j1:j2], 1.0, 'integrity', 'tgt_patch']
            elif tag == 'delete':
                if len(result_df) > 0:
                    last = result_df.at[result_df.index[-1], '번역문']
                    result_df.at[result_df.index[-1], '번역문'] = last.replace(output_tgt_all[i1:i2], '', 1)
    
    # 최종 저장
    result_df = result_df[final_columns]
    result_df.to_excel(output_file, index=False)
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    if use_spacy_tokenizer:
        print(f"🔗 spaCy + 토크나이저 융합 방식이 통합되었습니다")
    else:
        print(f"🔄 Vice Versa 토크나이저 방식이 통합되었습니다")
    
    return result_df