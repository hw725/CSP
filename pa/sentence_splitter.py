"""PA 문장 분할기 - SuPar-Kanbun & Stanza 기반 (spaCy 대체)"""
from typing import List, Tuple
import torch
import os
import json
import hashlib
import numpy as np
from pathlib import Path

# OpenAI wrapper 클래스 (SA와 동일)
class OpenAIWrapper:
    """OpenAI API 래퍼 - SA 시스템과 동일한 방식"""
    
    def __init__(self, api_key=None, model="text-embedding-3-large"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.cache_dir = Path("embeddings_cache_openai")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "openai_embeddings.json"
        self._embedding_cache = {}
        self._load_cache()
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
    
    def _load_cache(self):
        """캐시 파일에서 임베딩 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self._embedding_cache = {k: np.array(v) for k, v in cache_data.items()}
                print(f"📂 OpenAI 캐시 로드: {len(self._embedding_cache)}개 항목")
            except Exception as e:
                print(f"⚠️ 캐시 로드 실패: {e}")
                self._embedding_cache = {}
    
    def _save_cache(self):
        """임베딩을 캐시 파일에 저장"""
        try:
            cache_data = {k: v.tolist() for k, v in self._embedding_cache.items()}
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
            print(f"💾 OpenAI 캐시 저장: {len(self._embedding_cache)}개 항목")
        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """텍스트에 대한 캐시 키 생성"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def compute_embeddings_with_cache(self, texts, use_cache=True):
        """캐시를 사용한 OpenAI 임베딩 생성"""
        
        # 단일 텍스트 처리
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # 캐시에서 찾기
        cached_embeddings = {}
        missing_texts = []
        missing_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    cached_embeddings[i] = self._embedding_cache[cache_key]
                else:
                    missing_texts.append(text)
                    missing_indices.append(i)
        else:
            missing_texts = texts
            missing_indices = list(range(len(texts)))
        
        # 캐시 히트 로그
        if use_cache and cached_embeddings:
            print(f"📂 캐시 히트: {len(cached_embeddings)}개, 누락: {len(missing_texts)}개")
        
        # 누락된 텍스트들 API 호출
        new_embeddings = {}
        if missing_texts:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                
                print(f"🔄 OpenAI API 호출: {len(missing_texts)}개 텍스트")
                
                response = client.embeddings.create(
                    model=self.model,
                    input=missing_texts,
                    encoding_format="float"
                )
                
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                
                for i, (idx, embedding) in enumerate(zip(missing_indices, batch_embeddings)):
                    new_embeddings[idx] = embedding
                    
                    # 캐시에 저장
                    if use_cache:
                        cache_key = self._get_cache_key(missing_texts[i])
                        self._embedding_cache[cache_key] = embedding
                
                # 캐시 파일 저장
                if use_cache and new_embeddings:
                    self._save_cache()
                    
                print(f"✅ OpenAI 임베딩 생성: {len(batch_embeddings)}개 → 차원: {len(batch_embeddings[0])}")
                
            except Exception as e:
                print(f"❌ OpenAI API 호출 실패: {e}")
                raise
        
        # 결과 조합
        all_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            elif i in new_embeddings:
                all_embeddings.append(new_embeddings[i])
            else:
                raise ValueError(f"임베딩을 찾을 수 없습니다: {texts[i]}")
        
        if return_single:
            return all_embeddings[0]
        else:
            return all_embeddings

# SuPar-Kanbun과 Stanza 사용 (spaCy 대체)
import re
import regex
try:
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)  # CSP 디렉토리
    sys.path.insert(0, project_root)
    
    from common.new_parsers import (
        smart_sentence_split,
        split_source_with_supar, 
        split_target_with_stanza,
        fallback_split_by_punctuation,
        SUPAR_AVAILABLE,
        STANZA_AVAILABLE
    )
    print("✅ PA: SuPar-Kanbun & Stanza 파서 로드됨")
except ImportError as e:
    print(f"⚠️ PA: 새 파서 로드 실패, 폴백 모드 (정상): {e}")
    SUPAR_AVAILABLE = False
    STANZA_AVAILABLE = False
    
    def smart_sentence_split(text: str, is_source: bool = True) -> List[str]:
        """폴백: 정규식 기반 분할"""
        pattern = r'(?<=[。？！○])\s*|(?<=[.!?])\s+'
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

# 하이브리드 토크나이저 추가
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # 중국어: SikuBERT (기존 유지)
    from common.tokenizers import (
        get_siku_tokenizer,
        siku_get_embeddings,
        siku_similarity
    )
    
    # 🆕 한국어: 하이브리드 토크나이저 (RoBERTa-Hanja + Kiwipiepy)
    from common.tokenizers import (
        get_hybrid_korean_tokenizer,
        hybrid_tokenize_korean,
        get_roberta_hanja_tokenizer,
        get_kiwi_tokenizer
    )
    
    # 토크나이저 초기화
    siku_tokenizer = get_siku_tokenizer()
    
    # 🆕 하이브리드 한국어 토크나이저 초기화
    hybrid_korean_tokenizer = get_hybrid_korean_tokenizer()
    
    print("✅ PA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)")
    
except Exception as e:
    siku_tokenizer = None
    hybrid_korean_tokenizer = None
    print(f"⚠️ PA: 하이브리드 토크나이저 초기화 실패: {e}")

def analyze_script_segments(text: str) -> List[Tuple[str, str]]:
    """텍스트를 스크립트별로 분석"""
    segments = []
    current_segment = ""
    current_script = None
    
    for char in text:
        if regex.match(r'\p{Han}', char):  # 한자
            script = 'han'
        elif regex.match(r'\p{Hangul}', char):  # 한글
            script = 'hangul'
        else:  # 기타 (공백, 구두점 등)
            script = 'other'
        
        if current_script != script:
            if current_segment:
                segments.append((current_segment, current_script))
            current_segment = char
            current_script = script
        else:
            current_segment += char
    
    if current_segment:
        segments.append((current_segment, current_script))
    
    return segments

def preprocess_with_hybrid_tokenization(text: str) -> str:
    """
    ⚠️ 이 함수는 spaCy 전용 전처리입니다. 
    원문 분할에는 절대 사용하지 마세요! 어절 경계만 사용!
    - 중국어: SikuBERT (기존 유지)
    - 한국어: RoBERTa-Hanja (한자) + Kiwipiepy (한글)
    """
    # 원문 분할에는 사용 금지! 번역문 spaCy 전처리 전용!
    if not hybrid_korean_tokenizer or not siku_tokenizer:
        return preprocess_with_siku_tokenization_fallback(text)
    
    try:
        # 🆕 하이브리드 한국어 토큰화 (번역문 분석 전용!)
        if regex.search(r'\p{Hangul}', text):  # 한글이 포함된 경우
            korean_result = hybrid_korean_tokenizer.tokenize_korean_text(text, text_type="translation")
            
            # 하이브리드 결과를 spaCy가 이해할 수 있는 형태로 변환
            processed_tokens = []
            for segment in korean_result['segments']:
                if segment['type'] == 'hanja':
                    # 한자 부분: RoBERTa 결과 사용
                    processed_tokens.extend(segment['tokens'])
                elif segment['type'] == 'hangul':
                    # 한글 부분: Kiwi 결과 사용
                    processed_tokens.extend(segment['tokens'])
            
            if processed_tokens:
                return ' '.join(processed_tokens)
        
        # 중국어 처리 (기존 SikuBERT 유지, 번역문 분석 전용!)
        if regex.search(r'\p{Han}', text) and not regex.search(r'\p{Hangul}', text):
            segments = analyze_script_segments(text)
            processed_parts = []
            
            for segment_text, script_type in segments:
                if script_type == 'han':  # 한자 부분은 SikuBERT로 토크나이징
                    tokens = siku_tokenizer.tokenize_chinese(segment_text)
                    processed_parts.append(' '.join(tokens))
                else:
                    processed_parts.append(segment_text)
            
            return ' '.join(processed_parts)
        
        # 폴백: 원본 텍스트 반환
        return text
        
    except Exception as e:
        print(f"⚠️ 하이브리드 토크나이징 실패: {e}")
        return preprocess_with_siku_tokenization_fallback(text)

def preprocess_with_siku_tokenization_fallback(text: str) -> str:
    """SikuBERT 폴백: 한자 부분만 사전 토크나이징"""
    if not siku_tokenizer:
        return text
    
    segments = analyze_script_segments(text)
    processed_segments = []
    
    han_segments = []
    han_indices = []
    
    # 한자 부분만 추출
    for i, (segment, script) in enumerate(segments):
        if script == 'han' and len(segment.strip()) > 0:
            han_segments.append(segment)
            han_indices.append(i)
    
    # 배치로 SikuBERT 처리
    if han_segments:
        try:
            tokenized_han = []
            for han_text in han_segments:
                tokens = siku_tokenizer.tokenize_chinese(han_text)
                tokenized_han.append(' '.join(tokens))
            han_dict = dict(zip(han_indices, tokenized_han))
        except Exception as e:
            print(f"⚠️ SikuBERT 토크나이징 실패: {e}")
            han_dict = {}
    else:
        han_dict = {}
    
    # 결과 재조립
    for i, (segment, script) in enumerate(segments):
        if i in han_dict:
            # SikuBERT로 토크나이징된 한자 부분
            processed_segments.append(' '.join(han_dict[i]))
        else:
            # 원본 유지
            processed_segments.append(segment)
    
    return ''.join(processed_segments)

# 기존 함수명과의 호환성
def preprocess_with_siku_tokenization(text: str) -> str:
    """호환성을 위한 래퍼 - 하이브리드 토크나이저 우선 사용"""
    return preprocess_with_hybrid_tokenization(text)

def split_target_sentences_advanced(text: str, max_length: int = 150, splitter: str = "punctuation", use_siku_preprocessing: bool = True) -> List[str]:
    """
    번역문 분할 - SikuBERT 전처리 + 문장 종결부호 기준 분할
    """
    # SikuBERT 전처리 적용 (한자 부분만)
    if use_siku_preprocessing and contains_chinese(text):
        text = preprocess_with_siku_tokenization(text)
    
    # 종결부호(한글/한자/영문) + 공백 또는 텍스트 끝 기준 분할
    # 종결부호: . ? ! 。" ？ ！ ○ 등
    pattern = r'(?<=[.!?。？！○])\s+'  # 종결부호 뒤 공백 기준
    sentences = re.split(pattern, text.strip())
    # 빈 문장 제거하고 앞뒤 공백 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_with_new_parsers(text: str, is_target: bool = True, use_siku_preprocessing: bool = True) -> List[str]:
    """새로운 파서들(SuPar-Kanbun/Stanza)을 사용한 문장 분할"""
    # SikuBERT 전처리 적용
    if use_siku_preprocessing and contains_chinese(text):
        text = preprocess_with_siku_tokenization(text)
    
    try:
        # is_target이 True면 번역문(Stanza), False면 원문(SuPar-Kanbun)
        sentences = smart_sentence_split(text, is_source=not is_target)
        return sentences if sentences else [text]
    except Exception as e:
        print(f"⚠️ 새 파서 분할 실패, 폴백: {e}")
        return split_with_smart_punctuation_rules(text)

def split_with_smart_punctuation_rules(text: str) -> List[str]:
    """중국고전 문장 분할 패턴 강화"""
    # 🆕 중국고전 특화 분할 패턴 (문단식별자 2 문제 해결)
    classical_chinese_patterns = [
        r'(?<=[。？！])',  # 기본 문장 부호
        r'(?<=하고)\s*',   # '하고' 뒤 분할
        r'(?<=하여)\s*',   # '하여' 뒤 분할  
        r'(?<=하니)\s*',   # '하니' 뒤 분할
        r'(?<=이요)\s*',   # '이요' 뒤 분할 (고전 어미)
        r'(?<=이며)\s*',   # '이며' 뒤 분할
        r'(?<=라)\s*',     # '라' 뒤 분할 (고전 어미)
        r'(?<=것을)\s*',   # '것을' 뒤 분할
        r'(?<=것이)\s*',   # '것이' 뒤 분할
        r'(?<=한다)\s*\.', # '한다.' 뒤 분할
    ]
    
    # 첫 번째: 고전 한문 패턴 적용
    for pattern in classical_chinese_patterns:
        if re.search(pattern, text):
            segments = re.split(pattern, text)
            # 빈 문자열 제거하고 앞뒤 공백 정리
            segments = [seg.strip() for seg in segments if seg.strip()]
            if len(segments) > 1:
                return segments
    
    # 폴백: 기존 패턴
    pattern = r'(?<=[。？！○])|(?<=[.!?]\s)'
    segments = re.split(pattern, text)
    return [seg.strip() for seg in segments if seg.strip()]

def apply_legacy_rules(sentences: List[str], max_length: int = 150) -> List[str]:
    length_adjusted = []
    for sent in sentences:
        if len(sent) > max_length:
            length_adjusted.extend(split_long_sentence_semantically(sent, max_length))
        else:
            length_adjusted.append(sent)
    return merge_low_chinese_segments(length_adjusted)

def split_long_sentence_semantically(sentence: str, max_length: int) -> List[str]:
    if len(sentence) <= max_length:
        return [sentence]
    parts = []
    remaining = sentence
    while len(remaining) > max_length:
        split_pos = find_semantic_split_near_position(remaining, max_length)
        if split_pos > 0:
            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:]
        else:
            break
    if remaining:
        parts.append(remaining)
    return parts

def find_semantic_split_near_position(text: str, target_pos: int) -> int:
    start = max(0, target_pos - 20)
    end = min(len(text), target_pos + 20)
    search_text = text[start:end]
    split_patterns = [
        (r'[。！？○]', 1),
        (r'[.!?]\s', 2),
        (r'[：]', 1),
        (r'[:]\s', 2),
        (r'[，]\s*(?=.{10,})', 1),
        (r'[,]\s+(?=.{10,})', 2),
        (r'\s+', 1),
    ]
    for pattern, offset in split_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            return start + matches[0].end()
    return target_pos

def merge_low_chinese_segments(sentences: List[str]) -> List[str]:
    if not sentences:
        return []
    merged, buffer = [], ''
    for sent in sentences:
        han_count = len(regex.findall(r'\p{Han}', sent))
        if han_count <= 3:
            buffer += sent
        else:
            if buffer:
                merged.append(buffer)
                buffer = ''
            merged.append(sent)
    if buffer:
        if merged:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return [s for s in merged if s]

def contains_chinese(text: str) -> bool:
    chinese_count = len(regex.findall(r'\p{Han}', text))
    return chinese_count > len(text) * 0.3

def split_source_by_whitespace_and_align(
    source: str,
    target_count: int,
    target_sentences: List[str] = None,
    embedder_name: str = "bge",
    embedder_func=None,
    max_workers: int = 4,
    batch_size: int = 100,
) -> List[str]:
    """
    원문(한문) 분할: 어절 경계에서만 분할, 어절 내부 절대 분할 금지!
    
    Args:
        source: 원문 텍스트
        target_count: 분할할 개수
        target_sentences: 번역문 문장들 (의미적 매칭용)
        embedder_name: 사용할 임베더 이름 ("bge" 또는 "openai")
        embedder_func: 외부에서 전달된 임베더 함수 (선택적)
    """
    if not source.strip():
        return [''] * target_count
    
    # 1. 어절 단위로 분할 (공백 기준, 어절 내부 절대 분할 금지!)
    words = source.split()  # 어절 단위로 분할
    if not words:
        return [''] * target_count

    # 어절이 target_count보다 적으면 균등 분배
    if len(words) <= target_count:
        result = []
        for i in range(target_count):
            if i < len(words):
                result.append(words[i])
            else:
                result.append('')
        return result
    
    # 2. 임베딩 기반 의미적 분할 (어절 경계에서만!)
    if target_sentences and len(target_sentences) > 0:
        try:
            # 외부에서 전달된 임베더 함수가 있으면 우선 사용
            if embedder_func:
                embed_func = embedder_func
                target_embeddings = embed_func(target_sentences)
                print(f"✅ 외부 임베더 함수 사용 ({embedder_name})")
            
            # OpenAI 임베더 사용
            elif embedder_name == "openai":
                # 외부에서 전달된 임베더 함수를 사용해야 함
                if not embedder_func:
                    print("⚠️ OpenAI 임베더 함수가 전달되지 않았습니다. BGE로 폴백합니다.")
                    embedder_name = "bge"  # BGE로 폴백
                else:
                    embed_func = embedder_func
                    target_embeddings = embed_func(target_sentences)
                    print("✅ OpenAI 임베딩 사용 (외부 함수)")
            
            # BGE 임베더 사용 (기본값 또는 OpenAI 실패시 폴백)
            if embedder_name == "bge":
                from common.embedders.bge import get_embedding_manager
                
                embedder = get_embedding_manager()
                print("✅ BGE-M3 Multi-Vector 임베딩 사용 (SA 방식, 작은 배치)")
                
                # SA 방식: 작은 배치 크기 + Multi-vector로 안전하게 처리
                target_embeddings = embedder.compute_embeddings_with_cache(
                    target_sentences, 
                    batch_size=4, 
                    use_multi_vector=True
                )
                
                def bge_embed_source(texts):
                    # SA처럼 작은 배치 + Multi-vector로 처리
                    return embedder.compute_embeddings_with_cache(
                        texts, 
                        batch_size=4, 
                        use_multi_vector=True
                    )
                
                embed_func = bge_embed_source
                
        except Exception as e:
            print(f"⚠️ 임베딩 실패, 하이브리드로 폴백: {e}")
            # 폴백: 하이브리드 임베딩
            from common.tokenizers import siku_get_embeddings
            from common.tokenizers import get_roberta_hanja_tokenizer
            import numpy as np
            
            print("✅ 하이브리드 임베딩 폴백 (원문:SikuBERT + 번역문:RoBERTa)")
            
            roberta_tokenizer = get_roberta_hanja_tokenizer()
            
            def get_roberta_embeddings(texts, batch_size=32):
                embeddings = []
                for text in texts:
                    emb = roberta_tokenizer.get_embeddings(text)
                    embeddings.append(emb.cpu().numpy())
                return np.array(embeddings)
            
            target_embeddings = get_roberta_embeddings(target_sentences, batch_size=32)
            
            def hybrid_embed_source(texts):
                return siku_get_embeddings(texts, batch_size=32)
            
            embed_func = hybrid_embed_source
            
            # 어절 경계 기반 스팬 임베딩 계산
            N, W = target_count, len(words)
            span_cache = {}
            
            # 어절 단위로 스팬 생성 (어절 내부 분할 금지!)
            for i in range(W):
                for j in range(i + 1, W + 1):
                    span_words = words[i:j]
                    span_text = ' '.join(span_words).strip()
                    if span_text:
                        try:
                            span_emb = embed_func([span_text])[0]
                            span_cache[(i, j)] = span_emb
                        except:
                            continue
            
            # DP로 최적 분할점 찾기 (어절 경계만)
            import numpy as np
            dp = np.full((N+1, W+1), -np.inf)
            back = np.zeros((N+1, W+1), dtype=int)
            dp[0, 0] = 0.0
            
            # DP로 어절 경계에서만 최적 분할 찾기 (강제 분할점 고려)
            for i in range(1, N+1):
                target_emb = target_embeddings[i-1]
                target_norm = np.linalg.norm(target_emb)
                
                for j in range(i, W-(N-i)+1):
                    for k in range(i-1, j):
                        if (k, j) not in span_cache:
                            continue
                        
                        span_emb = span_cache[(k, j)]
                        span_norm = np.linalg.norm(span_emb)
                        
                        # 🔥 BGE-M3 임베딩 정규화 및 유사도 계산
                        if target_norm > 1e-8 and span_norm > 1e-8:
                            # 임베딩 정규화 (BGE-M3 추가 정규화)
                            target_emb_norm = target_emb / target_norm
                            span_emb_norm = span_emb / span_norm
                            
                            # 정규화된 코사인 유사도
                            base_cosine = float(np.dot(target_emb_norm, span_emb_norm))
                            
                            # 🎯 BGE-M3 유사도 기준 (현실적인 범위로 조정)
                            if base_cosine > 0.15:  # 0.15 이상 = 강한 매칭
                                embedding_sim = base_cosine * 5.0 + 0.5  # 강한 가중치
                                print(f"🎯 강신뢰 매칭: {' '.join(words[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 5.0")
                            elif base_cosine > 0.10:  # 0.10-0.15 = 좋은 매칭  
                                embedding_sim = base_cosine * 4.0 + 0.3  # 좋은 가중치
                                print(f"🔍 중강신뢰 매칭: {' '.join(words[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 4.0")
                            elif base_cosine > 0.05:  # 0.05-0.10 = 보통 매칭
                                embedding_sim = base_cosine * 3.0 + 0.2  # 보통 가중치
                                print(f"🔍 중신뢰 매칭: {' '.join(words[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 3.0")
                            elif base_cosine > 0.01:  # 0.01-0.05 = 약한 매칭
                                embedding_sim = base_cosine * 2.0 + 0.1  # 약한 가중치
                                print(f"⚠️ 저신뢰 매칭: {' '.join(words[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 2.0")
                            elif base_cosine > -0.05:  # -0.05-0.01 = 매우 약한 매칭
                                embedding_sim = base_cosine + 0.05  # 최소 보상
                                print(f"❌ 부정매칭: {' '.join(words[k:j])[:15]}... -> 유사도 {base_cosine:.3f} + 0.05")
                            else:
                                embedding_sim = base_cosine * 0.2  # 임계값 이하는 페널티
                                print(f"❌ 부정매칭: {span_text[:15]}... -> 유사도 {base_cosine:.3f} * 0.2")
                        else:
                            embedding_sim = 0.0
                        
                        # 🔥 구조적 가중치 대폭 강화 (의미+구조 균형)
                        span_words = words[k:j]
                        span_text = ' '.join(span_words).strip()
                        structural_bonus = 0.0
                        
                        # 🎯 실제 파서 기반 구조 분석
                        try:
                            from common.new_parsers import SUPAR_AVAILABLE, supar_parser
                            
                            if SUPAR_AVAILABLE and 'supar_parser' in globals():
                                # SuPar로 실제 구문 분석
                                parsed_result = supar_parser.predict([span_text], verbose=False)
                                
                                # 구문 트리에서 대구 구조 감지
                                if len(parsed_result.sentences) > 0:
                                    sentence = parsed_result.sentences[0]
                                    
                                    # 병렬 구조 감지
                                    pos_sequence = [word.upos for word in sentence.words]
                                    
                                    # 대구 패턴 감지 (품사 반복)
                                    if len(pos_sequence) >= 6:
                                        mid = len(pos_sequence) // 2
                                        first_half = pos_sequence[:mid]
                                        second_half = pos_sequence[mid:mid*2] if mid*2 <= len(pos_sequence) else pos_sequence[mid:]
                                        
                                        if len(first_half) == len(second_half):
                                            similarity = sum(1 for a, b in zip(first_half, second_half) if a == b) / len(first_half)
                                            if similarity > 0.5:
                                                structural_bonus = 0.5  # 실제 대구 구조
                                                print(f"🔥 파서 기반 대구 감지: '{span_text[:15]}...' -> +{structural_bonus}")
                                            else:
                                                structural_bonus = 0.2
                                        else:
                                            structural_bonus = 0.1
                                    else:
                                        structural_bonus = 0.1
                                else:
                                    structural_bonus = 0.0
                            else:
                                # 폴백: 기존 하드코딩 방식
                                if '이요' in span_text and span_text.endswith('니'):
                                    structural_bonus = 0.4  # 대구 구조 확실성 -> 강력한 보너스
                                    print(f"🔥 대구 구조 강화: '{span_text[:15]}...' -> +{structural_bonus}")
                                elif span_text.endswith('니') and len(span_text) > 10:
                                    structural_bonus = 0.2  # 긴 문장의 '니' 종결 -> 중간 보너스
                                elif span_text.endswith('이요') and len(span_text) > 8:
                                    structural_bonus = 0.15  # '이요' 종결 -> 소폭 보너스
                                elif span_text.endswith('라') or span_text.endswith('다'):
                                    structural_bonus = 0.1  # 일반 종결 -> 최소 보너스
                                else:
                                    structural_bonus = 0.0
                                    
                        except Exception as e:
                            print(f"⚠️ 구조 분석 실패, 폴백 사용: {e}")
                            # 폴백: 기존 하드코딩 방식
                            if '이요' in span_text and span_text.endswith('니'):
                                structural_bonus = 0.4  # 대구 구조 확실성 -> 강력한 보너스
                                print(f"🔥 대구 구조 강화: '{span_text[:15]}...' -> +{structural_bonus}")
                            elif span_text.endswith('니') and len(span_text) > 10:
                                structural_bonus = 0.2  # 긴 문장의 '니' 종결 -> 중간 보너스
                            elif span_text.endswith('이요') and len(span_text) > 8:
                                structural_bonus = 0.15  # '이요' 종결 -> 소폭 보너스
                            elif span_text.endswith('라') or span_text.endswith('다'):
                                structural_bonus = 0.1  # 일반 종결 -> 최소 보너스
                            else:
                                structural_bonus = 0.0
                        
                        # 🆕 의미+구조 균형: 토씨 매칭은 생략
                        particle_bonus = 0.0
                        
                        # 최종 점수 계산
                        final_sim = embedding_sim + structural_bonus + particle_bonus
                        
                        # 🔥 구조적 확실성 exponential 보너스
                        if structural_bonus >= 0.4:  # 대구 구조 확실성
                            structural_certainty_bonus = structural_bonus * 2.0  # 구조 확실성 2배 증폭
                            final_sim += structural_certainty_bonus
                            print(f"🚀 구조 확실성 보너스: +{structural_certainty_bonus:.2f}")
                        elif final_sim > 0.8:  # 의미적 고신뢰
                            quality_bonus = (final_sim - 0.8) * 1.5  # 의미 확실성 보너스
                            final_sim += quality_bonus
                        
                        # DP 점수 계산
                        score = dp[i-1, k] + final_sim
                        
                        if score > dp[i, j]:
                            dp[i, j] = score
                            back[i, j] = k
            
            # 최고 점수를 가진 분할 찾기 + 미세 조정
            max_score = dp[N, W]
            if max_score == -np.inf:
                # DP 실패시 폴백
                raise Exception("DP failed to find valid split")
            
            # 현재 최적해
            cuts = [W]
            curr = W
            for i in range(N, 0, -1):
                prev = back[i, curr]
                cuts.append(prev)
                curr = prev
            cuts = cuts[::-1]
            
            current_result = []
            for i in range(N):
                chunk_words = words[cuts[i]:cuts[i+1]]
                chunk = ' '.join(chunk_words)
                current_result.append(chunk)
            
            # 추가 후보들 시도 (다중 스케일 분할점 조정)
            candidate_results = [current_result]
            
            # 1단계: ±1 어절 조정
            for shift in [-1, 1]:
                try:
                    adjusted_cuts = cuts.copy()
                    for i in range(1, len(adjusted_cuts)-1):  # 첫번째와 마지막 제외
                        new_cut = adjusted_cuts[i] + shift
                        if new_cut > adjusted_cuts[i-1] and new_cut < adjusted_cuts[i+1]:
                            adjusted_cuts[i] = new_cut
                    
                    adjusted_result = []
                    for i in range(N):
                        if i < len(adjusted_cuts)-1:
                            chunk_words = words[adjusted_cuts[i]:adjusted_cuts[i+1]]
                            chunk = ' '.join(chunk_words)
                            adjusted_result.append(chunk)
                    
                    if len(adjusted_result) == N:
                        candidate_results.append(adjusted_result)
                except:
                    continue
            
            # 2단계: ±2 어절 조정 (더 큰 변화)
            for shift in [-2, 2]:
                try:
                    adjusted_cuts = cuts.copy()
                    for i in range(1, len(adjusted_cuts)-1):
                        new_cut = adjusted_cuts[i] + shift
                        if new_cut > adjusted_cuts[i-1] + 1 and new_cut < adjusted_cuts[i+1] - 1:
                            adjusted_cuts[i] = new_cut
                    
                    adjusted_result = []
                    for i in range(N):
                        if i < len(adjusted_cuts)-1:
                            chunk_words = words[adjusted_cuts[i]:adjusted_cuts[i+1]]
                            chunk = ' '.join(chunk_words)
                            adjusted_result.append(chunk)
                    
                    if len(adjusted_result) == N:
                        candidate_results.append(adjusted_result)
                except:
                    continue
            
            # 길이 기반 분할도 후보에 추가
            try:
                target_lengths = [len(s) for s in target_sentences]
                total_target_length = sum(target_lengths)
                if total_target_length > 0:
                    length_ratios = [l / total_target_length for l in target_lengths]
                    
                    length_result = []
                    start = 0
                    for i, ratio in enumerate(length_ratios):
                        # 비율에 따라 어절 수 할당 (최소 1개)
                        allocated_words = max(1, int(len(words) * ratio))
                        end = min(start + allocated_words, len(words))
                        
                        # 마지막 구간은 남은 모든 어절 포함
                        if i == len(length_ratios) - 1:
                            end = len(words)
                        
                        chunk_words = words[start:end]
                        if chunk_words:  # 빈 청크 방지
                            length_result.append(' '.join(chunk_words))
                        else:
                            length_result.append('')
                        start = end
                    
                    if len(length_result) == N and start >= len(words):
                        candidate_results.append(length_result)
            except Exception:
                pass
            
            # 균등 분할도 후보에 추가
            try:
                chunk_size = len(words) // target_count
                remainder = len(words) % target_count
                uniform_result = []
                start = 0
                for i in range(target_count):
                    current_size = chunk_size + (1 if i < remainder else 0)
                    end = start + current_size
                    chunk_words = words[start:end]
                    uniform_result.append(' '.join(chunk_words))
                    start = end
                candidate_results.append(uniform_result)
            except Exception:
                pass
            
            # 모든 후보 중 최고 유사도 선택
            best_result = current_result
            best_similarity = -1.0
            
            for candidate in candidate_results:
                try:
                    if len(candidate) == len(target_sentences):
                        cand_embeddings = embed_func(candidate)
                        cand_similarities = []
                        for cand_emb, tgt_emb in zip(cand_embeddings, target_embeddings):
                            cand_norm = np.linalg.norm(cand_emb)
                            tgt_norm = np.linalg.norm(tgt_emb)
                            if cand_norm > 1e-8 and tgt_norm > 1e-8:
                                sim = float(np.dot(cand_emb, tgt_emb) / (cand_norm * tgt_norm))
                            else:
                                sim = 0.0
                            cand_similarities.append(sim)
                        
                        avg_sim = np.mean(cand_similarities)
                        if avg_sim > best_similarity:
                            best_result = candidate
                            best_similarity = avg_sim
                except:
                    continue
            
            result = best_result
            
            # 최종 결과 로깅
            import logging
            logger = logging.getLogger(__name__)
            if best_similarity >= 0:
                logger.info(f"최적 분할 완료 (평균 유사도: {best_similarity:.3f})")
            
            return result
            
        except Exception as e:
            # 로깅으로 변경하여 진행바 간섭 방지
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"임베딩 기반 분할 실패, 균등 분배로 폴백: {e}")

    # 3. 폴백: 어절 기준 균등 분배
    chunk_size = len(words) // target_count
    remainder = len(words) % target_count
    
    result = []
    start = 0
    for i in range(target_count):
        current_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_size
        chunk_words = words[start:end]
        result.append(' '.join(chunk_words))
        start = end
    
    return result