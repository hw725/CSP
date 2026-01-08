"""PA 문장 분할기 - SuPar-Kanbun & Stanza 기반 (spaCy 대체)"""
from typing import List, Tuple
import torch
import os
import json
import hashlib
import numpy as np
from pathlib import Path
import re
import sys

# BGE Embedder import (의미 기반 경계 감지용)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
    from embedders import get_embedding_manager
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False

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
    번역문 분할 - 종결 구두점 우선, 구두점 없으면 의미 기반 경계 감지
    
    ⭐️ 개선: 의미 기반 경계 감지로 구두점 없는 텍스트도 분할 가능
    """
    # 1) 구두점 기반 분할 시도
    strong_end_pattern = r'(?<=[。！？.!?])\s+'
    sentences = re.split(strong_end_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 구두점이 없어서 분할 실패한 경우 의미 기반 경계 감지 (PA: 문장 단위)
    if len(sentences) == 1 and len(text) > 100 and BGE_AVAILABLE:
        try:
            offsets = detect_semantic_boundaries(text, window_size=80, threshold=0.75, min_segment_length=30)
            if len(offsets) > 1:
                sentences = [text[start:end].strip() for start, end in offsets if start < end]
                sentences = [s for s in sentences if s]
                print(f"✅ 의미 기반 경계 감지: {len(sentences)}개 문장")
        except Exception as e:
            print(f"⚠️ 의미 기반 경계 감지 실패: {e}")
    
    # 2) SikuBERT 전처리 (필요 시)
    if use_siku_preprocessing and contains_chinese(text):
        text = preprocess_with_siku_tokenization(text)

    # 3) 종결 구두점 기준 1차 분할 (강한 종결만)
    strong_end_pattern = r'(?<=[。！？.!?])\s+'  # 중국어/영문 종결부호
    sentences = re.split(strong_end_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # 4) 너무 긴 문장만 예외적으로 콤마에서 1회 분할 (괄호/브래킷 내부 제외)
    def split_long_by_comma_outside_brackets(s: str, limit: int) -> List[str]:
        if len(s) <= limit:
            return [s]
        # 괄호/브래킷 내부 콤마는 무시
        level = 0
        split_pos = -1
        for i, ch in enumerate(s):
            if ch in '([':
                level += 1
            elif ch in ')]' and level > 0:
                level -= 1
            elif ch == ',' and level == 0:
                split_pos = i
                break
        if split_pos > 0:
            left = s[:split_pos].strip()
            right = s[split_pos+1:].strip()
            # 비병렬 콤마 추정: 좌/우 구간이 모두 충분히 길고, 좌측이 종결구두점으로 끝나지 않는 경우
            if len(left) > limit*0.4 and len(right) > limit*0.4 and (not re.search(r'[。！？.!?]$', left)):
                return [left, right]
        return [s]

    adjusted = []
    for s in sentences:
        parts = split_long_by_comma_outside_brackets(s, max_length)
        adjusted.extend(parts)

    # 5) 닫는 따옴표 단독 문장 방지: ", ', ”, ’, 」, 』, 〉, 》 등 단독 토큰을 앞 문장에 붙인다
    def merge_lonely_closers(segs: List[str]) -> List[str]:
        if not segs:
            return segs
        closers_chars = set([
            ")", "]", "}", "\"", "'",  # ASCII closers including quote/double-quote
            "\u201D", "\u2019",            # ” ’
            "\u3009", "\u300B", "\u300D", "\u300F",  # 〉 》 」 』
            "\u3011", "\u3015",              # 】 〕
            "\uFF07", "\uFF3D", "\uFF5D"   # FULLWIDTH ' ] }
        ])
        # 닫는 따옴표 세트 (문장 시작 검사용)
        closing_quotes = set([
            "\"", "'",
            "\u201D", "\u2019",
            "\u300D", "\u300F",
        ])
        
        merged: List[str] = []
        for seg in segs:
            s = seg.strip()
            if not s:
                continue
            
            # Case 1: 전체가 닫는 기호로만 구성된 경우
            if all(ch in closers_chars for ch in s):
                if merged:
                    merged[-1] = merged[-1].rstrip() + s
                else:
                    merged.append(s)
                continue
            
            # Case 2: 문장이 닫는 따옴표로 시작하는 경우
            if s[0] in closing_quotes and merged:
                # 앞부분의 연속된 닫는 따옴표들 추출
                i = 0
                while i < len(s) and s[i] in closing_quotes:
                    i += 1
                leading = s[:i]
                remaining = s[i:].lstrip()
                
                # 이전 문장에 닫는 따옴표 붙이기
                merged[-1] = merged[-1].rstrip() + leading
                
                # 남은 텍스트가 있으면 현재 문장으로
                if remaining:
                    merged.append(remaining)
            else:
                merged.append(s)
        
        return merged

    adjusted = merge_lonely_closers(adjusted)
    return adjusted

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
    """중국고전 문장 분할 패턴 강화 + 의미 기반 경계 감지"""
    
    # 1) 구두점 패턴 시도
    classical_chinese_patterns = [
        r'(?<=[。？！])',  # 기본 문장 부호
        r'(?<=하고)\s*',   # '하고' 뒤 분할
        r'(?<=하여)\s*',   # '하여' 뒤 분할  
        r'(?<=하니)\s*',   # '하니' 뒤 분할
    ]
    
    for pattern in classical_chinese_patterns:
        if re.search(pattern, text):
            segments = re.split(pattern, text)
            segments = [seg.strip() for seg in segments if seg.strip()]
            if len(segments) > 1:
                return segments
    
    # 2) 구두점이 없으면 의미 기반 경계 감지 (PA: 문장 단위)
    if len(text) > 100 and BGE_AVAILABLE:
        try:
            offsets = detect_semantic_boundaries(text, window_size=80, threshold=0.75, min_segment_length=30)
            if len(offsets) > 1:
                sentences = [text[start:end].strip() for start, end in offsets if start < end]
                sentences = [s for s in sentences if s]
                if sentences:
                    print(f"✅ 의미 기반 경계 감지: {len(sentences)}개 문장")
                    return sentences
        except Exception as e:
            print(f"⚠️ 의미 기반 경계 감지 실패: {e}")
    
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

def detect_semantic_boundaries(text: str, window_size: int = 50, threshold: float = 0.75, min_segment_length: int = 20) -> List[Tuple[int, int]]:
    """
    의미 기반 경계 감지 (BGE 임베딩 + 코사인 유사도)
    
    슬라이딩 윈도우로 텍스트를 스캔하면서 의미 변화가 큰 지점을 경계로 판단
    
    Args:
        text: 분할할 텍스트
        window_size: 슬라이딩 윈도우 크기 (문자 단위)
                    - PA(문장 단위): 80-100 권장
                    - SA(구 단위): 20-30 권장
        threshold: 코사인 유사도 임계값 (낮을수록 경계로 판단)
        min_segment_length: 최소 세그먼트 길이 (이보다 짧으면 병합)
    
    Returns:
        [(0, 45), (45, 92), ...] 형태의 오프셋 쌍
    """
    if not BGE_AVAILABLE or len(text) < window_size * 2:
        return [(0, len(text))]
    
    try:
        embedder = get_embedding_manager()
        
        # 1. 슬라이딩 윈도우로 텍스트 세그먼트 추출
        segments = []
        segment_offsets = []
        step_size = window_size // 2  # 50% 오버랩
        
        for start in range(0, len(text) - window_size + 1, step_size):
            end = start + window_size
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
                segment_offsets.append((start, end))
        
        # 마지막 부분 처리
        if segment_offsets and segment_offsets[-1][1] < len(text):
            last_segment = text[segment_offsets[-1][1]:].strip()
            if last_segment:
                segments.append(last_segment)
                segment_offsets.append((segment_offsets[-1][1], len(text)))
        
        if len(segments) < 2:
            return [(0, len(text))]
        
        # 2. BGE 임베딩 계산
        embeddings = embedder.compute_embeddings_with_cache(
            segments, 
            batch_size=32,
            use_multi_vector=False  # Dense vector만 사용 (1024차원)
        )
        
        if embeddings is None or len(embeddings) == 0:
            return [(0, len(text))]
        
        # 3. 인접 세그먼트 간 코사인 유사도 계산
        similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
            else:
                cosine_sim = 1.0
            
            similarities.append(cosine_sim)
        
        # 4. 유사도가 threshold 이하인 지점을 경계로 판단
        boundary_candidates = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundary_pos = (segment_offsets[i][1] + segment_offsets[i + 1][0]) // 2
                boundary_candidates.append(boundary_pos)
        
        if not boundary_candidates:
            return [(0, len(text))]
        
        # 5. 경계로 문장 분할
        boundaries = []
        start = 0
        for pos in sorted(set(boundary_candidates)):
            if pos > start:
                boundaries.append((start, pos))
                start = pos
        
        if start < len(text):
            boundaries.append((start, len(text)))
        
        # 너무 짧은 세그먼트 병합
        merged_boundaries = []
        for start, end in boundaries:
            if len(merged_boundaries) > 0 and end - start < min_segment_length:
                prev_start, _ = merged_boundaries[-1]
                merged_boundaries[-1] = (prev_start, end)
            else:
                merged_boundaries.append((start, end))
        
        return merged_boundaries if merged_boundaries else [(0, len(text))]
        
    except Exception as e:
        print(f"⚠️ 의미 기반 경계 감지 실패: {e}")
        return [(0, len(text))]

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
    🎯 원본 텍스트 형태 절대 보존: augmentation은 임베딩 계산용만 사용
    
    Args:
        source: 원문 텍스트
        target_count: 분할할 개수
        target_sentences: 번역문 문장들 (의미적 매칭용)
        embedder_name: 사용할 임베더 이름 ("bge" 또는 "openai")
        embedder_func: 외부에서 전달된 임베더 함수 (선택적)
    """
    def augment_source_hanja(text: str) -> str:
        """원문 괄호 속 한자 노출 (임베딩용만)"""
        import regex as re
        han_regex = re.compile(r"\p{Han}")
        def repl(m: re.Match) -> str:
            inner = m.group(1)
            hanja = ''.join(ch for ch in inner if han_regex.match(ch))
            return f" {hanja} " if hanja else " "
        text_aug = re.sub(r"\(([^)]*)\)", repl, text)
        text_aug = re.sub(r"\[([^\]]*)\]", repl, text_aug)
        return ' '.join(text_aug.split())
    
    def mask_unaligned_segments(text: str):
        """비대응 표시 구간([- (...)])에서 [, -, ]만 토큰으로 마스킹"""
        pattern = re.compile(r"\[-\(([^)]*)\)\]")
        mapping = []  # (token, symbol)

        def repl(match):
            seq = len(mapping) // 3
            token_l = f"__UNALIGNED_L_{seq}__"
            token_h = f"__UNALIGNED_H_{seq}__"
            token_r = f"__UNALIGNED_R_{seq}__"
            # 순서대로 복원할 수 있도록 매핑 저장
            mapping.extend([
                (token_l, "["),
                (token_h, "-"),
                (token_r, "]"),
            ])
            inner = match.group(1)
            return f"{token_l}{token_h}({inner}){token_r}"

        masked = pattern.sub(repl, text)
        return masked, mapping

    def unmask_text(text: str, mapping):
        for token, original in mapping:
            text = text.replace(token, original)
        return text

    def unmask_list(chunks: List[str], mapping):
        return [unmask_text(chunk, mapping) for chunk in chunks]
    if not source.strip():
        return [''] * target_count

    # 0. 비대응 구간 마스킹 후 분할 처리
    masked_source, mask_map = mask_unaligned_segments(source)

    # 🎯 원본 어절 보존 (최종 출력용) - 어절 경계의 원문 오프셋을 수집하여 슬라이스로 재조립
    # 공백을 기준으로 어절(span)들을 추출하되, 최종 결과 조립은 원문 슬라이스로 수행
    word_spans = []  # List[Tuple[start, end]] for each non-whitespace token
    for m in re.finditer(r"\S+", masked_source):
        word_spans.append((m.start(), m.end()))
    words_original = [masked_source[s:e] for (s, e) in word_spans]

    def slice_segment_by_word_index(i_start: int, i_end: int) -> str:
        """어절 인덱스 구간 [i_start, i_end) 에 해당하는 원문 슬라이스를 그대로 반환
        - 경계 사이의 공백/개행 포함 원문을 보존
        - 인덱스가 유효하지 않으면 빈 문자열 반환
        """
        if i_start >= i_end or i_start < 0 or i_end > len(word_spans):
            return ''
        start_char = word_spans[i_start][0]
        end_char = word_spans[i_end - 1][1]
        return masked_source[start_char:end_char]
    
    # 🎯 Augmented 어절 생성 (임베딩 계산용만)
    source_augmented = augment_source_hanja(masked_source)
    words_aug = source_augmented.split()
    
    # 1. 어절 단위로 분할 (공백 기준, 어절 내부 절대 분할 금지!)
    if not words_original:
        return [''] * target_count

    # 어절이 target_count보다 적으면 균등 분배 (원본 사용!)
    if len(words_original) <= target_count:
        result = []
        for i in range(target_count):
            if i < len(words_original):
                # 단일 어절 슬라이스 (원문 보존)
                result.append(slice_segment_by_word_index(i, i+1))
            else:
                result.append('')
        return unmask_list(result, mask_map)
    
    # 1-1. PA 원문에 의미 기반 경계 감지 적용 (어절 경계만 지키면서 문장 단위로)
    # 구두점이 명확하지 않으면 BGE 임베딩으로 의미 변화 지점 감지
    if len(words_original) > 10 and BGE_AVAILABLE:  # 어절 10개 이상이면 의미 분석
        try:
            # 어절을 공백으로 재조합
            reconstructed_text = ' '.join(words_original)
            
            # 의미 기반 경계 감지 (PA: 문장 단위, window_size=80)
            offsets = detect_semantic_boundaries(
                reconstructed_text, 
                window_size=80, 
                threshold=0.75, 
                min_segment_length=30
            )
            
            if len(offsets) > 1 and len(offsets) <= target_count:
                # 오프셋을 어절 경계로 변환
                result = []
                for start, end in offsets:
                    # 재구성 텍스트의 오프셋 → 어절 인덱스 매핑
                    # reconstructed_text = ' '.join(words_original)
                    # 각 어절의 시작 오프셋을 누적하여 인덱스를 찾는다
                    recon_starts = []
                    pos = 0
                    for w in words_original:
                        recon_starts.append(pos)
                        pos += len(w) + 1  # 공백 포함 누적
                    recon_ends = [s + len(w) for s, w in zip(recon_starts, words_original)]

                    # 시작/끝 인덱스 추정
                    word_start = 0
                    for idx, s in enumerate(recon_starts):
                        if s >= start:
                            word_start = idx
                            break
                    else:
                        word_start = len(words_original)

                    word_end = len(words_original)
                    for idx, e in enumerate(recon_ends):
                        if e >= end:
                            word_end = idx + 1
                            break

                    if word_start < word_end:
                        segment = slice_segment_by_word_index(word_start, word_end)
                        result.append(segment)
                
                if len(result) > 0:
                    print(f"✅ PA 원문 의미 기반 분할: {len(result)}개 어절 그룹")
                    return unmask_list(result, mask_map)
        except Exception as e:
            print(f"⚠️ PA 원문 의미 기반 분할 실패: {e}")
    
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
                # PA에서도 SA와 동일한 안정적 멀티벡터 설정을 공유하므로, 용도를 명확히 표기
                print("✅ BGE-M3 Multi-Vector 임베딩 사용 (PA 분할용, 작은 배치)")
                
                # SA 방식: 작은 배치 크기 + Multi-vector로 안전하게 처리
                target_embeddings = embedder.compute_embeddings_with_cache(
                    target_sentences, 
                    batch_size=4, 
                    use_multi_vector=True
                )
                
                def bge_embed_source(texts):
                    # PA에서도 SA처럼 작은 배치 + Multi-vector로 처리
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
            
            # 어절 경계 기반 스팬 임베딩 계산 (augmented 사용!)
            N, W = target_count, len(words_aug)
            span_cache = {}
            
            # 어절 단위로 스팬 생성 (어절 내부 분할 금지!)
            for i in range(W):
                for j in range(i + 1, W + 1):
                    span_words_aug = words_aug[i:j]
                    span_text = ' '.join(span_words_aug).strip()
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
                                print(f"🎯 강신뢰 매칭: {' '.join(words_original[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 5.0")
                            elif base_cosine > 0.10:  # 0.10-0.15 = 좋은 매칭  
                                embedding_sim = base_cosine * 4.0 + 0.3  # 좋은 가중치
                                print(f"🔍 중강신뢰 매칭: {' '.join(words_original[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 4.0")
                            elif base_cosine > 0.05:  # 0.05-0.10 = 보통 매칭
                                embedding_sim = base_cosine * 3.0 + 0.2  # 보통 가중치
                                print(f"🔍 중신뢰 매칭: {' '.join(words_original[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 3.0")
                            elif base_cosine > 0.01:  # 0.01-0.05 = 약한 매칭
                                embedding_sim = base_cosine * 2.0 + 0.1  # 약한 가중치
                                print(f"⚠️ 저신뢰 매칭: {' '.join(words_original[k:j])[:15]}... -> 유사도 {base_cosine:.3f} * 2.0")
                            elif base_cosine > -0.05:  # -0.05-0.01 = 매우 약한 매칭
                                embedding_sim = base_cosine + 0.05  # 최소 보상
                                print(f"❌ 부정매칭: {' '.join(words_original[k:j])[:15]}... -> 유사도 {base_cosine:.3f} + 0.05")
                            else:
                                embedding_sim = base_cosine * 0.2  # 임계값 이하는 페널티
                                print(f"❌ 부정매칭: {span_text[:15]}... -> 유사도 {base_cosine:.3f} * 0.2")
                        else:
                            embedding_sim = 0.0
                        
                        # 🔥 구조적 가중치 대폭 강화 (의미+구조 균형)
                        span_words = words_original[k:j]
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
                chunk = slice_segment_by_word_index(cuts[i], cuts[i+1])
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
                            chunk = slice_segment_by_word_index(adjusted_cuts[i], adjusted_cuts[i+1])
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
                            chunk = slice_segment_by_word_index(adjusted_cuts[i], adjusted_cuts[i+1])
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
                        allocated_words = max(1, int(len(words_original) * ratio))
                        end = min(start + allocated_words, len(words_original))
                        
                        # 마지막 구간은 남은 모든 어절 포함
                        if i == len(length_ratios) - 1:
                            end = len(words_original)
                        
                        if start < end:
                            length_result.append(slice_segment_by_word_index(start, end))
                        else:
                            length_result.append('')
                        start = end
                    
                    if len(length_result) == N and start >= len(words_original):
                        candidate_results.append(length_result)
            except Exception:
                pass
            
            # 균등 분할도 후보에 추가
            try:
                chunk_size = len(words_original) // target_count
                remainder = len(words_original) % target_count
                uniform_result = []
                start = 0
                for i in range(target_count):
                    current_size = chunk_size + (1 if i < remainder else 0)
                    end = start + current_size
                    uniform_result.append(slice_segment_by_word_index(start, end))
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
            
            return unmask_list(result, mask_map)
            
        except Exception as e:
            # 로깅으로 변경하여 진행바 간섭 방지
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"임베딩 기반 분할 실패, 균등 분배로 폴백: {e}")

    # 3. 폴백: 어절 기준 균등 분배 (🎯 원본 어절 사용!)
    chunk_size = len(words_original) // target_count
    remainder = len(words_original) % target_count
    
    result = []
    start = 0
    for i in range(target_count):
        current_size = chunk_size + (1 if i < remainder else 0)
        end = start + current_size
        chunk_words_orig = words_original[start:end]
        result.append(' '.join(chunk_words_orig))
        start = end
    
    return unmask_list(result, mask_map)