"""PA 문장 분할기 - 하이브리드 토크나이저 기반 (RoBERTa-Hanja + Kiwipiepy + SikuBERT)"""
from typing import List, Tuple
import torch

# 번역문 분할에만 spaCy 사용 (폴백용)
import re
import regex
try:
    import spacy
    nlp_ko = spacy.load("ko_core_news_lg")
except Exception:
    nlp_ko = None
try:
    nlp_zh = spacy.load("zh_core_web_lg")
except Exception:
    nlp_zh = None

# 하이브리드 토크나이저 추가
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    # 중국어: SikuBERT/AnchiBERT (기존 유지)
    from common.tokenizers import (
        get_siku_tokenizer,
        get_anchi_tokenizer,
        siku_get_embeddings,
        siku_similarity,
        anchi_get_embeddings,
        anchi_similarity,
        detect_chinese_period,
        PeriodDetector
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
    anchi_tokenizer = get_anchi_tokenizer()
    period_detector = PeriodDetector()
    
    # 🆕 하이브리드 한국어 토크나이저 초기화
    hybrid_korean_tokenizer = get_hybrid_korean_tokenizer()
    
    print("✅ PA: 하이브리드 토크나이저 초기화 완료 (중국어: SikuBERT/AnchiBERT, 한국어: RoBERTa-Hanja+Kiwipiepy)")
    
except Exception as e:
    siku_tokenizer = None
    anchi_tokenizer = None  
    period_detector = None
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

def split_with_spacy(text: str, is_target: bool = True, use_siku_preprocessing: bool = True) -> List[str]:
    # SikuBERT 전처리 적용
    if use_siku_preprocessing and contains_chinese(text):
        text = preprocess_with_siku_tokenization(text)
    
    if contains_chinese(text):
        nlp_model = nlp_zh
    else:
        nlp_model = nlp_ko
    if not nlp_model:
        return []
    try:
        doc = nlp_model(text)
        return [sent.text for sent in doc.sents if sent.text]
    except Exception:
        return []

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

def split_source_by_whitespace_and_align(source: str, target_count: int, target_sentences: List[str] = None) -> List[str]:
    """
    원문(한문) 분할: 어절 경계에서만 분할, 어절 내부 절대 분할 금지!
    
    Args:
        source: 원문 텍스트
        target_count: 분할할 개수
        target_sentences: 번역문 문장들 (의미적 매칭용)
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
            # SikuBERT 임베딩 사용 
            from common.tokenizers import siku_get_embeddings, anchi_get_embeddings
            
            try:
                target_embeddings = siku_get_embeddings(target_sentences, batch_size=32)
                embed_func = lambda texts: siku_get_embeddings(texts, batch_size=32)
            except Exception as e:
                target_embeddings = anchi_get_embeddings(target_sentences, batch_size=32)
                embed_func = lambda texts: anchi_get_embeddings(texts, batch_size=32)
            
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
            
            # DP로 어절 경계에서만 최적 분할 찾기
            for i in range(1, N+1):
                target_emb = target_embeddings[i-1]
                target_norm = np.linalg.norm(target_emb)
                
                for j in range(i, W-(N-i)+1):
                    for k in range(i-1, j):
                        if (k, j) not in span_cache:
                            continue
                        
                        span_emb = span_cache[(k, j)]
                        span_norm = np.linalg.norm(span_emb)
                        
                        # 코사인 유사도 계산
                        if target_norm > 1e-8 and span_norm > 1e-8:
                            embedding_sim = float(np.dot(target_emb, span_emb) / (target_norm * span_norm))
                        else:
                            embedding_sim = 0.0
                        
                        # 구조적 힌트 보정 (어절 단위) - 하이브리드 방식
                        span_words = words[k:j]
                        span_text = ' '.join(span_words).strip()
                        structural_bonus = 0.0
                        
                        # 🆕 방법 1: Kiwipiepy 동적 어미 감지 (우선 시도)
                        morphological_bonus = 0.0
                        if span_text:
                            try:
                                from common.korean_particle_matcher import get_korean_particle_matcher
                                matcher = get_korean_particle_matcher()
                                
                                # 형태소 분석으로 어미 감지
                                morphs = matcher._get_kiwi_tokenizer().analyze(span_text, top_n=1)[0][0]
                                last_morph = morphs[-1] if morphs else None
                                
                                if last_morph:
                                    pos_tag = last_morph[1]  # 품사 태그
                                    morph_text = last_morph[0]  # 형태소 텍스트
                                    
                                    # 종결어미 자동 감지 (낮은 보조 가중치)
                                    if pos_tag in ['EF']:  # 종결어미
                                        if morph_text in ['도다', '로다', '구나']:
                                            morphological_bonus = 0.08  # 강한 감탄 종결 (0.35→0.08)
                                        elif morph_text in ['구려', '어라', '지어다']:
                                            morphological_bonus = 0.06  # 일반 감탄 종결 (0.3→0.06)
                                        else:
                                            morphological_bonus = 0.05  # 기타 종결어미 (0.25→0.05)
                                    
                                    # 연결어미 자동 감지 (미미한 가중치)
                                    elif pos_tag in ['EC']:  # 연결어미
                                        if morph_text in ['이고', '며', '여', '니']:
                                            morphological_bonus = 0.03  # 일반 연결어미 (0.2→0.03)
                                        else:
                                            morphological_bonus = 0.02  # 기타 연결어미 (0.15→0.02)
                                        
                            except Exception:
                                morphological_bonus = 0.0  # 형태소 분석 실패
                        
                        # 🆕 방법 2: 하드코딩 패턴 매칭 (폴백 + 보완)
                        hardcoded_bonus = 0.0
                        if span_text:
                            # 앞 부분 텍스트도 확인 (한문 원문과 한글 어미 분리 케이스)
                            preceding_text = ' '.join(words[max(0, k-10):k]).strip() if k > 0 else ""
                            full_context = preceding_text + " " + span_text if preceding_text else span_text
                            
                            # 한자 어미 패턴 (미미한 참고 가중치로 조정)
                            if re.search(r'曰\w+이요$', span_text):
                                hardcoded_bonus = 0.08  # 가장 높은 가중치 (0.4→0.08)
                            elif span_text.endswith('이요'):
                                hardcoded_bonus = 0.06  # (0.25→0.06)
                            elif span_text.endswith('라'):
                                hardcoded_bonus = 0.08  # (0.4→0.08)
                            elif span_text.endswith('也'):
                                hardcoded_bonus = 0.05  # (0.3→0.05)
                            elif span_text.endswith('矣'):
                                hardcoded_bonus = 0.08  # (0.4→0.08)
                            # 🆕 고어 감탄 종결어미 (확실한 패턴만, 참고용 가중치)
                            elif span_text.endswith('고') and not span_text.endswith('이고'):
                                # 현재 span이나 앞 부분에서 한문 WH의문사 확인
                                wh_question_words = ['何', '誰', '安', '孰', '焉', '奚', '胡', '曷']
                                has_wh = any(word in full_context for word in wh_question_words)
                                if has_wh:
                                    hardcoded_bonus = 0.08  # WH의문사 + "고" 조합 (0.4→0.08)
                                # else: 일반적인 "고"는 제외 (연결어미 가능성)
                            elif span_text.endswith('아') and len(span_text) > 1:
                                # "아" 감탄 종결어미: 한문 의문허사가 있을 때만
                                question_particles = ['乎', '耶', '歟', '與', '邪', '也']
                                has_question_particle = any(particle in full_context for particle in question_particles)
                                if has_question_particle:
                                    hardcoded_bonus = 0.07  # 의문허사 + "아" 조합 (0.35→0.07)
                                # else: 일반적인 "아"는 제외
                            elif span_text.endswith('오') and not span_text.endswith('이오') and len(span_text) > 1:
                                # "오" 감탁 종결어미: WH의문사가 있을 때만
                                wh_question_words = ['何', '誰', '安', '孰', '焉', '奚', '胡', '曷']
                                has_wh = any(word in full_context for word in wh_question_words)
                                if has_wh:
                                    hardcoded_bonus = 0.07  # WH의문사 + "오" 조합 (0.35→0.07)
                                # else: 일반적인 "오"는 제외
                            # 기존 고어 감탄 종결어미들 (참고용 미미한 가중치)
                            elif span_text.endswith('구나'):
                                hardcoded_bonus = 0.07  # (0.35→0.07)
                            elif span_text.endswith('도다'):
                                hardcoded_bonus = 0.07  # (0.35→0.07)
                            elif span_text.endswith('로다'):
                                hardcoded_bonus = 0.07  # (0.35→0.07)
                            elif span_text.endswith('구려'):
                                hardcoded_bonus = 0.05  # (0.3→0.05)
                            elif span_text.endswith('어라'):
                                hardcoded_bonus = 0.05  # (0.3→0.05)
                            elif span_text.endswith('을지어다'):
                                hardcoded_bonus = 0.05  # (0.3→0.05)
                            elif span_text.endswith('지어다'):
                                hardcoded_bonus = 0.05  # (0.3→0.05)
                            # 연결 어미들 (미미한 참고 가중치)
                            elif span_text.endswith('이고'):
                                hardcoded_bonus = 0.03  # 연결어미 "이고" (0.2→0.03)
                            elif span_text.endswith('하고'):
                                hardcoded_bonus = 0.03  # (0.2→0.03)
                            elif span_text.endswith('하여'):
                                hardcoded_bonus = 0.03  # (0.2→0.03)
                            elif span_text.endswith('하니'):
                                hardcoded_bonus = 0.03  # (0.2→0.03)
                            elif span_text.endswith('이며'):
                                hardcoded_bonus = 0.02  # (0.15→0.02)
                            elif span_text.endswith('것을'):
                                hardcoded_bonus = 0.02  # (0.15→0.02)
                            elif span_text.endswith('것이'):
                                hardcoded_bonus = 0.02  # (0.15→0.02)
                        
                        # 🆕 하이브리드 선택: 더 높은 가중치 사용
                        structural_bonus = max(morphological_bonus, hardcoded_bonus)
                        
                        # 디버깅용 로깅 (필요시)
                        if morphological_bonus > 0 or hardcoded_bonus > 0:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.debug(f"어미 감지: '{span_text}' -> 형태소:{morphological_bonus:.2f}, 하드코딩:{hardcoded_bonus:.2f}, 최종:{structural_bonus:.2f}")
                        
                        # 토씨 매칭 보정 (미미한 참고 가중치)
                        try:
                            from common.korean_particle_matcher import get_korean_particle_matcher
                            matcher = get_korean_particle_matcher()
                            target_sentence = target_sentences[i-1] if i <= len(target_sentences) else ""
                            span_particles = matcher.extract_particles_from_text(span_text)
                            target_particles = matcher.extract_particles_from_text(target_sentence)
                            particle_sim = matcher.calculate_particle_similarity(span_particles, target_particles)
                            particle_bonus = particle_sim * 0.05  # 0.3 → 0.05 (의미 우선, 토씨는 참고만)
                        except Exception:
                            particle_bonus = 0.0
                        
                        # 최종 점수
                        final_sim = embedding_sim + structural_bonus + particle_bonus
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