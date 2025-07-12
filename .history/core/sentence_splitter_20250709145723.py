"""PA 문장 분할기 - 번역문은 spaCy, 원문은 의미적 병합만 지원"""
from typing import List, Tuple

# 번역문 분할에만 spaCy 사용
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

def split_target_sentences_advanced(text: str, max_length: int = 150, splitter: str = "punctuation") -> List[str]:
    """
    번역문 분할 - 닫는 따옴표 홀로 분할 방지, 길이 초과 문장 의미 기반 재분할
    """
    # 1차: 기본 종결부호 기준 분할
    pattern = r'(?<=[.!?。？！○])(?=\s+[^ ])'
    sentences = re.split(pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 2차: 닫는 따옴표 홀로 분할된 것 병합
    merged = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        # 유니코드 값으로 따옴표 종류 지정
        quote_only = current in ['\u0022', '\u0027', '\u201d', '\u2019', '\u300d', '\u300f']
        # 정규식에도 유니코드 값 사용
        short_quote = (len(current.strip()) <= 3 and 
                      re.match(r'^[\u0022\u0027\u201d\u2019\u300d\u300f]+', current.strip()))
        
        if i > 0 and (quote_only or short_quote):
            merged[-1] += ' ' + current
        else:
            merged.append(current)
        i += 1
        
    if not merged:
        merged = [text]

    # 3차: max_length 초과 문장 spacy로 의미 기반 재분할
    final_sentences = []
    for sent in merged:
        if len(sent) > max_length:
            spacy_sents = split_with_spacy(sent)
            if spacy_sents:
                final_sentences.extend(spacy_sents)
            else:
                # spaCy 분할 실패 시 기존 문장 유지
                final_sentences.append(sent)
        else:
            final_sentences.append(sent)
            
    return final_sentences if final_sentences else [text]

def split_with_spacy(text: str, is_target: bool = True) -> List[str]:
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
    pattern = r'(?<=[。？！○])|(?<=[.!?]\s)'
    segments = re.split(pattern, text)
    return [seg for seg in segments if seg]

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
        (r'[：:]', 1),
        (r'[,，]\s*(?=.{10,})', 1),
        (r'\s+', 1),
    ]
    for pattern, offset in split_patterns:
        match = re.search(pattern, search_text)
        if match:
            return start + match.end()
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

def split_source_by_whitespace_and_align(source: str, target_count: int) -> List[str]:
    """
    원문(한문) 분할: 번역문 분할 개수에 맞춰 순차적으로 분할(병합/패딩), 모든 공백/포맷 100% 보존
    
    주어+발화동사+인용구 병합 시에도 공백이 손상되지 않도록 보장
    """
    if not source.strip():
        return [''] * target_count
    
    delimiter_pattern = r'([：。！？；、，\s]+)'
    parts = re.split(delimiter_pattern, source)
    
    tokens = []
    for part in parts:
        if part:
            tokens.append(part)
    
    if not tokens:
        return [''] * target_count
    
    if len(tokens) <= target_count:
        result = tokens + [''] * (target_count - len(tokens))
        return result[:target_count]
    else:
        chunk_size = len(tokens) // target_count
        remainder = len(tokens) % target_count
        
        result = []
        start = 0
        for i in range(target_count):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            chunk = ''.join(tokens[start:end])
            result.append(chunk)
            start = end
        
        return result

def split_source_by_meaning_units(source: str, target_count: int) -> List[str]:
    """
    원문(한문/한글) 의미 단위 분할: 한글 어미에서 분리되지 않도록 개선하고, 어절 내부 분할을 방지.
    """
    logger.debug(f"[split_source_by_meaning_units] Input source: '{source}', target_count: {target_count}")
    if not source.strip():
        logger.debug(f"[split_source_by_meaning_units] Source is empty, returning empty list.")
        return [''] * target_count
    
    meaning_pattern = r'([。！？；、，：]+)'
    parts = re.split(meaning_pattern, source)
    logger.debug(f"[split_source_by_meaning_units] Parts after re.split: {parts}")
    
    units = []
    i = 0
    while i < len(parts):
        if parts[i].strip():
            current = parts[i]
            if i + 1 < len(parts) and re.match(r'^[。！？；、，：]+', parts[i + 1]):
                current += parts[i + 1]
                i += 2
            else:
                i += 1
            units.append(current)
        else:
            i += 1
    logger.debug(f"[split_source_by_meaning_units] Units after initial grouping: {units}")
    
    def is_korean_ending(u):
        return bool(re.search(r'(다|니다|요|라|까|죠|네|군|구나|구요|네요|랍니다|랍니까|라니|라면|라서|라니까|라더라|라더군|라더라고|라더니|라더냐|라더라구요), u.strip()))

    merged = []
    for u in units:
        if merged and is_korean_ending(u):
            merged[-1] += u
        else:
            merged.append(u)
    units = merged
    logger.debug(f"[split_source_by_meaning_units] Units after Korean ending merge: {units}")
    
    if not units:
        return [''] * target_count
    
    if len(units) == target_count:
        logger.debug(f"[split_source_by_meaning_units] Units count matches target_count, returning: {units}")
        return units
    elif len(units) < target_count:
        logger.debug(f"[split_source_by_meaning_units] Units count less than target_count, attempting to split.")
        while len(units) < target_count:
            splittable_units = [(i, u) for i, u in enumerate(units) if ' ' in u.strip() and len(u) > 10]
            if not splittable_units:
                logger.debug(f"[split_source_by_meaning_units] No more splittable units.")
                break
            
            original_idx, longest_unit = max(splittable_units, key=lambda item: len(item[1]))
            mid = len(longest_unit) // 2
            
            left_pos = longest_unit.rfind(' ', 0, mid)
            right_pos = longest_unit.find(' ', mid)
            
            split_at = -1
            if left_pos != -1 and right_pos != -1:
                split_at = left_pos if mid - left_pos <= right_pos - mid else right_pos
            else:
                split_at = left_pos if left_pos != -1 else right_pos

            if split_at != -1:
                part1 = longest_unit[:split_at].strip()
                part2 = longest_unit[split_at+1:].strip()
                if part1 and part2:
                    units[original_idx:original_idx+1] = [part1, part2]
                    logger.debug(f"[split_source_by_meaning_units] Split unit: {longest_unit} -> {[part1, part2]}")
                else:
                    logger.debug(f"[split_source_by_meaning_units] Split resulted in empty part, breaking.")
                    break
            else:
                logger.debug(f"[split_source_by_meaning_units] No valid split point found, breaking.")
                break
        
        units.extend([''] * (target_count - len(units)))
        logger.debug(f"[split_source_by_meaning_units] Units after splitting and padding: {units}")
    else:
        logger.debug(f"[split_source_by_meaning_units] Units count more than target_count, attempting to merge.")
        while len(units) > target_count:
            min_len = float('inf')
            merge_idx = 0
            for i in range(len(units) - 1):
                combined_len = len(units[i]) + len(units[i + 1])
                if combined_len < min_len:
                    min_len = combined_len
                    merge_idx = i
            units[merge_idx:merge_idx+2] = [units[merge_idx] + units[merge_idx + 1]]
            logger.debug(f"[split_source_by_meaning_units] Merged units, current units: {units}")
    
    logger.debug(f"[split_source_by_meaning_units] Final units before slicing: {units}")
    return units[:target_count]