"""괄호 처리 및 마스킹 기능 모듈"""

import regex as re
from typing import List, Tuple

"""설정값과 상수를 관리하는 모듈"""

# 마스킹 템플릿
MASK_TEMPLATE = '[MASK{}]'

# 괄호 종류별 분류
HALF_WIDTH_BRACKETS = [
    ('(', ')'),
    ('[', ']'),
]

FULL_WIDTH_BRACKETS = [
    ('（', '）'),
    ('［', '］'),
]

TRANS_BRACKETS = [
    ('<', '>'),
    ('《', '》'),
    ('〈', '〉'),
    ('「', '」'),
    ('『', '』'),
    ('〔', '〕'),
    ('【', '】'),
    ('〖', '〗'),
    ('〘', '〙'),
    ('〚', '〛'),
]

ALL_BRACKETS = HALF_WIDTH_BRACKETS + FULL_WIDTH_BRACKETS + TRANS_BRACKETS

# 임베딩 설정
DEFAULT_BATCH_SIZE = 20
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'

# 처리 설정
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_CACHE_SIZE = 10000

def mask_brackets(text: str, text_type: str) -> Tuple[str, List[str]]:
    """Mask content within brackets according to rules."""
    assert text_type in {'source', 'target'}, "text_type must be 'source' or 'target'"

    masks: List[str] = []
    mask_id = [0]

    def safe_sub(pattern, repl, s):
        def safe_replacer(m):
            if '[MASK' in m.group(0):
                return m.group(0)
            return repl(m)
        return pattern.sub(safe_replacer, s)

    patterns: List[Tuple[re.Pattern, bool]] = []

    if text_type == 'source':
        for left, right in HALF_WIDTH_BRACKETS:
            patterns.append((re.compile(re.escape(left) + r'[^' + re.escape(left + right) + r']*?' + re.escape(right)), True))
        for left, right in FULL_WIDTH_BRACKETS:
            patterns.append((re.compile(re.escape(left)), False))
            patterns.append((re.compile(re.escape(right)), False))
    elif text_type == 'target':
        for left, right in HALF_WIDTH_BRACKETS + FULL_WIDTH_BRACKETS:
            patterns.append((re.compile(re.escape(left) + r'[^' + re.escape(left + right) + r']*?' + re.escape(right)), True))
        for left, right in TRANS_BRACKETS:
            patterns.append((re.compile(re.escape(left)), False))
            patterns.append((re.compile(re.escape(right)), False))

    def mask_content(s: str, pattern: re.Pattern, content_mask: bool) -> str:
        def replacer(match: re.Match) -> str:
            token = MASK_TEMPLATE.format(mask_id[0])
            masks.append(match.group())
            mask_id[0] += 1
            return token
        return safe_sub(pattern, replacer, s)

    for pattern, content_mask in patterns:
        if content_mask:
            text = mask_content(text, pattern, content_mask)
    for pattern, content_mask in patterns:
        if not content_mask:
            text = mask_content(text, pattern, content_mask)

    return text, masks

def restore_masks(text: str, masks: List[str]) -> str:
    """Restore masked tokens to their original content."""
    for i, original in enumerate(masks):
        text = text.replace(MASK_TEMPLATE.format(i), original)
    return text