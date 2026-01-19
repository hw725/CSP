"""텍스트 정규화 유틸리티

중복 방지를 위해 공통 함수를 여기에 모음.
"""

import re


def normalize_text(
    text: str,
    collapse_newlines: bool = True,
    strip_bracket_minus: bool = True
) -> str:
    """텍스트 정규화
    
    Args:
        text: 정규화할 텍스트
        collapse_newlines: 개행을 공백으로 치환
        strip_bracket_minus: [- ... ] 주석 제거
    
    Returns:
        정규화된 텍스트
    """
    if text is None:
        return ""
    
    s = str(text)
    
    if strip_bracket_minus:
        # [- ... ] 주석 제거 (비탐욕 + DOTALL로 줄바꿈 포함)
        s = re.sub(r"\[\-.*?\]", "", s, flags=re.DOTALL)
    
    if collapse_newlines:
        s = s.replace("\r", " ").replace("\n", " ")
    
    # 다중 공백 압축
    s = re.sub(r"\s+", " ", s).strip()
    
    return s
