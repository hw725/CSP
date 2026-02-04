"""
공통 텍스트 정규화 모듈 - SA/PA 모두 사용
"""

import re
from typing import Tuple

def normalize_text(text: str, normalize_brackets: bool = False) -> str:
    """
    텍스트 정규화: 공백, 개행, 특수문자 정리

    Args:
        text: 정규화할 텍스트
        normalize_brackets: [-...] 패턴 제거 여부 (기본: False, PA는 True)

    Returns:
        정규화된 텍스트
    """
    try:
        t = str(text).strip()
        if not t:
            return ""

        # 1단계: 전각 공백/제어문자/개행 정리 → 반각 공백
        t = t.replace("\u3000", " ")  # 전각 공백
        t = t.replace("\r\n", " ")  # Windows 개행
        t = t.replace("\r", " ")  # Mac 개행 (레거시)
        t = t.replace("\n", " ")  # Unix 개행
        t = t.replace("\t", " ")  # 탭

        # 2단계: [-...] 패턴 제거 (편집 주석, 선택적)
        if normalize_brackets:
            t = re.sub(r"\[-[^\]]*\]", "", t)

        # 3단계: 중복 공백 축소
        t = re.sub(r"\s+", " ", t)

        # 4단계: 문장부호 앞뒤 공백 정리
        # 구두점 앞: 공백 제거
        t = re.sub(r"\s+([.,!?!？，、；;：:])", r"\1", t)
        # 구두점 뒤: 공백 유지 (최대 1개)
        t = re.sub(r"([.,!?!？，、；;：:])\s*", r"\1 ", t)

        # 5단계: 마지막 공백 정리
        t = t.strip()

        # 6단계: 마지막 문자가 구두점이면 뒤의 공백 제거
        if t and t[-1] in ".,!?!？，、；;：:":
            t = t.rstrip()

        return t

    except Exception as e:
        print(f"⚠️ 정규화 오류: {e}")
        return str(text).strip()

def normalize_source_and_target(
    src_text: str, tgt_text: str, normalize_brackets_in_tgt: bool = False
) -> Tuple[str, str]:
    """
    원문/번역문 쌍 정규화

    Args:
        src_text: 원문
        tgt_text: 번역문
        normalize_brackets_in_tgt: 번역문의 [-...] 패턴 제거 여부

    Returns:
        (정규화된 원문, 정규화된 번역문)
    """
    src_norm = normalize_text(src_text, normalize_brackets=False)
    tgt_norm = normalize_text(tgt_text, normalize_brackets=normalize_brackets_in_tgt)

    return src_norm, tgt_norm

def normalize_for_similarity(text: str) -> str:
    """
    유사도 비교용 정규화 (더 적극적)
    - 공백, 개행 모두 제거
    - [-텍스트] 패턴 보존! (무결성 검증용)
    """
    t = normalize_text(text, normalize_brackets=False)  # ✅ 변경: [-텍스트] 보존
    # 모든 공백 제거
    t = re.sub(r"\s+", "", t)
    return t
