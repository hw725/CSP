"""
서지정보 추출기 — XML 원문/번역문에서 서지정보(bibliographic metadata) 추출

출력 양식:
<classics name="한글서명+권" id="jti_code">
  <서지정보>
    <대표서명>한문제목</대표서명>
    <대표서명한글>한글제목</대표서명한글>
    <저자></저자>
    <번역서명>역주유형+한문제목+권</번역서명>
    <번역서명한글>역주유형+한글제목+권</번역서명한글>
    <역자></역자>
    <번역서발행년도>YYYY</번역서발행년도>
    <번역서발행자></번역서발행자>
    <DB식별자>jti_code</DB식별자>
  </서지정보>
</classics>

저자, 역자는 XML에 포함되어 있지 않아 빈 값으로 출력.
번역서발행자는 '전통문화연구회'로 통일.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# 역주 유형 한문 매핑
BRACKET_TYPE_HANJA = {
    "역주": "譯註",
    "현토": "懸吐",
}


@dataclass
class BiblioInfo:
    """서지정보 데이터 클래스"""

    jti_code: str = ""  # jti_4c0227
    classics_name: str = ""  # 당송팔대가문초구양수7
    대표서명: str = ""  # 唐宋八大家文抄 歐陽脩
    대표서명한글: str = ""  # 당송팔대가문초 구양수
    저자: str = ""  # (XML에 없음)
    번역서명: str = ""  # 譯註唐宋八大家文抄 歐陽脩7
    번역서명한글: str = ""  # 역주당송팔대가문초구양수7
    역자: str = ""  # (XML에 없음)
    번역서발행년도: str = ""  # 2023
    번역서발행자: str = ""  # (XML에 없음)
    db_식별자: str = ""  # jti_4c0227
    source_file: str = ""  # 원본 파일명

    def to_xml(self, indent: str = "  ") -> str:
        """서지정보 XML 문자열 생성"""
        lines = [
            f'<classics name="{self.classics_name}" id="{self.jti_code}">',
            f"{indent}<서지정보>",
            f"{indent}{indent}<대표서명>{self.대표서명}</대표서명>",
            f"{indent}{indent}<대표서명한글>{self.대표서명한글}</대표서명한글>",
            f"{indent}{indent}<저자>{self.저자}</저자>",
            f"{indent}{indent}<번역서명>{self.번역서명}</번역서명>",
            f"{indent}{indent}<번역서명한글>{self.번역서명한글}</번역서명한글>",
            f"{indent}{indent}<역자>{self.역자}</역자>",
            f"{indent}{indent}<번역서발행년도>{self.번역서발행년도}</번역서발행년도>",
            f"{indent}{indent}<번역서발행자>{self.번역서발행자}</번역서발행자>",
            f"{indent}{indent}<DB식별자>{self.db_식별자}</DB식별자>",
            f"{indent}</서지정보>",
            "</classics>",
        ]
        return "\n".join(lines)


def _strip_volume(text: str) -> str:
    """제목에서 권수 표시 제거

    예: '唐宋八大家文抄 歐陽脩(7)' → '唐宋八大家文抄 歐陽脩'
        '周易傳義(上)' → '周易傳義'
        '당송팔대가문초구양수7' → '당송팔대가문초구양수'
    """
    if not text:
        return ""
    text = text.strip()
    # 괄호 안 숫자/상하/上下 제거
    text = re.sub(r"\s*\([\d상하上下]+\)\s*$", "", text)
    # 끝의 숫자 제거
    text = re.sub(r"\d+$", "", text)
    return text.strip()


def _extract_volume_kr(title: str) -> str:
    """한글 제목에서 권수 추출: '당송팔대가문초구양수7' → '7', '주역전의(상)' → '상'"""
    # 끝의 숫자
    m = re.search(r"(\d+)$", title)
    if m:
        return m.group(1)
    # 괄호 안 상/하/숫자
    m = re.search(r"\(([\d상하]+)\)$", title)
    if m:
        return m.group(1)
    return ""


def _extract_volume_hanja(title: str) -> str:
    """한문 제목에서 권수 추출: '唐宋八大家文抄 歐陽脩(7)' → '7', '周易傳義(上)' → '上'"""
    # 괄호 안 숫자/上下
    m = re.search(r"\(([\d上下]+)\)\s*$", title)
    if m:
        return m.group(1)
    # 끝의 숫자
    m = re.search(r"(\d+)\s*$", title)
    if m:
        return m.group(1)
    return ""


def extract_biblio_from_xml(xml_path: str) -> Optional[BiblioInfo]:
    """단일 XML 파일에서 서지정보 추출

    Args:
        xml_path: 원문 또는 번역문 XML 파일 경로

    Returns:
        BiblioInfo 또는 None (파싱 실패 시)
    """
    path = Path(xml_path)

    # 파일명 분석: jti_4c0227-[역주]당송팔대가문초구양수7_원문_x-C2023.xml
    m = re.match(
        r"(jti_\w+)-\[(.+?)\](.+?)_(?:원문|번역문)(?:_x)?-C(\d+)",
        path.stem,
    )
    if not m:
        return None

    fname_jti = m.group(1)  # jti_4c0227
    bracket_type = m.group(2)  # 역주
    fname_title = m.group(3)  # 당송팔대가문초구양수7
    fname_year = m.group(4)  # 2023

    volume_kr = _extract_volume_kr(fname_title)  # 7, 상, 하
    fname_title_base = _strip_volume(fname_title)

    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except ET.ParseError:
        return None

    # BaseID
    base_id_elem = root.find(".//BaseID")
    base_id_text = (base_id_elem.text or "") if base_id_elem is not None else ""
    jti_code = base_id_text.replace("ID:", "").strip()
    if not jti_code:
        jti_code = fname_jti

    # 고전정보
    info = root.find(".//고전정보")
    info_text = (info.text or "").strip() if info is not None else ""
    info_hangul = (info.get("한글", "") or "").strip() if info is not None else ""

    # 발행년도: root의 year 속성 (없으면 파일명에서)
    pub_year = root.get("year", "") or fname_year

    # === 서지정보 필드 생성 ===

    # 한문 제목에서 권수 추출 (上/下/숫자)
    volume_hanja = _extract_volume_hanja(info_text)

    # 대표서명 (한문, 권수 제거)
    representative_title = _strip_volume(info_text)

    # 대표서명한글 (한글, 권수 제거)
    if info_hangul:
        representative_title_kr = _strip_volume(info_hangul)
        # 한글 속성에 한자가 섞여있을 수 있음 (예: "당송팔대가문초 歐陽脩(1)")
        # 이 경우 파일명 기반으로 대체
        if any("\u4e00" <= ch <= "\u9fff" for ch in representative_title_kr):
            representative_title_kr = fname_title_base
    else:
        representative_title_kr = fname_title_base

    # 번역서명 (한문): 역주유형(한문) + 대표서명(한문) + 한문권수
    bracket_hanja = BRACKET_TYPE_HANJA.get(bracket_type, bracket_type)
    trans_title = f"{bracket_hanja}{representative_title}"
    if volume_hanja:
        trans_title += volume_hanja

    # 번역서명한글: 역주유형 + 한글제목 + 한글권수
    trans_title_kr = f"{bracket_type}{representative_title_kr}"
    if volume_kr:
        trans_title_kr += volume_kr

    # 저자 추출: 당송팔대가문초의 경우 대표서명에서 저자명 추출
    author = ""
    if "唐宋八大家文抄" in representative_title:
        parts = representative_title.split("唐宋八大家文抄")
        if len(parts) > 1 and parts[1].strip():
            author = parts[1].strip()

    # classics name: 한글제목 + 권수 (파일명 기반)
    classics_name = fname_title

    biblio = BiblioInfo(
        jti_code=jti_code,
        classics_name=classics_name,
        대표서명=representative_title,
        대표서명한글=representative_title_kr,
        저자=author,
        번역서명=trans_title,
        번역서명한글=trans_title_kr,
        역자="",
        번역서발행년도=pub_year,
        번역서발행자="전통문화연구회",
        db_식별자=jti_code,
        source_file=path.name,
    )

    return biblio


def extract_biblio_from_directory(
    directory: str, pattern: str = "*원문*.xml"
) -> list[BiblioInfo]:
    """디렉토리의 모든 원문 XML에서 서지정보 추출

    Args:
        directory: XML 파일 디렉토리
        pattern: 파일 패턴 (기본: 원문 파일)

    Returns:
        BiblioInfo 리스트
    """
    dir_path = Path(directory)
    results = []

    for xml_file in sorted(dir_path.glob(pattern)):
        biblio = extract_biblio_from_xml(str(xml_file))
        if biblio:
            results.append(biblio)

    return results


def export_biblio_xml(
    biblios: list[BiblioInfo], output_path: str = None
) -> str:
    """서지정보 리스트를 XML 파일로 출력

    Args:
        biblios: BiblioInfo 리스트
        output_path: 출력 파일 경로 (None이면 문자열만 반환)

    Returns:
        XML 문자열
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<서지정보목록>"]

    for biblio in biblios:
        lines.append("")
        # 각 항목을 2칸 들여쓰기
        for line in biblio.to_xml().split("\n"):
            lines.append(f"  {line}")

    lines.append("")
    lines.append("</서지정보목록>")

    xml_str = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

    return xml_str


def export_biblio_xlsx(
    biblios: list[BiblioInfo], output_path: str
) -> None:
    """서지정보 리스트를 XLSX 파일로 출력"""
    import pandas as pd

    data = []
    for b in biblios:
        data.append(
            {
                "DB식별자": b.db_식별자,
                "대표서명": b.대표서명,
                "대표서명한글": b.대표서명한글,
                "저자": b.저자,
                "번역서명": b.번역서명,
                "번역서명한글": b.번역서명한글,
                "역자": b.역자,
                "번역서발행년도": b.번역서발행년도,
                "번역서발행자": b.번역서발행자,
            }
        )

    df = pd.DataFrame(data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
