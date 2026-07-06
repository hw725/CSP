"""
XML Extractor Package — XML 원문/번역문에서 병렬 데이터 추출 (torch 불필요)

주요 모듈:
- xml_processor: XMLProcessor (문단/문장/구 추출), XMLPair, create_xml_pair_from_directory
- xml_unit_parser: XMLUnitParser (문장/어절 단위 추출)
- jti_code_mappings: JTI 코드 ↔ 텍스트명 매핑
- xml_file_browser: XML 쌍 스캔/선택
- xml_biblio_extractor: 서지정보 추출
- cli: 변환 전용 CLI
"""

from .xml_processor import XMLProcessor, XMLPair, create_xml_pair_from_directory
from .xml_unit_parser import XMLUnitParser
from .jti_code_mappings import (
    JTI_CODE_MAPPINGS,
    TEXT_TO_JTI_MAPPINGS,
    get_jti_by_text_name,
    get_text_name_by_jti,
)
from .xml_biblio_extractor import (
    BiblioInfo,
    extract_biblio_from_xml,
    extract_biblio_from_directory,
    export_biblio_xml,
    export_biblio_xlsx,
)

__version__ = "1.0.0"
__all__ = [
    "XMLProcessor",
    "XMLPair",
    "create_xml_pair_from_directory",
    "XMLUnitParser",
    "JTI_CODE_MAPPINGS",
    "TEXT_TO_JTI_MAPPINGS",
    "get_jti_by_text_name",
    "get_text_name_by_jti",
    "BiblioInfo",
    "extract_biblio_from_xml",
    "extract_biblio_from_directory",
    "export_biblio_xml",
    "export_biblio_xlsx",
]
