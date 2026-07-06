"""
XML Pipeline Package — PA/SA 정렬 파이프라인 (Docker/GPU 환경)

순수 XML→XLSX 추출은 xml_extractor 패키지 참조.

주요 모듈:
- cli: 명령줄 인터페이스
- processor: PA/SA 파이프라인 오케스트레이션
- similarity: 유사도 계산 (torch 필요)
- accuracy: 정확도 평가
"""

# xml_extractor에서 추출 관련 클래스 re-export (하위 호환)
from xml_extractor import XMLProcessor, XMLPair, create_xml_pair_from_directory, XMLUnitParser

from .xml_pipeline_cli import XMLPipelineManager
from .xml_pipeline_processor import XMLPipelineProcessor
from .xml_level_similarity import XMLLevelSimilarityCalculator
from .xml_file_browser import XMLFileBrowser
from .docker_xml_smart import DockerXMLSmart

__version__ = "1.0.0"
__all__ = [
    "XMLPipelineManager",
    "XMLPipelineProcessor",
    "XMLProcessor",
    "XMLPair",
    "create_xml_pair_from_directory",
    "XMLUnitParser",
    "XMLLevelSimilarityCalculator",
    "XMLFileBrowser",
    "DockerXMLSmart",
]
