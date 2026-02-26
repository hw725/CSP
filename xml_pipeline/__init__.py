"""
XML Pipeline Package

고전 문헌 XML 파일들을 처리하는 파이프라인 패키지입니다.

주요 모듈:
- cli: 명령줄 인터페이스
- processor: 핵심 처리 로직
- parser: XML 파싱 기능
- similarity: 유사도 계산
- accuracy: 정확도 평가
"""

from .xml_pipeline_cli import XMLPipelineManager
from .xml_pipeline_processor import XMLPipelineProcessor
from .xml_unit_parser import XMLUnitParser
from .xml_level_similarity import XMLLevelSimilarityCalculator
from .xml_file_browser import XMLFileBrowser
from .docker_xml_smart import DockerXMLSmart

__version__ = "1.0.0"
__all__ = [
    "XMLPipelineManager",
    "XMLPipelineProcessor", 
    "XMLUnitParser",
    "XMLLevelSimilarityCalculator",
    "XMLFileBrowser",
    "DockerXMLSmart"
]