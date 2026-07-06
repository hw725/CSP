"""하위 호환 shim — 정본은 xml_extractor.jti_code_mappings (2026-07-06 중복 제거).

로직 수정 금지: xml_extractor.jti_code_mappings 만 고친다. 이 파일은 기존 import 경로
(xml_pipeline.jti_code_mappings / from .jti_code_mappings import ...) 보존용 얇은 re-export 껍데기다.
"""
from xml_extractor.jti_code_mappings import *  # noqa: F401,F403
