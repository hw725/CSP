#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공통 환경 설정 모듈
- 환경 변수와 기본값을 중앙에서 관리하여 일관성 제공
"""

import os
import json
from pathlib import Path
from typing import Optional, Any, Dict
_CONFIG_CACHE: Dict[str, Any] | None = None

def _load_file_config() -> Dict[str, Any]:
    """프로젝트 루트의 csp_config.json을 읽어 캐시합니다 (없으면 빈 dict)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    # 프로젝트 루트 추정: 본 파일의 상위 상위 디렉토리
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "csp_config.json"
    if cfg_path.exists():
        try:
            _CONFIG_CACHE = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            _CONFIG_CACHE = {}
    else:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_results_dir() -> Path:
    """XLSX 파이프라인 결과 디렉토리"""
    cfg = _load_file_config()
    val = cfg.get("results_dir") or os.getenv("CSP_XLSX_RESULTS") or "xlsx_pipeline_results"
    return Path(val)


def get_embedder(kind: str = "pa") -> str:
    """
    임베더 선택 (문자열 식별자)
    - kind: "pa" 또는 "sa"
    우선순위: 전용 > 공통 > 기본(bge-m3)
    """
    cfg = _load_file_config()
    common = cfg.get("embedder") or os.getenv("CSP_EMBEDDER")
    if kind == "sa":
        return (
            cfg.get("sa_embedder")
            or os.getenv("CSP_SA_EMBEDDER")
            or common
            or "bge-m3"
        )
    return (
        cfg.get("pa_embedder")
        or os.getenv("CSP_PA_EMBEDDER")
        or common
        or "bge-m3"
    )


def get_device() -> Optional[str]:
    """장치 설정 (예: cuda:0, cpu). 미설정 시 None 반환."""
    cfg = _load_file_config()
    return cfg.get("device") or os.getenv("CSP_DEVICE")


def get_openai_api_key() -> Optional[str]:
    cfg = _load_file_config()
    return cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY")


def get_alignment_params() -> Dict[str, Any]:
    """PA/SA 정렬 파라미터 반환 (임계값, 페널티, 보너스 등)"""
    cfg = _load_file_config()
    params = cfg.get("alignment_params", {})
    
    # 기본값 정의
    defaults = {
        "similarity_threshold": 0.5,
        "length_penalty": 0.1,
        "distance_decay": 0.05,
        "boundary_bonus": 0.15,
        "particle_bonus": 0.2,
        "comma_bonus": 0.1,
        "dp_window": 2,
        "sim_gamma": 1.2,
    }
    
    # 설정 파일 값으로 덮어쓰기
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = default_val
    
    return params

def get_thresholds() -> Dict[str, Any]:
    """PA/SA 임계값 설정을 반환 (csp_config.json 우선, 없으면 기본값)"""
    cfg = _load_file_config()
    thresholds = cfg.get("thresholds", {})
    
    # 기본값 정의
    defaults = {
        "pa": {
            "unit": "row",
            "metrics": ["partial_match", "target_avg_similarity"],
            "levels": {
                "min": {"partial_match": 0.10, "target_avg_similarity": 0.10},
                "recommended": {"partial_match": 0.15, "target_avg_similarity": 0.19},
                "top": {"partial_match": 0.21, "target_avg_similarity": 0.26},
            },
        },
        "sa": {
            "unit": "row",
            "metrics": ["partial_match", "target_avg_similarity"],
            "levels": {
                "min": {"partial_match": 0.885, "target_avg_similarity": 0.769},
                "recommended": {"partial_match": 0.952, "target_avg_similarity": 0.905},
                "top": {"partial_match": 1.0, "target_avg_similarity": 1.0},
            },
        },
    }
    
    # 설정 파일 값으로 기본값 덮어쓰기 (deep merge)
    if "pa" in thresholds:
        defaults["pa"].update(thresholds["pa"])
    if "sa" in thresholds:
        defaults["sa"].update(thresholds["sa"])
    
    return defaults


def as_dict() -> dict:
    """디버깅/로깅용 현재 설정 스냅샷"""
    return {
        "results_dir": str(get_results_dir()),
        "pa_embedder": get_embedder("pa"),
        "sa_embedder": get_embedder("sa"),
        "device": get_device(),
        "openai_key_set": bool(get_openai_api_key()),
        "thresholds_loaded": bool(_load_file_config().get("thresholds")),
    }
