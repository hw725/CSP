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
    val = (
        cfg.get("results_dir")
        or os.getenv("CSP_XLSX_RESULTS")
        or "xlsx_pipeline_results"
    )
    return Path(val)

def get_embedder(kind: str = "pa") -> str:
    """
    임베더 선택 (문자열 식별자)
    - kind: "pa" 또는 "sa"
    우선순위: 전용 > 공통 > 기본(bge-m3)
    """
    cfg = _load_file_config()
    common = cfg.get("embedder") or os.getenv("CSP_EMBEDDER")
    if kind == "s2p":
        return (
            cfg.get("s2p_embedder") or os.getenv("CSP_S2P_EMBEDDER") or common or "bge-m3"
        )
    return cfg.get("p2s_embedder") or os.getenv("CSP_P2S_EMBEDDER") or common or "bge-m3"

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

    # 기본값 정의 (정확도 중심)
    defaults = {
        "similarity_threshold": 0.65,
        "length_penalty": 0.1,
        "distance_decay": 0.08,
        "boundary_bonus": 0.2,
        "particle_bonus": 0.25,
        "comma_bonus": 0.15,
        "dp_window": 4,
        "sim_gamma": 1.4,
        "hanja_bonus": 0.3,
        "hanja_strict": True,
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

def get_p2s_selection_params() -> Dict[str, Any]:
    """P2S strict의 후보 선택(prior/style bonus 등) 파라미터.

    우선순위:
    - csp_config.json의 p2s_selection_params
    - (없으면) 레거시 pa_selection_params로 fallback
    - (없으면) 안전한 기본값

    목적:
    - 휴리스틱(상수/토큰 리스트)을 코드에서 분리하여 재현/튜닝 가능하게 함
    - trace에서 기여도를 투명하게 남기기 위한 단일 진입점 제공
    """

    cfg = _load_file_config()
    # 새로운 p2s_selection_params 찾기, 없으면 레거시 pa_selection_params로 fallback
    params = cfg.get("p2s_selection_params") or cfg.get("pa_selection_params") or {}

    defaults: Dict[str, Any] = {
        # boundary-aware alignment matcher에서 (의미 유사도 vs 경계 일치) 결합 가중치
        # 0이면 의미 유사도만, 1이면 경계 일치만 반영.
        "boundary_aware_weight": 0.3,
        "candidate_prior_bonus_by_prefix": {
            "supar(": 0.015,
            "boundary(": 0.010,
        },
        "boundary_style_prior": {
            "enabled": True,  # Alignment score 우선, style은 보조 역할
            "weight_terminal": 0.006,  # 1/3로 약화 (의미 대응 우선)
            "weight_continuation": -0.010,  # 1/3로 약화
            "continuation_tokens": [
                "하며",
                "하며,",
                "하고",
                "하야",
                "하여",
                "하야,",
                "하여,",
            ],
            "continuation_tail_cjk": [
                "而",
                "以",
                "則",
                "乃",
                "故",
                "及",
                "與",
                "且",
            ],
            "terminal_suffixes": [
                "이라",
                "矣라",
                "也라",
                "耳라",
                "니라",
                "로라",
                "哉아",
                "邪아",
            ],
            "terminal_punct": [".", "!", "?", "。", "！", "？"],
        },
        "penalty_short_pairs": {
            "long_tgt_threshold": 40,
            "short_src_threshold": 12,
            "penalty_per_pair": 0.015,
        },
        "penalty_empty_src": 0.5,
        "max_candidates_multiplier": 12,
        "whitespace_dp_penalties": {
            "long_tgt_threshold": 80,
            "short_src_threshold": 25,
            "very_short_src_threshold": 8,
            "penalty_short": 0.070,
            "penalty_very_short": 0.090,
            "ratio_outlier": {
                "min_tgt_len": 80,
                "ratio_high_threshold": 3.8,
                "ratio_mid_threshold": 3.2,
                "median_margin_high": 1.2,
                "median_margin_mid": 1.0,
                "src_len_cap_high": 45,
                "src_len_cap_mid": 35,
                "penalty_high": 0.18,
                "penalty_mid": 0.12,
                "penalty_longest_shortest": 0.10,
            },
        },
    }

    # shallow + nested dict merge (필요 최소만)
    merged: Dict[str, Any] = dict(defaults)
    for key, val in params.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged

def as_dict() -> dict:
    """디버깅/로깅용 현재 설정 스냅샷"""
    return {
        "results_dir": str(get_results_dir()),
        "p2s_embedder": get_embedder("p2s"),
        "s2p_embedder": get_embedder("s2p"),
        "device": get_device(),
        "openai_key_set": bool(get_openai_api_key()),
        "thresholds_loaded": bool(_load_file_config().get("thresholds")),
    }
