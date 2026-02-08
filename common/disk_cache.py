"""
범용 디스크 캐시 유틸리티.

BGE 임베더의 pickle 캐시 패턴을 재사용한 공통 모듈.
토크나이저, 파서, particle matcher 등에서 사용.
"""

import atexit
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CACHE_BASE_DIR = Path(".cache")

# 전역 인스턴스 레지스트리 (atexit에서 flush)
_all_caches: list = []


def _flush_all_caches():
    for cache in _all_caches:
        try:
            cache.flush()
        except Exception:
            pass


atexit.register(_flush_all_caches)


class DiskCache:
    """메모리 + 디스크 이중 캐시.

    - 메모리에서 먼저 조회 (O(1))
    - 미스 시 디스크에서 로드 (첫 호출 때 한 번)
    - 새 항목 추가 시 주기적으로 디스크에 저장
    - 프로세스 종료 시 atexit로 안전하게 flush
    """

    def __init__(self, name: str, save_every: int = 100):
        self._name = name
        self._cache: dict = {}
        self._dirty_count = 0
        self._save_every = save_every
        self._cache_path = CACHE_BASE_DIR / f"{name}.pkl"
        self._loaded = False
        _all_caches.append(self)

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"[DiskCache:{self._name}] 로드: {len(self._cache)}개 항목")
            except Exception as e:
                logger.warning(f"[DiskCache:{self._name}] 로드 실패: {e}")
                self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        self._ensure_loaded()
        return self._cache.get(key)

    def put(self, key: str, value: Any):
        self._ensure_loaded()
        self._cache[key] = value
        self._dirty_count += 1
        if self._dirty_count >= self._save_every:
            self.flush()

    def flush(self):
        if self._dirty_count == 0:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            logger.debug(f"[DiskCache:{self._name}] 저장: {len(self._cache)}개 항목")
            self._dirty_count = 0
        except Exception as e:
            logger.warning(f"[DiskCache:{self._name}] 저장 실패: {e}")

    def __len__(self):
        self._ensure_loaded()
        return len(self._cache)

    def __contains__(self, key: str):
        self._ensure_loaded()
        return key in self._cache
