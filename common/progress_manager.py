"""
통합 진행률 관리자 (Common Progress Manager)
PA와 SA 모듈에서 공통으로 사용할 진행률 막대 관리
"""

import threading
from tqdm import tqdm
from typing import Optional, Dict, Any


class UnifiedProgressManager:
    """통합 진행률 관리자 - 싱글톤 패턴"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.progress_bar: Optional[tqdm] = None
            self.current_task = ""
            self.total_steps = 0
            self.completed_steps = 0
            self.task_info = {}
            self.initialized = True
    
    def start_progress(self, total: int, description: str = "🔄 처리 중", **kwargs) -> None:
        """진행률 막대 시작"""
        if self.progress_bar:
            self.progress_bar.close()
        
        # 기본 설정 - 깔끔한 프로그레스 바
        default_kwargs = {
            'unit': '항목',
            'ncols': 80,
            'bar_format': '{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}',
            'mininterval': 0.5,  # 최소 업데이트 간격 (0.5초)
            'maxinterval': 2.0,  # 최대 업데이트 간격 (2초)
            'smoothing': 0.1     # 진행률 평활화
        }
        default_kwargs.update(kwargs)
        
        self.progress_bar = tqdm(
            total=total,
            desc=description,
            **default_kwargs
        )
        
        self.current_task = description
        self.total_steps = total
        self.completed_steps = 0
        self.task_info = {}
    
    def update(self, n: int = 1, **postfix) -> None:
        """진행률 업데이트"""
        if self.progress_bar:
            self.progress_bar.update(n)
            self.completed_steps += n
            
            if postfix:
                self.task_info.update(postfix)
                self.progress_bar.set_postfix(self.task_info)
    
    def set_description(self, description: str) -> None:
        """진행률 막대 설명 변경"""
        if self.progress_bar:
            self.progress_bar.set_description(description)
            self.current_task = description
    
    def set_postfix(self, **kwargs) -> None:
        """후위 정보 설정"""
        if self.progress_bar:
            self.task_info.update(kwargs)
            self.progress_bar.set_postfix(self.task_info)
    
    def finish(self, message: str = "") -> None:
        """진행률 완료"""
        if self.progress_bar:
            if message:
                self.progress_bar.set_description(f"✅ {message}")
            self.progress_bar.close()
            self.progress_bar = None
        
        self.current_task = ""
        self.completed_steps = 0
        self.total_steps = 0
        self.task_info = {}
    
    def is_active(self) -> bool:
        """진행률 막대 활성 상태 확인"""
        return self.progress_bar is not None
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 정보 반환"""
        return {
            'active': self.is_active(),
            'task': self.current_task,
            'completed': self.completed_steps,
            'total': self.total_steps,
            'info': self.task_info
        }


# 전역 인스턴스
progress_manager = UnifiedProgressManager()


def start_unified_progress(total: int, description: str = "🔄 처리 중", **kwargs) -> None:
    """통합 진행률 시작 (편의 함수)"""
    progress_manager.start_progress(total, description, **kwargs)


def update_unified_progress(n: int = 1, **postfix) -> None:
    """통합 진행률 업데이트 (편의 함수)"""
    progress_manager.update(n, **postfix)


def finish_unified_progress(message: str = "") -> None:
    """통합 진행률 완료 (편의 함수)"""
    progress_manager.finish(message)


def set_progress_description(description: str) -> None:
    """진행률 설명 변경 (편의 함수)"""
    progress_manager.set_description(description)


def set_progress_postfix(**kwargs) -> None:
    """진행률 후위 정보 설정 (편의 함수)"""
    progress_manager.set_postfix(**kwargs)


def is_progress_active() -> bool:
    """진행률 활성 상태 확인 (편의 함수)"""
    return progress_manager.is_active()
