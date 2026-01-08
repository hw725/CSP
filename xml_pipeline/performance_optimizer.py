"""
XML 파이프라인 성능 최적화기
Performance optimizer for XML pipeline
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import concurrent.futures
import threading
from dataclasses import dataclass

# 프로젝트 루트 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

@dataclass
class PerformanceConfig:
    """성능 최적화 설정"""
    # 배치 처리 설정
    embedding_batch_size: int = 50  # 기본 20 -> 50으로 증가
    xml_processing_batch_size: int = 100  # XML 요소 배치 처리
    
    # 병렬 처리 설정
    max_workers: int = 4  # CPU 코어 수에 따라 조정
    enable_parallel_xml: bool = True  # XML 파싱 병렬화
    enable_parallel_embedding: bool = True  # 임베딩 생성 병렬화
    
    # 메모리 최적화
    enable_streaming: bool = True  # 스트리밍 처리
    chunk_size: int = 1000  # 청크 크기
    gc_interval: int = 100  # 가비지 컬렉션 간격
    
    # 캐시 설정
    enable_embedding_cache: bool = True
    cache_cleanup_interval: int = 500  # 캐시 정리 간격
    
    # 디버깅 및 로깅
    enable_profiling: bool = False
    verbose_logging: bool = False

class PerformanceOptimizer:
    """파이프라인 성능 최적화 관리자"""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.performance_stats = {}
        self.start_time = None
        
    def optimize_pa_command(self, cmd: List[str]) -> List[str]:
        """PA 명령어 성능 최적화"""
        optimized_cmd = cmd.copy()
        
        # 배치 크기 최적화
        if '--batch-size' not in cmd:
            optimized_cmd.extend(['--batch-size', str(self.config.embedding_batch_size)])
        else:
            # 기존 배치 크기를 더 큰 값으로 교체
            for i, arg in enumerate(optimized_cmd):
                if arg == '--batch-size' and i + 1 < len(optimized_cmd):
                    current_size = int(optimized_cmd[i + 1])
                    if current_size < self.config.embedding_batch_size:
                        optimized_cmd[i + 1] = str(self.config.embedding_batch_size)
        
        # 워커 수 최적화
        if '--max-workers' not in cmd:
            optimized_cmd.extend(['--max-workers', str(self.config.max_workers)])
        else:
            for i, arg in enumerate(optimized_cmd):
                if arg == '--max-workers' and i + 1 < len(optimized_cmd):
                    current_workers = int(optimized_cmd[i + 1])
                    if current_workers < self.config.max_workers:
                        optimized_cmd[i + 1] = str(self.config.max_workers)
        
        # 성능 모드 활성화
        if '--performance-mode' not in cmd:
            optimized_cmd.extend(['--performance-mode'])
            
        # 캐시 활성화
        if self.config.enable_embedding_cache and '--enable-cache' not in cmd:
            optimized_cmd.extend(['--enable-cache'])
            
        return optimized_cmd
    
    def optimize_sa_command(self, cmd: List[str]) -> List[str]:
        """SA 명령어 성능 최적화"""
        optimized_cmd = cmd.copy()
        
        # 배치 크기 최적화 (SA는 문장 단위라서 더 큰 배치 가능)
        batch_size = self.config.embedding_batch_size * 2
        
        if '--batch-size' not in cmd:
            optimized_cmd.extend(['--batch-size', str(batch_size)])
        else:
            for i, arg in enumerate(optimized_cmd):
                if arg == '--batch-size' and i + 1 < len(optimized_cmd):
                    current_size = int(optimized_cmd[i + 1])
                    if current_size < batch_size:
                        optimized_cmd[i + 1] = str(batch_size)
        
        # 워커 수 최적화
        if '--max-workers' not in cmd:
            optimized_cmd.extend(['--max-workers', str(self.config.max_workers)])
        
        # 스트리밍 처리 활성화
        if self.config.enable_streaming and '--streaming' not in cmd:
            optimized_cmd.extend(['--streaming'])
        
        return optimized_cmd
    
    def setup_environment_optimization(self):
        """환경 변수 최적화 설정"""
        
        # Python 최적화
        os.environ['PYTHONOPTIMIZE'] = '2'  # 최적화 모드
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # pyc 파일 생성 방지
        
        # NumPy 최적화
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.config.max_workers)
        os.environ['MKL_NUM_THREADS'] = str(self.config.max_workers)
        
        # TensorFlow/PyTorch 최적화 (사용하는 경우)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 로그 레벨 조정
        
        # 메모리 최적화
        if hasattr(sys, 'setswitchinterval'):
            sys.setswitchinterval(0.001)  # 스레드 컨텍스트 스위칭 간격 조정
    
    def start_profiling(self, operation_name: str):
        """성능 프로파일링 시작"""
        self.start_time = time.time()
        self.performance_stats[operation_name] = {
            'start_time': self.start_time,
            'memory_before': self._get_memory_usage()
        }
        
        if self.config.verbose_logging:
            print(f"🔄 {operation_name} 시작")
    
    def end_profiling(self, operation_name: str):
        """성능 프로파일링 종료"""
        if operation_name not in self.performance_stats:
            return
            
        end_time = time.time()
        stats = self.performance_stats[operation_name]
        
        stats['end_time'] = end_time
        stats['duration'] = end_time - stats['start_time']
        stats['memory_after'] = self._get_memory_usage()
        stats['memory_diff'] = stats['memory_after'] - stats['memory_before']
        
        if self.config.verbose_logging:
            print(f"✅ {operation_name} 완료: {stats['duration']:.2f}초")
            print(f"   메모리 사용량: {stats['memory_diff']:.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def print_performance_summary(self):
        """성능 요약 출력"""
        if not self.performance_stats:
            return
            
        print("\n" + "="*60)
        print("📊 성능 최적화 결과")
        print("="*60)
        
        total_time = 0
        for operation, stats in self.performance_stats.items():
            if 'duration' in stats:
                duration = stats['duration']
                memory_diff = stats.get('memory_diff', 0)
                print(f"🔸 {operation}: {duration:.2f}초 (메모리: {memory_diff:+.1f}MB)")
                total_time += duration
        
        print(f"\n⏱️ 총 처리 시간: {total_time:.2f}초")
        print("="*60)

def apply_quick_optimizations():
    """빠른 최적화 적용"""
    optimizer = PerformanceOptimizer()
    optimizer.setup_environment_optimization()
    
    print("🚀 성능 최적화 설정 적용 완료")
    print(f"   - 배치 크기: {optimizer.config.embedding_batch_size}")
    print(f"   - 최대 워커: {optimizer.config.max_workers}")
    print(f"   - 병렬 처리: 활성화")
    print(f"   - 캐시 사용: {optimizer.config.enable_embedding_cache}")
    
    return optimizer

def get_optimized_config_for_jti(jti_code: str) -> PerformanceConfig:
    """JTI 코드별 최적화 설정 반환"""
    
    # 텍스트 크기에 따른 설정 조정
    large_texts = ['4c0201', '4c0101', '1j0201']  # 당송팔대가문초, 관자 등
    
    if jti_code in large_texts:
        return PerformanceConfig(
            embedding_batch_size=100,  # 대용량은 더 큰 배치
            xml_processing_batch_size=200,
            max_workers=6,
            chunk_size=2000,
            gc_interval=50  # 더 자주 정리
        )
    else:
        return PerformanceConfig()  # 기본 설정

if __name__ == "__main__":
    # 테스트용
    optimizer = apply_quick_optimizations()
    
    # 샘플 명령어 최적화 테스트
    sample_pa_cmd = ["python", "pa/main.py", "input.xlsx", "output.xlsx"]
    optimized_cmd = optimizer.optimize_pa_command(sample_pa_cmd)
    
    print("\n🔧 최적화 전:")
    print(" ".join(sample_pa_cmd))
    print("\n⚡ 최적화 후:")
    print(" ".join(optimized_cmd))