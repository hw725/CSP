"""
Docker + GPU 가속화 최적화
Docker GPU Acceleration Optimizer
"""

import os
import sys
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path

class DockerGPUOptimizer:
    """Docker 환경에서 GPU 가속화 최적화"""
    
    def __init__(self):
        self.cuda_available = self._check_cuda_availability()
        self.gpu_info = self._get_gpu_info()
        self.torch_gpu = self._check_pytorch_gpu()
        
    def _check_cuda_availability(self) -> bool:
        """CUDA 가용성 확인"""
        try:
            # nvidia-smi 명령어로 GPU 확인
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 수집"""
        if not self.cuda_available:
            return {}
        
        try:
            # GPU 메모리 정보
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = {}
                for i, line in enumerate(lines):
                    total, used, free = map(int, line.split(', '))
                    gpu_info[f'gpu_{i}'] = {
                        'total_memory_mb': total,
                        'used_memory_mb': used,
                        'free_memory_mb': free,
                        'utilization_percent': (used / total) * 100 if total > 0 else 0
                    }
                return gpu_info
            
        except Exception as e:
            print(f"⚠️ GPU 정보 수집 실패: {e}")
        
        return {}
    
    def _check_pytorch_gpu(self) -> bool:
        """PyTorch GPU 지원 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_gpu_optimization_config(self) -> Dict[str, Any]:
        """GPU 최적화 설정 반환"""
        config = {
            'cuda_available': self.cuda_available,
            'torch_gpu_available': self.torch_gpu,
            'gpu_count': len(self.gpu_info),
            'recommended_batch_size': 32,
            'use_gpu_embedding': False,
            'gpu_memory_fraction': 0.8
        }
        
        if self.cuda_available and self.torch_gpu:
            # GPU 메모리에 따른 배치 사이즈 조정
            if self.gpu_info:
                gpu_0 = self.gpu_info.get('gpu_0', {})
                free_memory_gb = gpu_0.get('free_memory_mb', 0) / 1024
                
                if free_memory_gb >= 8:
                    config['recommended_batch_size'] = 128
                    config['use_gpu_embedding'] = True
                elif free_memory_gb >= 4:
                    config['recommended_batch_size'] = 64
                    config['use_gpu_embedding'] = True
                elif free_memory_gb >= 2:
                    config['recommended_batch_size'] = 32
                    config['use_gpu_embedding'] = True
                
                print(f"🚀 GPU 최적화 활성화:")
                print(f"   사용 가능 GPU 메모리: {free_memory_gb:.1f}GB")
                print(f"   권장 배치 사이즈: {config['recommended_batch_size']}")
                print(f"   GPU 임베딩 사용: {config['use_gpu_embedding']}")
        
        return config
    
    def create_gpu_optimized_dockerfile(self) -> str:
        """GPU 최적화된 Dockerfile 생성"""
        dockerfile_content = """# GPU 최적화된 Docker 이미지
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python 별칭 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

# CUDA 환경변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# PyTorch GPU 버전 설치
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 필요한 라이브러리 설치
RUN pip3 install \\
    sentence-transformers \\
    transformers \\
    accelerate \\
    pandas \\
    numpy \\
    openpyxl \\
    scikit-learn \\
    tqdm \\
    psutil

# 작업 디렉토리 설정
WORKDIR /workspace

# GPU 메모리 최적화 설정
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=1

# 기본 명령어
CMD ["bash"]
"""
        
        dockerfile_path = Path('Dockerfile.gpu')
        dockerfile_path.write_text(dockerfile_content)
        return str(dockerfile_path)
    
    def create_gpu_docker_compose(self) -> str:
        """GPU 최적화된 docker-compose.yml 생성"""
        compose_content = """version: '3.8'

services:
  csp-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: csp-gpu-workspace
    
    # GPU 리소스 할당
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0
      
    volumes:
      # 워크스페이스 (성능 최적화)
      - .:/workspace:delegated
      
      # GPU 캐시 볼륨들
      - gpu_cache:/workspace/.cache:rw
      - pip_cache:/root/.cache/pip:rw
      - huggingface_cache:/root/.cache/huggingface:rw
      - torch_cache:/root/.cache/torch:rw
      
      # GPU 임시 메모리
      - type: tmpfs
        target: /tmp/gpu_cache
        tmpfs:
          size: 4G
          mode: 1777
    
    working_dir: /workspace
    environment:
      - PYTHONPATH=/workspace
      
      # CUDA 최적화 환경변수
      - CUDA_VISIBLE_DEVICES=0
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      - CUDA_LAUNCH_BLOCKING=0
      - CUDA_CACHE_DISABLE=0
      - CUDA_CACHE_MAXSIZE=2147483648
      
      # Python 최적화
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONOPTIMIZE=2
      
      # PyTorch GPU 최적화
      - TORCH_CUDNN_BENCHMARK=true
      - TORCH_BACKENDS_CUDNN_BENCHMARK=true
      - TORCH_BACKENDS_CUDNN_DETERMINISTIC=false
      - TORCH_USE_CUDA_DSA=1
      
      # 수치 연산 최적화 (CPU 백업용)
      - OPENBLAS_NUM_THREADS=4
      - MKL_NUM_THREADS=4
      - OMP_NUM_THREADS=4
      
      # 캐시 최적화
      - HF_HOME=/root/.cache/huggingface
      - TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
      - TORCH_HOME=/root/.cache/torch
      - TMPDIR=/tmp/gpu_cache
    
    stdin_open: true
    tty: true
    
    # 리소스 최적화 (GPU 포함)
    deploy:
      resources:
        limits:
          cpus: '16.0'
          memory: 32G
        reservations:
          cpus: '8.0'
          memory: 16G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    restart: unless-stopped
    networks:
      - csp_gpu_network
    
    # 공유 메모리 및 IPC 최적화 (GPU용)
    ipc: host
    shm_size: 4gb

# GPU 전용 볼륨들
volumes:
  gpu_cache:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=4g,uid=0,gid=0,mode=1777
  
  pip_cache:
    driver: local
  
  huggingface_cache:
    driver: local
    
  torch_cache:
    driver: local

# GPU 전용 네트워크
networks:
  csp_gpu_network:
    driver: bridge
"""
        
        compose_path = Path('docker-compose-gpu.yml')
        compose_path.write_text(compose_content)
        return str(compose_path)
    
    def create_gpu_run_script(self) -> str:
        """GPU 최적화 실행 스크립트 생성"""
        script_content = """#!/bin/bash

echo "🚀 Docker GPU 최적화 모드 시작..."

# GPU 가용성 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU 드라이버가 설치되어 있지 않습니다."
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "❌ GPU에 접근할 수 없습니다. Docker GPU 지원을 확인하세요."
    exit 1
fi

echo "✅ GPU 환경 확인 완료"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# 기존 컨테이너 정리
echo "🔄 기존 GPU 컨테이너 정리..."
docker-compose -f docker-compose-gpu.yml down 2>/dev/null || true

# GPU 컨테이너 시작
echo "🐳 GPU 최적화 컨테이너 시작..."
docker-compose -f docker-compose-gpu.yml up -d

# 컨테이너 준비 대기
echo "⏳ GPU 컨테이너 초기화 대기..."
sleep 10

# GPU 상태 확인
echo "🔍 컨테이너 내 GPU 상태 확인..."
docker-compose -f docker-compose-gpu.yml exec -T csp-gpu nvidia-smi

# XML 파이프라인 실행
if [ "$1" != "" ]; then
    echo "🎯 GPU 가속 XML 파일 처리: $1"
    docker-compose -f docker-compose-gpu.yml exec csp-gpu python xml_pipeline/xml_pipeline_processor.py "$1"
else
    echo "📋 사용법: $0 <xml_file>"
    echo "예시: $0 관자4_병렬.xml"
    echo ""
    echo "🚀 GPU 가속 모드 대기 중... (Ctrl+C로 종료)"
    docker-compose -f docker-compose-gpu.yml logs -f csp-gpu
fi

echo "✅ GPU 가속 처리 완료"
"""
        
        script_path = Path('run_gpu_optimized.sh')
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        # Windows 배치 파일도 생성
        windows_script = """@echo off
echo 🚀 Docker GPU 최적화 모드 시작...

REM GPU 가용성 확인
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo ❌ NVIDIA GPU 드라이버가 설치되어 있지 않습니다.
    pause
    exit /b 1
)

nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo ❌ GPU에 접근할 수 없습니다. Docker GPU 지원을 확인하세요.
    pause
    exit /b 1
)

echo ✅ GPU 환경 확인 완료
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

REM 기존 컨테이너 정리
echo 🔄 기존 GPU 컨테이너 정리...
docker-compose -f docker-compose-gpu.yml down 2>nul

REM GPU 컨테이너 시작
echo 🐳 GPU 최적화 컨테이너 시작...
docker-compose -f docker-compose-gpu.yml up -d

REM 컨테이너 준비 대기
echo ⏳ GPU 컨테이너 초기화 대기...
timeout /t 10 /nobreak >nul

REM GPU 상태 확인
echo 🔍 컨테이너 내 GPU 상태 확인...
docker-compose -f docker-compose-gpu.yml exec -T csp-gpu nvidia-smi

REM XML 파이프라인 실행
if "%1"=="" (
    echo 📋 사용법: %0 ^<xml_file^>
    echo 예시: %0 관자4_병렬.xml
    pause
    exit /b 0
)

echo 🎯 GPU 가속 XML 파일 처리: %1
docker-compose -f docker-compose-gpu.yml exec csp-gpu python xml_pipeline/xml_pipeline_processor.py "%1"

echo ✅ GPU 가속 처리 완료
pause
"""
        
        windows_script_path = Path('run_gpu_optimized.bat')
        windows_script_path.write_text(windows_script, encoding='utf-8')
        
        return str(script_path)

def setup_gpu_optimization():
    """GPU 최적화 환경 설정"""
    optimizer = DockerGPUOptimizer()
    
    if not optimizer.cuda_available:
        print("❌ CUDA/GPU가 감지되지 않았습니다. CPU 최적화만 사용됩니다.")
        return None
    
    print("🚀 GPU 최적화 환경 설정 중...")
    
    # GPU 설정 분석
    gpu_config = optimizer.get_gpu_optimization_config()
    
    # GPU 최적화 파일들 생성
    dockerfile_path = optimizer.create_gpu_optimized_dockerfile()
    compose_path = optimizer.create_gpu_docker_compose()
    script_path = optimizer.create_gpu_run_script()
    
    print(f"📝 GPU Dockerfile 생성: {dockerfile_path}")
    print(f"📝 GPU Docker Compose 생성: {compose_path}")
    print(f"📝 GPU 실행 스크립트 생성: {script_path}")
    
    return {
        'gpu_config': gpu_config,
        'dockerfile': dockerfile_path,
        'compose_file': compose_path,
        'run_script': script_path
    }

if __name__ == "__main__":
    result = setup_gpu_optimization()
    if result:
        print("🎉 GPU 최적화 설정 완료!")
        print("사용법:")
        print("  Linux/Mac: ./run_gpu_optimized.sh <xml_file>")
        print("  Windows: run_gpu_optimized.bat <xml_file>")
    else:
        print("💻 GPU를 사용할 수 없어 CPU 모드로 실행됩니다.")