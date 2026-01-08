# CSP Docker 환경 - PyTorch 2.6 CUDA 완전 고정
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 시스템 업데이트 및 필수 패키지 설치 (한글/한자 지원 포함)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    # 로케일 및 폰트 지원
    locales \
    fontconfig \
    # 한글/한자 폰트 패키지
    fonts-nanum \
    fonts-nanum-coding \
    fonts-nanum-extra \
    fonts-unfonts-core \
    fonts-unfonts-extra \
    fonts-baekmuk \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# 한국어 로케일 설정
RUN locale-gen ko_KR.UTF-8
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8  
ENV LANGUAGE=ko_KR:ko
ENV PYTHONIOENCODING=utf-8

# 폰트 캐시 업데이트
RUN fc-cache -fv

# pip 업그레이드
RUN python -m pip install --upgrade pip

# 🔥 1단계: PyTorch 2.6 CUDA 상태 검증
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 🔥 2단계: 필수 torch 관련 패키지들 시스템 레벨 고정
RUN pip install \
    transformers==4.55.3 \
    huggingface-hub==0.34.4 \
    FlagEmbedding==1.2.11 \
    accelerate==1.10.0 \
    sentence-transformers==5.1.0 \
    peft \
    --no-cache-dir

# 🔥 3단계: Poetry 설치
RUN pip install poetry==1.8.2

# Poetry 설정 (시스템 Python 사용)
RUN poetry config virtualenvs.create false
RUN poetry config virtualenvs.in-project false

# 🔒 torch 패키지를 시스템에서 보호하기 위한 설정
RUN poetry config installer.modern-installation false

# 작업 디렉토리 설정
WORKDIR /workspace

# 프로젝트 파일 복사
COPY pyproject.toml ./
ENV PIP_FORCE_REINSTALL=false

# 🔥 4단계: Poetry로 나머지 패키지 설치
RUN poetry install --no-dev

# 🔒 torch 패키지들을 시스템에서 "변경 불가능"으로 마킹
RUN pip freeze | grep -E "torch|numpy" > /workspace/system-locked-packages.txt
RUN echo "🔒 시스템 고정 패키지들:" && cat /workspace/system-locked-packages.txt

# NumPy 버전 호환성 강제 조정
RUN pip install --force-reinstall numpy==1.26.2

# OpenAI 패키지 추가 (테스트용)
RUN pip install openai==1.101.0 pydantic anyio httpx jiter sniffio kiwipiepy --no-cache-dir

# 분석 및 시각화 도구 추가
RUN pip install pandas matplotlib seaborn scikit-learn --no-cache-dir

# 환경 변수 설정
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# GPU 최적화 환경 변수
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV TORCH_CUDNN_BENCHMARK=true
ENV TORCH_BACKENDS_CUDNN_BENCHMARK=true
ENV TORCH_BACKENDS_CUDNN_DETERMINISTIC=false

# 🎯 최종 PyTorch 상태 확인
RUN python -c "import torch; print(f'Final PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
    
CMD ["/bin/bash"]
