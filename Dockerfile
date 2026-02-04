# CSP Docker 환경 - PyTorch 2.6 기반
# ========================================
# 버전 동기화 참고 (Dockerfile vs requirements.txt)
# ========================================
# - torch: Docker 베이스 이미지 2.6.0 유지 (requirements.txt는 2.9.1)
#   → 이유: pytorch/pytorch:2.9.1 공식 이미지가 아직 없음
#   → 로컬 .venv는 2.9.1 사용 가능, Docker는 2.6.0 고정
# - 기타 패키지: requirements.txt와 동일 버전 유지
# ========================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 시스템 패키지 설치 (한글/한자 폰트)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ca-certificates \
    locales fontconfig \
    fonts-nanum fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# 한국어 로케일 설정
RUN locale-gen ko_KR.UTF-8
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko
ENV PYTHONIOENCODING=utf-8

# 폰트 캐시
RUN fc-cache -fv

# uv 설치 (pip 대비 2-10배 빠른 패키지 설치)
# 참고: https://github.com/astral-sh/uv
RUN pip install --no-cache-dir uv

# PyTorch 검증
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 필수 패키지 설치 (uv 사용으로 빌드 속도 향상)
# 주의: torch는 베이스 이미지(2.6.0)를 유지 (2.9.1 Docker 이미지 없음)
RUN uv pip install --system --no-cache \
    transformers==4.57.3 \
    datasets==3.2.0 \
    accelerate==1.10.0 \
    huggingface-hub==0.34.4 \
    FlagEmbedding>=1.2.0 \
    sentence-transformers>=2.2.0 \
    pandas==2.3.3 \
    tqdm==4.67.1 \
    numpy==1.26.4 \
    scikit-learn==1.8.0 \
    scipy \
    matplotlib \
    umap-learn \
    plotly \
    openpyxl==3.1.5 \
    lxml==5.1.0 \
    kiwipiepy==0.22.2 \
    regex==2023.12.25 \
    openai==2.14.0 \
    google-generativeai==0.7.2 \
    peft==0.15.0 \
    bitsandbytes==0.43.3 \
    stanza \
    suparkanbun \
    esupar

# Stanza 리소스 사전 다운로드
ENV CSP_STANZA_DIR=/opt/stanza_resources
RUN python -c "import os; import stanza; d=os.environ.get('CSP_STANZA_DIR','/opt/stanza_resources'); os.makedirs(d, exist_ok=True); stanza.download('ko', model_dir=d, verbose=False); stanza.download('zh', model_dir=d, verbose=False)"

# 환경 변수
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 워크스페이스
WORKDIR /workspace

# 기본 명령
CMD ["/bin/bash"]
