# CSP Docker 환경 - PyTorch 2.6 기반
# PyTorch 2.6.0이 Docker Hub의 최신 cuda12.4-cudnn9 이미지
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

# pip 업그레이드
RUN python -m pip install --upgrade pip wheel setuptools

# PyTorch 검증
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 필수 패키지 설치 (보안 및 성능 개선 버전)
# 주의: torch는 베이스 이미지(2.6.0)를 유지 (2.9.1 Docker 이미지 없음)
RUN pip install --no-cache-dir \
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
