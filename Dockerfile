# CSP Docker 환경 - PyTorch 2.6 + SQL Server 지원
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 시스템 패키지 설치 (한글/한자 + ODBC 지원)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ca-certificates \
    locales fontconfig \
    fonts-nanum fonts-noto-cjk fonts-dejavu-core \
    unixodbc unixodbc-dev odbcinst gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

# SQL Server ODBC Driver 17 설치
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
    rm -rf /var/lib/apt/lists/*

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

# 필수 패키지 설치 (requirements.txt 포함 pyodbc)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    transformers==4.55.3 \
    datasets==3.2.0 \
    accelerate==1.10.0 \
    huggingface-hub==0.34.4 \
    pandas==2.2.2 \
    pyodbc==5.1.0 \
    sqlalchemy==2.0.35 \
    tqdm==4.67.1 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    openpyxl==3.1.2 \
    lxml==5.1.0 \
    kiwipiepy==0.22.1

# 환경 변수
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 워크스페이스
WORKDIR /workspace

# 기본 명령
CMD ["/bin/bash"]
