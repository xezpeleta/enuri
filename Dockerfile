FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python3
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Madrid apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    cargo \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Upgrade pip and install PyTorch with CUDA support, plus required Python packages.
RUN pip3 install torch torchaudio && \
    pip3 install --no-cache-dir librosa soundfile faster-whisper

# Install uv
RUN pip3 install uv
RUN git clone https://github.com/xezpeleta/zuzeneko_whistreaminga.git

WORKDIR /app/zuzeneko_whistreaminga

RUN pip install -r requirements.web.txt
COPY samples .

ENTRYPOINT ["uv", "run", "whisper_online_webserver.py"]
CMD ["--task", "transcribe", "--buffer_trimming_sec", "10", "--backend", "faster-whisper", "--model","tiny.en", "--language", "en"]