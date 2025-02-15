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

# Copy your local code into the container
#COPY . /app

# Upgrade pip and install PyTorch with CUDA support, plus required Python packages.
RUN pip3 install torch torchaudio && \
    pip3 install --no-cache-dir librosa soundfile faster-whisper

# Install uv
RUN pip3 install uv
RUN git clone https://github.com/xezpeleta/whisper_streaming.git

WORKDIR /app/whisper_streaming

# Uv commands
RUN uv python install 3.10 && uv venv --python 3.10
RUN uv add --script whisper_online_server.py faster-whisper numpy librosa

ENTRYPOINT ["uv", "run", "whisper_online_server.py", "--backend", "faster-whisper","--language", "eu", "--host", "localhost", "--port", "43001"]
