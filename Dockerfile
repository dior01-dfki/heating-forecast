# Base image with CUDA & cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
        git \
        software-properties-common \
        curl \
        python3.10 \
        python3.10-distutils \
        python3.10-venv \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && python3.10 -m pip install --upgrade pip

# Set working directories
WORKDIR /app

# Copy forcateri first (framework)
COPY forcateri /forcateri

# Set PYTHONPATH so editable install works inside container
ENV PYTHONPATH=/forcateri:$PYTHONPATH

# Install forcateri in editable mode
RUN pip install -e /forcateri

# Copy heating-forecast repo after forcateri
COPY heating-forecast /heating-forecast
WORKDIR /heating-forecast

# Install heating-forecast requirements
RUN pip install --no-cache-dir -r requirements.txt


