FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*


RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y curl python3.10 python3.10-distutils python3.10-venv git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

RUN python3.10 -m pip --version

# Copy forcateri (now inside build context)
COPY forcateri /forcateri

# Copy heating-forecast
COPY heating-forecast /heating-forecast

WORKDIR /heating-forecast

# Install forcateri editable
RUN pip install -e /forcateri

# Install heating-forecast requirements
RUN pip install -r requirements.txt


