FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    uuid-runtime \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip first for better dependency resolution
RUN pip install --upgrade pip

# Install torch with CUDA support
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install a specific version of transformers known to work with this Python version
RUN pip install transformers==4.11.3 tokenizers==0.10.3

# Copy requirements file and install remaining dependencies
COPY requirements-cloud.txt /app/
RUN grep -v "torch\|transformers" requirements-cloud.txt > modified_requirements.txt && \
    sed -i 's/protobuf==3.20.1/protobuf>=3.7.0,<4.0.0/g' modified_requirements.txt && \
    pip install -r modified_requirements.txt

# Copy application code
COPY . /app/

# Make sure NVIDIA driver environment variables are set
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# entrypoint.sh is the file that will be executed when the container is run
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
