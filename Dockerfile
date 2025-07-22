# Start from the official Codespaces base image (Ubuntu + dev tools)
FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip

# Set python3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Optional: Set default shell to bash
SHELL ["/bin/bash", "-c"]
