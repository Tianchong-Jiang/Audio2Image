FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   ffmpeg \
                   bsm6 \
                   libxext6 \
                   python3.8 \
                   python3-pip \
                   python3.8-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio
RUN python3 -m pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        Jinja2 \
        librosa \
        numpy \
        scipy \
        tensorboard \
        transformers

RUN python3 -m pip install --no-cache-dir \
    jupyterlab \
    wandb \
    kornia \
    imageio \
    imageio-ffmpeg \
    moviepy \
    opencv-python

RUN python3 -m pip install --no-cache-dir \
    bitsandbytes


# Activate the venv as default!
CMD ["/bin/bash", "-c", ". /opt/venv/bin/activate && /bin/bash"]
