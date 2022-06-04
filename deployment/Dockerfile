FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy}

SHELL ["/bin/bash", "-c"]

# Fix issue https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904/3
RUN apt-key adv --keyserver-options http-proxy=${http_proxy} --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install common dependencies
RUN apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install \
        build-essential \
        ca-certificates \
        git \
        gnupg2 \
        libcairo2-dev \
        libgif-dev \
        libgirepository1.0-dev \
        libncurses-dev \
        libpango1.0-dev \
        librsvg2-dev \
        tmux \
        vim \
        ffmpeg \
        libsm6 \
        libxext6

RUN conda install av moviepy -c conda-forge && \
    pip install -U pip && \
    pip install \
        'git+https://github.com/facebookresearch/fvcore' \
        'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
        numpy \
        torch \
        iopath \
        simplejson \
        psutil \
        sklearn \
        opencv-python \
        torchvision \
        tensorboard \
        pytorchvideo \
        cython && \
    git clone https://github.com/facebookresearch/detectron2 /tmp/detectron2_repo && \
    pip install -e /tmp/detectron2_repo

ENV PYTHONPATH=/store/research/SlowFast:/tmp/slowfast:${PYTHONPATH}

RUN apt-get update && \
      apt-get -y install sudo && \
      echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
      chmod a+rwx /opt/conda/lib/python3.8/site-packages/

# ADD . /tmp/slowfast
# RUN cd /tmp/slowfast && \
#     python setup.py build develop
