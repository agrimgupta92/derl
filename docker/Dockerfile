# Base Image
ARG PARENT_IMAGE
FROM $PARENT_IMAGE

# Args need to be below FROM! Docker is stupid
ARG TORCH
ARG USER_ID

# Setup basic packages 
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    cmake \
    unzip \
    bzip2 \
    wget \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libopenmpi-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    ffmpeg \
    xpra \
    libglfw3 \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /user/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /user/.mujoco \
    && mv /user/.mujoco/mujoco200_linux /user/.mujoco/mujoco200 \
    && rm mujoco.zip
COPY docker/mjkey.txt /user/.mujoco/
ENV LD_LIBRARY_PATH /user/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH /user/.mujoco/mujoco200
ENV MUJOCO_PY_MJKEY_PATH /user/.mujoco/mjkey.txt

COPY docker/build_files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY docker/build_files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install conda 
WORKDIR /user
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -p /user/miniconda -b
RUN rm /user/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/user/miniconda/bin:${PATH}

# Create the conda env.
RUN conda create -n docker_env python=3.6
# Make python default to env python
ENV PATH=/user/miniconda/envs/docker_env/bin:${PATH}

RUN /bin/bash -c ". activate docker_env; conda install pytorch ${TORCH} -c pytorch"

COPY docker/build_files/requirements.txt /user/derl/docker/build_files/requirements.txt
RUN /bin/bash -c ". activate docker_env; cd derl/docker/build_files; pip install -r requirements.txt"

# Copy the code in the very end
COPY derl /user/derl/derl
COPY tools /user/derl/tools
COPY configs /user/derl/configs

COPY setup.py /user/derl/
RUN /bin/bash -c ". activate docker_env; cd derl; pip install -e ."

RUN conda init
RUN /bin/bash -c ". activate docker_env"

# Change permissions
RUN useradd --shell /bin/bash -u ${USER_ID} -o -d /user user
RUN chown -R user /user/miniconda/envs/docker_env/lib/python3.6/site-packages/mujoco_py/

# Set python ENV variables
ENV PYTHONUNBUFFERED=1