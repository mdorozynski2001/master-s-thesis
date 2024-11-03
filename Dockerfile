FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
SHELL ["/bin/bash", "-c", "-o", "pipefail"]

# Nvidia
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility,graphics
ENV CUDA_MODULE_LOADING=LAZY
ENV DEBIAN_FRONTEND=noninteractive

# libtcmalloc-minimal4 should be used to avoid memory leaks during training.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libtcmalloc-minimal4 \
    make \
    git  \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip:
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_ROOT_USER_ACTION=ignore \
    # poetry
    POETRY_VERSION=1.8.4 \
    POETRY_NO_INTERACTION=1 \
    # For development inside docker container we do not use virtualenv
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME='/opt/poetry' \
    LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4'

ENV PATH="$POETRY_HOME/bin:$PATH"
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -

# Project initialization:
WORKDIR /app
COPY poetry.lock /app/poetry.lock
COPY pyproject.toml /app/pyproject.toml

RUN poetry config cache-dir /app/.cache/pypoetry && \
    poetry install --no-interaction --no-ansi --no-root -vvv

RUN poetry run python3 --version
RUN python3 --version
RUN rm /app/poetry.lock /app/pyproject.toml

RUN git config --global --add safe.directory /app
