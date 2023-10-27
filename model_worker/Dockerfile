FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV PYTHONFAULTHANDLER=1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MPLCONFIGDIR /tmp/matplotlib
ENV TRANSFORMERS_CACHE=/models
# Install dependencies
RUN --mount=type=cache,target=/var/cache/apt,id=apt \
    apt-get update && \
    apt-get install --no-install-recommends -y git curl python3.10 python3-dev python3-pip tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

RUN ln -s /usr/bin/python3 /usr/bin/python

# Create appuser
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Create app directory and change ownership to appuser
RUN mkdir /model_worker && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /model_worker /home/appuser
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY ./requirements.txt /model_worker/requirements.txt 
RUN --mount=type=cache,target=~/.cache/pip \
    pip install -r /model_worker/requirements.txt
COPY ./model_worker/requirements.txt /model_worker/requirements.txt
RUN --mount=type=cache,target=~/.cache/pip \
    pip install -r /model_worker/requirements.txt

# # Set working directory to /app

COPY --chown=appuser:appuser ./model_worker /model_worker
WORKDIR /model_worker
RUN chmod +x ./run.sh
ENTRYPOINT ["tini", "--", "bash", "./run.sh"]   


