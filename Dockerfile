FROM python:3.11-slim

WORKDIR /app

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip:
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_ROOT_USER_ACTION=ignore \
    # poetry:
    POETRY_VERSION=2.1.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME="/usr/local" \
    POETRY_HTTP_TIMEOUT=120

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
    build-essential \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    libpq-dev \
    curl \
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}" "poetry-plugin-export"

COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt --without-hashes --output /tmp/requirements.txt \
  && awk '!/^nvidia-/' /tmp/requirements.txt \
     | sed '/^torch==/d;/^torchvision==/d' \
     > /tmp/requirements.cpu.txt \
  && pip install --no-cache-dir -r /tmp/requirements.cpu.txt \
     --extra-index-url https://download.pytorch.org/whl/cpu \
     torch==2.4.1+cpu torchvision==0.19.1+cpu

COPY src ./src

ENV PYTHONPATH="/app/src"

CMD ["python", "-m", "uvicorn", "docling_chat_bot.main:app", "--host", "0.0.0.0", "--port", "8000"]
