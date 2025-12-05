FROM python:3.11-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=30 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir poetry poetry-plugin-export
RUN poetry export -f requirements.txt --without-hashes --output requirements.txt \
    && awk '!/^nvidia-/' requirements.txt | sed '/^torch==/d;/^torchvision==/d' > requirements.cpu.txt \
    && pip install --no-cache-dir -r requirements.cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu torchvision==0.19.1+cpu

COPY src ./src

ENV PYTHONPATH="/app/src"

CMD ["poetry", "run", "uvicorn", "docling_chat_bot.main:app", "--host", "0.0.0.0", "--port", "8000"]