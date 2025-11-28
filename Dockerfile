FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    libxml2 \
    libxslt1.1 \
    libpq-dev \
    && apt-get clean

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry
RUN poetry install --no-root

COPY src ./src

ENV PYTHONPATH="/app/src"

CMD ["poetry", "run", "uvicorn", "docling_chat_bot.main:app", "--host", "0.0.0.0", "--port", "8000"]
