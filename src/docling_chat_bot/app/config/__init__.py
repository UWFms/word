"""
Environment-based configuration for the docling service.

This package exposes a single object called ``properties`` which holds
all configuration parameters.  Each attribute on ``properties`` is
populated from an environment variable if present, otherwise a
reasonable default is used.  This approach avoids the complexity of
YAML or other configuration loaders and makes it straightforward to
access settings throughout the codebase via ``properties.FOO``.
"""
import os
class Properties:
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    MILVUS_DB_HOST: str | None = os.getenv("MILVUS_DB_HOST")
    MILVUS_DB_PORT: int = int(os.getenv("MILVUS_DB_PORT", "19530"))
    MILVUS_DB_NAME: str | None = os.getenv("MILVUS_DB_NAME")
    MILVUS_COLLECTION_NAME: str | None = os.getenv("MILVUS_COLLECTION_NAME")
    MILVUS_USERNAME: str = os.getenv("MILVUS_USERNAME", "root")
    MILVUS_PASSWORD: str | None = os.getenv("MILVUS_PASSWORD")
    LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL_NAME: str | None = os.getenv("LLM_MODEL_NAME")
    LLM_CATALOG_ID_YANDEX: str | None = os.getenv("LLM_CATALOG_ID_YANDEX")
    LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    LLM_TOKENIZE_CONNECT_TIMEOUT: float = float(
        os.getenv("LLM_TOKENIZE_CONNECT_TIMEOUT", "1.0")
    )
    LLM_TOKENIZE_TIMEOUT: float = float(os.getenv("LLM_TOKENIZE_TIMEOUT", "3.0"))
    LLM_MODEL_MAX_TOKEN_INPUT: int = int(os.getenv("LLM_MODEL_MAX_TOKEN_INPUT", "32000"))
    LLM_MODEL_MAX_TOKEN_OUTPUT: int = int(os.getenv("LLM_MODEL_MAX_TOKEN_OUTPUT", "1000"))
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str | None = os.getenv("POSTGRES_DB")
    POSTGRES_USER: str | None = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str | None = os.getenv("POSTGRES_PASSWORD")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    DOCUMENT_PATH: str | None = os.getenv("DOCUMENT_PATH")

properties = Properties()
