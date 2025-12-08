"""
App configuration using Pydantic Settings for type safety, validation and environment variable support
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # The host address where the uvicorn server will run
    SERVER_HOST: str
    # The port number for the uvicorn server to listen on
    SERVER_PORT: int
    # Milvus database host address
    MILVUS_DB_HOST: str
    # Milvus database port
    MILVUS_DB_PORT: int
    # Milvus database name
    MILVUS_DB_NAME: str
    # Milvus collection name for storing vectors
    MILVUS_COLLECTION_NAME: str
    # Username for protect milvus
    MILVUS_USERNAME: str = "root"
    # Password for protect milvus
    MILVUS_PASSWORD: str
    # Model server url to connect llm server
    LLM_BASE_URL: str = "https://llm.api.cloud.yandex.net/v1"
    # Model api key is a secret account token for llm server
    LLM_API_KEY: str
    # Model ID used to generate the response
    LLM_MODEL_NAME: str
    # Model catalog id to connect llm server
    LLM_CATALOG_ID_YANDEX: str
    # seconds
    LLM_REQUEST_TIMEOUT: int = 120
    # seconds
    LLM_TOKENIZE_CONNECT_TIMEOUT: int = 1
    # seconds
    LLM_TOKENIZE_TIMEOUT: int = 3
    # Model input tokens is a context length
    LLM_MODEL_MAX_TOKEN_INPUT: int = 32000
    # Model output tokens is a context length
    LLM_MODEL_MAX_TOKEN_OUTPUT: int = 1000
    # Additional RAG params
    RAG_RETRIEVE_DOCUMENT_LIMIT: int = 5
    RAG_SCORE_THRESHOLD: float = 0.5
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    UPLOAD_DIR: str = "./uploads"

    # `+psycopg` driver notation from SQLAlchemy
    @property
    def DATABASE_URL(self):
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def DATABASE_URL_WITHOUT_DRIVER(self):
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_nested_delimiter='_',
        # case_sensitive=False,
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

# Create settings instance
properties = Settings()