import os
import yaml
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _set_nested(target: dict, path: tuple[str, ...], value):
    """Безопасно создаёт вложенные ключи и вставляет значение."""
    ref = target
    for key in path[:-1]:
        if key not in ref or not isinstance(ref[key], dict):
            ref[key] = {}
        ref = ref[key]
    ref[path[-1]] = value


def load_config() -> dict:
    # YAML может быть пустым
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Карта: ENV → путь в config
    env_map = {
        # server
        "SERVER_HOST": ("server", "host"),
        "SERVER_PORT": ("server", "port"),

        # uploads
        "UPLOAD_DIR": ("upload", "dir"),

        # milvus
        "MILVUS_DB_HOST": ("milvus", "db", "host"),
        "MILVUS_DB_PORT": ("milvus", "db", "port"),
        "MILVUS_DB_NAME": ("milvus", "db", "name"),
        "MILVUS_COLLECTION_NAME": ("milvus", "collection", "name"),
        "MILVUS_USERNAME": ("milvus", "auth", "username"),
        "MILVUS_PASSWORD": ("milvus", "auth", "password"),

        # llm
        "LLM_BASE_URL": ("llm", "base_url"),
        "LLM_API_KEY": ("llm", "api_key"),
        "LLM_MODEL_NAME": ("llm", "model_name"),
        "LLM_CATALOG_ID_YANDEX": ("llm", "catalog_id"),
        "LLM_REQUEST_TIMEOUT": ("llm", "timeout"),
        "LLM_MODEL_MAX_TOKEN_INPUT": ("llm", "max_token_input"),
        "LLM_MODEL_MAX_TOKEN_OUTPUT": ("llm", "max_token_output"),

        # postgres
        "POSTGRES_HOST": ("postgres", "host"),
        "POSTGRES_PORT": ("postgres", "port"),
        "POSTGRES_DB": ("postgres", "db_name"),
        "POSTGRES_USER": ("postgres", "user"),
        "POSTGRES_PASSWORD": ("postgres", "password"),
    }

    # применяем ENV
    for env_key, path in env_map.items():
        value = os.getenv(env_key)
        if value not in (None, ""):
            _set_nested(data, path, value)

    return data
