from typing import Any, Iterable, Sequence

import requests
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from pydantic import ConfigDict, PrivateAttr

from .config.config import properties
from .logger import logger


class YandexTokenizer(BaseTokenizer):
    """Tokenizer implementation backed by Yandex's tokenize API."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    api_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenize"

    _endpoint_reachable: bool | None = PrivateAttr(default=None)
    _connect_timeout: float = PrivateAttr(default=1.0)
    _read_timeout: float = PrivateAttr(default=3.0)

    def __init__(self) -> None:
        super().__init__()
        connect_timeout = max(0.25, float(properties.LLM_TOKENIZE_CONNECT_TIMEOUT or 1.0))
        read_timeout = max(connect_timeout, float(properties.LLM_TOKENIZE_TIMEOUT or 3.0))
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout

    @property
    def timeout_seconds(self) -> float:
        return self._read_timeout

    @property
    def connect_timeout_seconds(self) -> float:
        return self._connect_timeout

    @property
    def timeout(self) -> tuple[float, float]:
        """Return a (connect, read) timeout tuple for requests."""

        return (self._connect_timeout, self._read_timeout)

    def _is_api_reachable(self) -> bool:
        """Cheap reachability probe to avoid hanging on a dead endpoint."""

        if self._endpoint_reachable is not None:
            return self._endpoint_reachable

        try:
            response = requests.get(self.api_url, timeout=self.timeout)
            # Even 4xx (e.g., 405 Method Not Allowed) proves the host is reachable.
            reachable = response.status_code < 500
        except requests.RequestException as exc:
            logger.warning(
                "Yandex tokenize endpoint %s is unreachable: %s. Using word-based fallback.",
                self.api_url,
                exc,
            )
            reachable = False

        self._endpoint_reachable = reachable
        return reachable

    def _tokenize_via_api(self, text: str) -> list[str]:
        if not properties.LLM_MODEL_NAME:
            logger.warning(
                "LLM_MODEL_NAME is not configured; falling back to naive token counting.",
            )
            return []

        headers = {
            "Authorization": f"Bearer {properties.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"modelUri": properties.LLM_MODEL_NAME, "text": text}

        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("tokens", [])

    def count_tokens(self, text: str) -> int:  # type: ignore[override]
        """Get number of tokens for given text."""

        try:
            if not self._is_api_reachable():
                return len(text.split())

            tokens = self._tokenize_via_api(text)
            return len(tokens) if tokens else len(text.split())
        except requests.Timeout:
            logger.warning(
                "Yandex tokenization timed out after %ss; disabling token API for this run.",
                self.timeout_seconds,
            )
            self._endpoint_reachable = False
            return len(text.split())
        except requests.RequestException as exc:  # pragma: no cover - network fallback
            logger.error(
                "Error in Yandex token counting: %s. Falling back to word-based counting and disabling token API for this run.",
                exc,
                exc_info=True,
            )
            self._endpoint_reachable = False
            return len(text.split())

    def get_max_tokens(self) -> int:  # type: ignore[override]
        """Get maximum number of tokens allowed."""

        return int(properties.LLM_MODEL_MAX_TOKEN_INPUT)

    def get_tokenizer(self) -> Any:  # type: ignore[override]
        """Return a placeholder tokenizer object (API-based tokenizer)."""

        return None


def _count_content_tokens(content: Any, tokenizer: BaseTokenizer) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return tokenizer.count_tokens(content)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or str(content)
        return tokenizer.count_tokens(text)
    if isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
        return sum(_count_content_tokens(item, tokenizer) for item in content)
    return tokenizer.count_tokens(str(content))


def count_token_in_messages(messages: Sequence[Any]) -> int:
    """Count tokens across a collection of messages.

    Each message is expected to expose a ``content`` attribute which can be
    either a string, a list (for hierarchical content), or a mapping with a
    ``text``/``content`` field.
    """

    tokenizer = YandexTokenizer()
    total = 0
    for message in messages:
        content = getattr(message, "content", None)
        total += _count_content_tokens(content, tokenizer)
    return total
