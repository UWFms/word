import time
from typing import List, Optional

import docling.backend.msword_backend as msb  # type: ignore
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, InputFormat, WordFormatOption

from .logger import logger
from .tokenization import YandexTokenizer

# Флаг, чтобы патчить обработчик формул только один раз
_EQUATIONS_PATCHED: bool = False


def _disable_equations(*args, **kwargs):
    """
    Игнорирует любые OMML-формулы и предотвращает падения Docling.
    Возвращаем пустую строку и пустой список формул.
    """
    return "", []


def _patch_equations() -> None:
    """
    Находим в msword_backend класс(ы) с методом _handle_equations_in_text
    и подменяем этот метод на _disable_equations.
    """
    global _EQUATIONS_PATCHED
    if _EQUATIONS_PATCHED:
        return

    try:
        for name, obj in msb.__dict__.items():
            if isinstance(obj, type) and "_handle_equations_in_text" in obj.__dict__:
                logger.debug(f"Patching equation handler for {obj}")
                obj._handle_equations_in_text = _disable_equations  # type: ignore[attr-defined]
                _EQUATIONS_PATCHED = True
                break

        if not _EQUATIONS_PATCHED:
            logger.warning(
                "Could not find _handle_equations_in_text in msword_backend; "
                "equation patch was not applied."
            )
    except Exception as e:
        logger.error(f"Failed to patch equation handler: {e}", exc_info=True)


class DocProcessor:
    """
    Простой процессор, который умеет обработать один DOCX-файл и вернуть список чанков.

    Использование:
        processor = DocProcessor()
        chunks = processor.process_single_file("/app/uploads/WKR.docx", "WKR.docx")
    """

    def __init__(self) -> None:
        _patch_equations()

    def process_single_file(self, path: str, filename: str) -> List:
        logger.info(f"Converting DOCX with docling: {path}")

        _patch_equations()

        converter = DocumentConverter(
            format_options={InputFormat.DOCX: WordFormatOption()}
        )

        try:
            result = converter.convert(path)
        except Exception as e:
            logger.error(f"Failed to convert document '{path}': {e}", exc_info=True)
            raise

        doc = result.document
        logger.info("Document successfully converted to docling representation")

        # Токен-осведомлённое чанкование
        tokenizer = YandexTokenizer()
        max_tokens = tokenizer.get_max_tokens()
        logger.info(
            "Running HybridChunker with Yandex tokenizer (connect_timeout=%ss, read_timeout=%ss, max_tokens=%s)...",
            tokenizer.connect_timeout_seconds,
            tokenizer.timeout_seconds,
            max_tokens,
        )
        if not tokenizer._is_api_reachable():
            logger.warning(
                "Yandex tokenize endpoint %s is not reachable; HybridChunker will use "
                "word-based estimates for chunk sizing.",
                tokenizer.api_url,
            )

        chunker = HybridChunker(tokenizer=tokenizer)

        start_time = time.perf_counter()
        chunks = []
        try:
            for idx, chunk in enumerate(chunker.chunk(doc)):
                chunk_text: Optional[str] = getattr(chunk, "text", None) or getattr(
                    chunk, "content", None
                )
                text_preview = (chunk_text or "").strip().replace("\n", " ")
                if len(text_preview) > 120:
                    text_preview = text_preview[:117] + "..."

                meta_summary = getattr(chunk, "metadata", {}) or {}
                heading_meta = meta_summary.get("headings") or meta_summary.get(
                    "heading"
                )
                logger.debug(
                    "Chunk %s: text_len=%s, headings=%s, preview='%s'",
                    idx,
                    len(chunk_text or ""),
                    heading_meta,
                    text_preview,
                )
                chunks.append(chunk)
        except Exception:
            logger.error("HybridChunker failed while chunking the document", exc_info=True)
            raise

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Document chunked into %s segments in %.2fs (max_tokens=%s)",
            len(chunks),
            elapsed,
            max_tokens,
        )

        # Добавим имя документа в метаданные каждого чанка
        for ch in chunks:
            try:
                if hasattr(ch, "metadata") and isinstance(ch.metadata, dict):
                    ch.metadata.setdefault("document_name", filename)
            except Exception:
                continue

        return chunks
