import json
import os
import re
from typing import List, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from docling_chat_bot.app.config import properties
from docling_chat_bot.app.doc_processor import DocProcessor
from docling_chat_bot.app.logger import logger
from docling_chat_bot.app.milvus import (
    milvus_store,
    recreate_collection,
    search_embeddings,
    check_milvus_connection,
)

router = APIRouter()


class SimilarRequest(BaseModel):
    query: str
    top_k: int = 5


class SimilarHit(BaseModel):
    id: Any
    distance: float
    text: str | None = None
    metadata: Dict[str, Any] | None = None


class SimilarResponse(BaseModel):
    results: List[SimilarHit]


class IndexResponse(BaseModel):
    inserted: int


class HealthResponse(BaseModel):
    status: str
    milvus_ok: bool
    doc_loaded: bool


def _extract_headings_from_meta(meta_str: str) -> list[str] | None:
    """
    meta_str — это строка вида:
      "schema_name='...DocMeta' ... headings=['2 ', '2.4 ', '2.4.1 '] captions=None origin=..."
    Вытаскиваем из неё headings как список строк.
    """
    if not meta_str:
        return None

    m = re.search(r"headings=\[(.*?)]", meta_str)
    if not m:
        return None

    inside = m.group(1)

    values = re.findall(r"'(.*?)'", inside)
    headings = [v.strip() for v in values if v.strip()]

    return headings or None


def _extract_text_and_metadata(chunk: Any, idx: int, document_name: str) -> tuple[str, Dict[str, Any]]:
    """
    Достаём текст и полную metadata из docling chunk.
    """
    text = ""
    for attr in ("text", "content", "page_content"):
        if hasattr(chunk, attr):
            val = getattr(chunk, attr)
            if val:
                text = str(val)
                break
    if not text:
        text = str(chunk)
    text = text.strip()

    meta = {}

    if hasattr(chunk, "__dict__"):
        for key, value in chunk.__dict__.items():
            if key in ("doc_items", "origin", "model_fields", "model_config", "model_extra", "model_computed_fields"):
                continue
            try:
                json.dumps(value)
                meta[key] = value
            except Exception:
                meta[key] = str(value)

    # headings нормализуем
    if "headings" in meta:
        try:
            meta["headings"] = [str(h) for h in meta["headings"]] if isinstance(meta["headings"], list) else []
        except:
            meta["headings"] = []

    # chunk index
    meta["chunk_index"] = idx

    # document name
    meta["document_name"] = document_name

    # source
    meta.setdefault("source", "docling_upload")

    return text, meta


@router.get("/health", response_model=HealthResponse)
def health():
    """
    Простая health-проверка Milvus + загружена ли коллекция.

    doc_loaded: True, если коллекция существует и в ней есть хотя бы один объект.
    """
    milvus_ok = check_milvus_connection()

    try:
        stats = milvus_store.client.get_collection_stats(
            collection_name=properties.MILVUS_COLLECTION_NAME
        )
        row_count = int(stats.get("row_count", "0"))
        doc_loaded = row_count > 0
    except Exception:
        doc_loaded = False

    status = "ok" if milvus_ok and doc_loaded else "degraded"

    return HealthResponse(
        status=status,
        milvus_ok=milvus_ok,
        doc_loaded=doc_loaded,
    )


@router.post("/api/v1/doc/index", response_model=IndexResponse)
def index_document():
    """
    Пересоздаёт коллекцию в Milvus и заливает туда чанки
    для всех .docx-файлов из UPLOAD_DIR.
    """
    # Определяем папку с документами
    upload_dir = properties.UPLOAD_DIR or "./uploads"
    upload_dir_abs = os.path.abspath(upload_dir)

    if not os.path.isdir(upload_dir_abs):
        logger.error(f"Upload dir '{upload_dir_abs}' does not exist")
        raise HTTPException(
            status_code=500,
            detail=f"Upload dir '{upload_dir_abs}' does not exist",
        )

    # Собираем список .docx файлов
    doc_files: list[str] = [
        f for f in os.listdir(upload_dir_abs) if f.lower().endswith(".docx")
    ]

    if not doc_files:
        logger.error(f"No .docx files found in '{upload_dir_abs}'")
        return IndexResponse(inserted=0)

    # Пересоздаём коллекцию
    logger.info("Recreating Milvus collection before indexing...")
    recreate_collection()

    processor = DocProcessor()

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    # Обрабатываем каждый документ из uploads
    for filename in sorted(doc_files):
        doc_path = os.path.join(upload_dir_abs, filename)
        logger.info(f"Processing document for indexing: {doc_path}")

        try:
            chunks = processor.process_single_file(doc_path, filename)
        except Exception as e:
            logger.error(
                f"Failed to process document '{doc_path}': {e}",
                exc_info=True,
            )
            continue

        if not chunks:
            logger.warning(f"No chunks produced for document '{doc_path}'")
            continue

        for idx, chunk in enumerate(chunks):
            text, metadata = _extract_text_and_metadata(chunk, idx, document_name=filename)

            if not text:
                continue

            texts.append(text)
            metadatas.append(metadata)

    if not texts:
        logger.warning("No non-empty chunks to index after processing all documents")
        return IndexResponse(inserted=0)

    # Записываем в Milvus через langchain Milvus store
    logger.info(f"Inserting {len(texts)} chunks into Milvus via milvus_store.add_texts(...)")
    try:
        milvus_store.add_texts(texts=texts, metadatas=metadatas)
    except Exception as e:
        logger.error(f"Failed to insert chunks into Milvus: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to insert chunks into Milvus: {e}")

    logger.info(
        f"Successfully indexed {len(texts)} chunks into collection '{properties.MILVUS_COLLECTION_NAME}'"
    )
    return IndexResponse(inserted=len(texts))


@router.post("/api/v1/doc/similar", response_model=SimilarResponse)
def similar(payload: SimilarRequest):
    """
    Поиск похожих чанков по запросу пользователя.
    Использует search_embeddings из milvus.py (Yandex + MilvusClient.search).
    """
    logger.info(f"Searching similar for query='{payload.query}', top_k={payload.top_k}")

    try:
        raw_results = search_embeddings(payload.query, payload.top_k)
    except Exception as e:
        logger.error(f"Milvus search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Milvus search failed: {e}")

    hits: List[SimilarHit] = []
    for r in raw_results:
        meta_full = r.get("metadata") or {}

        meta_public = {}
        if "meta" in meta_full and isinstance(meta_full["meta"], str):
            headings = _extract_headings_from_meta(meta_full["meta"])
            if headings:
                meta_public["headings"] = headings
        if "document_name" in meta_full:
            meta_public["document_name"] = meta_full["document_name"]
        if "chunk_index" in meta_full:
            meta_public["chunk_index"] = meta_full["chunk_index"]

        hits.append(
            SimilarHit(
                id=r.get("id"),
                distance=r.get("distance"),
                text=r.get("text"),
                metadata=meta_public,
            )
        )

    return SimilarResponse(results=hits)
