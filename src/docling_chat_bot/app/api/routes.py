import json
import os
import re
from typing import List, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config.config import properties
from ..doc_processor import DocProcessor
from ..logger import logger
from ..milvus import (
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


class SectionChunksRequest(BaseModel):
    document_name: str
    chunk_index: int
    depth: int = 1


class SectionChunkHit(BaseModel):
    chunk_index: int
    document_name: str
    text: str | None = None
    headings: list[str] | None = None
    metadata: Dict[str, Any] | None = None


class SectionChunksResponse(BaseModel):
    depth_used: int
    target_heading: str | None
    results: list[SectionChunkHit]
    message: str | None = None


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


def _headings_from_metadata(meta_full: Dict[str, Any]) -> list[str] | None:
    """Extract headings from metadata JSON or embedded meta string."""

    if not meta_full:
        return None

    if "headings" in meta_full and isinstance(meta_full["headings"], list):
        normalized: list[str] = []
        for h in meta_full["headings"]:
            text = str(h).strip()
            if text:
                normalized.append(text)
        if normalized:
            return normalized

    if "meta" in meta_full and isinstance(meta_full["meta"], str):
        return _extract_headings_from_meta(meta_full["meta"])

    return None


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


@router.post("/api/v1/doc/chunks-by-heading", response_model=SectionChunksResponse)
def chunks_by_heading(payload: SectionChunksRequest):
    """
    Возвращает все чанки документа, находящиеся на том же уровне оглавления.

    depth=1 — подраздел (последний элемент headings), depth=2 — родительский раздел и т.д.
    """

    if payload.depth < 1:
        depth_requested = 1
    else:
        depth_requested = payload.depth

    try:
        doc_chunks = milvus_store.client.query(
            collection_name=properties.MILVUS_COLLECTION_NAME,
            filter=f"metadata['document_name'] == \"{payload.document_name}\"",
            output_fields=["text", "metadata"],
            limit=10000,
        )
    except Exception as e:
        logger.error(f"Milvus query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")

    if not doc_chunks:
        raise HTTPException(status_code=404, detail="Document not found in Milvus")

    def _as_int(val: Any) -> int | None:
        try:
            return int(val)
        except Exception:
            return None

    target_chunk = None
    for ch in doc_chunks:
        meta = (ch.get("metadata") or {}) if isinstance(ch, dict) else {}
        if _as_int(meta.get("chunk_index")) == payload.chunk_index:
            target_chunk = ch
            break

    if not target_chunk:
        raise HTTPException(status_code=404, detail="Chunk not found in document")

    target_meta = target_chunk.get("metadata") or {}
    headings = _headings_from_metadata(target_meta)

    if not headings:
        return SectionChunksResponse(
            depth_used=0,
            target_heading=None,
            results=[],
            message="У выбранного чанка не обнаружен раздел",
        )

    depth_used = min(depth_requested, len(headings))
    target_heading = headings[-depth_used]

    matches: list[SectionChunkHit] = []
    for ch in doc_chunks:
        meta = (ch.get("metadata") or {}) if isinstance(ch, dict) else {}
        chunk_headings = _headings_from_metadata(meta) or []

        if target_heading not in chunk_headings:
            continue

        idx_val = _as_int(meta.get("chunk_index"))
        match = SectionChunkHit(
            chunk_index=idx_val if idx_val is not None else -1,
            document_name=str(meta.get("document_name", payload.document_name)),
            text=ch.get("text"),
            headings=chunk_headings or None,
            metadata=meta or None,
        )
        matches.append(match)

    matches.sort(key=lambda m: m.chunk_index)

    return SectionChunksResponse(
        depth_used=depth_used,
        target_heading=target_heading,
        results=matches,
        message=None,
    )
