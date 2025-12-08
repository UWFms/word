from typing import List

from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, DataType

from .config.config import properties
from .logger import logger

TEXT_EMBEDDING_DIM = "тест"


class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._invocation_params["encoding_format"] = "float"

    def embed_documents(self, texts: List[str], chunk_size: int | None = None) -> List[List[float]]:
        # Ignoring chunk_size parameter
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        invocation_params = self._invocation_params.copy()
        if "model" in invocation_params:
            del invocation_params["model"]
        invocation_params["encoding_format"] = "float"
        response = self.client.create(model=self.model, input=text, **invocation_params)
        return response.data[0].embedding


embedding_function = CustomOpenAIEmbeddings(
    model=f"emb://{properties.LLM_CATALOG_ID_YANDEX}/text-search-doc/latest",
    openai_api_key=properties.LLM_API_KEY,
    openai_api_base=properties.LLM_BASE_URL,
)

milvus_store = Milvus(
    embedding_function=embedding_function,
    connection_args={
        "uri": f"{properties.MILVUS_DB_HOST}:{properties.MILVUS_DB_PORT}",
        "user": properties.MILVUS_USERNAME,
        "password": properties.MILVUS_PASSWORD,
        "db_name": properties.MILVUS_DB_NAME,
    },
    collection_name=properties.MILVUS_COLLECTION_NAME,
    vector_field="vector",
    text_field="text",
    metadata_field="metadata",
)


def recreate_collection():
    client: MilvusClient = milvus_store.client
    collection_name = properties.MILVUS_COLLECTION_NAME

    # 1) Release if loaded
    try:
        client.release_collection(collection_name=collection_name)
    except Exception as e:
        logger.debug(f"release_collection skipped: {e}")

    # 2) Drop if exists
    if client.has_collection(collection_name=collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Dropping it.")
        client.drop_collection(collection_name=collection_name)
        logger.info(f"Successfully dropped collection '{collection_name}'.")

    logger.info(f"Creating new collection: '{collection_name}'")

    embedding_dim = len(milvus_store.embeddings.embed_query(TEXT_EMBEDDING_DIM))

    # Define schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type="COSINE",
        consistency_level="Strong",
    )
    logger.info("Collection schema created successfully.")

    # Create an index for the vector field
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
        params={},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    logger.info("Successfully created index for the 'vector' field.")

    client.load_collection(collection_name=collection_name)


def search_embeddings(query: str, top_k: int = 5):
    vector = embedding_function.embed_query(query)

    results = milvus_store.client.search(
        collection_name=properties.MILVUS_COLLECTION_NAME,
        data=[vector],
        anns_field="vector",
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "metadata"],
    )

    output = []
    for hit in results[0]:
        logger.info(f"{hit=}")
        output.append({
            "id": hit.id,
            "distance": hit.distance,
            "text": hit.entity.get("text"),
            "metadata": hit.entity.get("metadata"),
        })

    return output


def check_milvus_connection() -> bool:
    """Checks the Milvus database connection."""
    logger.info(f"Checking Milvus connection to {properties.MILVUS_DB_HOST}:{properties.MILVUS_DB_PORT}...")
    try:
        milvus_store.client.list_collections()
        logger.info("Milvus connection successful.")
        return True
    except Exception as e:
        logger.error(f"Milvus connection failed: {e}", exc_info=True)
        return False
