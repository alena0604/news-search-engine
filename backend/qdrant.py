from typing import Optional, List
from bytewax.outputs import DynamicSink, StatelessSinkPartition
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from backend.models import EmbeddedDocument
from backend.settings import AppConfig
import logging
import os


class QdrantVectorOutput(DynamicSink):
    """A class representing a Qdrant vector output."""

    def __init__(
        self,
        config: AppConfig,
        vector_size: Optional[int] = None,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        step_id: str = "output",
    ):
        # Fetch vector size from the config or fallback to default
        self._vector_size = vector_size or config.get(
            "EMBEDDING_MODEL_MAX_INPUT_LENGTH", 384
        )
        self._collection_name = collection_name or config.get(
            "VECTOR_DB_OUTPUT_COLLECTION_NAME"
        )

        # Initialize Qdrant client if not passed explicitly
        if client:
            self.client = client
        else:
            self.client = self.build_qdrant_client(config)

        # Ensure the collection exists or create one if necessary
        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

    def build_qdrant_client(self, config: AppConfig) -> QdrantClient:
        """Build the Qdrant client using values from the configuration."""
        url = os.getenv("QDRANT_URL", config.get("QDRANT_URL"))
        api_key = os.getenv("QDRANT_API_KEY", config.get("QDRANT_API_KEY"))

        if not url or not api_key:
            raise KeyError(
                "QDRANT_URL and QDRANT_API_KEY must be set in environment variables or config."
            )

        return QdrantClient(url, api_key=api_key)

    def build(self, step_id, worker_index, worker_count) -> "QdrantVectorSink":
        """Builds a QdrantVectorSink object."""
        return QdrantVectorSink(self.client, self._collection_name)


class QdrantVectorSink(StatelessSinkPartition):
    """A sink that writes document embeddings to a Qdrant collection."""

    def __init__(self, client: QdrantClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def article_exists(self, article_url: str) -> bool:
        """Check if an article with a given URL already exists in the Qdrant collection."""
        try:
            # Build a filter to match the 'url' field in the payload
            filter_condition = Filter(
                must=[FieldCondition(key="url", match=MatchValue(value=article_url))]
            )

            # Use query_points with the filter to check for existing articles by URL
            response = self._client.query_points(
                collection_name=self._collection_name,
                query_filter=filter_condition,  # Filter by article URL
                limit=1,
                with_payload=True,
            )

            # Check if any points were returned in the response
            return len(response.points) > 0
        except Exception as e:
            logging.error(f"Error during article existence check: {e}")
            return False

    def write_batch(self, documents: List[EmbeddedDocument]):
        """Writes a batch of documents to Qdrant, skipping duplicates."""
        points = []
        for doc in documents:
            if not self.article_exists(doc.metadata["url"]):
                points.append(
                    PointStruct(
                        id=doc.doc_id, vector=doc.embeddings, payload=doc.metadata
                    )
                )
            else:
                logging.info(f"Duplicate article skipped: {doc.doc_id}")

        if points:  # Only upsert if there are new points to insert
            try:
                self._client.upsert(
                    collection_name=self._collection_name, points=points
                )
            except Exception as e:
                logging.error(f"Error during batch upsert: {e}")

    def write(self, document: EmbeddedDocument):
        """Writes a single document to Qdrant."""
        self.write_batch([document])
