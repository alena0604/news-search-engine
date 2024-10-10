from qdrant_client import QdrantClient

from backend.embeddings import TextEmbedder


class QdrantSearchClass:
    """Handles interaction with Qdrant, including vector search operations."""

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def query_index(self, query_text: str, embedder: TextEmbedder, top_k: int = 10):
        """
        Queries the Qdrant index for similar articles based on the query text embedding.

        Args:
            query_text (str): The search text input by the user.
            embedder (TextEmbedder): The text embedder to create query embeddings.
            top_k (int): Number of top results to return.

        Returns:
            list: A list of structured search results from Qdrant.
        """
        embeddings = embedder(query_text, to_list=True)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embeddings,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": result.score,
                "title": result.payload.get("title"),
                "image_url": result.payload.get("image_url"),
                "date": result.payload.get("published_at"),
                "url": result.payload.get("url"),
            }
            for result in search_result
        ]
