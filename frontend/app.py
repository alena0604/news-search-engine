import os

from qdrant_client import QdrantClient

from backend.settings import AppConfig
from frontend.news_search import NewsSearchApp
from frontend.qdrant_search import QdrantSearchClass


def main():
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../conf/config.yaml")
    )
    config = AppConfig(config_path=config_path)

    # Set up the Qdrant client and app logic
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL", config.get("QDRANT_URL")),
        api_key=os.getenv("QDRANT_API_KEY", config.get("QDRANT_API_KEY")),
    )
    qdrant_search = QdrantSearchClass(
        client=qdrant_client,
        collection_name=config.get("VECTOR_DB_OUTPUT_COLLECTION_NAME"),
    )
    # Run the app
    news_search_app = NewsSearchApp(qdrant_search, config)
    news_search_app.run()


if __name__ == "__main__":
    main()
