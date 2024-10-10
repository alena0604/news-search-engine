import streamlit as st

from backend.cleaners import clean_full
from backend.embeddings import TextEmbedder
from backend.settings import AppConfig
from frontend.qdrant_search import QdrantSearchClass
from frontend.ui import ArticleRenderer


class NewsSearchApp:
    """The class to control the Qdrant News Search Streamlit App."""

    def __init__(self, qdrant_search: QdrantSearchClass, config: AppConfig):
        self.qdrant_search = qdrant_search
        self.config = config

    def run(self):
        """Runs the main application."""
        st.title("Qdrant Real-Time News Search")
        question = st.text_input(
            "Explore the latest news", key="question", on_change=self._on_search
        )

    def _on_search(self):
        """Event handler for the search input."""
        question = st.session_state.get("question")
        if question:
            clean_question = clean_full(question)
            embedder = TextEmbedder(self.config)
            articles = self.qdrant_search.query_index(clean_question, embedder)
            ArticleRenderer.display_articles(articles)
