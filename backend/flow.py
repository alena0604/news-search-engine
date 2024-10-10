import logging as logger
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bytewax.operators as op
from bytewax.dataflow import Dataflow
from bytewax.inputs import FixedPartitionedSource, StatefulSourcePartition
from bytewax.outputs import DynamicSink

from backend.embeddings import TextEmbedder
from backend.models import ChunkedDocument, EmbeddedDocument, RefinedDocument
from backend.news_loader import ArticleFetcher
from backend.qdrant import QdrantVectorOutput
from backend.settings import AppConfig


class NewsStreamSource(StatefulSourcePartition):
    """Custom input source for Bytewax that yields articles from multiple APIs."""

    def __init__(self, fetch_func, resume_state=None):
        self.fetch_func = fetch_func
        self.last_fetched_timestamp = resume_state or datetime.now(tz=timezone.utc)

    def next_batch(self):
        """Get the next batch of articles from the fetcher."""
        articles = self.fetch_func()
        if not articles:
            logger.info("No articles fetched from this source.")
            return []  # No more articles, return empty list
        return articles

    def next_awake(self):
        """Set the next awake time (polling interval) for fetching more articles."""
        return datetime.now(tz=timezone.utc) + timedelta(seconds=5)

    def snapshot(self):
        """Save the state of the fetcher to resume later."""
        return self.last_fetched_timestamp


class NewsStreamInput(FixedPartitionedSource):
    """Input class to partition and stream news data from multiple sources."""

    def __init__(self, fetcher):
        """Initialize with the ArticleFetcher instance."""
        self.fetcher = fetcher

    def list_parts(self):
        """List all fetch functions (e.g., partitions) for each news source."""
        return [fetch.__name__ for fetch in self.fetcher.sources]

    def build_part(self, step_id, for_part, resume_state):
        """Build the partition for the specific fetch function."""
        fetch_func = getattr(self.fetcher, for_part)
        return NewsStreamSource(fetch_func, resume_state)


def build(model_cache_dir: Optional[Path] = None):
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../conf/config.yaml")
    )
    config = AppConfig(config_path=config_path)

    fetcher = ArticleFetcher(config=config)
    model = TextEmbedder(cache_dir=model_cache_dir, config=config)

    flow = Dataflow("new_stream")

    # Use the custom FetcherInput to fetch articles
    inp = op.input("input", flow, NewsStreamInput(fetcher))
    # op.inspect("dbg_input", inp)

    stream = op.map("refine", inp, RefinedDocument.from_common)
    # op.inspect("dbg_refine", stream)

    stream = op.flat_map(
        "chunkenize",
        stream,
        lambda refined_doc: ChunkedDocument.from_refined(refined_doc, model),
    )
    # op.inspect("dbg_chunkenize", stream)
    stream = op.map(
        "embed",
        stream,
        lambda chunked_doc: EmbeddedDocument.from_chunked(chunked_doc, model),
    )
    # op.inspect("dbg_embed", stream)
    op.output("output", stream, _build_output(model, config=config))
    return flow


def _build_output(model: TextEmbedder, config) -> DynamicSink:
    return QdrantVectorOutput(vector_size=model.max_input_length, config=config)
