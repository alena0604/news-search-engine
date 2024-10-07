from datetime import datetime
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from dateutil import parser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator
from unstructured.staging.huggingface import chunk_by_attention_window

from backend.cleaners import clean_full, normalize_whitespace, remove_html_tags
from backend.embeddings import TextEmbedder

# Configure logging with timestamps and better structure
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use UTC timestamp for consistency across different environments
CURRENT_TIMESTAMP = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter()


class DocumentSource(BaseModel):
    """Represents the source of a document or article."""

    id: Optional[str] = None
    name: str


class CommonDocument(BaseModel):
    """A unified model representing articles from various news sources."""

    article_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "N/A"
    url: str = "N/A"
    published_at: str = CURRENT_TIMESTAMP
    source_name: str = "Unknown"
    image_url: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None

    @field_validator("title", "description", "content", mode="before")
    def clean_text_fields(cls, value: Optional[str]) -> str:
        """Clean text fields by removing unwanted characters and normalizing whitespace."""
        return clean_full(value) if value else "N/A"

    @field_validator("url", "image_url", mode="before")
    def clean_url_fields(cls, value: Optional[str]) -> str:
        """Clean URL fields by removing HTML tags and normalizing them."""
        if value:
            value = remove_html_tags(value)
            value = normalize_whitespace(value)
        return value or "N/A"

    @field_validator("published_at", mode="before")
    def clean_date_field(cls, value: str) -> str:
        """Ensure the date is correctly parsed and formatted, falling back to the current timestamp."""
        try:
            parsed_date = parser.parse(value)
            return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            logger.error(
                f"Date parsing failed for '{value}', using the current timestamp."
            )
            return CURRENT_TIMESTAMP

    @classmethod
    def from_json(cls, data: dict) -> "CommonDocument":
        """Create an instance of `CommonDocument` from a JSON payload."""
        return cls(**data)


class NewsDataIOModel(BaseModel):
    """Represents an article model from NewsDataIO."""

    article_id: str
    title: str
    link: str
    description: Optional[str]
    pubDate: str
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    source_icon: Optional[str] = None
    creator: Optional[List[str]] = None
    image_url: Optional[str] = None
    content: Optional[str] = None

    def to_common(self) -> CommonDocument:
        """Convert NewsDataIOModel to CommonDocument format."""
        return CommonDocument(
            article_id=self.article_id,
            title=self.title,
            description=self.description,
            url=self.link,
            published_at=self.pubDate,
            source_name=self.source_id or "Unknown",
            image_url=self.image_url,
            content=self.content,
        )


class NewsAPIModel(BaseModel):
    """Represents an article model from NewsAPI."""

    source: DocumentSource
    author: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: str
    urlToImage: Optional[str] = None
    publishedAt: str
    content: Optional[str] = None

    def to_common(self) -> CommonDocument:
        """Convert NewsAPIModel to CommonDocument format."""
        return CommonDocument(
            title=self.title,
            description=self.description,
            url=self.url,
            published_at=self.publishedAt,
            source_name=self.source.name,
            image_url=self.urlToImage,
            content=self.content,
        )


class RefinedDocument(BaseModel):
    """Represents a refined version of a common document for further processing."""

    doc_id: str
    full_text: str = ""
    metadata: Dict[str, Any] = {}

    @classmethod
    def from_common(cls, common: CommonDocument) -> "RefinedDocument":
        """Convert a `CommonDocument` into a `RefinedDocument`."""
        return cls(
            doc_id=common.article_id,
            full_text=".".join([common.title, common.description]),
            metadata={
                "title": common.title,
                "url": common.url,
                "published_at": common.published_at,
                "source_name": common.source_name,
                "image_url": common.image_url,
            },
        )


class ChunkedDocument(BaseModel):
    """Represents a chunk of a larger document, typically used for embeddings."""

    doc_id: str
    chunk_id: str
    full_raw_text: str
    text: str
    metadata: Dict[str, Union[str, Any]]

    @classmethod
    def from_refined(
        cls, refined_doc: RefinedDocument, embedding_model: TextEmbedder
    ) -> List["ChunkedDocument"]:
        """Chunk a refined document and return the corresponding `ChunkedDocument` list."""
        chunks = cls.chunk_text(refined_doc.full_text, embedding_model)

        return [
            cls(
                doc_id=refined_doc.doc_id,
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                full_raw_text=refined_doc.full_text,
                text=chunk,
                metadata=refined_doc.metadata,
            )
            for chunk in chunks
        ]

    @staticmethod
    def chunk_text(text: str, embedding_model: TextEmbedder) -> List[str]:
        """Split text into smaller sections and further chunk based on the attention window."""
        text_sections = RECURSIVE_SPLITTER.split_text(text=text)
        chunks = []
        for section in text_sections:
            chunks.extend(chunk_by_attention_window(section, embedding_model.tokenizer))
        return chunks


class EmbeddedDocument(BaseModel):
    """Represents a chunk of a document with its corresponding embeddings."""

    doc_id: str
    chunk_id: str
    full_raw_text: str
    text: str
    embeddings: List[float]
    metadata: Dict[str, Union[str, Any]] = {}

    @classmethod
    def from_chunked(
        cls, chunked_doc: ChunkedDocument, embedding_model: TextEmbedder
    ) -> "EmbeddedDocument":
        """Create an embedded document from a chunked document using a text embedding model."""
        return cls(
            doc_id=chunked_doc.doc_id,
            chunk_id=chunked_doc.chunk_id,
            full_raw_text=chunked_doc.full_raw_text,
            text=chunked_doc.text,
            embeddings=embedding_model(chunked_doc.text, to_list=True),
            metadata=chunked_doc.metadata,
        )

    def to_payload(self) -> tuple[str, List[float], Dict[str, Any]]:
        """Prepare the embedded document for further processing or transmission."""
        return self.chunk_id, self.embeddings, self.metadata

    def __repr__(self) -> str:
        return f"EmbeddedDocument(doc_id={self.doc_id}, chunk_id={self.chunk_id})"
