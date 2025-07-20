"""
Data models for the Memory System based on the schema in general.md
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from tabulate import tabulate


class MemoryItemRaw(BaseModel):
    """Model for creating a new MemoryItem via the API"""

    content_type: str = Field(..., description="Type of item, e.g., 'message', 'image', 'web_link'")
    text_content: Optional[str] = Field(None, description="Raw text content")
    data_uri: Optional[str] = Field(None, description="Pointer to binary data like S3 URL")
    event_timestamp: datetime = Field(..., description="Real-world timestamp of the event")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the origin")
    reply_to_id: Optional[UUID] = Field(None, description="ID of the item this is replying to")


class MemoryItem(BaseModel):
    """Full MemoryItem model as stored in database"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    content_type: str
    text_content: Optional[str] = None
    analyzed_text: Optional[str] = None
    data_uri: Optional[str] = None
    embedding: Optional[List[float]] = None
    embedding_model_version: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    event_timestamp: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        headers = ["Field", "Value"]
        table = [
            ["id", str(self.id)],
            ["parent_id", str(self.parent_id) if self.parent_id else "None"],
            ["content_type", self.content_type],
            ["text_content", self.text_content[:60] if self.text_content else "None"],
            ["analyzed_text", self.analyzed_text[:60] if self.analyzed_text else "None"],
            ["data_uri", self.data_uri[:60] if self.data_uri else "None"],
            ["embedding_model_version", self.embedding_model_version or "None"],
            ["meta", str(self.meta)[:60] if self.meta else "None"],
            ["event_timestamp", self.event_timestamp.isoformat()],
            ["created_at", self.created_at.isoformat()],
            ["updated_at", self.updated_at.isoformat()],
        ]
        return "\n" + tabulate(table, headers=headers, tablefmt="rounded_outline")


class Relationship(BaseModel):
    """Model for relationships between MemoryItems"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    source_node_id: UUID
    target_node_id: UUID
    relationship_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryItemResponse(BaseModel):
    """Response model for retrieved MemoryItems"""

    item: MemoryItem
    score: Optional[float] = None


class RetrievalResponse(BaseModel):
    """Response model for retrieval API"""

    query: str
    results: List[MemoryItemResponse]


class IngestionResponse(BaseModel):
    """Response model for ingestion API"""

    status: str
    item_id: UUID


class RetrievalRequest(BaseModel):
    """Model for retrieval query parameters"""

    query: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    content_types: Optional[List[str]] = None
    include_context: bool = False
