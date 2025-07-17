"""
Data models for the Memory System based on the schema in general.md
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


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
