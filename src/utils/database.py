import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
import numpy as np
from loguru import logger
from pgvector.asyncpg import register_vector

from src.utils.models import MemoryItem, MemoryItemRaw, Relationship


class DatabaseManager:
    """Manages database connections and operations for the Memory System"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url, min_size=5, max_size=20, command_timeout=60
        )

        # Register pgvector types
        async with self.pool.acquire() as conn:
            await register_vector(conn)

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()

    @logger.catch()
    async def create_memory_item(
        self,
        item_data: MemoryItemRaw,
        analyzed_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        embedding_model_version: Optional[str] = None,
        parent_id: Optional[UUID] = None,
    ) -> MemoryItem:
        """Create a new MemoryItem in the database"""
        async with self.pool.acquire() as conn:
            # Convert embedding to numpy array if provided
            embedding_array = np.array(embedding, dtype=np.float16) if embedding else None

            row = await conn.fetchrow(
                """
                INSERT INTO MemoryItems (
                    parent_id, content_type, text_content, analyzed_text, data_uri, 
                    embedding, embedding_model_version, meta, event_timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING *
            """,
                parent_id,
                item_data.content_type,
                item_data.text_content,
                analyzed_text or item_data.text_content,
                item_data.data_uri,
                embedding_array,
                embedding_model_version,
                json.dumps(item_data.meta) if item_data.meta else None,
                item_data.event_timestamp,
            )

            # Convert the row to a MemoryItem
            item = self._row_to_memory_item(row)

            # Create relationship if this is a reply
            if item_data.reply_to_id:
                await self.create_relationship(
                    source_node_id=item.id,
                    target_node_id=item_data.reply_to_id,
                    relationship_type="reply_to",
                )

            return item

    async def create_relationship(
        self, source_node_id: UUID, target_node_id: UUID, relationship_type: str
    ) -> Relationship:
        """Create a new relationship between MemoryItems"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO Relationships (source_node_id, target_node_id, relationship_type)
                VALUES ($1, $2, $3)
                RETURNING *
            """,
                source_node_id,
                target_node_id,
                relationship_type,
            )

            return self._row_to_relationship(row)

    async def get_memory_item(self, item_id: UUID) -> Optional[MemoryItem]:
        """Get a MemoryItem by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM MemoryItems WHERE id = $1
            """,
                item_id,
            )

            return self._row_to_memory_item(row) if row else None

    async def search_memory_items(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        content_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        """Search MemoryItems using vector similarity"""
        async with self.pool.acquire() as conn:
            # Query conditions
            conditions = ["embedding IS NOT NULL"]
            params: List[Any] = [np.array(query_embedding, dtype=np.float16)]
            param_idx = 2

            if content_types:
                conditions.append(f"content_type = ANY(${param_idx})")
                params.append(content_types)
                param_idx += 1

            if start_date:
                conditions.append(f"event_timestamp >= ${param_idx}")
                params.append(start_date)
                param_idx += 1

            if end_date:
                conditions.append(f"event_timestamp <= ${param_idx}")
                params.append(end_date)
                param_idx += 1

            # Add support for JSON filters on meta field
            if filters:
                for key, value in filters.items():
                    conditions.append(f"meta->>$${param_idx} = $${param_idx + 1}")
                    params.extend([key, str(value)])
                    param_idx += 2

            where_clause = " AND ".join(conditions)

            query = f"""
                SELECT *, (embedding <=> $1) as distance
                FROM MemoryItems
                WHERE {where_clause}
                ORDER BY embedding <=> $1
                LIMIT {top_k}
            """

            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                item = self._row_to_memory_item(row)
                score = 1.0 / (1.0 + row["distance"])
                results.append((item, score))

            return results

    async def get_related_items(
        self, item_id: UUID, relationship_types: Optional[List[str]] = None
    ) -> List[Tuple[MemoryItem, str]]:
        """Get items related to the given item"""
        async with self.pool.acquire() as conn:
            conditions = ["(r.source_node_id = $1 OR r.target_node_id = $1)"]
            params: List[Any] = [item_id]

            if relationship_types:
                conditions.append("r.relationship_type = ANY($2)")
                params.append(relationship_types)

            where_clause = " AND ".join(conditions)

            query = f"""
                SELECT m.*, r.relationship_type,
                       CASE WHEN r.source_node_id = $1 THEN 'outgoing' ELSE 'incoming' END as direction
                FROM MemoryItems m
                JOIN Relationships r ON (
                    (r.source_node_id = m.id AND r.target_node_id = $1) OR
                    (r.target_node_id = m.id AND r.source_node_id = $1)
                )
                WHERE {where_clause}
            """

            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                item = self._row_to_memory_item(row)
                relationship_info = f"{row['direction']}:{row['relationship_type']}"
                results.append((item, relationship_info))

            return results

    async def get_chunk_siblings(self, chunk_id: UUID) -> List[MemoryItem]:
        """Get all chunks that belong to the same parent article"""
        async with self.pool.acquire() as conn:
            # First get the parent_id of this chunk
            chunk = await self.get_memory_item(chunk_id)
            if not chunk or not chunk.parent_id:
                return []

            # Get all chunks with the same parent
            rows = await conn.fetch(
                """
                SELECT * FROM MemoryItems 
                WHERE parent_id = $1 AND content_type = 'article_chunk'
                ORDER BY (meta->>'chunk_index')::int
                """,
                chunk.parent_id,
            )

            return [self._row_to_memory_item(row) for row in rows]

    async def aggregate_chunk_results(
        self, search_results: List[Tuple[MemoryItem, float]]
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Aggregate chunk results by parent article to avoid redundant results
        Returns the best-scoring chunk per parent article
        """
        parent_groups: Dict[Optional[UUID], List[Tuple[MemoryItem, float]]] = {}
        standalone_items: List[Tuple[MemoryItem, float]] = []

        # Group chunks by parent article
        for item, score in search_results:
            if item.content_type == "article_chunk" and item.parent_id:
                if item.parent_id not in parent_groups:
                    parent_groups[item.parent_id] = []
                parent_groups[item.parent_id].append((item, score))
            else:
                standalone_items.append((item, score))

        # For each parent group, keep only the best-scoring chunk
        aggregated_results = standalone_items.copy()

        for _parent_id, chunk_results in parent_groups.items():
            # Sort by score and take the best one
            chunk_results.sort(key=lambda x: x[1], reverse=True)
            best_chunk, best_score = chunk_results[0]

            # Optionally enhance with context from surrounding chunks
            # For now, just add the best chunk
            aggregated_results.append((best_chunk, best_score))

        # Sort final results by score
        aggregated_results.sort(key=lambda x: x[1], reverse=True)
        return aggregated_results

    def _row_to_memory_item(self, row) -> MemoryItem:
        """Convert database row to MemoryItem model"""
        embedding = row["embedding"].tolist() if row["embedding"] is not None else None
        meta = json.loads(row["meta"]) if row["meta"] else None

        return MemoryItem(
            id=row["id"],
            parent_id=row["parent_id"],
            content_type=row["content_type"],
            text_content=row["text_content"],
            analyzed_text=row["analyzed_text"],
            data_uri=row["data_uri"],
            embedding=embedding,
            embedding_model_version=row["embedding_model_version"],
            meta=meta,
            event_timestamp=row["event_timestamp"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_relationship(self, row) -> Relationship:
        """Convert database row to Relationship model"""
        return Relationship(
            id=row["id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            relationship_type=row["relationship_type"],
            created_at=row["created_at"],
        )
