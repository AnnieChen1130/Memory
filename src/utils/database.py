import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import asyncpg
import numpy as np
from loguru import logger
from pgvector.asyncpg import register_vector
from tabulate import tabulate

from src.utils.models import MemoryItem, MemoryItemRaw, Relationship


class DatabaseManager:
    """Manages database connections and operations for the Memory System"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool
        logger.debug(f"DatabaseManager initialized with URL: {database_url}")

    async def __aenter__(self):
        async def init_conn(conn: asyncpg.Connection):
            await register_vector(conn)

        logger.debug("pgvector extension registered for connection.")

        logger.info("Creating asyncpg connection pool...")
        self.pool = await asyncpg.create_pool(
            self.database_url, min_size=5, max_size=20, command_timeout=60, init=init_conn
        )
        logger.info("Database connection pool created.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed.")

    @logger.catch()
    async def create_memory_item(
        self,
        item_data: MemoryItemRaw,
        analyzed_text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_model_version: Optional[str] = None,
        parent_id: Optional[UUID] = None,
    ) -> MemoryItem:
        logger.debug(
            f"Creating MemoryItem: {item_data}, analyzed_text={analyzed_text}, parent_id={parent_id}"
        )
        async with self.pool.acquire() as conn:
            embedding_array: Optional[np.ndarray] = None
            if embedding is not None:
                if isinstance(embedding, np.ndarray) and embedding.dtype == np.float16:
                    embedding_array = embedding
                elif isinstance(embedding, np.ndarray):
                    embedding_array = embedding.astype(np.float16)
                else:
                    embedding_array = np.array(embedding, dtype=np.float16)

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

            item = self._row_to_memory_item(row)
            logger.info(f"MemoryItem created with ID: {item.id}")

            # TODO: Create relationship
            # if item_data.reply_to_id:
            #     await self.create_relationship(
            #         source_node_id=item.id,
            #         target_node_id=item_data.reply_to_id,
            #         relationship_type="reply_to",
            #     )

            return item

    @logger.catch()
    async def create_relationship(
        self, source_node_id: UUID, target_node_id: UUID, relationship_type: str
    ) -> Relationship:
        logger.debug(
            f"Creating Relationship: source={source_node_id}, target={target_node_id}, type={relationship_type}"
        )
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

            logger.info(f"Relationship created: {row['id']}")
            return self._row_to_relationship(row)

    @logger.catch()
    async def get_memory_item(self, item_id: UUID) -> Optional[MemoryItem]:
        logger.debug(f"Fetching MemoryItem with ID: {item_id}")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM MemoryItems WHERE id = $1
                """,
                item_id,
            )

            logger.info(f"Fetched MemoryItem: {item_id} found={row is not None}")
            return self._row_to_memory_item(row) if row else None

    @logger.catch()
    async def update_memory_item(
        self,
        item_id: UUID,
        analyzed_text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_model_version: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        headers = ["Field", "Value"]
        params = [
            ["item_id", str(item_id)],
            ["analyzed_text", analyzed_text[:60] if analyzed_text else "None"],
            ["embed_model", embedding_model_version or "None"],
        ]
        logger.debug(
            "Update params:\n" + tabulate(params, headers=headers, tablefmt="rounded_outline")
        )

        async with self.pool.acquire() as conn:
            # Embedding
            embedding_array: Optional[np.ndarray] = None
            if embedding is not None:
                if isinstance(embedding, np.ndarray) and embedding.dtype == np.float16:
                    embedding_array = embedding
                elif isinstance(embedding, np.ndarray):
                    embedding_array = embedding.astype(np.float16)
                else:
                    embedding_array = np.array(embedding, dtype=np.float16)

            # Prepare update fields
            update_fields = ["updated_at = NOW()"]
            params = []
            param_idx = 1

            if analyzed_text is not None:
                update_fields.append(f"analyzed_text = ${param_idx}")
                params.append(analyzed_text)
                param_idx += 1

            if embedding_array is not None:
                update_fields.append(f"embedding = ${param_idx}")
                params.append(embedding_array)
                param_idx += 1

            if embedding_model_version is not None:
                update_fields.append(f"embedding_model_version = ${param_idx}")
                params.append(embedding_model_version)
                param_idx += 1

            # Nothing to update
            if len(params) == 0:
                return await self.get_memory_item(item_id)

            # Add item_id as the last parameter
            params.append(item_id)

            query = f"""
                UPDATE MemoryItems 
                SET {", ".join(update_fields)}
                WHERE id = ${param_idx}
                RETURNING *
            """

            row = await conn.fetchrow(query, *params)
            logger.info(f"Updated MemoryItem: {item_id} found={row is not None}")
            return self._row_to_memory_item(row) if row else None

    async def search_memory_items(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        content_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        logger.debug(
            f"Searching MemoryItems: top_k={top_k}, content_types={content_types}, start_date={start_date}, end_date={end_date}, filters={filters}"
        )
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

            logger.info(f"Search returned {len(results)} results.")
            return results

    async def get_related_items(
        self, item_id: UUID, relationship_types: Optional[List[str]] = None
    ) -> List[Tuple[MemoryItem, str]]:
        logger.debug(
            f"Getting related items for: {item_id}, relationship_types={relationship_types}"
        )
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

            logger.info(f"Related items found: {len(results)} for item {item_id}")
            return results

    async def get_chunk_siblings(self, chunk_id: UUID) -> List[MemoryItem]:
        logger.debug(f"Getting chunk siblings for: {chunk_id}")
        async with self.pool.acquire() as conn:
            chunk = await self.get_memory_item(chunk_id)
            if not chunk or not chunk.parent_id:
                return []

            rows = await conn.fetch(
                """
                SELECT * FROM MemoryItems 
                WHERE parent_id = $1 AND content_type = 'article_chunk'
                ORDER BY (meta->>'chunk_index')::int
                """,
                chunk.parent_id,
            )

            logger.info(
                f"Chunk siblings found: {len(rows)} for parent_id={chunk.parent_id if chunk else None}"
            )
            return [self._row_to_memory_item(row) for row in rows]

    async def aggregate_chunk_results(
        self, search_results: List[Tuple[MemoryItem, float]]
    ) -> List[Tuple[MemoryItem, float]]:
        logger.debug(f"Aggregating chunk results, input count: {len(search_results)}")
        parent_groups: defaultdict[UUID, List[Tuple[MemoryItem, float]]] = defaultdict(list)
        standalone_items: List[Tuple[MemoryItem, float]] = []

        # Group chunks by parent article
        for item, score in search_results:
            if item.content_type == "text_chunk" and item.parent_id:
                parent_groups[item.parent_id].append((item, score))
            else:
                standalone_items.append((item, score))

        # For each parent group, keep only the best-scoring chunk(s)
        aggregated_results = standalone_items.copy()

        for _parent_id, chunk_results in parent_groups.items():
            chunk_results.sort(key=lambda x: x[1], reverse=True)
            top_chunks = chunk_results[:3]
            aggregated_results.extend(top_chunks)

        aggregated_results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Aggregated results count: {len(aggregated_results)}")
        return aggregated_results

    def _row_to_memory_item(self, row) -> MemoryItem:
        embedding: Optional[List[float]] = None
        if row["embedding"] is not None:
            embedding = row["embedding"].to_list()
        meta = json.loads(row["meta"]) if row["meta"] else None

        logger.debug(f"Converted DB row to MemoryItem: {row['id']}")
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
        logger.debug(f"Converted DB row to Relationship: {row['id']}")
        return Relationship(
            id=row["id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            relationship_type=row["relationship_type"],
            created_at=row["created_at"],
        )
