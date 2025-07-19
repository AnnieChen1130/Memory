from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from uuid import UUID

import torch
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.services.embedding import EmbeddingService
from src.services.rerank import RerankingService
from src.uri_handler.task_queue import BackgroundTaskManager
from src.utils.config import settings
from src.utils.database import DatabaseManager
from src.utils.models import (
    IngestionResponse,
    MemoryItem,
    MemoryItemRaw,
    MemoryItemResponse,
    RetrievalResponse,
)

torch.set_float32_matmul_precision("high")

# Global instances - will be initialized during startup
db_manager: Optional[DatabaseManager] = None
embedding_service: Optional[EmbeddingService] = None
reranking_service: Optional[RerankingService] = None
task_manager: Optional[BackgroundTaskManager] = None

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events"""
    global db_manager, embedding_service, reranking_service, task_manager

    try:
        # Database
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.initialize()

        # Embedding service
        embedding_service = EmbeddingService(settings.embedding_model)
        await embedding_service.initialize()

        # Reranking service
        reranking_service = RerankingService()
        await reranking_service.initialize()

        # TODO: use redis or RabbitMQ in the future.
        # Background task manager
        task_manager = BackgroundTaskManager(db_manager, embedding_service)
        await task_manager.start_workers()

        yield
    finally:
        if task_manager:
            await task_manager.stop_workers()
        if db_manager:
            await db_manager.close()


# Global instances
app = FastAPI(
    title="Memory System API",
    description="A backend service for ingesting, storing, and retrieving event logs",
    version="0.1.0",
    lifespan=lifespan,
)

security_dependency = Security(security)


def get_api_key_verification(
    credentials: Optional[HTTPAuthorizationCredentials] = security_dependency,
) -> Optional[HTTPAuthorizationCredentials]:
    """Verify API key if configured"""
    if settings.api_key:
        if not credentials or credentials.credentials != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials


AuthDep = Depends(get_api_key_verification)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.post("/api/v1/ingest", response_model=IngestionResponse)
async def ingest_memory_item(
    item_data: MemoryItemRaw,
    credentials: Optional[HTTPAuthorizationCredentials] = AuthDep,
):
    """
    Generates embeddings and stores the item in the database.
    For complex data types, triggers asynchronous processing.
    """
    if not db_manager or not embedding_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # Determine if this should trigger async processing
        should_process_async = False
        async_content_types = ["web_link", "image", "audio_clip"]

        if item_data.content_type in async_content_types and item_data.data_uri:
            should_process_async = True

        analyzed_text = item_data.text_content or ""
        if item_data.data_uri and not should_process_async:
            # Only parse URIs synchronously for simple types
            analyzed_text += parse(item_data.data_uri)

        # Generate embedding if we have text to analyze
        embedding = None
        embedding_model_version = None
        if analyzed_text:
            embedding = embedding_service.encode(analyzed_text)
            embedding_model_version = embedding_service.get_model_version()

        # Create the memory item in database
        memory_item = await db_manager.create_memory_item(
            item_data=item_data,
            analyzed_text=analyzed_text,
            embedding=embedding,
            embedding_model_version=embedding_model_version,
        )

        # Trigger async processing if needed
        if should_process_async and task_manager and item_data.data_uri:
            from src.uri_handler.task_queue import (
                create_audio_transcription_task,
                create_image_analysis_task,
                create_web_scraping_task,
            )

            if item_data.content_type == "web_link":
                task = create_web_scraping_task(memory_item.id, item_data.data_uri)
                await task_manager.enqueue_task(task)
            elif item_data.content_type == "image":
                task = create_image_analysis_task(memory_item.id, item_data.data_uri)
                await task_manager.enqueue_task(task)
            elif item_data.content_type == "audio_clip":
                task = create_audio_transcription_task(memory_item.id, item_data.data_uri)
                await task_manager.enqueue_task(task)

        return IngestionResponse(status="ingested", item_id=memory_item.id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest memory item: {str(e)}",
        ) from e


@app.get("/api/v1/retrieve", response_model=RetrievalResponse)
async def retrieve_memory_items(
    query: str,
    top_k: int = 10,
    filters: Optional[str] = None,  # JSON string
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    content_types: Optional[str] = None,  # Comma-separated
    include_context: bool = False,
    enable_reranking: bool = True,  # New parameter for reranking
    credentials: Optional[HTTPAuthorizationCredentials] = AuthDep,
):
    """
    Retrieve and rank MemoryItems based on a query.

    Performs semantic search using vector similarity and optionally
    includes related items through the relationship graph.
    """
    if not db_manager or not embedding_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        query_embedding = embedding_service.encode(query, is_query=True)

        # Parse filters
        filters_dict = None
        if filters:
            import json

            try:
                filters_dict = json.loads(filters)
            except json.JSONDecodeError as json_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid filters JSON format",
                ) from json_err

        # Parse content_types
        content_types_list = None
        if content_types:
            content_types_list = [ct.strip() for ct in content_types.split(",")]

        # Initial vector search - get more candidates for reranking
        initial_k = min(top_k * 3, 100) if enable_reranking and reranking_service else top_k

        search_results = await db_manager.search_memory_items(
            query_embedding=query_embedding,
            top_k=initial_k,
            content_types=content_types_list,
            start_date=start_date,
            end_date=end_date,
            filters=filters_dict,
        )

        # Aggregate chunks to avoid redundant results from the same article
        search_results = await db_manager.aggregate_chunk_results(search_results)

        # Apply reranking if enabled and service available
        if enable_reranking and reranking_service and search_results:
            search_results = reranking_service.rerank(
                query, search_results, top_k
            )  # Convert to response format
        results = []
        for item, score in search_results:
            results.append(MemoryItemResponse(item=item, score=score))

        # Implement context retrieval if include_context is True
        if include_context:
            # Fetch related items for each result
            enriched_results = []
            for result in results:
                related_items = await db_manager.get_related_items(result.item.id)

                # Add related items as context
                context = []
                for related_item, relationship_info in related_items:
                    context.append({"item": related_item, "relationship": relationship_info})

                # Create enriched response with context
                enriched_result = {"item": result.item, "score": result.score, "context": context}
                enriched_results.append(enriched_result)

            return {"query": query, "results": enriched_results}

        return RetrievalResponse(query=query, results=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory items: {str(e)}",
        ) from e


@app.get("/api/v1/items/{item_id}", response_model=MemoryItem)
async def get_memory_item(
    item_id: UUID, credentials: Optional[HTTPAuthorizationCredentials] = AuthDep
):
    """Get a specific MemoryItem by ID"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        item = await db_manager.get_memory_item(item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Memory item not found"
            )
        return item

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory item: {str(e)}",
        ) from e


@app.get("/api/v1/items/{item_id}/related")
async def get_related_items(
    item_id: UUID,
    relationship_types: Optional[str] = None,  # Comma-separated
    credentials: Optional[HTTPAuthorizationCredentials] = AuthDep,
):
    """Get items related to the specified MemoryItem"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        # Parse relationship types
        rel_types_list = None
        if relationship_types:
            rel_types_list = [rt.strip() for rt in relationship_types.split(",")]

        related_items = await db_manager.get_related_items(
            item_id=item_id, relationship_types=rel_types_list
        )

        # Format the response
        results = []
        for item, relationship_info in related_items:
            results.append({"item": item, "relationship": relationship_info})

        return {"item_id": item_id, "related_items": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get related items: {str(e)}",
        ) from e


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(
    task_id: UUID,
    credentials: Optional[HTTPAuthorizationCredentials] = AuthDep,
):
    """Get the status of a background processing task"""
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    try:
        task = await task_manager.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": task.status.value,
            "source_item_id": task.source_item_id,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error_message": task.error_message,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
    )
