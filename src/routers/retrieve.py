from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.routers.dependencies import (
    get_db_manager,
    get_embedding_service,
    get_rerank_service,
)
from src.services.embedding import EmbeddingService
from src.services.rerank import RerankingService
from src.utils.database import DatabaseManager
from src.utils.models import MemoryItemResponse, RetrievalResponse

router = APIRouter(prefix="/api/v1", tags=["retrieve"])


@router.get("/retrieve", response_model=RetrievalResponse)
async def retrieve_memory_items(
    query: str,
    top_k: int = 10,
    filters: Optional[str] = None,  # JSON string
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    content_types: Optional[str] = None,  # Comma-separated
    include_context: bool = False,
    enable_reranking: bool = True,
    db_manager: DatabaseManager = Depends(get_db_manager),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    reranking_service: RerankingService = Depends(get_rerank_service),
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

        # Initial vector search (get more candidates for reranking)
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
            search_results = reranking_service.rerank(query, search_results, top_k)

        # Convert to response format
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
