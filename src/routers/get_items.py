from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.routers.dependencies import get_db_manager
from src.utils.database import DatabaseManager
from src.utils.models import MemoryItem

router = APIRouter(prefix="/api/v1/items", tags=["items"])


@router.get("/{item_id}", response_model=MemoryItem)
async def get_memory_item(
    item_id: UUID,
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> MemoryItem:
    """Get a specific MemoryItem by ID"""
    logger.info(f"Fetching MemoryItem with ID: {item_id}")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        item = await db_manager.get_memory_item(item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Memory item not found"
            )
        logger.info(f"MemoryItem found: {item}")
        return item

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory item: {str(e)}",
        ) from e


@router.get("/{item_id}/related")
async def get_related_items(
    item_id: UUID,
    relationship_types: Optional[str] = None,  # Comma-separated
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get items related to the specified MemoryItem"""
    logger.info(
        f"Fetching related items for ID: {item_id}, relationship_types: {relationship_types}"
    )
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

        logger.info(f"Related items found: {len(results)} for item {item_id}")
        return {"item_id": item_id, "related_items": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get related items: {str(e)}",
        ) from e
