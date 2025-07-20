from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.routers.dependencies import get_task_manager
from src.uri_handler.task_queue import BackgroundTaskManager

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.get("/{task_id}")
async def get_task_status(
    task_id: UUID,
    task_manager: BackgroundTaskManager = Depends(get_task_manager),
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
            "source_item_id": task.source_item,
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
