from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.routers.dependencies import (
    get_db_manager,
    get_embedding_service,
    get_task_manager,
)
from src.services.embedding import EmbeddingService
from src.uri_handler.task_queue import BackgroundTaskManager
from src.utils.database import DatabaseManager
from src.utils.models import IngestionResponse, MemoryItemRaw

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_memory_item(
    item_data: MemoryItemRaw,
    db_manager: DatabaseManager = Depends(get_db_manager),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    task_manager: BackgroundTaskManager = Depends(get_task_manager),
) -> IngestionResponse:
    """
    Generates embeddings and stores the item in the database.
    For complex data types, triggers asynchronous processing.
    """
    logger.debug(
        "\n"
        "┌─────────────────┬────────────────────────────────────────────────────────────────┐\n"
        f"│ content_type    │ {item_data.content_type!r:<60} │\n"
        f"│ text_content    │ {item_data.text_content!r:<60} │\n"
        f"│ data_uri        │ {item_data.data_uri!r:<60} │\n"
        f"│ event_timestamp │ {item_data.event_timestamp.isoformat():<60} │\n"
        f"│ meta            │ {str(item_data.meta)!r:<60} │\n"
        f"│ reply_to_id     │ {str(item_data.reply_to_id)!r:<60} │\n"
        "└─────────────────┴────────────────────────────────────────────────────────────────┘"
    )

    try:
        analyzed_text = item_data.text_content

        # Generate embedding
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

        # Determine if this should trigger async processing
        should_process_async = False
        async_content_types = ["web_link", "image", "audio", "video", "long_text"]

        # If contains paragraphs or more than 20 sentences, mark as long_text
        if item_data.text_content:
            num_paragraphs = item_data.text_content.count("\n\n")
            num_sentences = sum(item_data.text_content.count(p) for p in ".!?。！？……")
            if num_paragraphs > 0 or num_sentences > 20:
                item_data = item_data.model_copy(update={"content_type": "long_text"})

        if item_data.content_type in async_content_types:
            if item_data.data_uri:
                should_process_async = True
            if item_data.content_type == "long_text":
                should_process_async = True

        # Trigger async processing
        if should_process_async and task_manager:
            from src.uri_handler.task_queue import (
                create_media_analysis_task,
                create_web_scraping_task,
            )

            assert item_data.data_uri, "Data URI must be provided for async processing"
            if item_data.content_type == "web_link":
                task = create_web_scraping_task(memory_item, item_data.data_uri)
                await task_manager.enqueue_task(task)
            elif item_data.content_type in ["image", "video", "audio"]:
                task = create_media_analysis_task(
                    memory_item, item_data.data_uri, item_data.content_type
                )
                await task_manager.enqueue_task(task)
            logger.info(
                f"Triggered async processing for item: {memory_item.id}, type: {item_data.content_type}"
            )

        return IngestionResponse(status="ingested", item_id=memory_item.id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest memory item: {str(e)}",
        ) from e
