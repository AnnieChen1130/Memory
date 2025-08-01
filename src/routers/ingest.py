from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from tabulate import tabulate

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
    headers = ["Field", "Value"]
    params = [
        ["content_type", item_data.content_type],
        ["text_content", item_data.text_content[:60] if item_data.text_content else "None"],
        ["data_uri", item_data.data_uri[:60] if item_data.data_uri else "None"],
        ["event_timestamp", item_data.event_timestamp.isoformat()],
        ["meta", str(item_data.meta)[:60]],
        ["reply_to_id", str(item_data.reply_to_id)[:60]],
    ]
    logger.debug("Got raw data:\n" + tabulate(params, headers=headers, tablefmt="rounded_outline"))

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
        logger.info(f"MemoryItem ingested: {memory_item}")

        # Determine if this should trigger async processing
        # If contains paragraphs or more than 20 sentences, mark as long_text
        if item_data.text_content:
            num_paragraphs = item_data.text_content.count("\n\n")
            num_sentences = sum(1 for char in item_data.text_content if char in ".!?。！？……")
            if num_paragraphs > 0 or num_sentences > 20:
                item_data = item_data.model_copy(update={"content_type": "long_text"})

        # Trigger async processing
        if task_manager:
            from src.uri_handler.task_queue import (
                create_media_analysis_task,
                create_text_analysis_task,
                create_web_scraping_task,
            )

            task = None
            if item_data.content_type in ["web_link", "image", "video", "audio"]:
                assert item_data.data_uri, "Data URI must be provided for async processing"
                if item_data.content_type == "web_link":
                    task = create_web_scraping_task(memory_item, item_data.data_uri)
                elif item_data.content_type in ["image", "video", "audio"]:
                    task = create_media_analysis_task(
                        memory_item, item_data.data_uri, item_data.content_type
                    )
            elif item_data.content_type == "long_text":
                assert item_data.text_content, "Text content must be provided for text analysis"
                task = create_text_analysis_task(memory_item, item_data.text_content)
            else:
                logger.debug(
                    f"Content type {item_data.content_type} doesn't trigger async processing."
                )
            if task:
                await task_manager.enqueue_task(task)
                logger.info(
                    f"Enqueued task: {task.task_id} for item: {memory_item.id}, type: {item_data.content_type}"
                )

        return IngestionResponse(status="ingested", item_id=memory_item.id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest memory item: {str(e)}",
        ) from e
