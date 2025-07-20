"""
Background task processing for the Memory System

Handles asynchronous processing of complex data types like web links, images, and audio clips.
Uses a simple in-memory queue for now, but can be extended to use Redis, RabbitMQ, or other queue systems.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from loguru import logger

from src.services.embedding import EmbeddingService
from src.services.media_analysis import MediaAnalysisService
from src.utils.database import DatabaseManager
from src.utils.models import MemoryItem


class TaskType(Enum):
    WEB_SCRAPING = "web_scraping"
    IMAGE_ANALYSIS = "image_analysis"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    TEXT_ANALYSIS = "text_analysis"
    MEDIA_ANALYSIS = "media_analysis"


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    task_id: UUID
    task_type: TaskType
    source_item: MemoryItem  # The original MemoryItem that triggered this task
    data: Dict[str, Any]  # Task-specific data
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class BackgroundTaskManager:
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.media_analysis_service: Optional[MediaAnalysisService] = None
        self.task_queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.running_tasks: Dict[UUID, BackgroundTask] = {}
        self.worker_count = 3  # Number of worker coroutines
        self.workers: List[asyncio.Task] = []

    async def __aenter__(self):
        logger.info(f"Starting {self.worker_count} background workers")

        try:
            self.media_analysis_service = MediaAnalysisService()
        except Exception as e:
            logger.error(f"Failed to create media analysis service: {e}")

        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker_loop(f"Worker-{i}"))
            self.workers.append(worker)

        logger.debug("Background workers started.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("Stopping background workers")
        for worker in self.workers:
            worker.cancel()

        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.debug("Background workers stopped.")

    async def enqueue_task(self, task: BackgroundTask) -> UUID:
        await self.task_queue.put(task)
        logger.info(f"Enqueued task {task.task_id} of type {task.task_type}")
        return task.task_id

    async def get_task_status(self, task_id: UUID) -> Optional[BackgroundTask]:
        logger.debug(f"Getting status for task {task_id}")
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]

        # TODO: Check completed tasks in database/persistent storage
        return None

    async def _worker_loop(self, worker_name: str):
        logger.info(f"{worker_name} started")

        while True:
            try:
                task = await self.task_queue.get()
                logger.debug(f"{worker_name} picked up task {task.task_id} ({task.task_type})")

                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now(timezone.utc)
                self.running_tasks[task.task_id] = task

                logger.info(
                    f"{worker_name} processing task {task.task_id} of type {task.task_type}"
                )

                # Process the task
                try:
                    await self._process_task(task)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now(timezone.utc)
                    logger.info(f"Task {task.task_id} completed successfully")

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.now(timezone.utc)
                    logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

                finally:
                    self.task_queue.task_done()
                    del self.running_tasks[task.task_id]

            except asyncio.CancelledError:
                logger.info(f"{worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"{worker_name} error: {e}", exc_info=True)

    async def _process_task(self, task: BackgroundTask):
        if task.task_type == TaskType.WEB_SCRAPING:
            await self._process_web_scraping(task)
        elif task.task_type == TaskType.MEDIA_ANALYSIS:
            await self._process_media_analysis(task)
        elif task.task_type == TaskType.AUDIO_TRANSCRIPTION:
            await self._process_audio_transcription(task)
        elif task.task_type == TaskType.TEXT_ANALYSIS:
            await self._process_text_analysis(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    async def _process_web_scraping(self, task: BackgroundTask):
        pass

    async def _process_media_analysis(self, task: BackgroundTask):
        """Process media analysis task - generate description of media"""
        media_uri = task.data.get("media_uri")
        if not media_uri:
            raise ValueError("No media URI provided for media analysis task")

        media_type = task.data.get("media_type")
        if not media_type:
            raise ValueError("No media type provided for media analysis task")
        if media_type not in ["image", "video", "audio"]:
            raise ValueError(f"Illegal media type: {media_type}. Should be image/video/audio.")
        logger.info(f"Processing media analysis for {media_uri} of type {media_type}")

        original_item = task.source_item
        if not original_item:
            raise ValueError(f"Original memory item {task.source_item} not found")

        existing_caption = original_item.text_content

        try:
            assert self.media_analysis_service, "MediaAnalysisService not initialized"
            async with self.media_analysis_service as media_service:
                analyzed_text = await media_service.analyze(media_uri, media_type, existing_caption)

            # Generate embedding for the analyzed text
            embedding = self.embedding_service.encode(analyzed_text)
            embedding_model_version = self.embedding_service.get_model_version()

            # Update the original memory item with the analysis
            await self.db_manager.update_memory_item(
                item_id=task.source_item.id,
                analyzed_text=analyzed_text,
                embedding=embedding,
                embedding_model_version=embedding_model_version,
            )

            logger.info(f"Successfully analyzed media {media_uri} for item {task.source_item.id}")

        except Exception as e:
            logger.error(f"Failed to analyze media {media_uri}: {e}")
            raise

    async def _process_audio_transcription(self, task: BackgroundTask):
        pass

    async def _process_text_analysis(self, task: BackgroundTask):
        pass


# Factory function to create tasks
def create_web_scraping_task(source_item: MemoryItem, url: str) -> BackgroundTask:
    """Create a web scraping background task"""
    return BackgroundTask(
        task_id=uuid4(),
        task_type=TaskType.WEB_SCRAPING,
        source_item=source_item,
        data={"url": url},
    )


def create_media_analysis_task(
    source_item: MemoryItem, media_uri: str, media_type: str
) -> BackgroundTask:
    return BackgroundTask(
        task_id=uuid4(),
        task_type=TaskType.MEDIA_ANALYSIS,
        source_item=source_item,
        data={"media_uri": media_uri, "media_type": media_type},
    )


def create_audio_transcription_task(source_item: MemoryItem, audio_uri: str) -> BackgroundTask:
    """Create an audio transcription background task"""
    return BackgroundTask(
        task_id=uuid4(),
        task_type=TaskType.AUDIO_TRANSCRIPTION,
        source_item=source_item,
        data={"audio_uri": audio_uri},
    )


# Import the missing function
__all__ = [
    "BackgroundTaskManager",
    "BackgroundTask",
    "TaskType",
    "TaskStatus",
    "create_web_scraping_task",
    "create_media_analysis_task",
    "create_audio_transcription_task",
]
