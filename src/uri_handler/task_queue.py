"""
Background task processing for the Memory System

Handles asynchronous processing of complex data types like web links, images, and audio clips.
Uses a simple in-memory queue for now, but can be extended to use Redis, RabbitMQ, or other queue systems.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from src.utils.database import DatabaseManager
from src.utils.embedding import EmbeddingService


class TaskType(Enum):
    """Types of background tasks"""

    WEB_SCRAPING = "web_scraping"
    IMAGE_ANALYSIS = "image_analysis"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    TEXT_ANALYSIS = "text_analysis"


class TaskStatus(Enum):
    """Status of background tasks"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    """Represents a background processing task"""

    task_id: UUID
    task_type: TaskType
    source_item_id: UUID  # The original MemoryItem that triggered this task
    data: Dict[str, Any]  # Task-specific data
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class BackgroundTaskManager:
    """Manages background task processing"""

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.task_queue: asyncio.Queue[BackgroundTask] = asyncio.Queue()
        self.running_tasks: Dict[UUID, BackgroundTask] = {}
        self.worker_count = 2  # Number of worker coroutines
        self.workers: List[asyncio.Task] = []
        self.logger = logging.getLogger(__name__)

    async def start_workers(self):
        """Start background worker coroutines"""
        self.logger.info(f"Starting {self.worker_count} background workers")
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker)

    async def stop_workers(self):
        """Stop all background workers"""
        self.logger.info("Stopping background workers")
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def enqueue_task(self, task: BackgroundTask) -> UUID:
        """Add a task to the processing queue"""
        await self.task_queue.put(task)
        self.logger.info(f"Enqueued task {task.task_id} of type {task.task_type}")
        return task.task_id

    async def get_task_status(self, task_id: UUID) -> Optional[BackgroundTask]:
        """Get the status of a specific task"""
        # Check running tasks first
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]

        # TODO: Check completed tasks in database/persistent storage
        return None

    async def _worker_loop(self, worker_name: str):
        """Main worker loop - processes tasks from the queue"""
        self.logger.info(f"Worker {worker_name} started")

        while True:
            try:
                # Get next task from queue
                task = await self.task_queue.get()

                # Mark task as processing
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.utcnow()
                self.running_tasks[task.task_id] = task

                self.logger.info(f"Worker {worker_name} processing task {task.task_id}")

                # Process the task
                try:
                    await self._process_task(task)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    self.logger.info(f"Task {task.task_id} completed successfully")

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    self.logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

                finally:
                    # Mark task as done and remove from running tasks
                    self.task_queue.task_done()
                    # Keep completed tasks for a while for status checking
                    # In production, move to persistent storage

            except asyncio.CancelledError:
                self.logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}", exc_info=True)

    async def _process_task(self, task: BackgroundTask):
        """Process a specific task based on its type"""
        if task.task_type == TaskType.WEB_SCRAPING:
            await self._process_web_scraping(task)
        elif task.task_type == TaskType.IMAGE_ANALYSIS:
            await self._process_image_analysis(task)
        elif task.task_type == TaskType.AUDIO_TRANSCRIPTION:
            await self._process_audio_transcription(task)
        elif task.task_type == TaskType.TEXT_ANALYSIS:
            await self._process_text_analysis(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    async def _process_web_scraping(self, task: BackgroundTask):
        """Process web scraping task - extract content from URL and create chunks"""
        url = task.data.get("url")
        if not url:
            raise ValueError("No URL provided for web scraping task")

        # TODO: Implement actual web scraping
        # For now, simulate a long article
        from ..utils.models import MemoryItemRaw

        # Simulate web scraping - in real implementation, use libraries like:
        # - requests/httpx for fetching
        # - BeautifulSoup/selectolax for parsing
        # - newspaper3k/readability for article extraction

        # Simulate a long article content
        simulated_content = f"""
        This is a simulated long article from {url}. 
        
        Section 1: Introduction
        This article discusses various aspects of artificial intelligence and machine learning. 
        The field has evolved significantly over the past decades, with numerous breakthroughs 
        in deep learning, natural language processing, and computer vision.
        
        Section 2: Historical Context
        The history of AI dates back to the 1950s when researchers first began exploring 
        computational models of intelligence. Early pioneers like Alan Turing and John McCarthy 
        laid the groundwork for what would become modern artificial intelligence.
        
        Section 3: Current Applications
        Today, AI is used in numerous applications including search engines, recommendation 
        systems, autonomous vehicles, medical diagnosis, and natural language understanding. 
        Companies like Google, OpenAI, and Anthropic are pushing the boundaries of what's possible.
        
        Section 4: Future Prospects
        Looking ahead, AI is expected to play an increasingly important role in society. 
        Areas like artificial general intelligence (AGI), robotics, and quantum computing 
        may lead to unprecedented technological advances.
        
        Section 5: Ethical Considerations
        As AI becomes more powerful, it's crucial to consider the ethical implications. 
        Issues like bias, privacy, job displacement, and AI safety need careful attention 
        from researchers, policymakers, and society as a whole.
        """

        # Create the main parsed article item (summary/overview)
        article_summary = "Article about AI and machine learning covering history, applications, and future prospects"

        # Generate embedding for summary
        summary_embedding = self.embedding_service.encode(article_summary)

        # Create main parsed article MemoryItem
        parsed_item = await self.db_manager.create_memory_item(
            item_data=MemoryItemRaw(
                content_type="parsed_article",
                text_content=simulated_content[:500] + "..."
                if len(simulated_content) > 500
                else simulated_content,
                data_uri=url,
                event_timestamp=datetime.utcnow(),
                meta={"parsed_from": str(task.source_item_id), "is_chunked": True},
                reply_to_id=None,
            ),
            analyzed_text=article_summary,
            embedding=summary_embedding,
            embedding_model_version=self.embedding_service.get_model_version(),
        )

        # Create chunks if content is long
        chunks = self.embedding_service.chunk_text(simulated_content, max_tokens=200, overlap=20)

        if len(chunks) > 1:
            self.logger.info(f"Creating {len(chunks)} chunks for article {parsed_item.id}")

            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Generate embedding for this chunk
                chunk_embedding = self.embedding_service.encode(chunk_text)

                # Create chunk MemoryItem
                chunk_item = await self.db_manager.create_memory_item(
                    item_data=MemoryItemRaw(
                        content_type="article_chunk",
                        text_content=chunk_text,
                        data_uri=url,
                        event_timestamp=datetime.utcnow(),
                        meta={
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "parent_article": str(parsed_item.id),
                        },
                        reply_to_id=None,
                    ),
                    analyzed_text=chunk_text,
                    embedding=chunk_embedding,
                    embedding_model_version=self.embedding_service.get_model_version(),
                    parent_id=parsed_item.id,  # Set hierarchical relationship
                )

                # Create relationship between chunk and parent article
                await self.db_manager.create_relationship(
                    source_node_id=chunk_item.id,
                    target_node_id=parsed_item.id,
                    relationship_type="chunk_of",
                )

        # Create relationship between parsed article and original web link
        await self.db_manager.create_relationship(
            source_node_id=parsed_item.id,
            target_node_id=task.source_item_id,
            relationship_type="parsed_from",
        )

        self.logger.info(
            f"Created parsed article {parsed_item.id} with {len(chunks)} chunks from {url}"
        )

    async def _process_image_analysis(self, task: BackgroundTask):
        """Process image analysis task - generate description of image"""
        image_uri = task.data.get("image_uri")
        if not image_uri:
            raise ValueError("No image URI provided for image analysis task")

        # TODO: Implement actual image analysis using multimodal models
        # For now, create a placeholder description

        analyzed_text = f"Image analysis for {image_uri} - [Image analysis not implemented yet]"

        # Generate embedding for future use
        _ = self.embedding_service.encode(analyzed_text)

        # Update the original item with the analysis
        # In a real implementation, you might create a separate analyzed item
        # and link it via relationships
        self.logger.info(f"Generated image description for {image_uri}")

    async def _process_audio_transcription(self, task: BackgroundTask):
        """Process audio transcription task"""
        audio_uri = task.data.get("audio_uri")
        if not audio_uri:
            raise ValueError("No audio URI provided for transcription task")

        # TODO: Implement actual speech-to-text
        # Use services like OpenAI Whisper, Google Speech-to-Text, etc.

        analyzed_text = f"Transcription of {audio_uri} - [Audio transcription not implemented yet]"

        # Generate embedding for future use
        _ = self.embedding_service.encode(analyzed_text)

        self.logger.info(f"Generated transcription for {audio_uri}")

    async def _process_text_analysis(self, task: BackgroundTask):
        """Process advanced text analysis task"""
        text = task.data.get("text")
        if not text:
            raise ValueError("No text provided for analysis task")

        # TODO: Implement advanced text analysis like:
        # - Entity extraction
        # - Sentiment analysis
        # - Topic modeling
        # - Summary generation

        self.logger.info(f"Performed text analysis on {len(text)} characters")


# Factory function to create tasks
def create_web_scraping_task(source_item_id: UUID, url: str) -> BackgroundTask:
    """Create a web scraping background task"""
    return BackgroundTask(
        task_id=uuid4(),
        task_type=TaskType.WEB_SCRAPING,
        source_item_id=source_item_id,
        data={"url": url},
    )


def create_image_analysis_task(source_item_id: UUID, image_uri: str) -> BackgroundTask:
    """Create an image analysis background task"""
    return BackgroundTask(
        task_id=uuid4(),
        task_type=TaskType.IMAGE_ANALYSIS,
        source_item_id=source_item_id,
        data={"image_uri": image_uri},
    )


def create_audio_transcription_task(source_item_id: UUID, audio_uri: str) -> BackgroundTask:
    """Create an audio transcription background task"""
    return BackgroundTask(
        task_id=uuid4(),
        task_type=TaskType.AUDIO_TRANSCRIPTION,
        source_item_id=source_item_id,
        data={"audio_uri": audio_uri},
    )


# Import the missing function
__all__ = [
    "BackgroundTaskManager",
    "BackgroundTask",
    "TaskType",
    "TaskStatus",
    "create_web_scraping_task",
    "create_image_analysis_task",
    "create_audio_transcription_task",
]
