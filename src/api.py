from contextlib import asynccontextmanager
from datetime import datetime

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.routers import get_items, ingest, retrieve, task_status
from src.services.embedding import EmbeddingService
from src.services.rerank import RerankingService
from src.uri_handler.task_queue import BackgroundTaskManager
from src.utils.config import settings
from src.utils.database import DatabaseManager

torch.set_float32_matmul_precision("high")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with (
        DatabaseManager(settings.database_url) as db_manager,
        EmbeddingService(settings.embedding_model) as embedding_service,
        RerankingService() as reranking_service,
    ):
        async with BackgroundTaskManager(db_manager, embedding_service) as task_manager:
            app.state.db_manager = db_manager
            app.state.embedding_service = embedding_service
            app.state.reranking_service = reranking_service
            app.state.task_manager = task_manager

            yield

            logger.info("Application Shutting Down...")

    logger.info("Application Shutdown Complete.")


# Global instances
app = FastAPI(
    title="Memory System API",
    description="A backend service for ingesting, storing, and retrieving event logs",
    version="0.1.0",
    lifespan=lifespan,
)

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


# Include routers (authentication is handled per router)
app.include_router(ingest.router)
app.include_router(retrieve.router)
app.include_router(get_items.router)
app.include_router(task_status.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
    )
