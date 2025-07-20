from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.services.embedding import EmbeddingService
from src.services.rerank import RerankingService
from src.uri_handler.task_queue import BackgroundTaskManager
from src.utils.config import settings
from src.utils.database import DatabaseManager

security = HTTPBearer(auto_error=False)


def get_db_manager(request: Request) -> DatabaseManager:
    return request.app.state.db_manager


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_rerank_service(request: Request) -> RerankingService:
    return request.app.state.reranking_service


def get_task_manager(request: Request) -> BackgroundTaskManager:
    return request.app.state.task_manager


def get_api_key_verification(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
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


# Create a reusable dependency
AuthDep = Depends(get_api_key_verification)
