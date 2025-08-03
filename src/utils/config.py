"""
Configuration settings for the Memory System
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database configuration
    database_url: str = "postgresql://sunya@/postgres?host=/home/sunya/dev/memory/pg_data"

    # MinIO configuration
    minio_url: str = "http://localhost:9876"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None

    # Embedding configuration
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dimension: int = 2560

    # Reranking configuration
    reranking_model: str = "Qwen/Qwen3-Reranker-0.6B"

    # Performance configuration
    max_batch_size: int = 32
    request_timeout: int = 30

    # Development settings
    debug: bool = False
    reload: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "MEMORY_"


# Global settings instance
settings = Settings()
