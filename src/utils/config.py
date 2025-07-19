"""
Configuration settings for the Memory System
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database configuration
    database_url: str = "postgresql://sunya@/postgres?host=/home/sunya/dev/memory/pg_data"

    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None

    # Embedding configuration
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dimension: int = 2560

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
