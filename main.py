"""
Main entry point for the Memory System
"""

import uvicorn
from loguru import logger

from src.utils.config import settings

# logger.remove()

logger.add(
    "logs/{time}.log",
    level="DEBUG",
    rotation="500 MB",
    compression="zip",
    enqueue=True,
    serialize=False,
)


def main():
    """Run the Memory System API server"""
    logger.info("Starting Memory System API...")
    logger.info(f"Database URL: {settings.database_url}")
    print(f"Embedding Model: {settings.embedding_model}")

    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level="info" if not settings.debug else "debug",
    )


if __name__ == "__main__":
    main()
