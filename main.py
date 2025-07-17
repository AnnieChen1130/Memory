"""
Main entry point for the Memory System
"""

import uvicorn

from src.utils.config import settings


def main():
    """Run the Memory System API server"""
    print("Starting Memory System API...")
    print(f"Database URL: {settings.database_url}")
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
