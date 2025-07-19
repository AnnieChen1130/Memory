"""
Embedding generation service for the Memory System
"""

import gc
from typing import Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for generating text embeddings using Qwen3-Embedding-4B"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False

    async def __aenter__(self):
        """Initialize the embedding model as an async context manager"""
        if self._initialized:
            return self

        model_kwargs = {}
        tokenizer_kwargs = {}

        if torch.cuda.is_available():
            # Enable flash attention for better acceleration and memory saving
            model_kwargs = {
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": torch.float16,
            }
            tokenizer_kwargs = {"padding_side": "left"}

        try:
            self.model = SentenceTransformer(
                self.model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )
            logger.info(f"Embedding model {self.model_name} initialized with flash attention.")
        except Exception as e:
            logger.error(f"Failed to initialize model with flash attention: {e}")
            # Fallback to basic initialization if flash attention fails
            self.model = SentenceTransformer(self.model_name)
            logger.warning(f"Using Embedding model {self.model_name} without flash attention.")

        self._initialized = True
        return self

    def get_model_version(self) -> str:
        """Get the current model version string"""
        return f"{self.model_name}"

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        if self.model is None:
            return 2560
        return self.model.get_sentence_embedding_dimension() or 2560

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Generate embedding for text content.

        Args:
            text: Text to encode
            is_query: If True, uses query prompt. If False, encodes content without prompt.

        Returns:
            Single embedding vector
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if not text or not text.strip():
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)

        prompt_name = "query" if is_query else None
        embedding = self.model.encode(text, prompt_name=prompt_name, convert_to_numpy=True)
        logger.debug(f"Generated embedding for text: {text[:50]}... with shape {embedding.shape}")

        return embedding

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context manager"""
        if not self._initialized:
            return

        logger.info(f"Exiting EmbeddingService. Releasing resources for {self.model_name}...")

        del self.model
        self.model = None
        self._initialized = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Resources for EmbeddingService: {self.model_name} have been released.")
