"""
Embedding generation service for the Memory System
"""

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

    async def initialize(self):
        """Initialize the embedding model"""
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
