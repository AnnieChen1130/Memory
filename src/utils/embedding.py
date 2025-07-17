"""
Embedding generation service for the Memory System
"""

from typing import List, Optional

import torch
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
            }
            tokenizer_kwargs = {"padding_side": "left"}

        try:
            self.model = SentenceTransformer(
                self.model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        except Exception:
            # Fallback to basic initialization if flash attention fails
            self.model = SentenceTransformer(self.model_name)

    def get_model_version(self) -> str:
        """Get the current model version string"""
        return f"{self.model_name}"

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        if self.model is None:
            return 2560
        return self.model.get_sentence_embedding_dimension() or 2560

    def chunk_text(
        self, text: str, max_tokens: int = 512, overlap: int = 15
    ) -> List[str]:
        """
        Split long text into chunks for embedding.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (approximate)
            overlap: Number of words to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Simple word-based chunking (can be enhanced with proper tokenization)
        words = text.split()
        if len(words) <= max_tokens:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end >= len(words):
                break
            start = end - overlap

        return chunks

    def encode(
        self, text: str, is_query: bool = False, max_tokens: int = 512
    ) -> List[float]:
        """
        Generate embedding for text content.

        Args:
            text: Text to encode
            is_query: If True, uses query prompt. If False, encodes content without prompt.
            max_tokens: Maximum tokens per chunk for long content (ignored for queries)

        Returns:
            Single embedding vector
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        if not text or not text.strip():
            return [0.0] * self.get_embedding_dimension()

        # Queries
        if is_query:
            embedding = self.model.encode(
                text, prompt_name="query", convert_to_numpy=True
            )
            return embedding.tolist()

        # Content
        words = text.split()
        if len(words) <= max_tokens:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        # Long content
        chunks = self.chunk_text(text, max_tokens)
        if not chunks:
            return [0.0] * self.get_embedding_dimension()

        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        chunk_embeddings = embeddings.tolist()

        # Average the embeddings
        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0]

        # Compute element-wise average
        dim = len(chunk_embeddings[0])
        averaged = [0.0] * dim

        for embedding in chunk_embeddings:
            for i in range(dim):
                averaged[i] += embedding[i]

        # Normalize by number of chunks
        num_chunks = len(chunk_embeddings)
        return [val / num_chunks for val in averaged]
