"""
Reranking service for the Memory System

Implements a two-stage retrieval process:
1. Initial vector search for candidate documents
2. Reranking using a more sophisticated model for final ordering
"""

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.models import MemoryItem


class RerankingService:
    """Service for reranking search results using Qwen3-Reranker-0.6B"""

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

        # Model-specific tokens and settings
        self.token_false_id: Optional[int] = None
        self.token_true_id: Optional[int] = None
        self.max_length = 8192
        self.prefix_tokens: List[int] = []
        self.suffix_tokens: List[int] = []

        # Default instruction for relevance judgment
        self.default_instruction = (
            "Given a content matching query, retrieve relevant passages that answer the query"
        )

    async def __aenter__(self):
        if self._initialized:
            return self

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

            if torch.cuda.is_available():
                # Use flash attention for better performance if available
                try:
                    self.model = (
                        AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16,
                            attn_implementation="flash_attention_2",
                        )
                        .cuda()
                        .eval()
                    )
                except Exception:
                    # Fallback without flash attention
                    self.model = (
                        AutoModelForCausalLM.from_pretrained(
                            self.model_name, torch_dtype=torch.float16
                        )
                        .cuda()
                        .eval()
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()

            # Initialize special tokens
            if self.tokenizer:
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

                # Initialize prompt templates
                prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

                self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

            print(f"Successfully loaded {self.model_name}")
            self._initialized = True
            return self

        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            raise RuntimeError(f"Could not initialize model {self.model_name}. ") from e

    def _format_instruction(self, instruction: Optional[str], query: str, doc: str) -> str:
        """Format the instruction for the reranker model"""
        if instruction is None:
            instruction = self.default_instruction
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]) -> Dict[str, Any]:
        """Process input pairs for the model"""
        if not self.tokenizer or not self.model:
            raise ValueError("Tokenizer or model not initialized")

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        for i, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + input_ids + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        return inputs

    @torch.no_grad()
    def _compute_logits(self, inputs: Dict[str, Any]) -> List[float]:
        """Compute relevance scores from model logits"""
        if not self.model:
            raise ValueError("Model not initialized")

        outputs = self.model(**inputs)
        batch_scores = outputs.logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[MemoryItem, float]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Rerank candidate items using Qwen3-Reranker model

        Args:
            query: The search query
            candidates: List of (MemoryItem, initial_score) tuples
            top_k: Optional limit on number of results to return
            instruction: Optional custom instruction for relevance judgment

        Returns:
            List of (MemoryItem, rerank_score) tuples ordered by relevance
        """
        if not self.model or not self.tokenizer or not candidates:
            return candidates

        # Prepare input pairs for the reranker
        pairs = []
        items = []

        for item, _ in candidates:
            # Use analyzed_text if available, otherwise fall back to text_content
            text = item.analyzed_text or item.text_content or ""
            if text.strip():  # Only include items with actual text content
                formatted_input = self._format_instruction(instruction, query, text)
                pairs.append(formatted_input)
                items.append(item)

        if not pairs:
            return candidates

        # Get reranking scores
        try:
            inputs = self._process_inputs(pairs)
            scores = self._compute_logits(inputs)

            # Pair items with their new scores
            reranked = list(zip(items, scores, strict=True))

            # Sort by reranking score (higher is better)
            reranked.sort(key=lambda x: float(x[1]), reverse=True)

            # Convert back to the expected type format
            result: List[Tuple[MemoryItem, float]] = [
                (item, float(score)) for item, score in reranked
            ]

            # Apply top_k limit if specified
            if top_k:
                result = result[:top_k]

            return result

        except Exception as e:
            print(f"Reranking failed, returning original order: {e}")
            return candidates

    def get_model_version(self) -> str:
        """Get the model name/version for tracking"""
        return self.model_name

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context manager"""
        if not self._initialized:
            return

        print(f"Exiting RerankingService. Releasing resources for {self.model_name}...")

        del self.model, self.tokenizer
        self.model, self.tokenizer = None, None
        self._initialized = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Resources for RerankingService: {self.model_name} have been released.")
