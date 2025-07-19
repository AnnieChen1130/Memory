import asyncio
import gc
import io
from typing import Optional

import httpx
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class ImageAnalysisService:
    def __init__(self, model_id: str = "google/gemma-3n-E4B-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._initialized = False

    async def __aenter__(self):
        if self._initialized:
            return self

        def load_model():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                ).eval()

                processor = AutoProcessor.from_pretrained(self.model_id)
                return model, processor

            except Exception as e:
                logger.error(f"Failed to load model {self.model_id}: {e}", exc_info=True)
                return None, None

        logger.info(f"Initializing image analysis model: {self.model_id}")
        try:
            loop = asyncio.get_event_loop()
            self.model, self.processor = await loop.run_in_executor(None, load_model)

            if self.model is None or self.processor is None:
                raise RuntimeError(f"Failed to load model {self.model_id}.")

            self._initialized = True
            logger.info(f"Image analysis model {self.model_id} initialized successfully")
            return self

        except Exception as e:
            self._initialized = False
            self.model, self.processor = None, None
            logger.exception(f"Failed to initialize model {self.model_id}: {e}")
            raise RuntimeError(f"Could not initialize model {self.model_id}. ") from e

    async def analyze_image(self, image_uri: str, existing_caption: Optional[str] = None) -> str:
        """
        Analyze an image and generate a detailed description

        Args:
            image_uri: URI of the image (URL or local path)
            existing_caption: Optional existing caption to enhance the analysis

        Returns:
            Generated description of the image
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Use within an 'async with' block.")

        logger.info(f"Analyzing image: {image_uri}")
        try:
            image = await self._load_image(image_uri)
            # image = image_uri

            # Prepare the prompt based on existing caption
            if existing_caption:
                prompt_text = (
                    f"Here is an image with this caption: '{existing_caption}'. "
                    "Please provide a detailed description of what you see in the image, "
                    "including any details not mentioned in the caption."
                )
            else:
                prompt_text = "Describe this image in detail."

            logger.debug(f"Prompt text: {prompt_text}")
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(
                None, self._model_inference, image, prompt_text
            )

            # Combine with existing caption if available
            if existing_caption:
                final_description = f"Caption: {existing_caption}\nDescription: {description}"
            else:
                final_description = description

            logger.info(f"Generated model-based image description: {description[:100]}...")
        except Exception as e:
            logger.error(f"Image analysis failed: {e}", exc_info=True)
            final_description = f"Caption: {existing_caption}" if existing_caption else ""
        return final_description

    # def _model_inference(self, image: Image.Image, prompt_text: str) -> str:
    @logger.catch()
    def _model_inference(self, image: Image.Image, prompt_text: str) -> str:
        assert self.model is not None, "Model not initialized. Call initialize() first."
        assert self.processor is not None, "Processor not initialized. Call initialize() first."

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You describe images in detail."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=120, do_sample=False)

        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.debug(f"Generated description: {decoded[:50]}...")

        return decoded.strip()

    async def _load_image(self, image_uri: str) -> Image.Image:
        """Load image from URI (URL or local path)"""
        try:
            if image_uri.startswith(("http://", "https://")):
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(image_uri, timeout=30.0)
                    response.raise_for_status()
                    image_data = response.content

                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_uri)

            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as e:
            logger.error(f"Failed to load image from {image_uri}: {e}")
            raise ValueError(f"Unable to load image from {image_uri}: {e}") from e

    def get_model_info(self) -> str | None:
        """Get model information string"""
        if self.model is not None:
            return f"{self.model_id}"
        else:
            return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._initialized:
            return

        logger.info(f"Exiting ImageAnalysisService. Releasing resources for {self.model_id}...")

        del self.model, self.processor
        self.model, self.processor = None, None
        self._initialized = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Resources for Image Analysis: {self.model_id} have been released.")
