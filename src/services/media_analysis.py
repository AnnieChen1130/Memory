import asyncio
import gc
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch
from dotenv import load_dotenv
from loguru import logger
from minio import Minio
from transformers import AutoModelForImageTextToText, AutoProcessor


class MediaAnalysisService:
    def __init__(self, model_id: str = "google/gemma-3n-E4B-it"):
        self.minio_client = None
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._initialized = False

    async def __aenter__(self):
        if self._initialized:
            return self

        def load_model():
            try:
                model = AutoModelForImageTextToText.from_pretrained(
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

        load_dotenv()
        minio_endpoint = os.getenv("MEMORY_MINIO_URL", "http://localhost:9876")
        minio_access_key = os.getenv("MEMORY_MINIO_ACCESS_KEY", "minioadmin")
        minio_secret_key = os.getenv("MEMORY_MINIO_SECRET_KEY", "minioadmin")
        minio_secure = os.getenv("MEMORY_MINIO_SECURE", "false").lower() == "true"

        self.minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
        )
        logger.info(f"Connected to MinIO at {minio_endpoint} with secure={minio_secure}")

        logger.info(f"Initializing media analysis model: {self.model_id}")
        try:
            loop = asyncio.get_event_loop()
            self.model, self.processor = await loop.run_in_executor(None, load_model)

            if self.model is None or self.processor is None:
                raise RuntimeError(f"Failed to load model {self.model_id}.")

            self._initialized = True
            logger.info(f"Media analysis model {self.model_id} initialized successfully")
            return self

        except Exception as e:
            self._initialized = False
            self.model, self.processor = None, None
            logger.exception(f"Failed to initialize model {self.model_id}: {e}")
            raise RuntimeError(f"Could not initialize model {self.model_id}. ") from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._initialized:
            return

        logger.info(f"Exiting ImageAnalysisService. Releasing resources for {self.model_id}...")

        del self.model, self.processor, self.minio_client
        self.model, self.processor = None, None
        self._initialized = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Resources for Image Analysis: {self.model_id} have been released.")

    async def analyze(
        self, media_uri: str, media_type: str, existing_caption: Optional[str] = None
    ) -> str:
        """
        Analyze an image and generate a detailed description

        Args:
            media_uri: URI of the media (URL or local path)
            media_type: Type of media (possible values: "image", "video", "audio")
            existing_caption: Optional existing caption to enhance the analysis

        Returns:
            Generated description of the media
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Use within an `async with` block.")

        if media_type not in ["image", "video", "audio"]:
            raise ValueError(f"Illegal type: {media_type}. Should be image/video/audio.")

        logger.info(f"Analyzing media: {media_uri}")
        try:
            accessible_url = await self._get_accessible_url(media_uri)
            logger.info(f"Using accessible url: {accessible_url[:100]}...")

            if existing_caption:
                prompt_text = (
                    f"Here is an caption to the {media_type}: '{existing_caption}'. "
                    "Please provide a detailed description of the media, "
                    "including any details not mentioned in the caption."
                )
            else:
                prompt_text = f"Describe the {media_type} in detail."

            logger.info(f"Analyzing media with Prompt text: {prompt_text}")

            loop = asyncio.get_event_loop()

            content: List[Dict[str, Any]] = []
            if media_type == "video":
                async with self._extract_video_content(accessible_url) as (frames, audio_path):
                    if frames:
                        content.extend([{"type": "image", "image": frame} for frame in frames])
                    if audio_path:
                        content.append({"type": "audio", "audio": audio_path})
                    content.append({"type": "text", "text": prompt_text})
                    description = await loop.run_in_executor(None, self._model_inference, content)
            else:
                content.append({"type": media_type, media_type: accessible_url})
                content.append({"type": "text", "text": prompt_text})
                description = await loop.run_in_executor(None, self._model_inference, content)

            # Combine with existing caption if available
            if existing_caption:
                final_description = f"Caption: {existing_caption}\nDescription: {description}"
            else:
                final_description = description

            logger.info(f"Generated model-based media description: {description}")
        except Exception as e:
            logger.error(f"Media analysis failed: {e}", exc_info=True)
            final_description = f"Caption: {existing_caption}" if existing_caption else ""
        return final_description

    @logger.catch()
    def _model_inference(self, content: List[Dict[str, Any]]) -> str:
        logger.debug(f"Running model inference on content: {content}")
        assert self.model is not None, "Model not initialized. Call initialize() first."
        assert self.processor is not None, "Processor not initialized. Call initialize() first."

        messages = [
            {
                "role": "user",
                "content": content,
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

        return decoded.strip()

    @asynccontextmanager
    async def _extract_video_content(
        self, video_uri: str
    ) -> AsyncGenerator[Tuple[List[str], Optional[str]], None]:
        logger.debug(f"Extracting video content from: {video_uri}")
        """Extract frames and audio from video"""
        temp_dir = tempfile.TemporaryDirectory()
        frames_dir = os.path.join(temp_dir.name, "frames")
        audio_file = os.path.join(temp_dir.name, "audio.acc")
        os.makedirs(frames_dir, exist_ok=True)

        frames: List[str] = []
        audio_path: Optional[str] = None
        loop = asyncio.get_event_loop()

        try:
            logger.debug(f"Extracting video content in temporary directory: {temp_dir.name}")

            # Extract frames at 1 fps
            frame_pattern = os.path.join(frames_dir, "%04d.jpg")
            _ = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["ffmpeg", "-i", video_uri, "-vf", "fps=1", frame_pattern, "-y"],
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            )

            for file in sorted(os.listdir(frames_dir)):
                if file.endswith(".jpg"):
                    frames.append(os.path.join(frames_dir, file))

            logger.info(f"Extracted {len(frames)} frames from video.")

            # Extract audio
            try:
                _ = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["ffmpeg", "-i", video_uri, "-vn", "-c:a", "copy", audio_file, "-y"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ),
                )
                audio_path = audio_file
                logger.info("Successfully extracted audio from video.")
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Could not extract audio from video {video_uri}. FFMPEG stderr: {e.stderr}"
                )

            yield frames, audio_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to process video {video_uri}: {e}")
            raise RuntimeError(f"Video processing failed: {e}") from e
        finally:
            logger.debug(f"Cleaning up temporary directory: {temp_dir.name}")
            temp_dir.cleanup()

    def get_model_info(self) -> str | None:
        """Get model information string"""
        if self.model is not None:
            return f"{self.model_id}"
        else:
            return None

    async def _get_accessible_url(self, uri: str) -> str:
        if not uri.startswith(("http://", "https://")):
            logger.debug(f"URI '{uri}' is a local path, not generating presigned URL.")
            return uri

        try:
            loop = asyncio.get_event_loop()
            bucket, key = await loop.run_in_executor(None, _parse_minio_url, uri)

            logger.info(f"Generating presigned URL for object: {key} in bucket: {bucket}")

            assert self.minio_client is not None, "need a minio client to get pre-signed url"
            presigned_url = await loop.run_in_executor(
                None,
                self.minio_client.get_presigned_url,
                "GET",
                bucket,
                key,
                timedelta(minutes=15),
            )
            return presigned_url
        except ValueError:
            logger.warning(
                f"Couldn't parse '{uri}' as MinIO URL. Assuming it's publicly accessible."
            )
            return uri
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {uri}: {e}", exc_info=True)
            raise


def _parse_minio_url(url: str) -> Tuple[str, str]:
    """
    Parse MinIO URL into components

    Args:
        url: MinIO URL in the format 'http://<host>:<port>/<bucket>/<object>'

    Returns:
        Tuple of (bucket, object)
    """
    parsed_url = urlparse(url)
    url_parts = parsed_url.path.strip("/").split("/", 1)
    if len(url_parts) < 2:
        raise ValueError(f"Invalid MinIO URL format: {url}")

    return url_parts[0], url_parts[1]
