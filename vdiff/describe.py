"""LLM-powered change description — used sparingly for ambiguous detections."""

import base64
import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_base64(img: Image.Image, max_width: int = 512) -> str:
    """Convert PIL Image to base64-encoded JPEG, resized for LLM efficiency."""
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ChangeDescriber:
    """Describes visual content using an LLM — only called for ambiguous detections."""

    def __init__(self, config: dict):
        self.provider = config.get("provider", "ollama")
        self.host = config.get("host", "http://localhost:11434")
        self.model = config.get("model", "llava")
        self.timeout = config.get("timeout", 60)
        self._client = None

    def _get_client(self):
        """Lazy-init the Ollama client."""
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def describe(
        self,
        prev_image: Image.Image,
        curr_image: Image.Image,
        region_description: str = "",
    ) -> str:
        """
        Describe what is visible in the current image where changes were detected.
        Only the current image is sent to the LLM to avoid hallucination.
        Falls back to region-based description on LLM failure.
        """
        fallback = self._fallback_description(region_description)

        try:
            return self._describe_with_llm(curr_image, region_description)
        except Exception as e:
            logger.warning(f"LLM description failed, using fallback: {e}")
            return fallback

    def describe_cropped(
        self,
        cropped_image: Image.Image,
        detected_class: str,
        confidence: float,
    ) -> str:
        """
        Describe a cropped detection region to clarify ambiguous objects.
        Called only for low-confidence YOLO detections.
        Returns a brief clarification string, or empty string on failure.
        """
        try:
            client = self._get_client()
            img_b64 = _image_to_base64(cropped_image, max_width=256)

            prompt = (
                f"YOLO detected this as '{detected_class}' with {confidence:.0%} confidence. "
                f"Look at this cropped image carefully. "
                f"In one short sentence, what is actually in this image? "
                f"If it matches '{detected_class}', confirm it. "
                f"If not, say what it actually is. Be brief and factual."
            )

            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
            )

            result = response["message"]["content"].strip()
            logger.info(f"LLM clarification for '{detected_class}': {result}")
            return result
        except Exception as e:
            logger.warning(f"LLM clarification failed for '{detected_class}': {e}")
            return ""

    def _describe_with_llm(
        self,
        curr_image: Image.Image,
        region_hint: str,
    ) -> str:
        """Send only the current image to the LLM for description."""
        client = self._get_client()

        curr_b64 = _image_to_base64(curr_image)

        prompt = (
            "This is a security camera image. Motion was detected in this frame. "
            "Describe ONLY what you can actually see in this image in 1-2 short sentences. "
            "Focus on: people, vehicles, animals, open/closed doors, packages, or anything notable. "
            "If you cannot clearly identify something, say so. "
            "Do NOT guess or assume things that are not clearly visible. "
            "Be brief and factual."
        )
        if region_hint:
            prompt += f"\nMotion was detected in these areas: {region_hint}."

        response = client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [curr_b64],
                }
            ],
        )

        description = response["message"]["content"].strip()
        logger.info(f"LLM description: {description}")
        return description

    @staticmethod
    def _fallback_description(region_description: str) -> str:
        """Non-LLM fallback using region-based info from the diff engine."""
        if region_description:
            return f"Movement detected in: {region_description}"
        return "Visual change detected in the frame."
