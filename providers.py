"""
Image generation providers with a common interface.
Supports easy switching between different AI image generation APIs.
"""

import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Protocol

from PIL import Image


class ImageProvider(ABC):
    """Abstract base class for image generation providers."""

    name: str
    description: str

    @abstractmethod
    def generate_from_sketch(
        self, sketch: Image.Image, style: str, prompt: str | None = None
    ) -> Image.Image:
        """
        Generate a new image from a sketch in the specified style.

        Args:
            sketch: The input sketch/drawing as a PIL Image
            style: The artistic style to apply
            prompt: Optional additional prompt text

        Returns:
            Generated image as a PIL Image
        """
        pass

    def _build_prompt(self, style: str, prompt: str | None = None) -> str:
        """Build the full prompt for image generation."""
        base_prompt = (
            f"Transform this child's sketch/drawing into a beautiful, detailed image "
            f"in {style} style. Keep the main elements and composition from the sketch "
            f"but make it look professionally rendered and polished."
        )
        if prompt:
            base_prompt += f" Additional instructions: {prompt}"
        return base_prompt


class OpenAIProvider(ImageProvider):
    """OpenAI GPT Image model provider."""

    name = "OpenAI GPT-Image-1.5"
    description = "OpenAI's latest image generation model with excellent sketch interpretation"

    def __init__(self, api_key: str | None = None, model: str = "gpt-image-1.5"):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-image-1.5)
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate_from_sketch(
        self, sketch: Image.Image, style: str, prompt: str | None = None
    ) -> Image.Image:
        """Generate image using OpenAI's image edit API."""
        # Convert PIL Image to bytes with proper format
        img_buffer = BytesIO()
        # Ensure image is in RGB mode for PNG
        if sketch.mode == "RGBA":
            sketch = sketch.convert("RGB")
        sketch.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        full_prompt = self._build_prompt(style, prompt)

        # Use the images.edit endpoint for sketch transformation
        # Pass as tuple with filename and mime type for proper detection
        response = self.client.images.edit(
            model=self.model,
            image=("sketch.png", img_bytes, "image/png"),
            prompt=full_prompt,
            n=1,
            size="1024x1024",
        )

        # Decode the base64 response
        image_data = base64.b64decode(response.data[0].b64_json)
        return Image.open(BytesIO(image_data))


class GeminiVertexProvider(ImageProvider):
    """Google Gemini provider via Vertex AI."""

    name = "Gemini 3 Pro Image (Vertex AI)"
    description = "Google's Gemini model with Nano Banana image generation via Vertex AI"

    def __init__(
        self,
        project: str | None = None,
        location: str = "global",
        model: str = "gemini-3-pro-image-preview",
    ):
        """
        Initialize the Gemini Vertex AI provider.

        Args:
            project: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            location: GCP location (default: global)
            model: Model to use (default: gemini-3-pro-image-preview)
        """
        from google import genai
        from google.genai.types import GenerateContentConfig, Modality

        # Set up environment for Vertex AI
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        if project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project
        if location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = location

        self.client = genai.Client()
        self.model = model
        self._config_class = GenerateContentConfig
        self._modality = Modality

    def generate_from_sketch(
        self, sketch: Image.Image, style: str, prompt: str | None = None
    ) -> Image.Image:
        """Generate image using Gemini's multimodal capabilities."""
        from google.genai.types import GenerateContentConfig, Modality

        full_prompt = self._build_prompt(style, prompt)

        # Gemini accepts PIL Images directly
        response = self.client.models.generate_content(
            model=self.model,
            contents=[sketch, full_prompt],
            config=GenerateContentConfig(
                response_modalities=[Modality.TEXT, Modality.IMAGE],
            ),
        )

        # Extract the image from the response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                return Image.open(BytesIO(image_data))

        raise RuntimeError("No image was generated in the response")


# Registry of available providers
PROVIDERS: dict[str, type[ImageProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiVertexProvider,
}


def get_provider(name: str, **kwargs) -> ImageProvider:
    """
    Get an initialized provider by name.

    Args:
        name: Provider name (e.g., "openai", "gemini")
        **kwargs: Additional arguments passed to the provider constructor

    Returns:
        Initialized provider instance
    """
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider: {name}. Available: {available}")

    return PROVIDERS[name](**kwargs)


def list_providers() -> list[tuple[str, str, str]]:
    """
    List all available providers.

    Returns:
        List of (key, name, description) tuples
    """
    result = []
    for key, cls in PROVIDERS.items():
        result.append((key, cls.name, cls.description))
    return result
