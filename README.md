# Sketch to Image Transformer

Transform children's sketches and drawings into beautiful AI-generated images!

Upload a photo of a hand-drawn sketch, select an artistic style, and watch as AI transforms it into a polished image while preserving the original composition and creativity.

## Features

- **Multiple AI Providers**: Switch between OpenAI and Google Gemini
- **Various Artistic Styles**: Cartoon, watercolor, oil painting, anime, and more
- **Custom Instructions**: Add your own guidance for the AI
- **Easy to Extend**: Add new providers by implementing the `ImageProvider` interface

## Installation

```bash
# Clone and enter the project
cd detsky_den2026

# Install dependencies using uv
uv sync
```

## Configuration

### OpenAI

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Google Gemini (Vertex AI)

1. Authenticate with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```

2. Set your project:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   ```

## Usage

Run the app:

```bash
uv run python main.py
```

Or using the installed script:

```bash
uv run sketch-to-image
```

Then open http://localhost:7860 in your browser.

## How It Works

1. **Upload**: Take a photo of a child's drawing or upload any sketch
2. **Select Style**: Choose from various artistic styles (cartoon, watercolor, etc.)
3. **Choose Provider**: Select OpenAI or Gemini for image generation
4. **Transform**: Click the button and wait for your transformed image!

## Adding New Providers

To add a new image generation provider, create a class that inherits from `ImageProvider`:

```python
from providers import ImageProvider, PROVIDERS

class MyNewProvider(ImageProvider):
    name = "My Provider"
    description = "Description of the provider"

    def generate_from_sketch(
        self, sketch: Image.Image, style: str, prompt: str | None = None
    ) -> Image.Image:
        # Your implementation here
        pass

# Register the provider
PROVIDERS["my_provider"] = MyNewProvider
```

## Supported Models

### OpenAI
- `gpt-image-1.5` (default) - Latest image generation model
- `gpt-image-1` - Previous generation model
- `gpt-image-1-mini` - Smaller, faster model

### Google Gemini (Vertex AI)
- `gemini-3-pro-image-preview` (default) - High-fidelity image generation
- `gemini-2.5-flash-image` - Faster, optimized for speed

## Project Structure

```
detsky_den2026/
├── main.py          # Gradio UI and app entry point
├── providers.py     # Image generation provider implementations
├── pyproject.toml   # Project dependencies
└── README.md        # This file
```
