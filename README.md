# @elizaos/plugin-google-genai

Google Generative AI integration for ElizaOS, supporting the latest Google Gemini 2.5 and 3 series models.

## Installation

```bash
npm install @elizaos/plugin-google-genai
```

## Configuration

1. Get your Google AI API key from [Google AI Studio](https://aistudio.google.com/)
2. Set the API key in your environment. Both the new `GOOGLE_GEMINI_API_KEY` and legacy `GOOGLE_GENERATIVE_AI_API_KEY` are supported.

```bash
GOOGLE_GEMINI_API_KEY=your_api_key_here
```

## Usage

Add the plugin to your character configuration:

```json
{
  "plugins": ["@elizaos/plugin-google-genai"]
}
```

## Supported Models (February 2026)

- **Text Generation**:
  - Small: `gemini-3-flash` (default) - Fast and cost-effective performance.
  - Large: `gemini-3.1-pro` (default) - Advanced reasoning and complex problem-solving.
- **Text Embeddings**: `text-embedding-004` (default)
- **Image Analysis**: `gemini-3-flash` (default)

## Environment Variables

- `GOOGLE_GEMINI_API_KEY` (required): Your Google Gemini API key
- `GOOGLE_GENERATIVE_AI_API_KEY` (legacy): Fallback API key
- `GOOGLE_SMALL_MODEL` (optional): Override small model (default: `gemini-3-flash`)
- `GOOGLE_LARGE_MODEL` (optional): Override large model (default: `gemini-3.1-pro`)
- `GOOGLE_IMAGE_MODEL` (optional): Override image model (default: `gemini-3-flash`)
- `GOOGLE_EMBEDDING_MODEL` (optional): Override embedding model (default: `text-embedding-004`)

## Model Types Provided

- `TEXT_SMALL` - Fast text generation using Gemini 3 Flash
- `TEXT_LARGE` - High-quality reasoning using Gemini 3.1 Pro
- `TEXT_EMBEDDING` - Text embeddings for similarity search
- `OBJECT_SMALL` - JSON object generation (small model)
- `OBJECT_LARGE` - Complex JSON object generation (large model)
- `IMAGE_DESCRIPTION` - Advanced image analysis and description

## Features

- Native integration with the latest Google Gemini 3 series models
- Massive 2M+ context window support (depending on model)
- State-of-the-art vision and reasoning capabilities
- Backward compatibility for legacy configuration
- Automated token usage tracking
- System instruction support from character configuration
