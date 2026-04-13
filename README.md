# Ollama Anthropic Proxy

A Node.js proxy service that converts Anthropic-style API requests to Ollama-compatible format.

## Features

- ✅ Accepts Anthropic `/v1/messages` format
- ✅ Transforms to Ollama `/v1/chat/completions` (OpenAI-compatible)
- ✅ Streaming responses support
- ✅ Tool/function calling support
- ✅ System prompts handling
- ✅ Multiple messages support
- ✅ Configurable remote Ollama URL and model

## Setup

```bash
npm install
```

## Configuration

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
PORT=3000
```

## Usage

```bash
npm start
```

## API Endpoint

### POST /v1/messages

Accepts Anthropic-format requests and returns Anthropic-format responses.

#### Example Request

```json
{
  "model": "claude-3-sonnet-20240229",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather today?"
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city name"
          }
        },
        "required": ["location"]
      }
    }
  ]
}
```

#### Example Response

```json
{
  "id": "msg_xxxxxxxxxxxxx",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Let me check the weather for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_xxxxxxxxxxxxx",
      "name": "get_weather",
      "input": {
        "location": "New York"
      }
    }
  ],
  "model": "llama3.1",
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 50,
    "output_tokens": 30
  }
}
```

## Streaming

Add `"stream": true` to your request to receive Server-Sent Events (SSE).
