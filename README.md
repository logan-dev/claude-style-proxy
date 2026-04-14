# claude-ollama-proxy v2

A production-grade Node.js proxy that lets **Claude Code** (and any Anthropic SDK client) run against a **local or remote Ollama instance** — with full support for tools, streaming, and multi-turn conversations.

```
Claude Code  ──→  localhost:4000 (this proxy)  ──→  Ollama (local or RunPod)
   Anthropic /v1/messages                          /v1/chat/completions
```

---

## What's new in v2

| Feature | v1 | v2 |
|---|---|---|
| SSE streaming event sequence | Partial / incorrect | Full Anthropic spec (`message_start` → `content_block_*` → `message_stop`) |
| Tool-call handling | Basic | Native tools + fallback JSON detection |
| Tool result round-trip | Basic | Full `tool_result` → Ollama `tool` role mapping |
| Multi-turn tool loops | ❌ | ✅ |
| Model alias map | ❌ | ✅ via `MODEL_MAP` env |
| `/v1/models` endpoint | ❌ | ✅ proxied from Ollama `/api/tags` |
| HTTP client | axios | Native `fetch` (Node 18+, zero extra deps) |
| Logging | `console.log` | Leveled, colored, per-request IDs |
| Error responses | Mixed | Always Anthropic error shape |
| Graceful shutdown | ❌ | ✅ |

---

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/logan-dev/claude-style-proxy
cd claude-style-proxy
npm install

# 2. Configure
cp .env.example .env
# Edit .env — set OLLAMA_BASE_URL to your remote host

# 3. Run
npm start

# 4. Point Claude Code at the proxy
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=ollama          # any non-empty string
claude
```

---

## Configuration

All options are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `4000` | Port the proxy listens on |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL of your Ollama server |
| `DEFAULT_MODEL` | `mistral` | Ollama model used when Claude Code sends a `claude-*` name |
| `MODEL_MAP` | `{}` | JSON map of `claude-name → ollama-name`, e.g. `{"claude-3-opus-20240229":"mixtral"}` |
| `ENABLE_STREAMING` | `true` | Set `false` to disable SSE |
| `REQUEST_TIMEOUT` | `120000` | Upstream timeout in ms |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |

### Model aliases example

```env
MODEL_MAP={"claude-3-opus-20240229":"mixtral","claude-3-sonnet-20240229":"mistral","claude-3-haiku-20240307":"llama3.2"}
```

### Per-request model override

You can override the model for a single request by sending the `X-Ollama-Model` header:

```bash
curl http://localhost:4000/v1/messages \
  -H "X-Ollama-Model: llama3.3:70b" \
  -d '{ "model": "claude-3-opus", "messages": [...] }'
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Main proxy endpoint (Anthropic → Ollama) |
| `GET` | `/v1/models` | Lists available Ollama models in Anthropic format |
| `GET` | `/health` | Health check — also verifies Ollama connectivity |

---

## How tool calls work

Claude Code sends tools like this:

```json
{
  "tools": [{ "name": "bash", "input_schema": { ... } }],
  "messages": [{ "role": "user", "content": "List files in /tmp" }]
}
```

The proxy:
1. Converts tools to OpenAI function format and sends them to Ollama natively
2. Maps Ollama `tool_calls` back to Anthropic `tool_use` content blocks
3. On the next turn, maps `tool_result` blocks → Ollama `tool` role messages
4. Also detects fallback JSON tool calls from models that don't support native tools

---

## Testing

```bash
# Run against a live proxy
PROXY_URL=http://localhost:4000 MODEL=mistral node test-client.js
```

Tests cover: health, model list, basic message, system prompt, multi-turn, tool call, tool result round-trip, and streaming.

---

## Shell alias (Claude Code integration)

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
alias claude-local='ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=ollama claude'
```

Then just run `claude-local` in any project directory.

---

## Requirements

- Node.js ≥ 18 (uses native `fetch` and async iterators — no `axios` needed)
- A running Ollama instance (local or remote RunPod)
