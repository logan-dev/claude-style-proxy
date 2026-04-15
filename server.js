/**
 * claude-ollama-proxy  v2.0
 * ─────────────────────────────────────────────────────────────
 * Translates Anthropic /v1/messages requests (sent by Claude Code)
 * into Ollama /api/chat  ←→  /v1/chat/completions requests,
 * then maps responses back to Anthropic format.
 *
 * Key improvements over v1:
 *  ✅ Proper Anthropic SSE streaming (message_start → content_block_start
 *     → content_block_delta → content_block_stop → message_delta → message_stop)
 *  ✅ Robust tool-call handling incl. streamed partial JSON assembly
 *  ✅ Modular transformation layer (easy to unit-test each piece)
 *  ✅ Model aliasing via MODEL_MAP env var
 *  ✅ /v1/models endpoint so Claude Code model picker works
 *  ✅ Structured leveled logger with per-request IDs
 *  ✅ Clean error responses in Anthropic error shape
 *  ✅ No axios dependency — uses Node 18+ native fetch
 *  ✅ Graceful shutdown
 */

import 'dotenv/config';
import express from 'express';
import { randomBytes } from 'node:crypto';

// ─── Config ──────────────────────────────────────────────────────────────────

const PORT            = parseInt(process.env.PORT || '4000', 10);
const OLLAMA_BASE_URL = (process.env.OLLAMA_BASE_URL || 'http://localhost:11434').replace(/\/$/, '');
const DEFAULT_MODEL   = process.env.DEFAULT_MODEL || process.env.OLLAMA_MODEL || 'mistral';
const TIMEOUT_MS      = parseInt(process.env.REQUEST_TIMEOUT || '120000', 10);
const LOG_LEVEL       = process.env.LOG_LEVEL || 'info';
const ENABLE_STREAM   = process.env.ENABLE_STREAMING !== 'false';

// Optional JSON map of  { "claude-model-name": "ollama-model-name" }
let MODEL_MAP = {};
try {
  if (process.env.MODEL_MAP) MODEL_MAP = JSON.parse(process.env.MODEL_MAP);
} catch { console.warn('[config] MODEL_MAP is not valid JSON — ignoring'); }

// ─── Logger ───────────────────────────────────────────────────────────────────

const LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };
const CLR    = { debug:'\x1b[36m', info:'\x1b[32m', warn:'\x1b[33m', error:'\x1b[31m', r:'\x1b[0m' };
const curLvl = LEVELS[LOG_LEVEL] ?? LEVELS.info;

function makeLog(prefix = '') {
  const fn = (lvl, msg, meta) => {
    if ((LEVELS[lvl] ?? 0) < curLvl) return;
    const ts  = new Date().toISOString();
    const tag = `${CLR[lvl]}[${lvl.toUpperCase().padEnd(5)}]${CLR.r}`;
    const line = `${ts} ${tag} ${prefix}${msg}`;
    if (meta !== undefined) (console[lvl === 'error' ? 'error' : 'log'])(line, '\n' + JSON.stringify(meta, null, 2));
    else                    (console[lvl === 'error' ? 'error' : 'log'])(line);
  };
  return {
    debug: (m,x) => fn('debug',m,x), info:  (m,x) => fn('info',m,x),
    warn:  (m,x) => fn('warn', m,x), error: (m,x) => fn('error',m,x),
    child: (id) => makeLog(`[${id}] `),
  };
}
const log = makeLog();

// ─── ID helpers ───────────────────────────────────────────────────────────────

const uid   = (prefix) => prefix + randomBytes(8).toString('hex');
const msgId = () => uid('msg_');
const tuId  = () => uid('toolu_');
const reqId = () => uid('req_');

// ─── Model resolution ─────────────────────────────────────────────────────────

function resolveModel(requestedModel, headerOverride) {
  if (headerOverride)                     return headerOverride;
  if (!requestedModel)                    return DEFAULT_MODEL;
  if (MODEL_MAP[requestedModel])          return MODEL_MAP[requestedModel];
  if (!requestedModel.startsWith('claude')) return requestedModel;   // already an ollama name
  return DEFAULT_MODEL;
}

// ─── Request transformation: Anthropic → Ollama ───────────────────────────────

/**
 * Flatten Anthropic content (string | ContentBlock[]) → plain string.
 * Used when building Ollama messages (which only accept string content).
 */
function flattenContent(content) {
  if (!content) return '';
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return String(content);

  return content.map(block => {
    switch (block.type) {
      case 'text':
        return block.text ?? '';
      case 'tool_result': {
        const c = block.content;
        if (typeof c === 'string') return `[Tool Result for ${block.tool_use_id}]: ${c}`;
        if (Array.isArray(c))      return `[Tool Result for ${block.tool_use_id}]: ${c.map(b => b.text ?? JSON.stringify(b)).join('\n')}`;
        return `[Tool Result for ${block.tool_use_id}]: ${JSON.stringify(c)}`;
      }
      case 'tool_use':
        // Serialize tool calls so the model can see what was requested
        return `[Tool Call: ${block.name}(${JSON.stringify(block.input ?? {})})]`;
      case 'image':
        return '[image]';
      default:
        return '';
    }
  }).join('');
}

/**
 * Build the system prompt string.
 * Optionally appends tool-use instructions for models that don't natively support
 * function calling (most Ollama models fall into this bucket unless they were
 * fine-tuned for tools).
 *
 * NOTE: When Ollama is accessed via its OpenAI-compat endpoint (/v1/chat/completions)
 * AND the running model supports function calling (e.g. llama3.1, mistral-nemo),
 * we pass tools natively and skip the text instructions.  For /api/chat we always
 * inject instructions.
 */
// Prepended to every request so models don't refuse agentic tasks
const AGENTIC_PREAMBLE = `You are an expert software engineering assistant running inside Claude Code.
You MUST attempt every task using the tools available to you — never refuse or say you cannot help.
When asked to do something, always try to use a tool rather than just describing what you would do.
If you are unsure, make a best-effort attempt.`.trim();

// Claude Code's system prompt is huge (thousands of tokens of tool schemas).
// At 4096 ctx it consumes everything before the user message even arrives.
// We strip it down to just the first paragraph + any <env> block.
const SYS_TRIM_THRESHOLD = parseInt(process.env.SYS_TRIM_THRESHOLD || '1500', 10);

function trimClaudeCodeSystemPrompt(raw) {
  if (!raw || raw.length < SYS_TRIM_THRESHOLD) return raw;

  // Keep the <env> block — has cwd, platform info that's useful
  const envMatch = raw.match(/<env>([\s\S]*?)<\/env>/);
  const envBlock = envMatch ? envMatch[0] : '';

  // Keep just the first paragraph (usually the most important instruction)
  const firstPara = raw.split('\n\n')[0]?.trim() ?? '';

  const kept = [firstPara, envBlock].filter(Boolean).join('\n\n');
  return kept || raw.slice(0, 600);
}

function buildSystemPrompt(systemInput, tools, injectToolInstructions = false) {
  let rawBase = '';
  if (typeof systemInput === 'string') rawBase = systemInput;
  else if (Array.isArray(systemInput)) rawBase = systemInput.map(b => b.text ?? '').join('\n');
  const trimmedSys = trimClaudeCodeSystemPrompt(rawBase);
  let base = trimmedSys ? `${AGENTIC_PREAMBLE}\n\n${trimmedSys}` : AGENTIC_PREAMBLE;

  if (!injectToolInstructions || !tools?.length) return base || undefined;

  const toolDocs = tools.map(t => {
    const props = t.input_schema?.properties ?? {};
    const req   = t.input_schema?.required ?? [];
    const params = Object.entries(props).map(([name, p]) =>
      `    ${name}${req.includes(name) ? ' (required)' : ''}: ${p.description ?? p.type ?? ''}`
    ).join('\n');
    return `• ${t.name}: ${t.description ?? ''}\n  Parameters:\n${params || '    (none)'}`;
  }).join('\n\n');

  const instructions = `
You have access to the following tools. Call them when the user's request requires it.

${toolDocs}

To call a tool, reply ONLY with a JSON object on a single line in this exact format:
{"type":"tool_use","name":"<tool_name>","id":"<unique_id>","input":{<params>}}

Do not add explanation before or after the JSON when calling a tool.
After receiving a tool result, continue normally.
`.trim();

  return base ? `${base}\n\n${instructions}` : instructions;
}

/**
 * Convert Anthropic messages array → Ollama messages array.
 *
 * Anthropic roles: "user" | "assistant"
 * Ollama roles:    "system" | "user" | "assistant" | "tool"
 *
 * The tricky parts:
 *  - user messages can contain tool_result blocks (not just text)
 *  - assistant messages can contain tool_use blocks
 *  - system prompt is a top-level field in Anthropic but a role in Ollama
 */
function convertMessages(anthropicMessages, systemPrompt) {
  const out = [];

  if (systemPrompt) {
    out.push({ role: 'system', content: systemPrompt });
  }

  for (const msg of anthropicMessages) {
    // ── user turn ──────────────────────────────────────────────────────────
    if (msg.role === 'user') {
      if (Array.isArray(msg.content)) {
        const toolResults = msg.content.filter(b => b.type === 'tool_result');
        const textBlocks  = msg.content.filter(b => b.type !== 'tool_result');

        // Each tool_result becomes an Ollama "tool" role message
        for (const tr of toolResults) {
          let content = '';
          if (typeof tr.content === 'string') content = tr.content;
          else if (Array.isArray(tr.content)) content = tr.content.map(c => c.text ?? JSON.stringify(c)).join('\n');
          else content = JSON.stringify(tr.content ?? '');

          out.push({ role: 'tool', tool_call_id: tr.tool_use_id, content });
        }

        // Remaining text content
        const textContent = textBlocks.map(b => b.text ?? flattenContent(b)).join('');
        if (textContent) out.push({ role: 'user', content: textContent });
      } else {
        out.push({ role: 'user', content: flattenContent(msg.content) });
      }
    }
    // ── assistant turn ────────────────────────────────────────────────────
    else if (msg.role === 'assistant') {
      if (Array.isArray(msg.content)) {
        const textBlocks    = msg.content.filter(b => b.type === 'text');
        const toolUseBlocks = msg.content.filter(b => b.type === 'tool_use');
        const textContent   = textBlocks.map(b => b.text ?? '').join('');

        const ollamaMsg = { role: 'assistant', content: textContent };

        if (toolUseBlocks.length) {
          ollamaMsg.tool_calls = toolUseBlocks.map(tu => ({
            id: tu.id,
            type: 'function',
            function: {
              name: tu.name,
              arguments: JSON.stringify(tu.input ?? {}),
            },
          }));
        }

        out.push(ollamaMsg);
      } else {
        out.push({ role: 'assistant', content: flattenContent(msg.content) });
      }
    }
  }

  return out;
}

/**
 * Convert Anthropic tools → OpenAI/Ollama tools format.
 */
function convertTools(anthropicTools) {
  if (!anthropicTools?.length) return undefined;
  return anthropicTools.map(t => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description ?? '',
      parameters: t.input_schema ?? { type: 'object', properties: {} },
    },
  }));
}

// ─── Response transformation: Ollama → Anthropic ─────────────────────────────

/**
 * Parse tool_calls from an Ollama message object.
 * Returns an array of Anthropic tool_use content blocks.
 */
function parseToolCalls(toolCalls) {
  if (!toolCalls?.length) return [];
  return toolCalls.map(tc => {
    let input = {};
    try {
      const raw = tc.function?.arguments;
      input = typeof raw === 'string' ? JSON.parse(raw) : (raw ?? {});
    } catch { /* keep empty */ }
    return {
      type: 'tool_use',
      id: tc.id ?? tuId(),
      name: tc.function?.name ?? tc.name ?? 'unknown',
      input,
    };
  });
}

/**
 * Map Ollama finish_reason → Anthropic stop_reason.
 */
function mapStopReason(finishReason, hasToolUse) {
  if (hasToolUse)                return 'tool_use';
  if (finishReason === 'stop')   return 'end_turn';
  if (finishReason === 'length') return 'max_tokens';
  return 'end_turn';
}

/**
 * Build a complete Anthropic response object from an Ollama non-streaming response.
 */
function buildAnthropicResponse(ollamaData, requestedModel, messageId) {
  const choice  = ollamaData.choices?.[0];
  const message = choice?.message ?? {};
  const content = [];

  if (message.content) {
    // Some models return tool_use as JSON in the text field when native tools aren't supported
    const parsed = tryParseFallbackToolUse(message.content);
    if (parsed) {
      content.push(parsed);
    } else {
      content.push({ type: 'text', text: message.content });
    }
  }

  const toolUses = parseToolCalls(message.tool_calls);
  content.push(...toolUses);

  return {
    id:            messageId ?? msgId(),
    type:          'message',
    role:          'assistant',
    content,
    model:         requestedModel,
    stop_reason:   mapStopReason(choice?.finish_reason, toolUses.length > 0),
    stop_sequence: null,
    usage: {
      input_tokens:  ollamaData.usage?.prompt_tokens     ?? 0,
      output_tokens: ollamaData.usage?.completion_tokens ?? 0,
    },
  };
}

/**
 * Some models (without native tool support) emit tool calls as raw JSON text.
 * Try to detect and parse that pattern so Claude Code still gets tool_use blocks.
 */
function tryParseFallbackToolUse(text) {
  if (!text?.includes('"type":"tool_use"') && !text?.includes('"type": "tool_use"')) return null;
  try {
    const trimmed = text.trim();
    // Handle JSON possibly wrapped in markdown fences
    const jsonStr = trimmed.startsWith('```')
      ? trimmed.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim()
      : trimmed;
    const obj = JSON.parse(jsonStr);
    if (obj.type === 'tool_use' && obj.name) {
      return {
        type: 'tool_use',
        id:   obj.id ?? tuId(),
        name: obj.name,
        input: typeof obj.input === 'object' ? obj.input : {},
      };
    }
  } catch { /* not a tool call */ }
  return null;
}

// ─── Streaming ────────────────────────────────────────────────────────────────

/**
 * Proxy a streaming Ollama response back to the client using the full
 * Anthropic SSE event sequence that Claude Code expects:
 *
 *   message_start
 *   content_block_start  (index 0, type=text)
 *   content_block_delta* (text_delta chunks)
 *   content_block_stop   (index 0)
 *   [ for each tool call:
 *     content_block_start  (index N, type=tool_use)
 *     content_block_delta* (input_json_delta chunks)
 *     content_block_stop   (index N) ]
 *   message_delta        (stop_reason, usage)
 *   message_stop
 */
async function handleStreaming(res, ollamaStreamResponse, requestedModel, messageId, log) {
  const writeSSE = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  // Kick off the message
  writeSSE({
    type: 'message_start',
    message: {
      id: messageId, type: 'message', role: 'assistant',
      content: [], model: requestedModel, stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  // Open text block
  writeSSE({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });

  let textAccum       = '';
  let toolCallMap     = {};   // index → { id, name, argsAccum }
  let nextBlockIndex  = 1;
  let inputTokens     = 0;
  let outputTokens    = 0;
  let finishReason    = 'stop';
  let buffer          = '';

  const reader = ollamaStreamResponse.body;
  const decoder = new TextDecoder();

  try {
    for await (const rawChunk of reader) {
      buffer += decoder.decode(rawChunk, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;

        const jsonStr = trimmed.slice(6);
        if (jsonStr === '[DONE]') continue;

        let chunk;
        try { chunk = JSON.parse(jsonStr); } catch { continue; }

        const delta       = chunk.choices?.[0]?.delta;
        const choiceStop  = chunk.choices?.[0]?.finish_reason;
        if (choiceStop) finishReason = choiceStop;

        if (chunk.usage) {
          inputTokens  = chunk.usage.prompt_tokens     ?? inputTokens;
          outputTokens = chunk.usage.completion_tokens ?? outputTokens;
        }

        if (!delta) continue;

        // ── Text delta ────────────────────────────────────────────────────
        if (typeof delta.content === 'string' && delta.content) {
          textAccum += delta.content;
          writeSSE({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: delta.content } });
        }

        // ── Tool-call delta ───────────────────────────────────────────────
        if (delta.tool_calls?.length) {
          for (const tc of delta.tool_calls) {
            const tcIdx = tc.index ?? 0;

            if (!toolCallMap[tcIdx]) {
              // New tool call — open a block
              const blockIndex = nextBlockIndex++;
              toolCallMap[tcIdx] = { blockIndex, id: tc.id ?? tuId(), name: tc.function?.name ?? '', argsAccum: '' };
              writeSSE({
                type: 'content_block_start',
                index: blockIndex,
                content_block: { type: 'tool_use', id: toolCallMap[tcIdx].id, name: toolCallMap[tcIdx].name, input: {} },
              });
            }

            const entry = toolCallMap[tcIdx];
            if (tc.function?.name && !entry.name) entry.name = tc.function.name;
            if (tc.id && !entry.id.startsWith('toolu_')) entry.id = tc.id;

            if (tc.function?.arguments) {
              entry.argsAccum += tc.function.arguments;
              writeSSE({
                type: 'content_block_delta',
                index: entry.blockIndex,
                delta: { type: 'input_json_delta', partial_json: tc.function.arguments },
              });
            }
          }
        }
      }
    }
  } catch (err) {
    log.error('Stream read error', { message: err.message });
  }

  // Close text block
  writeSSE({ type: 'content_block_stop', index: 0 });

  // Close tool-call blocks
  for (const entry of Object.values(toolCallMap)) {
    writeSSE({ type: 'content_block_stop', index: entry.blockIndex });
  }

  const hasTools   = Object.keys(toolCallMap).length > 0;
  const stopReason = mapStopReason(finishReason, hasTools);

  // message_delta carries the stop reason
  writeSSE({
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });

  writeSSE({ type: 'message_stop' });
  res.end();

  log.info(`Stream complete — stop_reason=${stopReason} tools=${hasTools} out_tokens=${outputTokens}`);
}

// ─── Ollama fetch helpers ─────────────────────────────────────────────────────

async function fetchOllama(path, body, signal) {
  const url = `${OLLAMA_BASE_URL}${path}`;
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
    signal,
  });
  return res;
}

// ─── Express App ─────────────────────────────────────────────────────────────

const app = express();
app.use(express.json({ limit: '50mb' }));

// Per-request logger middleware
app.use((req, _res, next) => {
  req.log = log.child(reqId());
  next();
});

// ── POST /v1/messages ─────────────────────────────────────────────────────────

app.post('/v1/messages', async (req, res) => {
  const rlog = req.log;

  const {
    model:           requestedModel = DEFAULT_MODEL,
    max_tokens       = 4096,
    system,
    messages         = [],
    tools,
    tool_choice,
    stream           = false,
    temperature,
    top_p,
    stop_sequences,
  } = req.body;

  rlog.info(`→ POST /v1/messages  model=${requestedModel} stream=${stream} tools=${tools?.length ?? 0} msgs=${messages.length}`);
  rlog.debug('Request body', req.body);

  // ── Validation ──────────────────────────────────────────────────────────────
  if (!messages.length) {
    return res.status(400).json(anthropicError('invalid_request_error', 'messages array must not be empty'));
  }

  // ── Model resolution ────────────────────────────────────────────────────────
  const ollamaModel   = resolveModel(requestedModel, req.headers['x-ollama-model']);
  rlog.info(`  model resolved: ${requestedModel} → ${ollamaModel}`);

  // ── Log system prompt size for diagnosis ──────────────────────────────────
  const rawSysLen = typeof system === 'string' ? system.length
    : Array.isArray(system) ? system.map(b=>b.text??'').join('').length : 0;
  rlog.info(`  system prompt raw=${rawSysLen} chars  tools=${tools?.length??0}`);

  // ── Strip oversized tool schemas ────────────────────────────────────────────
  // Ollama serializes full tool JSON schemas into the system prompt internally.
  // Claude Code sends 20+ tools with large schemas — this alone fills 4096 tokens.
  // Solution: keep tools but strip their schema 'description' fields down to a
  // single line so the model knows the tool exists without the token cost.
  const MAX_TOOLS = parseInt(process.env.MAX_TOOLS || '20', 10);
  const SLIM_TOOLS = process.env.SLIM_TOOLS !== 'false'; // default true
  function slimTools(toolList) {
  if (!toolList?.length) return toolList;

  return toolList.slice(0, MAX_TOOLS).map(t => ({
    ...t,
    description: (t.description ?? '').split('\n')[0].slice(0, 120),
    input_schema: {
      ...t.input_schema,
      properties: Object.fromEntries(
        Object.entries(t.input_schema?.properties ?? {}).map(([key, val]) => [
          key,
          {
            ...val,
            description: (val.description ?? '').split('\n')[0].slice(0, 80)
          }
        ])
      )
    }
  }));
}
  const effectiveTools = SLIM_TOOLS ? slimTools(tools) : tools;

  // ── Decide if we need fallback tool instructions in the system prompt ───────
  // Ollama's OpenAI-compat layer supports native tools for certain models.
  // We always try native tools first; only inject text instructions if needed.
  const useNativeTools     = !!(effectiveTools?.length);
  const injectToolHints    = false; // set to true if your model ignores native tools

  const systemPrompt = buildSystemPrompt(system, effectiveTools, injectToolHints);
  const ollamaMessages = convertMessages(messages, systemPrompt);
  const ollamaTools    = useNativeTools ? convertTools(effectiveTools) : undefined;

  const ollamaBody = {
    model:    ollamaModel,
    messages: ollamaMessages,
    stream:   ENABLE_STREAM && stream,
    options:  {
      ...(temperature !== undefined && { temperature }),
      ...(top_p       !== undefined && { top_p }),
      num_predict: max_tokens ?? 4096,
      num_ctx:     parseInt(process.env.NUM_CTX || '32768', 10),
      ...(stop_sequences?.length && { stop: stop_sequences }),
    },
    ...(ollamaTools && { tools: ollamaTools }),
  };

  rlog.debug('Ollama request body', ollamaBody);

  const controller = new AbortController();
  const timeout    = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const ollamaRes = await fetchOllama('/v1/chat/completions', ollamaBody, controller.signal);

    if (!ollamaRes.ok) {
      const errText = await ollamaRes.text();
      rlog.error(`Ollama HTTP ${ollamaRes.status}`, errText);
      return res.status(502).json(anthropicError('api_error', `Upstream Ollama error ${ollamaRes.status}: ${errText}`));
    }

    // ── Streaming ──────────────────────────────────────────────────────────
    if (ENABLE_STREAM && stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders();
      await handleStreaming(res, ollamaRes, requestedModel, msgId(), rlog);
      return;
    }

    // ── Non-streaming ──────────────────────────────────────────────────────
    const data          = await ollamaRes.json();
    rlog.debug('Ollama response', data);

    const anthropicResp = buildAnthropicResponse(data, requestedModel, msgId());
    rlog.info(`← 200  stop_reason=${anthropicResp.stop_reason}  tools_in_response=${anthropicResp.content.filter(b=>b.type==='tool_use').length}`);
    res.json(anthropicResp);

  } catch (err) {
    if (err.name === 'AbortError') {
      rlog.error('Request timed out');
      return res.status(504).json(anthropicError('api_error', `Request to Ollama timed out after ${TIMEOUT_MS}ms`));
    }
    rlog.error('Proxy error', { message: err.message, stack: err.stack });
    res.status(500).json(anthropicError('internal_server_error', err.message));
  } finally {
    clearTimeout(timeout);
  }
});

// ── GET /v1/models ────────────────────────────────────────────────────────────
// Claude Code calls this on startup. We proxy it from Ollama and dress it up.

app.get('/v1/models', async (req, res) => {
  try {
    const ollamaRes = await fetch(`${OLLAMA_BASE_URL}/api/tags`);
    if (!ollamaRes.ok) throw new Error(`Ollama /api/tags returned ${ollamaRes.status}`);
    const { models = [] } = await ollamaRes.json();

    const data = models.map(m => ({
      id:       m.name,
      object:   'model',
      created:  Math.floor(new Date(m.modified_at ?? Date.now()).getTime() / 1000),
      owned_by: 'ollama',
    }));

    // Also expose the alias entries so Claude Code can see mapped names
    for (const [alias] of Object.entries(MODEL_MAP)) {
      if (!data.find(m => m.id === alias)) {
        data.push({ id: alias, object: 'model', created: 0, owned_by: 'alias' });
      }
    }

    res.json({ object: 'list', data });
  } catch (err) {
    log.warn('Could not fetch models from Ollama', err.message);
    res.json({ object: 'list', data: [{ id: DEFAULT_MODEL, object: 'model', created: 0, owned_by: 'ollama' }] });
  }
});

// ── GET /health ───────────────────────────────────────────────────────────────

app.get('/health', async (_req, res) => {
  let ollamaOk = false;
  try {
    const r = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: AbortSignal.timeout(3000) });
    ollamaOk = r.ok;
  } catch { /* unreachable */ }

  res.status(ollamaOk ? 200 : 502).json({
    status:      ollamaOk ? 'ok' : 'degraded',
    proxy:       'claude-ollama-proxy v2',
    ollama_url:  OLLAMA_BASE_URL,
    ollama_ok:   ollamaOk,
    default_model: DEFAULT_MODEL,
    model_aliases: MODEL_MAP,
    streaming:   ENABLE_STREAM,
  });
});

// ─── Error shape helper ───────────────────────────────────────────────────────

function anthropicError(type, message) {
  return { type: 'error', error: { type, message } };
}

// ─── Start ────────────────────────────────────────────────────────────────────

const server = app.listen(PORT, () => {
  log.info(`🚀 claude-ollama-proxy v2 listening on :${PORT}`);
  log.info(`📡 Ollama backend : ${OLLAMA_BASE_URL}`);
  log.info(`🤖 Default model  : ${DEFAULT_MODEL}`);
  log.info(`📺 Streaming      : ${ENABLE_STREAM ? 'enabled' : 'disabled'}`);
  if (Object.keys(MODEL_MAP).length) log.info('🗺️  Model aliases  :', MODEL_MAP);
});

// Graceful shutdown
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => {
    log.info(`${sig} received — shutting down…`);
    server.close(() => process.exit(0));
  });
}
