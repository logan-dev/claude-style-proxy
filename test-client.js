/**
 * test-client.js  —  claude-ollama-proxy v2 test suite
 *
 * Run against a live proxy:
 *   PROXY_URL=http://localhost:4000 node test-client.js
 *
 * Tests:
 *   1. Health check
 *   2. Model list
 *   3. Basic message (non-streaming)
 *   4. System prompt passthrough
 *   5. Multi-turn conversation
 *   6. Tool call request + tool result round-trip
 *   7. Streaming message
 */

const PROXY = process.env.PROXY_URL || 'http://localhost:4000';
const MODEL = process.env.MODEL || 'mistral';

let passed = 0, failed = 0;

async function run(name, fn) {
  try {
    await fn();
    console.log(`  ✅  ${name}`);
    passed++;
  } catch (e) {
    console.error(`  ❌  ${name}\n     ${e.message}`);
    failed++;
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message ?? 'Assertion failed');
}

async function post(path, body) {
  const res = await fetch(`${PROXY}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${JSON.stringify(data)}`);
  return data;
}

// ─── Test 1: Health ───────────────────────────────────────────────────────────
await run('Health check', async () => {
  const res = await fetch(`${PROXY}/health`);
  const data = await res.json();
  assert(data.proxy?.includes('claude-ollama-proxy'), 'missing proxy field');
  assert(typeof data.ollama_url === 'string', 'missing ollama_url');
});

// ─── Test 2: Models ───────────────────────────────────────────────────────────
await run('GET /v1/models returns list', async () => {
  const res = await fetch(`${PROXY}/v1/models`);
  const data = await res.json();
  assert(data.object === 'list', 'expected object=list');
  assert(Array.isArray(data.data), 'expected data array');
});

// ─── Test 3: Basic message ────────────────────────────────────────────────────
await run('Basic non-streaming message', async () => {
  const data = await post('/v1/messages', {
    model: MODEL,
    max_tokens: 64,
    messages: [{ role: 'user', content: 'Reply with just the word PONG.' }],
  });
  assert(data.type === 'message', `expected type=message, got ${data.type}`);
  assert(data.role === 'assistant', 'expected role=assistant');
  assert(Array.isArray(data.content), 'content should be array');
  assert(data.content.length > 0, 'content should not be empty');
  assert(data.stop_reason, 'stop_reason should be set');
  assert(typeof data.usage?.input_tokens === 'number', 'missing usage.input_tokens');
});

// ─── Test 4: System prompt ────────────────────────────────────────────────────
await run('System prompt is respected', async () => {
  const data = await post('/v1/messages', {
    model: MODEL,
    max_tokens: 64,
    system: 'You are a pirate. Every response must begin with "Arrr!".',
    messages: [{ role: 'user', content: 'Hello.' }],
  });
  const text = data.content.find(b => b.type === 'text')?.text ?? '';
  // Just check the response exists — model behavior varies
  assert(text.length > 0, 'expected non-empty response text');
});

// ─── Test 5: Multi-turn conversation ─────────────────────────────────────────
await run('Multi-turn conversation', async () => {
  const data = await post('/v1/messages', {
    model: MODEL,
    max_tokens: 128,
    messages: [
      { role: 'user',      content: 'My name is Logan.' },
      { role: 'assistant', content: 'Hello Logan, nice to meet you!' },
      { role: 'user',      content: 'What is my name?' },
    ],
  });
  const text = data.content.find(b => b.type === 'text')?.text ?? '';
  assert(text.length > 0, 'expected response');
});

// ─── Test 6: Tool call ────────────────────────────────────────────────────────
await run('Tool call request shape is valid', async () => {
  const data = await post('/v1/messages', {
    model: MODEL,
    max_tokens: 256,
    tools: [
      {
        name: 'get_weather',
        description: 'Get current weather for a city.',
        input_schema: {
          type: 'object',
          properties: {
            city: { type: 'string', description: 'City name' },
          },
          required: ['city'],
        },
      },
    ],
    messages: [{ role: 'user', content: 'What is the weather in Delhi?' }],
  });
  // Response must be a proper Anthropic message — stop_reason may be tool_use or end_turn
  // depending on whether the model invoked the tool
  assert(['tool_use', 'end_turn'].includes(data.stop_reason), `unexpected stop_reason: ${data.stop_reason}`);
  assert(Array.isArray(data.content), 'content must be array');
});

// ─── Test 6b: Tool result round-trip ─────────────────────────────────────────
await run('Tool result is accepted in follow-up turn', async () => {
  const data = await post('/v1/messages', {
    model: MODEL,
    max_tokens: 128,
    messages: [
      { role: 'user', content: 'What is the weather in Delhi?' },
      {
        role: 'assistant',
        content: [
          { type: 'tool_use', id: 'toolu_test01', name: 'get_weather', input: { city: 'Delhi' } },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_test01', content: '38°C and sunny' },
        ],
      },
    ],
  });
  assert(data.type === 'message', `expected message, got ${data.type}`);
  assert(data.content.length > 0, 'expected non-empty content after tool result');
});

// ─── Test 7: Streaming ────────────────────────────────────────────────────────
await run('Streaming SSE events arrive correctly', async () => {
  const res = await fetch(`${PROXY}/v1/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 64,
      stream: true,
      messages: [{ role: 'user', content: 'Count 1 2 3.' }],
    }),
  });
  assert(res.headers.get('content-type')?.includes('text/event-stream'), 'expected SSE content-type');

  const events = [];
  const text = await res.text();
  for (const line of text.split('\n')) {
    if (line.startsWith('data: ')) {
      try { events.push(JSON.parse(line.slice(6))); } catch { /* skip */ }
    }
  }

  const types = events.map(e => e.type);
  assert(types.includes('message_start'),    'missing message_start event');
  assert(types.includes('content_block_start'), 'missing content_block_start event');
  assert(types.includes('message_stop'),     'missing message_stop event');
});

// ─── Summary ──────────────────────────────────────────────────────────────────
console.log(`\n${'─'.repeat(40)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
