const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config({
  path: require('path').resolve(__dirname, '.env')
});

const app = express();
const PORT = process.env.PORT ;
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ;
const DEFAULT_MODEL = process.env.OLLAMA_MODEL ;
console.log('Using Ollama Base URL:', OLLAMA_BASE_URL);
console.log('Using Default Model:', DEFAULT_MODEL); 

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Generate unique IDs
function generateMessageId() {
  return 'msg_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

function generateToolUseId() {
  return 'toolu_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

/**
 * Convert Anthropic tools format to OpenAI functions format
 */
function convertToolsToFunctions(tools) {
  if (!tools || !Array.isArray(tools)) {
    return undefined;
  }
  
  return tools.map(tool => ({
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema
  }));
}

/**
 * Flatten Anthropic content array to string
 */
function flattenContent(content) {
  if (typeof content === 'string') {
    return content;
  }
  
  if (Array.isArray(content)) {
    return content
      .map(block => {
        if (typeof block === 'string') return block;
        if (block.type === 'text') return block.text;
        if (block.type === 'tool_use') {
          return `[Tool Use: ${block.name}]`;
        }
        if (block.type === 'tool_result') {
          if (typeof block.content === 'string') {
            return `Tool Result: ${block.content}`;
          }
          if (Array.isArray(block.content)) {
            return `Tool Result: ${block.content.map(c => c.text || JSON.stringify(c)).join('')}`;
          }
          return `Tool Result: ${JSON.stringify(block.content)}`;
        }
        return '';
      })
      .join('');
  }
  
  return String(content || '');
}

/**
 * Convert Anthropic messages to OpenAI format
 */
function convertMessages(anthropicMessages, systemPrompt) {
  const openaiMessages = [];
  
  // Add system message if present
  if (systemPrompt) {
    openaiMessages.push({
      role: 'system',
      content: typeof systemPrompt === 'string' ? systemPrompt : flattenContent(systemPrompt)
    });
  }
  
  // Convert each message
  for (const msg of anthropicMessages) {
    const content = flattenContent(msg.content);
    
    // Handle tool results - they come as assistant messages with tool_result blocks
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      const toolResultBlocks = msg.content.filter(b => b.type === 'tool_result');
      const textBlocks = msg.content.filter(b => b.type === 'text');
      
      if (toolResultBlocks.length > 0) {
        // For tool results, we need to add them as separate messages
        for (const toolResult of toolResultBlocks) {
          const toolContent = toolResult.content;
          let contentStr;
          if (typeof toolContent === 'string') {
            contentStr = toolContent;
          } else if (Array.isArray(toolContent)) {
            contentStr = toolContent.map(c => c.text || JSON.stringify(c)).join('');
          } else {
            contentStr = JSON.stringify(toolContent);
          }
          
          openaiMessages.push({
            role: 'tool',
            tool_call_id: toolResult.tool_use_id,
            content: contentStr
          });
        }
        
        // Add any text content
        if (textBlocks.length > 0) {
          openaiMessages.push({
            role: 'user',
            content: textBlocks.map(b => b.text).join('')
          });
        }
        continue;
      }
    }
    
    openaiMessages.push({
      role: msg.role,
      content: content
    });
  }
  
  return openaiMessages;
}

/**
 * Convert OpenAI tool calls to Anthropic tool_use format
 */
function convertToolCallsToAnthropic(toolCalls) {
  if (!toolCalls || !Array.isArray(toolCalls)) {
    return [];
  }
  
  return toolCalls.map(tc => ({
    type: 'tool_use',
    id: tc.id || generateToolUseId(),
    name: tc.function?.name || tc.name,
    input: (() => {
      try {
        if (tc.function?.arguments) {
          return JSON.parse(tc.function.arguments);
        }
        if (tc.arguments) {
          return typeof tc.arguments === 'string' ? JSON.parse(tc.arguments) : tc.arguments;
        }
        return {};
      } catch (e) {
        return {};
      }
    })()
  }));
}

/**
 * Build enhanced system prompt based on available tools
 */
function buildEnhancedSystemPrompt(baseSystemPrompt, tools) {
  let systemPrompt = baseSystemPrompt || '';
  
  if (tools && tools.length > 0) {
    const toolDescriptions = tools.map(tool => {
      const params = tool.input_schema || tool.parameters || {};
      const requiredParams = params.required || [];
      const paramList = Object.entries(params.properties || {}).map(([name, prop]) => {
        const req = requiredParams.includes(name) ? ' (required)' : '';
        return `  - ${name}: ${prop.description || 'No description'}${req}`;
      }).join('\n');
      
      return `${tool.name}: ${tool.description || 'No description'}\n${paramList}`;
    }).join('\n\n');
    
    const toolInstructions = `
You have access to the following tools. When the user's request matches a tool's purpose, 
you MUST use the appropriate tool by calling it with the correct parameters.

Available Tools:
${toolDescriptions}

To use a tool, respond with a tool_use block in this format:
{
  "type": "tool_use",
  "name": "tool_name",
  "input": { "param1": "value1" }
}

Always provide all required parameters when using a tool.
`;
    
    systemPrompt = systemPrompt ? `${systemPrompt}\n\n${toolInstructions}` : toolInstructions;
  }
  
  return systemPrompt;
}

/**
 * Main endpoint - Anthropic-compatible /v1/messages
 */
app.post('/v1/messages', async (req, res) => {
  try {
    const {
      model = DEFAULT_MODEL,
      max_tokens = 1024,
      system,
      messages = [],
      tools,
      stream = false,
      temperature,
      top_p,
      stop_sequences
    } = req.body;

    console.log('Received request:', JSON.stringify(req.body));
    
    // Validate request
    if (!messages || messages.length === 0) {
      return res.status(400).json({
        error: {
          type: 'invalid_request_error',
          message: 'At least one message is required'
        }
      });
    }
    
    // Build enhanced system prompt with tool instructions
    const enhancedSystemPrompt = buildEnhancedSystemPrompt(system, tools);
    
    // Convert messages to OpenAI format
    const openaiMessages = convertMessages(messages, enhancedSystemPrompt);
    
    // Convert tools to OpenAI functions format
    const functions = convertToolsToFunctions(tools);
    
    // Prepare Ollama request
    const ollamaRequest = {
      model: model,
      messages: openaiMessages,
      stream: stream,
      options: {
        temperature: temperature,
        top_p: top_p,
        num_predict: max_tokens,
        stop: stop_sequences
      }
    };
    
    // Add functions/tools if present
    if (functions && functions.length > 0) {
      ollamaRequest.tools = functions;
    }
    
    console.log('Sending to Ollama:', JSON.stringify(ollamaRequest, null, 2));
    
    // Handle streaming
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      const messageId = generateMessageId();
      let accumulatedContent = '';
      let accumulatedToolCalls = [];
      let inputTokens = 0;
      let outputTokens = 0;
      
      const response = await axios.post(`${OLLAMA_BASE_URL}/v1/chat/completions`, ollamaRequest, {
        responseType: 'stream',
        timeout: 120000
      });
      
      response.data.on('data', async (chunk) => {
        const lines = chunk.toString().split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              // Send final message
              const finalContent = [];
              
              if (accumulatedContent.trim()) {
                finalContent.push({
                  type: 'text',
                  text: accumulatedContent.trim()
                });
              }
              
              for (const tc of accumulatedToolCalls) {
                finalContent.push(tc);
              }
              
              const stopReason = accumulatedToolCalls.length > 0 ? 'tool_use' : 'end_turn';
              
              res.write(`data: ${JSON.stringify({
                type: 'message_stop',
                message: {
                  id: messageId,
                  type: 'message',
                  role: 'assistant',
                  content: finalContent,
                  model: model,
                  stop_reason: stopReason,
                  stop_sequence: null,
                  usage: {
                    input_tokens: inputTokens,
                    output_tokens: outputTokens
                  }
                }
              })}\n\n`);
              res.end();
              return;
            }
            
            try {
              const parsed = JSON.parse(data);
              const delta = parsed.choices?.[0]?.delta;
              
              if (delta) {
                if (delta.content) {
                  accumulatedContent += delta.content;
                  res.write(`data: ${JSON.stringify({
                    type: 'content_block_delta',
                    index: 0,
                    delta: {
                      type: 'text_delta',
                      text: delta.content
                    }
                  })}\n\n`);
                }
                
                if (delta.tool_calls) {
                  for (const tc of delta.tool_calls) {
                    const toolCall = {
                      type: 'tool_use',
                      id: tc.id || generateToolUseId(),
                      name: tc.function?.name || tc.name,
                      input: {}
                    };
                    
                    if (tc.function?.arguments) {
                      try {
                        toolCall.input = JSON.parse(tc.function.arguments);
                      } catch (e) {
                        // Partial arguments, will be completed later
                      }
                    }
                    
                    accumulatedToolCalls.push(toolCall);
                    
                    res.write(`data: ${JSON.stringify({
                      type: 'content_block_start',
                      index: accumulatedToolCalls.length,
                      content_block: {
                        type: 'tool_use',
                        id: toolCall.id,
                        name: toolCall.name,
                        input: toolCall.input
                      }
                    })}\n\n`);
                  }
                }
              }
              
              // Track token usage
              if (parsed.usage) {
                inputTokens = parsed.usage.prompt_tokens || 0;
                outputTokens = parsed.usage.completion_tokens || 0;
              }
            } catch (e) {
              console.error('Error parsing stream chunk:', e);
            }
          }
        }
      });
      
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.write(`data: ${JSON.stringify({
          type: 'error',
          error: {
            type: 'internal_server_error',
            message: err.message
          }
        })}\n\n`);
        res.end();
      });
      
      return;
    }
    
    // Non-streaming request
    const ollamaResponse = await axios.post(`${OLLAMA_BASE_URL}/v1/chat/completions`, ollamaRequest, {
      timeout: 120000
    });
    
    console.log('Ollama response:', JSON.stringify(ollamaResponse.data, null, 2));
    
    const choice = ollamaResponse.data.choices?.[0];
    const message = choice?.message;
    
    // Build content array
    const content = [];
    
    if (message?.content) {
      content.push({
        type: 'text',
        text: message.content
      });
    }
    
    // Convert tool calls
    const toolUses = convertToolCallsToAnthropic(message?.tool_calls);
    content.push(...toolUses);
    
    // Determine stop reason
    let stopReason = 'end_turn';
    if (toolUses.length > 0) {
      stopReason = 'tool_use';
    } else if (choice?.finish_reason === 'stop') {
      stopReason = 'end_turn';
    } else if (choice?.finish_reason === 'length') {
      stopReason = 'max_tokens';
    }
    
    // Build Anthropic-style response
    const anthropicResponse = {
      id: generateMessageId(),
      type: 'message',
      role: 'assistant',
      content: content,
      model: model,
      stop_reason: stopReason,
      stop_sequence: null,
      usage: {
        input_tokens: ollamaResponse.data.usage?.prompt_tokens || 0,
        output_tokens: ollamaResponse.data.usage?.completion_tokens || 0
      }
    };
    
    res.json(anthropicResponse);
    
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        type: error.response?.data?.error?.type || 'internal_server_error',
        message: error.response?.data?.error?.message || error.message || 'Unknown error occurred'
      }
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', ollama_url: OLLAMA_BASE_URL });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Ollama Anthropic Proxy running on port ${PORT}`);
  console.log(`📡 Forwarding requests to: ${OLLAMA_BASE_URL}`);
  console.log(`🤖 Default model: ${DEFAULT_MODEL}`);
});
