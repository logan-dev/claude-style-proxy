const axios = require('axios');

async function testBasicRequest() {
  console.log('=== Testing Basic Request ===\n');
  
  const response = await axios.post('http://localhost:3000/v1/messages', {
    model: 'llama3.1',
    max_tokens: 512,
    system: 'You are a helpful assistant.',
    messages: [
      {
        role: 'user',
        content: 'Say hello!'
      }
    ]
  });
  
  console.log('Response:', JSON.stringify(response.data, null, 2));
}

async function testToolRequest() {
  console.log('\n=== Testing Tool Request ===\n');
  
  const response = await axios.post('http://localhost:3000/v1/messages', {
    model: 'llama3.1',
    max_tokens: 1024,
    system: 'You are a helpful assistant.',
    messages: [
      {
        role: 'user',
        content: 'What is the weather in New York?'
      }
    ],
    tools: [
      {
        name: 'get_weather',
        description: 'Get the current weather for a location',
        input_schema: {
          type: 'object',
          properties: {
            location: {
              type: 'string',
              description: 'The city and state, e.g. San Francisco, CA'
            }
          },
          required: ['location']
        }
      }
    ]
  });
  
  console.log('Response:', JSON.stringify(response.data, null, 2));
}

async function testStreamingRequest() {
  console.log('\n=== Testing Streaming Request ===\n');
  
  const response = await axios.post('http://localhost:3000/v1/messages', {
    model: 'llama3.1',
    max_tokens: 256,
    stream: true,
    messages: [
      {
        role: 'user',
        content: 'Count from 1 to 5'
      }
    ]
  }, {
    responseType: 'stream'
  });
  
  response.data.on('data', (chunk) => {
    process.stdout.write(chunk.toString());
  });
  
  return new Promise((resolve) => {
    response.data.on('end', () => {
      console.log('\nStream ended');
      resolve();
    });
  });
}

async function runTests() {
  try {
    // Test basic request
    await testBasicRequest();
    
    // Test tool request
    await testToolRequest();
    
    // Test streaming
    await testStreamingRequest();
    
    console.log('\n✅ All tests completed!');
  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
    process.exit(1);
  }
}

runTests();
