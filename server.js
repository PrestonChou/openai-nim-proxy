// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
  console.log(`[incoming] ${req.method} ${req.path}`);
  next();
});

process.on('unhandledRejection', (err) => {
  console.error('[unhandled rejection]', err);
});

app.all('/', (req, res) => {
  res.status(200).json({ status: 'proxy is running' });
});
  
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// ─────────────────────────────────────────────────────────────
// MODEL MAPPING
// Key   = the model name the CLIENT (Janitor AI) sends you.
// Value = the actual NIM model ID to call on NVIDIA's side.
// Put whatever "friendly" names you use in Janitor AI on the left;
// put the real, current build.nvidia.com/models slug on the right.
// ─────────────────────────────────────────────────────────────
const MODEL_MAPPING = {
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'kimi-k3':            'moonshotai/kimi-k3',
  'gemma-4':            'google/gemma-4-31b-it',
  'nemotron-3-ultra':   'nvidia/nemotron-3-ultra-550b-a55b'
};

// Ordered fallback chain to try, in order, if the requested/mapped model
// comes back 404 (not found / bad slug) or 410 (retired). Keep this to
// models you've confirmed are currently live on the free tier.
const FALLBACK_CHAIN = [
  'nvidia/nemotron-3-ultra-550b-a55b',
  'z-ai/glm-5-3',
  'nvidia/nemotron-3.5-lightning-30b-a3b'
];

// Cache of models we've confirmed are dead this run, so we don't
// re-try a known-410 model on every single request.
const deadModels = new Set();

// Some models enforce a narrower temperature range than the usual 0–2.
// Add an entry here (keyed by the real NIM model id) for any model that
// rejects out-of-range values; everything else falls back to [0, 2].
const TEMPERATURE_LIMITS = {
  'moonshotai/kimi-k3': [0, 1]
};

function clampTemperature(nimModel, temperature) {
  const [min, max] = TEMPERATURE_LIMITS[nimModel] || [0, 2];
  return Math.min(Math.max(temperature ?? 0.6, min), max);
}

function isRetiredStatus(status) {
  return status === 404 || status === 410;
}

// Calls NIM with a specific model id. Throws with a tagged error if the
// model itself is the problem (so the caller can decide to fall back)
// vs. some other failure (auth, rate limit, etc.) which should surface
// to the client as-is.
async function callNim(nimModel, payload, stream) {
  try {
    console.log(`[nim] calling ${nimModel}...`);
    const clampedPayload = {
      ...payload,
      temperature: clampTemperature(nimModel, payload.temperature)
    };
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, {
      ...clampedPayload,
      model: nimModel
    }, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 0, // no timeout — let the model take as long as it needs
      validateStatus: (status) => status < 500 || isRetiredStatus(status)
    });
    console.log(`[nim] got response from ${nimModel}: ${response.status}`);

    if (response.status >= 400 && !isRetiredStatus(response.status)) {
      let errorBody = response.data;
      if (stream) {
        // response.data is a readable stream here, not a JSON object —
        // read it into a string first so we can actually see what NVIDIA said.
        try {
          const chunks = [];
          for await (const chunk of response.data) chunks.push(chunk);
          errorBody = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        } catch (parseErr) {
          errorBody = { error: { message: 'could not parse streamed error body' } };
        }
      }
      console.error(`[nim] error body:`, JSON.stringify(errorBody));
      const err = new Error(errorBody?.error?.message || `NIM returned ${response.status}`);
      err.status = response.status;
      throw err;
    }

    if (isRetiredStatus(response.status)) {
      deadModels.add(nimModel);
      const err = new Error(`Model ${nimModel} returned ${response.status}`);
      err.modelRetired = true;
      err.status = response.status;
      throw err;
    }
    return response;
  } catch (error) {
    if (error.modelRetired) throw error;
    if (error.code === 'ECONNABORTED') {
      console.error(`[nim] TIMEOUT calling ${nimModel} after 30s`);
    }
    // Network-level or axios-thrown HTTP error (e.g. status >= 500)
    if (error.response && isRetiredStatus(error.response.status)) {
      deadModels.add(nimModel);
      const err = new Error(`Model ${nimModel} returned ${error.response.status}`);
      err.modelRetired = true;
      err.status = error.response.status;
      throw err;
    }
    throw error;
  }
}

// Resolves the client's requested model to a NIM model id, then calls
// it, cascading through FALLBACK_CHAIN if the chosen model is 404/410.
async function resolveAndCall(clientModel, payload, stream) {
  const candidates = [];

  const mapped = MODEL_MAPPING[clientModel];
  if (mapped && !deadModels.has(mapped)) candidates.push(mapped);

  // If the client sent something not in our map, try it verbatim first
  // (covers people who type a real NIM slug straight into Janitor AI).
  if (!mapped && !deadModels.has(clientModel)) candidates.push(clientModel);

  for (const fb of FALLBACK_CHAIN) {
    if (!candidates.includes(fb) && !deadModels.has(fb)) candidates.push(fb);
  }

  let lastError;
  for (const nimModel of candidates) {
    try {
      const response = await callNim(nimModel, payload, stream);
      if (nimModel !== mapped) {
        console.warn(`[fallback] "${clientModel}" served by "${nimModel}" instead`);
      }
      return { response, nimModel };
    } catch (error) {
      lastError = error;
      if (!error.modelRetired) throw error; // real error, don't keep trying
      console.warn(`[retired] ${nimModel} is unavailable (${error.status}), trying next candidate`);
    }
  }

  throw lastError || new Error('No candidate models available');
}

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    known_dead_models: Array.from(deadModels)
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const payload = {
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    const { response, nimModel } = await resolveAndCall(model, payload, stream);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: nimModel,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.status || error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.status || error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});
