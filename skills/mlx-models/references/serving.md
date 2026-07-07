last updated: 2026-07-07

# MLX Model Serving

Both mlx-lm and mlx-vlm provide OpenAI-compatible HTTP servers for local model
inference on Apple silicon. They share the same API surface but have
fundamentally different architectures targeting different use cases.

## When to Use Which

| Scenario | Use | Why |
|----------|-----|-----|
| Text-only LLM, multiple concurrent users | `mlx_lm.server` | BatchGenerator multiplexes requests |
| Vision-language or audio models | `mlx_vlm.server` | VisionFeatureCache, multimodal inputs |
| Distributed inference (multi-GPU) | `mlx_lm.server` | Pipeline and tensor parallelism |
| Speculative decoding | `mlx_lm.server` | Draft model support |
| Single-user, simple deployment | Either | Both work; mlx-vlm is simpler |
| Multi-turn image conversations | `mlx_vlm.server` | VisionFeatureCache avoids re-encoding |

## Architecture Comparison

| Aspect | mlx-lm | mlx-vlm |
|--------|--------|---------|
| HTTP framework | ThreadingHTTPServer | FastAPI (async) |
| Concurrency | Continuous batching via BatchGenerator | Sequential per-request |
| Prompt cache | LRU prefix trie (segmented) | None (relies on VisionFeatureCache) |
| Vision cache | None | VisionFeatureCache (LRU, 20 items) |
| Distributed | Pipeline + tensor parallelism | Single-device only |
| Speculative decoding | Yes (draft model) | No |
| State machine | SequenceStateMachine (Aho-Corasick) | Post-generation parsing |
| Model hot-swap | ModelProvider (lazy load) | model_cache dict |
| API compatibility | OpenAI only | OpenAI + Anthropic Messages API (`/v1/messages`) |

## mlx-lm Server

### Architecture

```
HTTP Request -> APIHandler -> ResponseGenerator (queue) ->
  Generation Thread -> BatchGenerator -> Model -> SSE Response
```

Key components:

- **ResponseGenerator**: Manages a dedicated generation thread. Requests are
  queued and dequeued by the generation thread for batched processing.
- **BatchGenerator**: Continuous batching with two phases -- prompt processing
  batch (parallel prefill) and generation batch (parallel token generation).
  Configurable via `--decode-concurrency` (default 32) and
  `--prompt-concurrency` (default 8).
- **ModelProvider**: Lazy model loading. Caches the loaded model and reloads
  only when the model path changes. Supports adapter loading and draft models.
- **SequenceStateMachine**: Aho-Corasick trie for efficient multi-pattern
  matching during generation. Tracks state transitions between `normal`,
  `reasoning` (thinking), and `tool` (function calling) states. Matches stop
  sequences and state-specific control tokens.

### Batching Strategy

Requests are batchable when they share the same model, the model architecture
supports batching, and no request has a fixed random seed. The BatchGenerator:

1. Collects compatible requests from the queue
2. Processes prompts in parallel (up to `--prompt-concurrency`)
3. Generates tokens in parallel (up to `--decode-concurrency`)
4. Drains the batch when an incompatible request arrives or the model changes

### Prompt Cache (LRU Prefix Trie)

The server maintains an LRU prompt cache that stores KV cache state keyed by
token sequences using a prefix trie:

- **Segment types**: `system`, `user`, `assistant` -- with priority-based
  eviction (system cached longest, assistant evicted first)
- **Lookup**: `fetch_nearest_cache()` finds exact or partial prefix matches
- **Size limits**: `--prompt-cache-size` (max 10 distinct sequences) and
  `--prompt-cache-bytes` (optional byte limit)
- **Extraction**: Caches are extracted at segment boundaries during generation
  and stored for cross-request reuse

### Streaming

Server-Sent Events (SSE) format:
- Content: `data: {json}\n\n`
- Completion: `data: [DONE]\n\n`
- Keepalive during long prefill: `keepalive {processed}/{total}\n\n`
- Usage stats with `stream_options: {"include_usage": true}`

### Distributed Inference

With `--pipeline`, the server uses layer-by-layer pipeline parallelism:
- Rank 0 runs the HTTP server; all ranks run the generation thread
- Ranks coordinate via `mx.distributed.init()` with separate pipeline and
  tensor groups
- Time budget enforcement syncs across ranks every 10 iterations

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## mlx-vlm Server

### Architecture

```
HTTP Request -> FastAPI route -> generate/stream_generate -> Response
```

Key components:

- **FastAPI lifespan**: Startup/shutdown via async context manager
- **model_cache**: Global dict holding one model + processor + config at a time
- **VisionFeatureCache**: LRU cache (max 20 items) for vision encoder outputs.
  Keys derived from file paths, URLs, or PIL image SHA256 hashes. Avoids
  expensive re-encoding when the same image appears across conversation turns.
- **Sequential processing**: No batching -- each request runs to completion
  before the next starts

### VisionFeatureCache

Caches the output of the vision encoder + embedding projection (the most
expensive step in VLM inference):

- **Hash keying**: File paths (string), URLs, PIL images (SHA256 of content),
  or composite keys for multi-image requests (pipe-delimited)
- **Scope**: Per-model; cleared on model unload
- **Impact**: Multi-turn conversations about the same image(s) skip the vision
  encoding stage entirely after the first turn

### Multimodal Input Handling

Supports multiple input types per request:
- `input_image` with detail levels (high/low/auto)
- `input_audio` for audio inputs
- `image_url` for URL-based images
- Text content intermixed with media
- Max 10 images per request (`MAX_IMAGES = 10`)

### Thinking Support

Both servers support extended thinking for models that have it:
- `enable_thinking`, `thinking_budget`, `thinking_start_token`,
  `thinking_end_token` parameters
- mlx-lm tracks thinking state in SequenceStateMachine
- mlx-vlm passes through `template_kwargs` to the chat template

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/responses` | POST | OpenAI responses.create() format. Stateful -- responses are stored server-side and can be retrieved, cancelled, or listed by ID (`GET`/`DELETE /v1/responses/{response_id}`, `POST .../cancel`) |
| `/v1/messages` | POST | Anthropic Messages API-compatible chat endpoint |
| `/v1/messages/count_tokens` | POST | Token counting for an Anthropic-format request |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/unload` | POST | Unload current model and free memory |

## Tool Calling

Both servers support function/tool calling, but implement it differently:

**mlx-lm**: Integrated into the generation state machine. Tool call boundaries
are tracked during token generation via SequenceStateMachine state transitions
(`normal` -> `tool` -> `normal`). The server auto-detects the correct parser
from the model's tokenizer config. Supports streaming tool calls.

**mlx-vlm**: Post-generation parsing. Tool parser is inferred from the
chat template via `_infer_tool_parser()` -> `load_tool_module()`. Tool calls
are extracted from the complete model output after generation finishes.

## CLI Reference

### mlx_lm.server

```bash
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --port 8080 --host 0.0.0.0
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | required | Model path or HuggingFace ID |
| `--adapter-path` | str | None | Trained adapter weights path |
| `--host` | str | 127.0.0.1 | Server bind address |
| `--port` | int | 8080 | Server port |
| `--allowed-origins` | str | `"*"` | CORS origins (comma-separated) |
| `--draft-model` | str | None | Model for speculative decoding |
| `--num-draft-tokens` | int | 3 | Tokens to draft per step |
| `--chat-template` | str | `""` | Custom chat template |
| `--chat-template-args` | JSON | `"{}"` | Template arguments (e.g., `'{"enable_thinking":false}'`) |
| `--temp` | float | 0.0 | Default sampling temperature |
| `--top-p` | float | 1.0 | Default nucleus sampling threshold |
| `--top-k` | int | 0 | Default top-k (0 = disabled) |
| `--min-p` | float | 0.0 | Default min-p (0 = disabled) |
| `--max-tokens` | int | 512 | Default max generation tokens |
| `--decode-concurrency` | int | 32 | Max parallel decode requests |
| `--prompt-concurrency` | int | 8 | Max parallel prompt processing |
| `--prefill-step-size` | int | 2048 | Tokens per prefill chunk |
| `--prompt-cache-size` | int | 10 | Max distinct KV caches |
| `--prompt-cache-bytes` | size | None | Max cache memory (e.g., `4G`) |
| `--pipeline` | flag | False | Use pipeline parallelism |
| `--log-level` | str | INFO | Logging level |
| `--trust-remote-code` | flag | False | Trust remote tokenizer code |

### mlx_vlm.server

```bash
python -m mlx_vlm.server --model mlx-community/gemma-4-4b-it-4bit \
    --port 8080
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | None | Model path or HuggingFace ID |
| `--adapter-path` | str | None | Trained adapter weights path |
| `--host` | str | 127.0.0.1 | Server bind address |
| `--port` | int | 8080 | Server port |
| `--prefill-step-size` | int | 2048 | Tokens per prefill chunk (lower = less peak memory) |
| `--kv-bits` | float | 0 | KV cache quantization bits (fractional = TurboQuant) |
| `--kv-quant-scheme` | str | uniform | `uniform` or `turboquant` |
| `--kv-group-size` | int | 64 | Group size for uniform KV quantization |
| `--max-kv-size` | int | 0 | Max KV cache tokens (0 = unlimited) |
| `--quantized-kv-start` | int | 5000 | Token index to start KV quantization |
| `--trust-remote-code` | flag | False | Trust remote tokenizer/processor code |
| `--reload` | flag | False | Auto-reload on file changes (dev only) |

### Shared Flags

Both servers accept `--model`, `--adapter-path`, `--host`, `--port`,
`--prefill-step-size`, and `--trust-remote-code` with the same semantics.

**mlx-lm unique**: Sampling defaults (`--temp`, `--top-p`, `--top-k`, `--min-p`,
`--max-tokens`), batching (`--decode-concurrency`, `--prompt-concurrency`),
prompt cache (`--prompt-cache-size`, `--prompt-cache-bytes`), distributed
(`--pipeline`), speculative decoding (`--draft-model`, `--num-draft-tokens`),
chat template (`--chat-template`, `--chat-template-args`).

**mlx-vlm unique**: KV quantization (`--kv-bits`, `--kv-quant-scheme`,
`--kv-group-size`, `--max-kv-size`, `--quantized-kv-start`), development
(`--reload`), model unloading (`/unload` endpoint).

## Common Usage Patterns

### Text LLM with batching

```bash
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --decode-concurrency 16 \
    --prompt-cache-size 20 \
    --prefill-step-size 4096
```

### Vision-language model with KV cache quantization

```bash
python -m mlx_vlm.server \
    --model mlx-community/gemma-4-4b-it-4bit \
    --kv-bits 3.5 \
    --port 8080
```

### Speculative decoding for faster generation

```bash
mlx_lm.server --model mlx-community/Llama-3.2-70B-Instruct-4bit \
    --draft-model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --num-draft-tokens 5
```

### Client request (both servers)

```python
import requests

response = requests.post("http://localhost:8080/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100,
    "stream": True,
})
```
