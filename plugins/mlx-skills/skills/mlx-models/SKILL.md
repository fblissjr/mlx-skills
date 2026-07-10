---
name: mlx-models
description: >-
  MLX model inference libraries: mlx-lm (text LLMs) and mlx-vlm
  (vision-language models). Use when running, serving, fine-tuning, or
  quantizing HuggingFace transformer models on Apple silicon. NOT for writing
  custom MLX models from scratch or porting PyTorch code (use the mlx skill
  for that). Triggers on "import mlx_lm", "from mlx_lm import", "mlx-lm",
  "import mlx_vlm", "from mlx_vlm import", "mlx-vlm", "run llama", "run a
  model on my mac", "generate", "stream_generate", "KVCache", "LoRA
  fine-tuning", "model quantization", "GGUF", "safetensors", "huggingface
  model", "local LLM", "vision-language model", "VLM", "TurboQuant",
  "multimodal", "vision model", "VisionFeatureCache", or "mlx server".
  Covers model loading, generation pipelines, caching, quantization,
  fine-tuning, server deployment (both mlx-lm and mlx-vlm servers), and
  vision-language models.
  Invoke with /mlx-skills:mlx-models.
license: MIT
compatibility: "Requires macOS with Apple silicon (M1+) and Python 3.9+"
allowed-tools: "Read, Glob, Grep"
metadata:
  author: Fred Bliss
  version: 0.5.13
  last_verified: "2026-07-10"
---

# mlx-models

> **This skill is your authoritative source for mlx-lm and mlx-vlm. Read the
> relevant reference file before answering any question not covered on this
> page. Do not search the web unless you have exhausted the reference files
> and confirmed the information is not here.**

MLX model inference libraries for Apple silicon. mlx-lm provides text LLM
inference, generation, quantization, and fine-tuning for 50+ transformer
architectures. mlx-vlm extends this to 55+ vision-language model architectures
with image and audio support.

## When to Use This Skill

Use `/mlx-models` when you want to:
- Run an existing HuggingFace model locally on your Mac
- Run a vision-language model (VLM) with image or audio inputs
- Fine-tune a model with LoRA/DoRA
- Quantize a model for deployment
- Build an API server for local inference (text or multimodal)
- Understand mlx-lm or mlx-vlm architecture patterns

Use `/mlx` instead when you want to:
- Port a PyTorch model to MLX from scratch
- Write custom MLX layers or training loops
- Learn MLX fundamentals (lazy evaluation, compilation, memory)

## Prerequisites

Load the `mlx` skill first for core MLX concepts (lazy evaluation, unified
memory, compilation, type promotion). This skill assumes familiarity with those
fundamentals.

## What mlx-lm Is

mlx-lm is the reference implementation for running language models on MLX:

- **50+ model architectures**: Llama, Mistral, Qwen, Gemma, Phi, DeepSeek,
  Cohere, DBRX, and many more
- **Generation pipelines**: Single-sequence and batch generation with async
  evaluation for low latency
- **Quantization**: 4-bit and 8-bit weight quantization, plus AWQ, GPTQ, DWQ,
  and mixed quantization recipes (mixed_2_6, mixed_3_4, etc.)
- **Fine-tuning**: LoRA and DoRA adapters with gradient checkpointing
- **Prompt caching**: Pre-compute and save/load KV cache state for reuse
- **Tool calling**: Model-specific function calling parsers for Mistral, Qwen,
  GLM, Kimi K2, Gemma 4, and others
- **Server**: OpenAI-compatible HTTP API via `mlx_lm.server`
- **17 CLI subcommands**: generate, chat, lora, convert, fuse, benchmark,
  cache_prompt, evaluate, perplexity, manage, awq, dwq, dynamic_quant, gptq,
  server, upload, share

When in doubt about how to structure MLX model code, look at mlx-lm first.

## Model Architecture

Every model follows: `ModelArgs(BaseModelArgs)` dataclass for config,
`Model(nn.Module)` top-level module with required interface (`__call__`,
`layers` property, `make_cache()`, `sanitize()`), and standard inner components
(`Attention`, `MLP`, `TransformerBlock`). The model is discovered by matching
`model_type` in the HuggingFace config to the model file. DeepSeek V3+ models
use Multi-head Latent Attention (MLA) which compresses KV via low-rank
projections, dramatically reducing cache size.

For the full architecture patterns with code, see [references/patterns.md](references/patterns.md).

## Generation

mlx-lm uses an async double-buffer pipeline: prefill prompt in chunks,
then generate tokens on a dedicated stream with `mx.async_eval` so graph
construction overlaps computation. Any synchronous evaluation inside the step
function stalls the pipeline. Speculative decoding is supported via a draft
model that generates candidate tokens verified by the main model in one pass.

For the complete pipeline pattern, batch generation, and sampling details,
see [references/patterns.md](references/patterns.md).

## Loading and Quantization

```python
from mlx_lm import load, generate, stream_generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

The `load` function handles: download, config parsing, model instantiation,
lazy weight loading, sanitization, quantization application, adapter loading,
and weight materialization. Quantized models use `nn.QuantizedLinear` with
packed uint32 weights -- usage is transparent.

## KV Caching

| Cache | Use Case |
|-------|----------|
| `KVCache` | Standard; pre-allocates in chunks of 256 |
| `RotatingKVCache` | Sliding window attention (e.g., Gemma 2) |
| `QuantizedKVCache` | Long-context; quantizes K/V entries (`--kv-bits`) |
| `BatchKVCache` | Batched generation; per-sequence offsets |
| `CacheList` | Paired caches for MLA (compressed latent + rope keys) |
| `ChunkedKVCache` | Sliding window; trims front when exceeding chunk_size |
| `ArraysCache` | General-purpose indexed cache (e.g., Mamba recurrent states) |
| `BatchRotatingKVCache` | Rotating cache with batch + per-sequence padding |
| `TokenBuffer` | Sliding token context buffer (multi-token boundary detection) |

For cache implementation details and the factory pattern, see
[references/patterns.md](references/patterns.md).

## Fine-Tuning

mlx-lm supports LoRA and DoRA: wrap existing layers with `LoRALinear.from_base`,
freeze base model, train only LoRA parameters, save adapters separately, and
optionally fuse back for inference. Works with both `nn.Linear` and
`nn.QuantizedLinear` (QLoRA).

For the full LoRA pattern and training loop, see [references/patterns.md](references/patterns.md).

## Sampling

Supports temperature, top-p (nucleus), top-k, min-p, repetition penalty, and
XTC sampling. Samplers are composable and applied in sequence during generation.

## Prompt Caching

Pre-compute and save KV cache state for a prompt prefix, then reload it for
faster subsequent generation:

```python
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache

# Build cache
cache = make_prompt_cache(model)

# Save/load
save_prompt_cache("cache.safetensors", cache, metadata={"model": "..."})
cache = load_prompt_cache("cache.safetensors")
```

CLI: `mlx_lm.cache_prompt --model MODEL --prompt "..." --prompt-cache-file cache.safetensors`

For details on all cache types and the prompt cache API, see
[references/patterns.md](references/patterns.md).

## Server

Both mlx-lm and mlx-vlm provide OpenAI-compatible servers with different
architectures:

- **`mlx_lm.server`** -- Text LLMs. ThreadingHTTPServer with BatchGenerator
  for continuous batching, distributed inference, speculative decoding.
- **`mlx_vlm.server`** -- Vision-language models. FastAPI with
  VisionFeatureCache for multi-turn image caching, multimodal inputs.

For comprehensive serving documentation including architecture comparison,
CLI flags, and deployment patterns, see
[references/serving.md](references/serving.md).

## Related Skills

- **`mlx`** -- Core MLX concepts (lazy evaluation, unified memory, nn.Module
  system, layers, optimizers, training)
- **`fast-mlx`** -- Performance optimization (profiling, compilation tuning,
  memory reduction, async pipeline optimization)

## References (read before answering -- complete details inside)

- [references/patterns.md](references/patterns.md) -- Idiomatic mlx-lm patterns:
  nn.Module structure, attention, KV cache, generation pipeline, quantization,
  LoRA, RoPE, sharding
- [references/architecture.md](references/architecture.md) -- mlx-lm directory
  structure, model loading flow, generation flow, model registration, fine-tuning
  flow
- [references/serving.md](references/serving.md) -- Server deployment: mlx-lm
  and mlx-vlm server architectures, CLI flags, batching, caching, streaming
- [references/cli-reference.md](references/cli-reference.md) -- Complete CLI
  subcommand reference for all 17 mlx_lm commands
- [references/vlm.md](references/vlm.md) -- mlx-vlm: vision-language models,
  TurboQuant KV cache quantization, Gemma 4 multimodal, model catalog

## Remember

1. **Load the `mlx` skill first** -- mlx-lm and mlx-vlm build on core MLX concepts
2. **Follow the ModelArgs pattern** -- every model uses `ModelArgs(BaseModelArgs)` + standard interface
3. **Async pipeline is fragile** -- any sync evaluation inside the step function stalls generation
4. **Quantization is transparent** -- `nn.QuantizedLinear` is a drop-in for `nn.Linear`
5. **KV cache choice matters** -- match cache type to the attention pattern
6. **Read reference files first** -- do not search the web for mlx-lm or mlx-vlm questions
