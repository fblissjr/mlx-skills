---
name: mlx-lm
description: >
  Use when working with mlx-lm, Apple's official language model library for MLX.
  Scan project imports for "import mlx_lm", "from mlx_lm import" to determine
  applicability. Also triggers on "mlx-lm", "generate", "stream_generate",
  "KVCache", "LoRA fine-tuning", "model quantization", "GGUF", "safetensors",
  or any LLM inference/training on Apple silicon using MLX. Covers model
  architecture, generation pipelines, caching, quantization, fine-tuning, and
  server deployment. For core MLX concepts, load the mlx skill first.
---

# mlx-lm

Apple's official language model library for MLX. Provides inference, generation,
quantization, and fine-tuning for 40+ transformer architectures on Apple silicon.

## Prerequisites

Load the `mlx` skill first for core MLX concepts (lazy evaluation, unified
memory, compilation, type promotion). This skill assumes familiarity with those
fundamentals.

## What mlx-lm Is

mlx-lm is the reference implementation for running language models on MLX:

- **40+ model architectures**: Llama, Mistral, Qwen, Gemma, Phi, DeepSeek,
  Cohere, DBRX, and many more
- **Generation pipelines**: Single-sequence and batch generation with async
  evaluation for low latency
- **Quantization**: 4-bit and 8-bit weight quantization via `nn.QuantizedLinear`
- **Fine-tuning**: LoRA and DoRA adapters with gradient checkpointing
- **Server**: OpenAI-compatible HTTP API via `mlx_lm.server`

When in doubt about how to structure MLX model code, look at mlx-lm first.

## Model Architecture

Every model follows: `ModelArgs(BaseModelArgs)` dataclass for config,
`Model(nn.Module)` top-level module with required interface (`__call__`,
`layers` property, `make_cache()`, `sanitize()`), and standard inner components
(`Attention`, `MLP`, `TransformerBlock`). The model is discovered by matching
`model_type` in the HuggingFace config to the model file.

For the full architecture patterns with code, see `references/patterns.md`.

## Generation

mlx-lm uses an async double-buffer pipeline: prefill prompt in chunks,
then generate tokens on a dedicated stream with `mx.async_eval` so graph
construction overlaps computation. Any synchronous evaluation inside the step
function stalls the pipeline.

For the complete pipeline pattern, batch generation, and sampling details,
see `references/patterns.md`.

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

For cache implementation details and the factory pattern, see
`references/patterns.md`.

## Fine-Tuning

mlx-lm supports LoRA and DoRA: wrap existing layers with `LoRALinear.from_base`,
freeze base model, train only LoRA parameters, save adapters separately, and
optionally fuse back for inference. Works with both `nn.Linear` and
`nn.QuantizedLinear` (QLoRA).

For the full LoRA pattern and training loop, see `references/patterns.md`.

## Sampling

Supports temperature, top-p (nucleus), top-k, min-p, repetition penalty, and
XTC sampling. Samplers are composable and applied in sequence during generation.

## Server

```bash
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Exposes `/v1/chat/completions` and `/v1/completions` endpoints. Uses
`BatchGenerator` for concurrent request handling.

## Related Skills

- **`mlx`** -- Core MLX concepts (lazy evaluation, unified memory, nn.Module
  system, layers, optimizers, training)
- **`fast-mlx`** -- Performance optimization (profiling, compilation tuning,
  memory reduction, async pipeline optimization)

## References

Load these on demand for deeper guidance:

- `references/patterns.md` -- Idiomatic mlx-lm patterns: nn.Module structure,
  attention, KV cache, generation pipeline, quantization, LoRA, RoPE, sharding
- `references/architecture.md` -- mlx-lm directory structure, model loading flow,
  generation flow, model registration, fine-tuning flow, server integration
