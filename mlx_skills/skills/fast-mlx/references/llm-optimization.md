last updated: 2026-02-23

# LLM Optimization Guide

Performance optimization techniques specific to language model inference and
training on MLX. All patterns sourced from mlx-lm.

For mlx-lm architecture, generation pipeline, and KV cache internals, load the
`mlx-lm` skill. This guide focuses on optimization techniques.

## KV Cache Optimization

### Choose the Right Cache Type

| Cache | Use Case | Memory | Speed |
|-------|----------|--------|-------|
| `KVCache` | General purpose, unlimited context | Grows linearly | Fast (in-place update) |
| `RotatingKVCache` | Fixed context window (sliding attention) | Bounded | Fast after warmup |
| `QuantizedKVCache` | Very long sequences (>5K tokens) | ~2-4x reduction | Slower (quantize/dequantize) |
| `BatchKVCache` | Batched generation | Per-sequence tracking | Overhead from padding |

### KV Cache Quantization

For long-context generation, quantize the KV cache after the initial prompt:

```python
generate_step(
    prompt, model,
    kv_bits=8,               # 8-bit KV quantization
    kv_group_size=64,        # Quantization group size
    quantized_kv_start=5000, # Start quantizing after 5000 tokens
)
```

Start quantizing after the prompt is processed (`quantized_kv_start`) to
maintain full precision for the initial context. The `to_quantized()` method
on `KVCache` converts in-place:

```python
cache.to_quantized(group_size=64, bits=4)
```

### Preallocated Buffer Strategy

`KVCache` preallocates in chunks of `step=256` tokens. For known maximum
lengths, you can set `step` to a larger value to reduce allocation overhead:

```python
cache = KVCache()
cache.step = 1024  # Fewer, larger allocations
```

### Cache Trimming

Trim unused cache entries to reclaim memory or enable reprocessing:

```python
from mlx_lm.models.cache import trim_prompt_cache

tokens_trimmed = trim_prompt_cache(cache, num_tokens=100)
```

### Prompt Cache Persistence

Save and reload prompt caches to avoid recomputing long system prompts:

```python
from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache

# Save
save_prompt_cache("system_prompt.safetensors", cache, {"model": model_name})

# Load and reuse
cache = load_prompt_cache("system_prompt.safetensors")
generate_step(user_prompt, model, prompt_cache=cache)
```

### MLA Cache Optimization

Models using Multi-head Latent Attention (MLA), such as DeepSeek V3, store
compressed latent vectors instead of full K/V per head. This dramatically
reduces cache memory:

- Standard: `cache_per_layer = 2 * batch * n_kv_heads * head_dim * seq_len * dtype_bytes`
- MLA: `cache_per_layer = batch * (kv_lora_rank + qk_rope_head_dim) * 2 * dtype_bytes * seq_len`

With typical values (`kv_lora_rank=512`, `qk_rope_head_dim=64` vs
`n_heads * head_dim` often 8192+), MLA reduces cache by ~14x per layer.

The cache uses `CacheList` pairs (latent cache + rope key cache) per layer.
For MLA models, `QuantizedKVCache` quantizes the latent vectors, compounding
memory savings.

## Async Generation Pipeline

### The Double-Buffer Pattern

The core latency optimization for token generation:

```python
generation_stream = mx.new_stream(mx.default_device())

def _step(tokens):
    with mx.stream(generation_stream):
        logits = model(tokens[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        return sampler(logprobs), logprobs.squeeze(0)

# First step
y, logprobs = _step(prompt_end)
mx.async_eval(y, logprobs)

# Pipeline: build next while previous computes
while generating:
    next_y, next_logprobs = _step(y)
    mx.async_eval(next_y, next_logprobs)
    mx.eval(y)
    yield y.item(), logprobs
    y, logprobs = next_y, next_logprobs
```

**Rules for the pipeline to work:**
1. `_step` must not contain any synchronous evaluations
2. The generation stream must be separate from any stream doing synchronous work
3. Wait for previous result only AFTER dispatching next computation

### Wired Memory Limit

Pin model weights in physical memory to prevent OS paging:

```python
from mlx_lm.generate import wired_limit

with wired_limit(model, [generation_stream]):
    for response in stream_generate(model, tokenizer, prompt):
        ...
```

This sets the wired limit to `max_recommended_working_set_size` and restores
it after generation completes.

## Prefill Optimization

### Chunked Prefill

Process long prompts in chunks to avoid OOM on very long inputs:

```python
prefill_step_size = 2048  # Process 2048 tokens at a time

while len(prompt) > 1:
    chunk = prompt[:prefill_step_size]
    model(chunk[None], cache=prompt_cache)
    mx.eval([c.state for c in prompt_cache])
    prompt = prompt[prefill_step_size:]
    mx.clear_cache()  # Reclaim memory between chunks
```

The `mx.clear_cache()` call is important -- without it, intermediate buffers
from different chunk sizes accumulate.

### Prefill Batch Size

In `BatchGenerator`, `prefill_batch_size` controls how many prompts are
processed together during prefill. Larger batches amortize overhead but
increase peak memory:

```python
gen = BatchGenerator(
    model,
    prefill_batch_size=8,       # Process 8 prompts together
    completion_batch_size=32,   # Up to 32 concurrent completions
    prefill_step_size=2048,     # Chunk size for long prompts
)
```

## Batch Generation

### Efficient Batch Management

`BatchGenerator` handles dynamic batching:

- Prompts are sorted by length to minimize padding waste
- Left-padding aligns sequences for batched prefill
- Finished sequences are filtered in-place without reallocating
- New prompts can be inserted while generation is running

```python
gen = BatchGenerator(model, max_tokens=128, stop_tokens=eos_tokens)

# Insert prompts
uids = gen.insert(prompts, max_tokens=[100, 200, 150])

# Generate
while responses := gen.next():
    for r in responses:
        if r.finish_reason is not None:
            print(f"Done: {r.uid}")
gen.close()
```

### Memory Considerations for Batching

- Batch size directly multiplies KV cache memory
- Left-padding wastes compute on padding tokens
- Use `filter()` to remove finished sequences promptly
- Consider `max_kv_size` to cap per-sequence cache size

## Speculative Decoding

### When to Use

Speculative decoding uses a small draft model to generate candidate tokens
that the main model verifies in parallel. It helps when:

- The draft model is significantly smaller (3-5x fewer parameters)
- The acceptance rate is high (similar model families)
- Latency matters more than throughput

```python
model, tokenizer = load("mlx-community/Llama-3.2-70B-Instruct-4bit")
draft_model, _ = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

for response in stream_generate(
    model, tokenizer, prompt,
    draft_model=draft_model,
    num_draft_tokens=3,
):
    print(response.text, end="")
```

### Tuning Draft Tokens

`num_draft_tokens` controls the speculation depth:
- Too few: verification overhead dominates
- Too many: low acceptance rate wastes draft computation
- Sweet spot is typically 2-4 for models in the same family
- CLI: `--draft-model <path>` and `--num-draft-tokens <n>` (default 3)

Draft and main model must share the same tokenizer. On rejection,
`trim_prompt_cache` rewinds both caches to the last accepted token.

## Training Optimization

### Gradient Checkpointing

Trade compute for memory during fine-tuning:

```python
from mlx_lm.tuner.trainer import grad_checkpoint

for layer in model.layers:
    grad_checkpoint(layer)
```

This recomputes activations during the backward pass instead of storing them.
Useful when fine-tuning large models on limited memory.

### Gradient Accumulation

Simulate larger batch sizes without increasing memory:

```python
training_args = TrainingArgs(
    batch_size=4,
    grad_accumulation_steps=4,  # Effective batch size = 16
)
```

### Releasing Gradients Before Evaluation

```python
# Always wrap training step in a function
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss  # grads released here

for x, y in dataset:
    loss = step(x, y)
    mx.eval(model.parameters(), optimizer.state)
```

## Memory Budget Estimation

Quick formulas for LLM memory planning:

| Component | Formula |
|-----------|---------|
| Model weights (fp16) | `params * 2 bytes` |
| Model weights (4-bit) | `params * 0.5 bytes` (approx) |
| KV cache per layer | `2 * batch * n_kv_heads * seq_len * head_dim * dtype_bytes` |
| Total KV cache | `kv_per_layer * num_layers` |
| LoRA adapters | `2 * rank * (input_dim + output_dim) * num_adapted_layers * dtype_bytes` |

Example: Llama 3.2 3B at 4-bit with 4096 context:
- Weights: ~1.5 GB
- KV cache (fp16): ~0.4 GB per batch item
- Total for batch=1: ~2 GB
