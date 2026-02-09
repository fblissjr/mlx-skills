# Idiomatic MLX Patterns

Patterns extracted from mlx-lm, the gold-standard MLX codebase. Use these as
the reference for how to write MLX code.

## Model Architecture Pattern

### ModelArgs Dataclass

Every model starts with a `ModelArgs` dataclass inheriting from `BaseModelArgs`.
This pattern is universal across all mlx-lm model implementations:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .base import BaseModelArgs

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
```

`BaseModelArgs.from_dict(params)` filters unknown keys via `inspect.signature`,
so config files can have extra fields without breaking.

### nn.Module Structure

MLX's `nn.Module` is similar to PyTorch but with key differences:

- Parameters are stored as attributes (like PyTorch)
- `__call__` is the forward pass (no separate `forward` method)
- No `nn.Parameter` wrapper -- any `mx.array` attribute is a parameter
- `model.parameters()` returns a nested dict of all parameters
- `model.trainable_parameters()` returns only trainable params (respects `freeze`)

```python
class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
```

### Standard Transformer Block

The standard block pattern from mlx-lm:

```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
```

Key observations:
- Pre-norm architecture (norm before attention/MLP)
- Residual connections via simple addition
- Cache is passed through but optional
- No explicit activation function calls -- handled inside MLP

### Top-Level Model

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LlamaModel(args)  # Inner model with layers
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        # Filter out weights that don't belong to this model
        return {k: v for k, v in weights.items()
                if "rotary_emb.inv_freq" not in k}
```

Important methods:
- `make_cache()` -- Creates the appropriate cache type per layer
- `sanitize(weights)` -- Filters loaded weights before applying
- `layers` property -- Gives mlx-lm access to iterate over layers
- `shard()` -- Distributes model across multiple devices

## Attention Pattern

The standard attention implementation from mlx-lm:

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or dim // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = initialize_rope(self.head_dim, args.rope_theta, ...)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape: (B, L, n_heads, head_dim) -> (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
```

Critical details:
- Shape is `(B, n_heads, L, head_dim)` for attention computation
- RoPE applied with `offset=cache.offset` for correct position encoding
- `cache.update_and_fetch` returns the full K/V history including new tokens
- GQA handled automatically -- `n_kv_heads < n_heads` with repeat in SDPA
- Always use `mx.fast.scaled_dot_product_attention` (via the `base.py` wrapper)

### Scaled Dot Product Attention

mlx-lm routes through a wrapper that handles quantized vs regular caches:

```python
def scaled_dot_product_attention(queries, keys, values, cache, scale, mask):
    if hasattr(cache, "bits"):
        # Quantized KV cache -- use mx.quantized_matmul
        return quantized_scaled_dot_product_attention(...)
    else:
        # Standard -- use the fast op
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
```

### Attention Mask Creation

```python
def create_attention_mask(h, cache=None, window_size=None, return_array=False):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None  # Single token generation needs no mask
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"  # String shortcut for the fast SDPA kernel
```

The `"causal"` string is a fast path -- `mx.fast.scaled_dot_product_attention`
recognizes it and applies causal masking internally without materializing
a mask tensor.

## KV Cache Pattern

### Standard KVCache

The workhorse cache with pre-allocated buffers and in-place updates:

```python
class KVCache:
    step = 256  # Allocation granularity

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            # Allocate or expand buffer in steps of self.step
            ...
        self.offset += keys.shape[2]
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
```

Key design choices:
- Pre-allocates in chunks of 256 to avoid frequent reallocation
- Uses slice assignment for in-place updates (graph nodes, not truly in-place)
- Returns sliced view of only the valid portion
- `offset` tracks how many tokens have been cached

### RotatingKVCache

For sliding window attention (used by models like Gemma 2):

```python
cache = RotatingKVCache(max_size=4096, keep=4)
```

- Maintains a fixed-size buffer that rotates when full
- `keep` tokens are always preserved at the start (attention sinks)
- Two update paths: `_update_concat` for prefill, `_update_in_place` for
  single-token generation

### QuantizedKVCache

Quantizes K/V entries to reduce memory during long-context generation:

```python
cache = QuantizedKVCache(group_size=64, bits=8)
```

Used when `--kv-bits` is specified. Trades precision for memory -- useful
for very long sequences.

### Cache Factory Pattern

Models define `make_cache()` to create the correct cache per layer:

```python
def make_cache(self):
    return [
        RotatingKVCache(max_size=self.sliding_window) if layer.use_sliding
        else KVCache()
        for layer in self.layers
    ]
```

The `make_prompt_cache()` function delegates to this method or falls back
to creating `KVCache` / `RotatingKVCache` based on `max_kv_size`.

## Generation Pattern

### Async Generation Pipeline

The generation loop in mlx-lm is carefully structured for latency:

```python
generation_stream = mx.new_stream(mx.default_device())

def _step(input_tokens):
    with mx.stream(generation_stream):
        logits = model(input_tokens[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampled = sampler(logprobs)
        return sampled, logprobs.squeeze(0)

# Prefill: process prompt in chunks
while prompt_remaining > 1:
    model(prompt_chunk[None], cache=prompt_cache)
    mx.eval([c.state for c in prompt_cache])
    mx.clear_cache()

# First token
y, logprobs = _step(last_prompt_token)
mx.async_eval(y, logprobs)

# Generation loop
while True:
    next_y, next_logprobs = _step(y)
    mx.async_eval(next_y, next_logprobs)
    mx.eval(y)
    yield y.item(), logprobs
    y, logprobs = next_y, next_logprobs
```

Pattern elements:
1. Dedicated stream for generation (`generation_stream`)
2. Prefill with chunking and explicit evaluation of cache state
3. Async pipeline: build next graph while previous evaluates
4. Periodic `mx.clear_cache()` every 256 tokens

### Batch Generation

mlx-lm supports batched generation via `BatchGenerator`:

- Left-pads shorter prompts in a batch
- Uses `BatchKVCache` which tracks per-sequence offsets
- Processes prefills in groups of `prefill_batch_size`
- Generates completions up to `completion_batch_size` concurrently
- Filters finished sequences in-place from the batch

## Quantization Patterns

### Weight Quantization

Models are quantized offline and loaded as `nn.QuantizedLinear`:

```python
# Loading handles this automatically via utils.load()
# QuantizedLinear stores:
#   - weight: uint32 packed quantized values
#   - scales: per-group scale factors
#   - biases: per-group bias values
#   - group_size: typically 64
#   - bits: typically 4

# Usage is transparent -- same interface as nn.Linear
output = quantized_layer(x)  # dequantize-multiply-accumulate in one kernel
```

### Model Loading with Quantization

```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

The `load` function handles:
- Downloading from HuggingFace
- Loading safetensors weights
- Applying `model.sanitize()` to filter weights
- Applying quantization config if present
- Loading adapters if specified

## LoRA Pattern

### Wrapping Existing Layers

LoRA in mlx-lm wraps existing linear layers without modifying the base model:

```python
class LoRALinear(nn.Module):
    @staticmethod
    def from_base(linear, r=8, dropout=0.0, scale=20.0):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, r, dropout, scale)
        lora_lin.linear = linear  # Keep original frozen
        return lora_lin

    def __call__(self, x):
        y = self.linear(x)                          # Frozen base
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b  # Low-rank update
        return y + (self.scale * z).astype(x.dtype)
```

Key design:
- `lora_a` initialized with small random values, `lora_b` initialized to zeros
- Scale factor applied to the low-rank update
- Result cast back to input dtype to prevent promotion
- `from_base()` works with both `nn.Linear` and `nn.QuantizedLinear`
- `fuse()` method merges LoRA weights back into the base for inference

### Freezing and Unfreezing

```python
model.freeze()  # Freeze all parameters
model.apply_to_modules(lambda k, m: m.unfreeze() if "lora" in k else None)
```

## RoPE Pattern

### Standard RoPE

```python
self.rope = nn.RoPE(dims=head_dim, traditional=False, base=10000)
# In attention:
queries = self.rope(queries, offset=cache.offset)
keys = self.rope(keys, offset=cache.offset)
```

### Custom RoPE (Llama3, Yarn, etc.)

All custom RoPE implementations follow the same interface:

```python
class Llama3RoPE(nn.Module):
    def __init__(self, dims, max_position_embeddings, base, scaling_config):
        super().__init__()
        # Compute custom frequency table
        self._freqs = computed_freqs

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x, self.dims, traditional=False,
            base=None, scale=1.0, offset=offset,
            freqs=self._freqs  # Custom frequencies
        )
```

The `initialize_rope()` factory dispatches based on `rope_type` in the config.
All variants ultimately call `mx.fast.rope` with custom frequency tables.

## Weight Sanitization Pattern

Models implement `sanitize(weights)` to clean up loaded weights:

```python
def sanitize(self, weights):
    # Remove precomputed rotary frequencies (recomputed at init)
    weights = {k: v for k, v in weights.items()
               if "rotary_emb.inv_freq" not in k}
    # Remove duplicate lm_head when tied
    if self.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
```

## Distributed / Sharding Pattern

mlx-lm supports model parallelism via `shard()`:

```python
def shard(self, group=None):
    group = group or mx.distributed.init()
    for layer in self.model.layers:
        layer.self_attn.q_proj = shard_linear(
            layer.self_attn.q_proj, "all-to-sharded", group=group
        )
        layer.self_attn.o_proj = shard_linear(
            layer.self_attn.o_proj, "sharded-to-all", group=group
        )
        layer.self_attn.n_heads //= group.size()
        layer.self_attn.n_kv_heads //= group.size()
        # Similarly for MLP layers
```

Sharding modes:
- `"all-to-sharded"`: Input replicated, output split across devices
- `"sharded-to-all"`: Input split, output all-reduced

## Training Loop Pattern

```python
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

for x, y in dataset:
    loss = step(x, y)
    mx.eval(model.parameters(), optimizer.state)
```

Key: `grads` is released before evaluation (function scope), reducing peak
memory. The `mx.eval` call evaluates both model params and optimizer state
together for efficiency.
