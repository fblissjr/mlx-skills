# mlx-lm Architecture

Architecture and integration patterns for mlx-lm, Apple's official language
model library for MLX.

## Directory Structure

```
mlx_lm/
  models/
    base.py             BaseModelArgs, create_attention_mask, scaled_dot_product_attention
    cache.py            KVCache, RotatingKVCache, QuantizedKVCache, BatchKVCache, etc.
    rope_utils.py       RoPE variants: Llama3RoPE, YarnRoPE, SuScaledRoPE
    mla.py              MultiLinear for Multi-head Latent Attention
    llama.py            Reference model implementation
    deepseek_v3.py      DeepSeek V3 (MLA attention)
    qwen2.py, mistral.py, gemma2.py, ...  (50+ models)
    activations.py      swiglu and other activation functions
    switch_layers.py    MoE (Mixture of Experts) layers
  tuner/
    lora.py             LoRALinear, LoRASwitchLinear, LoRAEmbedding
    dora.py             Weight-decomposed LoRA
    trainer.py          Training loop, gradient checkpointing
    datasets.py         Data loading and batching
    losses.py           Training loss functions
  generate.py           generate_step, stream_generate, batch_generate, BatchGenerator
  utils.py              load(), quantize(), model I/O
  server.py             OpenAI-compatible HTTP server
  sample_utils.py       Temperature, top-p, top-k, min-p, XTC samplers
  tokenizer_utils.py    TokenizerWrapper for unified tokenizer interface
  tool_parsers/         Model-specific tool/function calling parsers
  chat_templates/       Chat template files for tool message formatting
```

## Model Loading Flow

```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

What happens internally:

1. Download model from HuggingFace (or use local path)
2. Load `config.json` -> determine model architecture
3. Instantiate `ModelArgs` from config via `BaseModelArgs.from_dict()`
4. Instantiate `Model(args)` -- creates all layers
5. Load safetensors weights (lazy -- no data loaded yet)
6. Call `model.sanitize(weights)` to filter unnecessary keys
7. Apply quantization config if present (replaces `nn.Linear` with `nn.QuantizedLinear`)
8. Load adapter weights if `adapter_path` specified
9. Materialize weights by evaluating model parameters

## Generation Flow

```
Input: prompt string
  |
  v
Tokenize -> mx.array of token IDs
  |
  v
Prefill: process prompt in chunks of prefill_step_size
  |  - model(chunk, cache=prompt_cache) for each chunk
  |  - Evaluate cache state after each chunk
  |  - mx.clear_cache() after each chunk
  |
  v
First token: _step(last_token) -> (y, logprobs)
  |  - mx.async_eval(y, logprobs)
  |
  v
Generation loop (async pipeline):
  |  while n < max_tokens:
  |    next_y, next_logprobs = _step(y)
  |    mx.async_eval(next_y, next_logprobs)
  |    Wait for previous result
  |    yield y.item(), logprobs
  |    y, logprobs = next_y, next_logprobs
  |
  v
Detokenize -> text
```

## Model Registration

Each model implements a standard interface:

```python
# Required in each model file:
class ModelArgs(BaseModelArgs):
    model_type: str  # Must match config.json's model_type
    ...

class Model(nn.Module):
    def __init__(self, args: ModelArgs): ...
    # inputs: (B, L) token IDs -> returns (B, L, vocab_size) logits
    def __call__(self, inputs, cache=None, input_embeddings=None): ...

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        return filtered_weights
```

The model is discovered by matching `model_type` in the config to the model
file. The convention is `model_type = "llama"` maps to `models/llama.py`.

### Required Interface

| Method/Property | Purpose |
|-----------------|---------|
| `__call__(inputs, cache, input_embeddings)` | Forward pass returning logits |
| `layers` property | Access to transformer layers for iteration |
| `make_cache()` | Create appropriate cache type per layer |
| `sanitize(weights)` | Filter/rename loaded weights before applying |
| `shard(group)` (optional) | Distribute model across multiple devices |

## Fine-Tuning Flow

```python
from mlx_lm import lora

# Apply LoRA adapters
lora.apply(model, lora_config)  # Wraps target layers with LoRALinear

# Train
model.freeze()
# Only LoRA parameters are trainable
model.apply_to_modules(lambda k, m: m.unfreeze() if "lora" in k else None)

# Standard training loop with gradient checkpointing
for batch in dataset:
    loss = step(batch)
    # Evaluate model parameters and optimizer state
    mx.eval(model.parameters(), optimizer.state)

# Set val_batches=0 to skip validation (validation set is optional)

# Save adapters (only LoRA weights)
mx.save_safetensors(
    "adapters.safetensors",
    dict(tree_flatten(model.trainable_parameters()))
)

# Fuse adapters for inference
for name, module in model.named_modules():
    if hasattr(module, "fuse"):
        parent[name] = module.fuse()  # Merges LoRA weights into base
```

## Integration Patterns

### Using mlx-lm as a Library

```python
from mlx_lm import load, generate, stream_generate

# Load
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Simple generation
text = generate(model, tokenizer, "Hello", max_tokens=100)

# Streaming generation
for response in stream_generate(model, tokenizer, "Hello", max_tokens=100):
    print(response.text, end="", flush=True)

# Batch generation
from mlx_lm import batch_generate
results = batch_generate(model, tokenizer, prompts, max_tokens=100)
```

### Custom Model Integration

To add a new model to mlx-lm:

1. Create `models/your_model.py`
2. Define `ModelArgs(BaseModelArgs)` with `model_type` matching HuggingFace config
3. Implement `Model(nn.Module)` with required interface
4. Implement `sanitize()` to handle weight name mismatches
5. Optionally implement `make_cache()` for custom cache types
6. Optionally implement `shard()` for distributed inference

### Server Integration

mlx-lm provides an OpenAI-compatible server:

```bash
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

This exposes `/v1/chat/completions` and `/v1/completions` endpoints using
the `BatchGenerator` for concurrent request handling.

## Sampling

mlx-lm implements several sampling strategies in `sample_utils.py`:

| Strategy | Description |
|----------|-------------|
| Temperature | Scale logits before softmax to control randomness |
| Top-p (nucleus) | Sample from smallest set whose cumulative probability exceeds p |
| Top-k | Sample from the k most likely tokens |
| Min-p | Filter tokens below a minimum probability threshold |
| Repetition penalty | Reduce probability of recently generated tokens |
| XTC | Extended temperature-controlled sampling |

Samplers are composable and applied in sequence during the `_step` function
within the generation loop.

## Tool Calling

The mlx-lm server supports function/tool calling via model-specific parsers in
`mlx_lm/tool_parsers/`. Each parser implements `parse_tool_call(text, tools)`
to extract structured function calls from model output. Available parsers
include Mistral, Pythonic, GLM-4.7, Kimi K2, LongCat, Qwen3-Coder, and others.
Chat templates in `mlx_lm/chat_templates/` handle tool message formatting for
the corresponding models.

## mlx-vlm

mlx-vlm is a third-party library that extends mlx-lm patterns for
vision-language models. It supports 48+ VLM architectures including audio
(Qwen3-Omni-MoE).

### Key Differences from mlx-lm

1. **Two-stage architecture**: Vision encoder + Language model
2. **Processor-centric**: Depends on HuggingFace Transformers processors for
   image/audio preprocessing
3. **Multi-modal inputs**: Both text tokens and image/audio embeddings
4. **Shared utilities**: Uses mlx-lm's `make_sampler`, `make_logits_processors`,
   and `maybe_quantize_kv_cache` directly

### Trust Level

mlx-vlm is third-party code. When using its patterns:

- Verify attention implementations match mlx-lm conventions
- Check that KV cache usage follows the standard pattern
- Confirm fast ops are used where appropriate
- Watch for type promotion issues in the vision encoder
