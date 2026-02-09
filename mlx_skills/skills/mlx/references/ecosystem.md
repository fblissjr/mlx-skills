# MLX Ecosystem

Architecture and integration patterns for the MLX ecosystem packages.

## Ecosystem Overview

```
mlx (core)
  |
  +-- mlx.core (mx)      Array operations, lazy evaluation, compilation
  +-- mlx.nn              Neural network layers, optimizers, losses
  +-- mlx.utils           Tree utilities, serialization
  +-- mlx.optimizers      SGD, Adam, AdamW, Lion, etc.
  |
  +-- mlx-lm (official)   Language model inference and fine-tuning
  |     +-- models/        40+ model architectures (Llama, Mistral, Qwen, etc.)
  |     +-- tuner/         LoRA, DoRA fine-tuning
  |     +-- generate.py    Single and batch generation
  |     +-- server.py      OpenAI-compatible API server
  |
  +-- mlx-vlm (third-party) Vision-language models
        +-- models/         VLM architectures (LLaVA, Qwen-VL, etc.)
        +-- utils.py        Model loading, generation
```

## mlx-lm Architecture

### Directory Structure

```
mlx_lm/
  models/
    base.py             BaseModelArgs, create_attention_mask, scaled_dot_product_attention
    cache.py            KVCache, RotatingKVCache, QuantizedKVCache, BatchKVCache, etc.
    rope_utils.py       RoPE variants: Llama3RoPE, YarnRoPE, SuScaledRoPE
    llama.py            Reference model implementation
    qwen2.py, mistral.py, gemma2.py, ...  (40+ models)
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
```

### Model Loading Flow

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
9. Call `mx.eval(model.parameters())` to materialize weights

### Generation Flow

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

### Model Registration

Each model implements a standard interface:

```python
# Required in each model file:
class ModelArgs(BaseModelArgs):
    model_type: str  # Must match config.json's model_type
    ...

class Model(nn.Module):
    def __init__(self, args: ModelArgs): ...
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

### Fine-Tuning Flow

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
    mx.eval(model.parameters(), optimizer.state)

# Save adapters (only LoRA weights)
mx.save_safetensors("adapters.safetensors", dict(tree_flatten(model.trainable_parameters())))

# Fuse adapters for inference
for name, module in model.named_modules():
    if hasattr(module, "fuse"):
        parent[name] = module.fuse()  # Merges LoRA weights into base
```

## mlx-vlm Architecture

mlx-vlm extends mlx-lm patterns for vision-language models.

### Key Differences from mlx-lm

1. **Two-stage architecture**: Vision encoder + Language model
2. **Image preprocessing**: Resizing, normalization, patch extraction
3. **Multi-modal inputs**: Both text tokens and image embeddings
4. **Cross-attention or projection**: Different VLMs connect vision to language differently

### Common VLM Pattern

```python
class VLMModel(nn.Module):
    def __init__(self, config):
        self.vision_model = VisionEncoder(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size
        )

    def __call__(self, input_ids, pixel_values=None, cache=None):
        if pixel_values is not None:
            image_features = self.vision_model(pixel_values)
            image_features = self.multi_modal_projector(image_features)
            # Merge image and text embeddings
            inputs_embeds = self._merge_embeddings(input_ids, image_features)
        else:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        return self.language_model(input_ids, cache=cache, input_embeddings=inputs_embeds)
```

### Trust Level

mlx-vlm is third-party code. When using its patterns:

- Verify attention implementations match mlx-lm conventions
- Check that KV cache usage follows the standard pattern
- Confirm fast ops are used where appropriate
- Watch for type promotion issues in the vision encoder

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

## Core MLX Layers Reference

Key layers from `mlx.nn`:

| Layer | Notes |
|-------|-------|
| `nn.Linear` | Standard linear; stores `weight` as `(output, input)` |
| `nn.QuantizedLinear` | Quantized weights; transparent interface |
| `nn.Embedding` | Standard embedding lookup |
| `nn.RMSNorm` | Root mean square normalization |
| `nn.LayerNorm` | Standard layer normalization |
| `nn.RoPE` | Rotary position embeddings |
| `nn.Dropout` | Standard dropout (only during training) |
| `nn.MultiHeadAttention` | Built-in MHA (mlx-lm implements custom) |
| `nn.Transformer` | Built-in transformer (mlx-lm implements custom) |

Note: mlx-lm implements its own attention and transformer blocks rather than
using `nn.MultiHeadAttention`/`nn.Transformer`, because the custom
implementations integrate better with KV caching, RoPE variants, and
quantization.

## Optimizers Reference

From `mlx.optimizers`:

| Optimizer | Key Args |
|-----------|----------|
| `SGD` | `learning_rate`, `momentum`, `weight_decay` |
| `Adam` | `learning_rate`, `betas`, `eps` |
| `AdamW` | `learning_rate`, `betas`, `eps`, `weight_decay` |
| `Adagrad` | `learning_rate`, `eps` |
| `Lion` | `learning_rate`, `betas`, `weight_decay` |

Usage:
```python
optimizer = optim.AdamW(learning_rate=1e-5, weight_decay=0.01)
optimizer.update(model, grads)
```

With learning rate schedules:
```python
schedule = optim.linear_schedule(1e-5, 1e-6, steps=1000)
optimizer = optim.AdamW(learning_rate=schedule)
```
