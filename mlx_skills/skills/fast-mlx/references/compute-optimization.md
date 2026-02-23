last updated: 2026-02-23

# General Compute Optimization Guide

Performance optimization techniques for general MLX computations: matrix ops,
element-wise operations, data pipelines, and custom algorithms.

## Matrix Operation Optimization

### Transpose Strategy

The order of transpose matters for performance:

```python
# Vector-matrix (common in single-token inference):
# FAST: transpose on the right
result = x @ W.T

# SLOW: no transpose
result = x @ W

# Matrix-vector:
# FAST: no transpose needed
result = W @ x

# SLOW: transpose on left
result = W.T @ x
```

### Fused Operations

Use `mx.addmm` for matmul-then-add (linear layer with bias):

```python
# SLOW: separate ops
result = x @ W.T + bias

# FAST: fused
result = mx.addmm(bias, x, W)
```

### Quantized Matrix Operations

For mixed-precision computation with quantized weights:

```python
# mx.quantized_matmul handles dequantize-multiply in one kernel
result = mx.quantized_matmul(
    x, weight, scales, biases,
    transpose=True, group_size=64, bits=4
)
```

This is what `nn.QuantizedLinear` uses internally.

## Element-wise Fusion via Compilation

### What to Compile

Functions with chains of element-wise operations benefit most from compilation:

```python
# GOOD candidate: many element-wise ops
@mx.compile
def gelu_approx(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

# GOOD candidate: activation + scale + residual
@mx.compile
def fused_residual(x, residual, scale):
    return residual + scale * mx.fast.rms_norm(x, weight, eps=eps)
```

### Compile Entire Forward Passes

When input shapes are fixed, compile the whole model call:

```python
compiled_model = mx.compile(model)
output = compiled_model(fixed_shape_input)
```

For variable shapes, compile subfunctions with fixed shapes:

```python
# Compile just the transformer block (shapes fixed within a call)
for layer in model.layers:
    layer.__call__ = mx.compile(layer.__call__)
```

### Element-wise Precision

```python
# Fast ops handle precision internally:
mx.fast.rms_norm(x, w, eps=1e-5)       # Accumulates in higher precision
mx.fast.layer_norm(x, w, b, eps=1e-5)  # Same
mx.softmax(x, axis=-1, precise=True)    # Higher precision accumulation
```

No need to manually upcast and downcast around these operations.

## Vectorization with vmap

### When vmap Helps

`mx.vmap` vectorizes a function over a batch dimension. It is useful when:

- You have a function that works on single examples
- Broadcasting alone does not express the operation
- You want to avoid explicit loops

```python
# Single-example function
def per_example_loss(x, y, mask):
    pred = model(x)
    loss = ((pred - y) ** 2 * mask).sum() / mask.sum()
    return loss

# Vectorized over batch
batched_loss = mx.vmap(per_example_loss)
losses = batched_loss(batch_x, batch_y, batch_masks)
```

### vmap with Grad

Compose `vmap` with `grad` for per-example gradients:

```python
per_example_grad = mx.vmap(mx.grad(per_example_loss))
grads = per_example_grad(batch_x, batch_y)
```

### vmap Limitations

- Not all MLX operations support vmap
- Complex control flow inside the vmapped function may fail
- For simple batched operations, explicit broadcasting is often simpler and faster

## Streaming and Pipelines

### CPU-GPU Pipelining

Use separate streams to overlap data preparation (CPU) with computation (GPU):

```python
compute_stream = mx.new_stream(mx.gpu)

# Prepare first batch
batch = prepare_data(raw_data[0])

for i in range(1, len(raw_data)):
    # Start GPU computation
    with mx.stream(compute_stream):
        result = model(batch)
        mx.async_eval(result)

    # Prepare next batch while GPU works (on default/CPU stream)
    next_batch = prepare_data(raw_data[i])

    # Wait for GPU result
    mx.eval(result)
    process_result(result)

    batch = next_batch
```

### Multi-Stream Computation

For independent computations, use separate streams:

```python
stream_a = mx.new_stream(mx.gpu)
stream_b = mx.new_stream(mx.gpu)

with mx.stream(stream_a):
    result_a = compute_a(x)
    mx.async_eval(result_a)

with mx.stream(stream_b):
    result_b = compute_b(y)
    mx.async_eval(result_b)

# Both computations can run concurrently
mx.eval(result_a, result_b)
```

## Data Pipeline Optimization

### Lazy Loading

Use `mx.load` with string paths (not file handles) for lazy loading:

```python
# GOOD: lazy -- data loaded on demand
data = mx.load("data.safetensors")

# Then cast before evaluation for minimal peak memory
data = {k: v.astype(mx.float16) for k, v in data.items()}
mx.eval(data)
```

### Batch Construction

```python
# SLOW: building arrays one at a time
batch = []
for item in dataset:
    batch.append(mx.array(item))
batch = mx.stack(batch)

# FAST: build as numpy first, convert once
import numpy as np
batch = np.stack([item for item in dataset])
batch = mx.array(batch)
```

### Avoid Repeated Small Evaluations

```python
# BAD: evaluation per item
processed = []
for item in data:
    result = transform(mx.array(item))
    mx.eval(result)
    processed.append(result)

# GOOD: batch transform and single evaluation
batch = mx.array(data)
results = transform(batch)
mx.eval(results)
```

## Broadcasting Optimization

### Prefer Broadcasting Over Concatenation

```python
# SLOW: concatenation creates copies
expanded = mx.concatenate([x] * n, axis=0)

# FAST: broadcasting with repeat
expanded = mx.repeat(x, n, axis=0)

# FASTEST when possible: reshape + broadcast
# e.g., to apply same weights to all batch items
# weights shape (d,) -> (1, d) broadcasts with (B, d) input
result = x * weights[None, :]
```

### Gather/Scatter Operations

```python
# SLOW: fancy indexing
selected = x[indices]

# FAST: explicit gather
selected = mx.take_along_axis(x, indices[:, None], axis=0)

# For scatter (writing at indices):
mx.put_along_axis(x, indices[:, None], values, axis=0)
```

## Numerical Stability

### Softmax

```python
# Use precise=True for numerical stability without manual casting
probs = mx.softmax(logits, axis=-1, precise=True)
```

### Log-Sum-Exp

```python
# MLX provides a numerically stable logsumexp
log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
```

### Loss Computation

```python
# Cast loss to float32 for accumulation, even if model runs in fp16
ce = nn.losses.cross_entropy(logits, targets)
ce = ce.astype(mx.float32).sum() / num_tokens
```

This pattern is used in mlx-lm's training to prevent numerical issues from
accumulating half-precision losses.

## Memory Optimization

### Release Intermediates

```python
# Wrap computation to release intermediates
def compute_and_reduce(x):
    features = expensive_features(x)
    reduced = aggregate(features)
    return reduced  # features released after return

result = compute_and_reduce(input_data)
mx.eval(result)
```

### Periodic Cache Clearing

For any long-running computation with varying shapes:

```python
for i, batch in enumerate(batches):
    result = process(batch)
    mx.eval(result)
    if i % 100 == 0:
        mx.clear_cache()
```

### Memory Monitoring

```python
mx.metal.reset_peak_memory()

# ... your computation ...

print(f"Peak memory: {mx.metal.get_peak_memory() / 1e9:.2f} GB")
print(f"Active: {mx.metal.get_active_memory() / 1e9:.2f} GB")
print(f"Cache: {mx.metal.get_cache_memory() / 1e9:.2f} GB")
```
