# Debugging MLX Code

Practical guide for diagnosing common issues in MLX programs.

## Shape Debugging

### Inspecting Shapes Without Evaluation

MLX arrays know their shape before evaluation. Use this for debugging without
triggering computation:

```python
x = model.layer1(input)
print(x.shape)   # Works! Shape is known from the graph
print(x.dtype)   # Also available without evaluation
# print(x)       # DON'T -- this forces evaluation
```

### Common Shape Issues

**Reshape vs transpose confusion:**
```python
# Attention reshape pattern: (B, L, n_heads * head_dim) -> (B, n_heads, L, head_dim)
# WRONG: reshape alone changes element order
x = x.reshape(B, self.n_heads, L, -1)  # Elements in wrong positions!

# CORRECT: reshape then transpose
x = x.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
```

**Batch dimension assumptions:**
```python
# mlx-lm always uses batch dimension, even for single sequences
# Model input should be (B, L), not just (L,)
logits = model(tokens)      # Wrong if tokens is 1-D
logits = model(tokens[None]) # Correct: adds batch dimension
```

**KV cache shapes:**
```python
# Cache expects (B, n_kv_heads, L, head_dim)
# Common mistake: forgetting that keys/values are already transposed
keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
# Now keys is (B, n_kv_heads, L, head_dim) -- correct for cache
```

## Evaluation Issues

### Symptoms of Missing Evaluation

- Program hangs or runs very slowly (graph growing unboundedly)
- Out-of-memory errors on small inputs (graph accumulating)
- Results are `mx.array` objects instead of values

**Fix:** Add `mx.eval()` at iteration boundaries.

### Symptoms of Over-Evaluation

- Slow performance despite simple operations
- GPU utilization is low and bursty
- Each iteration takes suspiciously similar time regardless of batch size

**Fix:** Remove intermediate evaluations. Consolidate to one per iteration.

### Detecting Accidental Evaluations

Common implicit evaluation triggers to watch for:

```python
# These all trigger evaluation:
if x > threshold:       # bool() on array
    ...

for item in array:      # Iteration forces evaluation
    ...

len(x)                  # May force evaluation if shape is dynamic

f"Loss: {loss}"         # String formatting evaluates
print(f"Shape: {x}")    # Print evaluates

np.array(x)             # NumPy conversion evaluates
x.tolist()              # Conversion evaluates
x.item()                # Scalar extraction evaluates
```

### Async Pipeline Debugging

If your async generation pipeline stalls:

1. **Check for synchronous evaluation inside `_step`**: Any `mx.eval`, `.item()`,
   print, or NumPy conversion inside the step function will stall the pipeline.

2. **Check stream conflicts**: Operations on the same stream as the pipeline
   that trigger synchronous evaluation will block.

3. **Verify the double-buffer pattern**:
   ```python
   # Must build NEXT computation BEFORE waiting for current
   next_y = _step(y)          # Build next graph
   mx.async_eval(next_y)      # Dispatch
   mx.eval(y)                 # NOW wait for previous
   ```

## Memory Debugging

### Monitoring Memory

```python
import mlx.core as mx

# Check current allocation
print(f"Active: {mx.metal.get_active_memory() / 1e9:.2f} GB")
print(f"Peak:   {mx.metal.get_peak_memory() / 1e9:.2f} GB")
print(f"Cache:  {mx.metal.get_cache_memory() / 1e9:.2f} GB")

# Reset peak tracking
mx.metal.reset_peak_memory()
```

### Common Memory Issues

**Graph accumulation:**
```python
# BAD: graph grows every iteration
losses = []
for x, y in dataset:
    loss = loss_fn(model, x, y)
    losses.append(loss)  # Holds reference to entire graph!
# Fix: evaluate and convert to Python float
losses.append(loss.item())
```

**Unreleased temporaries:**
```python
# BAD: grads held during evaluation
loss, grads = nn.value_and_grad(model, fn)(model, x, y)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)  # grads still alive

# GOOD: use function scope to release
def step(x, y):
    loss, grads = nn.value_and_grad(model, fn)(model, x, y)
    optimizer.update(model, grads)
    return loss  # grads released when function returns
```

**Cache bloat:**
```python
# Variable-size computation caches grow unboundedly
# Check with:
print(f"Cache: {mx.metal.get_cache_memory() / 1e9:.2f} GB")

# Fix: periodic cache clearing
mx.clear_cache()
```

### Model Size Estimation

```python
from mlx.utils import tree_flatten, tree_reduce

# Total model size in bytes
model_bytes = tree_reduce(
    lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
    model, 0
)
print(f"Model size: {model_bytes / 1e9:.2f} GB")

# Parameter count
num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Parameters: {num_params / 1e6:.1f}M")

# Check against device limits
max_rec = mx.device_info()["max_recommended_working_set_size"]
print(f"Max recommended: {max_rec / 1e9:.2f} GB")
print(f"Model/limit ratio: {model_bytes / max_rec:.2%}")
```

## Common Errors

### "Expected mx.array but got ..."

Usually a type mismatch. Check that you are passing `mx.array` and not NumPy
arrays or Python lists where MLX arrays are expected.

```python
# Fix: convert explicitly
x = mx.array(numpy_array)
```

### "Shapes don't match" in Compiled Functions

Shape changes trigger recompilation. If you see unexpected shape errors in
compiled code:

1. Check if input shapes are changing between calls
2. Consider if `shapeless=True` is appropriate (use with caution)
3. Verify that all paths produce the same output shape

### "Cannot convert array with dtype float64"

MLX does not support float64 on GPU. This usually happens when porting
NumPy code:

```python
# Fix: use float32
x = mx.array(data, dtype=mx.float32)

# Or when converting from NumPy:
x = mx.array(numpy_array.astype(np.float32))
```

### Slow First Iteration

The first iteration is always slower due to:
- Lazy weight loading (first access materializes from disk)
- Compilation (first call through `mx.compile` traces the graph)
- Cache allocation (first token allocates KV cache buffers)

This is normal. Measure from the second iteration onwards.

### Out of Memory

1. Check model size vs available memory (see estimation above)
2. Look for graph accumulation (references held across iterations)
3. Check for unreleased temporaries
4. Try `mx.clear_cache()` periodically
5. Use gradient checkpointing for training
6. Consider quantized models (4-bit reduces memory ~4x)
7. Use `mx.set_wired_limit()` to prevent OS paging

## Profiling

### GPU Utilization

Check GPU utilization first with a tool like `mactop`:
```bash
# Install: brew install mactop (or similar)
mactop
```

If GPU utilization is not near 100%, the bottleneck is likely:
- Data loading or preprocessing
- Python overhead
- Too-frequent evaluations

### Metal Debugger

For detailed kernel-level profiling, use Xcode's Metal debugger:

1. Capture a Metal frame
2. Look at the GPU timeline
3. Identify expensive kernels
4. Check for gaps (idle time between kernels)

See: https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html

### Timing Individual Operations

```python
import time

# Ensure previous work is done
mx.synchronize()
tic = time.perf_counter()

result = operation(x)
mx.eval(result)  # Must evaluate to measure actual compute time

toc = time.perf_counter()
print(f"Time: {toc - tic:.4f}s")
```

Always `mx.synchronize()` before timing and evaluate the result. Otherwise
you measure graph construction time, not computation time.

### Generation Metrics

mlx-lm's `GenerationResponse` includes built-in metrics:

```python
for response in stream_generate(model, tokenizer, prompt):
    pass  # Get last response

print(f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.1f} tok/s")
print(f"Generation: {response.generation_tokens} tokens, {response.generation_tps:.1f} tok/s")
print(f"Peak memory: {response.peak_memory:.2f} GB")
```
