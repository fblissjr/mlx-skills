last updated: 2026-02-23

# MLX Fundamentals

Deep reference for MLX core concepts. Read SKILL.md first for the overview.

## Lazy Evaluation In Depth

MLX builds a computation graph lazily. Every operation (`mx.add`, `mx.matmul`,
indexing, reshaping) returns an `mx.array` that represents the result but does
not contain computed data yet. The graph is only executed when evaluation is
triggered.

### Explicit Evaluation

```python
mx.eval(x)           # Evaluate x (and its full dependency graph)
mx.eval(x, y, z)     # Evaluate multiple arrays -- single graph dispatch
```

Always evaluate multiple arrays together when possible. Each `mx.eval` call
has dispatch overhead, and evaluating together allows the scheduler to
optimize across the full graph.

### Implicit Evaluation

These operations force evaluation automatically:

- `.item()` -- extracts a Python scalar
- `.tolist()` -- converts to Python list
- `np.array(x)` -- converts to NumPy
- `print(x)` -- printing evaluates
- `bool(x)` -- truth testing
- `len(x)` when shape is not yet known
- Python control flow that depends on array values (`if x > 0:`)

### Asynchronous Evaluation

`mx.async_eval` dispatches the graph for computation and returns immediately.
The result is available when explicitly synchronized (e.g., by calling
`mx.eval` on the same or dependent array).

This enables pipelining: build the next graph while the previous one computes.
mlx-lm uses this pattern in its generation loop:

```python
generation_stream = mx.new_stream(mx.default_device())

# First step
y, logprobs = _step(prompt)
mx.async_eval(y, logprobs)

while True:
    # Build next graph while previous computes
    next_y, next_logprobs = _step(y)
    mx.async_eval(next_y, next_logprobs)

    # Wait for previous result
    mx.eval(y)
    yield y.item(), logprobs

    y, logprobs = next_y, next_logprobs
```

Critical constraint: `_step` cannot contain any synchronous evaluations
internally, or the pipeline stalls.

### Evaluation Strategy Rules

1. **Iterative algorithms**: Evaluate once per iteration
2. **Training**: `mx.eval(model.parameters(), optimizer.state)` after each step
3. **Generation**: One evaluation per token via async pipeline
4. **Data loading**: Evaluate after cast/transform, before next batch

## Unified Memory Model

### How It Works

Apple silicon uses a unified memory architecture (UMA). CPU and GPU access the
same physical memory through a shared address space. MLX leverages this:

- Arrays are allocated once in shared memory
- No explicit transfers needed
- Both CPU and GPU can read/write the same arrays
- The memory limit is the total system memory (unified RAM), not a separate VRAM pool

### Memory Management

MLX caches recently freed memory buffers for reuse rather than returning them
to the system:

```python
mx.metal.get_active_memory()   # Currently allocated bytes
mx.metal.get_peak_memory()     # Peak allocation since last reset
mx.metal.get_cache_memory()    # Cached (freed but held) bytes
mx.metal.set_memory_limit(n)   # Set soft memory limit
mx.metal.set_cache_limit(n)    # Set cache size limit
mx.clear_cache()               # Free cached buffers
```

For long-running programs (servers, generation loops), periodically call
`mx.clear_cache()` to prevent the cache from growing unboundedly, especially
with variable-shape computations.

### Wired Memory

For large models, use `mx.set_wired_limit()` to pin model weights in
physical memory, preventing the OS from paging them out:

```python
max_rec_size = mx.device_info()["max_recommended_working_set_size"]
mx.set_wired_limit(max_rec_size)
```

This is important for models that approach the device's memory capacity.

## Streams

### Concept

A stream is an ordered sequence of operations. Within a stream, operations
execute in the order they were enqueued. Across streams, operations can run
concurrently (subject to data dependencies).

### Default Stream

All operations go to the default GPU stream unless explicitly directed elsewhere.
For most code, this is sufficient.

### Multiple Streams

Multiple streams enable concurrent execution on CPU and GPU, or pipelining:

```python
s = mx.new_stream(mx.gpu)

with mx.stream(s):
    result = heavy_computation(x)

# Operations outside the context go to the default stream
other_result = another_computation(y)
```

### Synchronization

- `mx.eval(x)` synchronizes: waits for x and all its dependencies
- `mx.synchronize()` waits for all outstanding work on all streams
- `mx.synchronize(stream)` waits for all work on a specific stream

The async generation pattern in mlx-lm uses a dedicated `generation_stream`
and synchronizes explicitly when reading results.

## Compilation

### How mx.compile Works

`mx.compile` traces a function by running it with symbolic inputs, capturing
the operation graph, and producing an optimized version. Benefits:

- **Operation fusion**: Multiple element-wise ops become a single kernel
- **Memory optimization**: Intermediate buffers can be reused
- **Reduced dispatch overhead**: One kernel launch instead of many

### When to Compile

Good candidates for compilation:

- Functions with many element-wise operations (activation functions, normalization)
- Loss functions
- Functions called repeatedly with the same shapes

Poor candidates:

- Functions with heavy control flow dependent on array values
- Functions that call `mx.eval` internally
- One-shot computations where compile overhead is not amortized

### Recompilation Triggers

```python
@mx.compile
def fn(x, scale):
    return x * scale

fn(mx.ones(10), 3)    # Compiles
fn(mx.ones(10), 4)    # Recompiles! scale is a constant (Python int)
fn(mx.ones(20), 3)    # Recompiles! shape changed
```

Fix constant recompilation by making varying inputs into `mx.array`:

```python
fn(mx.ones(10), mx.array(3))    # Compiles
fn(mx.ones(10), mx.array(4))    # Reuses compiled graph -- value changed, not shape
```

### Compiling Closures

When a compiled function captures an `mx.array` from an outer scope, the
captured array's entire computation is included in the compiled graph:

```python
y = expensive_computation()

@mx.compile
def fn(x):
    return x + y  # y's full computation graph is compiled in!
```

Fix by declaring `y` as an explicit input:

```python
y = expensive_computation()

@partial(mx.compile, inputs=[y])
def fn(x):
    return x + y  # y tracked as external input, not recomputed
```

Or pass it as a function argument:

```python
@mx.compile
def fn(x, y):
    return x + y
```

### Shapeless Compilation

`mx.compile(fn, shapeless=True)` avoids recompilation when input shapes change.
Use with extreme caution -- the compiled graph assumes all shapes are valid.
This can silently produce wrong results if shape-dependent logic exists.

## Function Transformations

### Gradients

MLX computes gradients via function transformations, not backpropagation on
tensors:

```python
def loss_fn(x):
    return (x ** 2).sum()

grad_fn = mx.grad(loss_fn)
g = grad_fn(mx.array([1.0, 2.0, 3.0]))  # g = [2.0, 4.0, 6.0]
```

For multiple return values, use `mx.value_and_grad`:

```python
loss, grads = mx.value_and_grad(loss_fn)(x)
```

For models, use the `nn` variant which differentiates w.r.t. model parameters:

```python
loss_fn = lambda model, x, y: nn.losses.cross_entropy(model(x), y).mean()
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
```

### Gradient Checkpointing

`mx.checkpoint` recomputes activations during the backward pass instead of
storing them, trading compute for memory:

```python
def grad_checkpoint(layer):
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn
```

This is used in mlx-lm's trainer for memory-constrained fine-tuning.

### Vectorization (vmap)

`mx.vmap` transforms a function that operates on single examples into one that
operates on batches:

```python
def single_loss(x, y):
    return ((x - y) ** 2).sum()

batched_loss = mx.vmap(single_loss)
losses = batched_loss(batch_x, batch_y)  # Vectorized over batch dim
```

## Type System

### Supported Types

| Category | Types |
|----------|-------|
| Float | `float16`, `bfloat16`, `float32` |
| Integer | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32` |
| Boolean | `bool_` |
| Complex | `complex64` |

Note: `float64` is **not supported** on GPU. This is a common gotcha when
porting NumPy code.

### Type Promotion Rules

MLX follows NumPy-like promotion but with an important distinction for scalar
types:

**Strong types** (`mx.array`): promote to the wider type
```python
mx.array(1.0, mx.float32) * mx.array(1.0, mx.float16)  # -> float32
mx.array(1.0, mx.bfloat16) * mx.array(1.0, mx.float16)  # -> float32
```

**Weak types** (Python scalars): adapt to the array type
```python
mx.array(1.0, mx.float16) * 2.0    # -> float16 (Python float is weak)
mx.array(1, mx.int8) + 1            # -> int8 (Python int is weak)
```

### Common Promotion Pitfalls

```python
# BAD: mx.array(2.0) defaults to float32, promotes everything
result = my_fp16_tensor * mx.array(2.0)  # -> float32!

# GOOD: Python scalar preserves dtype
result = my_fp16_tensor * 2.0  # -> float16

# BAD: default dtype in zeros
mask = mx.zeros(shape)  # float32 by default
result = my_fp16_tensor * mask  # -> float32!

# GOOD: explicit dtype
mask = mx.zeros(shape, dtype=mx.float16)
result = my_fp16_tensor * mask  # -> float16
```

## Tree Utilities

MLX provides utilities for working with nested structures (pytrees) of arrays,
following JAX conventions. These are essential for model parameter manipulation:

```python
from mlx.utils import tree_flatten, tree_unflatten, tree_map

# Flatten model parameters to list of (key, value) pairs
flat = tree_flatten(model.parameters())

# Apply a function to all arrays in a nested structure
half_params = tree_map(lambda x: x.astype(mx.float16), model.parameters())

# Count parameters
num_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
```

`tree_map` is used extensively in mlx-lm for operations like quantizing
KV caches, filtering parameters, and applying dtype conversions.

## Random Number Generation

MLX uses explicit PRNG keys (like JAX), not global state (like NumPy/PyTorch):

```python
mx.random.seed(42)                          # Set global seed
key = mx.random.key(42)                     # Create explicit key
x = mx.random.normal(shape=(3, 4))          # Uses global state
x = mx.random.normal(shape=(3, 4), key=key) # Uses explicit key
```

For reproducibility in parallel contexts, split keys:

```python
key1, key2 = mx.random.split(key)
```

## Indexing and Slicing

MLX supports basic indexing and slicing but has limitations compared to NumPy:

- **Supported**: Basic indexing, slicing, boolean masking, `mx.take_along_axis`
- **Limited**: Fancy indexing with integer arrays (less efficient)
- **Preferred**: `mx.take_along_axis` and `mx.put_along_axis` for gather/scatter

```python
# Prefer take_along_axis over fancy indexing
indices = mx.array([0, 2, 4])
result = mx.take_along_axis(x, indices[:, None], axis=0)

# In-place update syntax (builds graph node, not truly in-place)
x[0] = 1.0  # Creates a new computation node
```

Note: "in-place" updates like `x[i] = v` actually create new graph nodes. They
work correctly but do not save memory the way true in-place operations would.

## mx.fast Complete API

### rms_norm

```python
mx.fast.rms_norm(x, weight, eps, *, stream=None) -> array
```

Root Mean Square normalization over the last axis. `weight` is optional (pass
`None` to skip scaling). Accumulates in higher precision internally -- do not
manually upcast.

### layer_norm

```python
mx.fast.layer_norm(x, weight, bias, eps, *, stream=None) -> array
```

Layer normalization over the last axis. Both `weight` and `bias` are optional.
Accumulates in higher precision internally.

### rope

```python
mx.fast.rope(a, dims, *, traditional, base, scale, offset, freqs=None, stream=None) -> array
```

Rotary position embedding.

| Param | Type | Description |
|-------|------|-------------|
| `a` | array | Input, shape `(..., L, D)` |
| `dims` | int | Feature dimensions to rotate. If < D, only first `dims` are rotated |
| `traditional` | bool | `True` for the original RoPE formulation, `False` for the GPT-NeoX variant |
| `base` | float or None | Angular frequency base. Exactly one of `base` and `freqs` must be `None` |
| `scale` | float | Scale for positions (default 1.0) |
| `offset` | int or array | Position offset. Can be a scalar or `(B,)` array for per-batch offsets |
| `freqs` | array or None | Custom frequency table (used by Llama3RoPE, YarnRoPE, etc.) |

### scaled_dot_product_attention

```python
mx.fast.scaled_dot_product_attention(q, k, v, *, scale, mask=None, sinks=None, stream=None) -> array
```

Fast multi-head attention: `O = softmax(Q @ K.T * scale) @ V`.

| Param | Type | Description |
|-------|------|-------------|
| `q` | array | Queries, shape `(B, N_q, T_q, D)` |
| `k` | array | Keys, shape `(B, N_kv, T_kv, D)` |
| `v` | array | Values, shape `(B, N_kv, T_kv, D)` |
| `scale` | float | Typically `1.0 / sqrt(D)` |
| `mask` | None, str, or array | `None` (no mask), `"causal"` (fast-path lower-right causal), or additive mask `(B, N, T_q, T_kv)` |
| `sinks` | array or None | Attention sink scores for rotating caches with `keep` tokens |

Supports MHA, GQA, and MQA -- do not pre-tile K/V for GQA. Softmax computed in
float32 regardless of input dtype.

## Device and Stream API

| Function | Description |
|----------|-------------|
| `mx.Device` | Device object (`mx.cpu`, `mx.gpu`) |
| `mx.Stream` | Stream object for ordering operations |
| `mx.default_device()` | Get the default device |
| `mx.set_default_device(device)` | Set the default device |
| `mx.default_stream(device)` | Get the default stream for a device |
| `mx.new_stream(device)` | Create a new stream on a device |
| `mx.set_default_stream(stream)` | Set the default stream |
| `mx.stream(s)` | Context manager to route ops to stream `s` |
| `mx.synchronize(stream=None)` | Wait for all work (or work on a specific stream) |
| `mx.device_count(kind)` | Count available devices of a kind |
| `mx.device_info()` | Dict with device info including `max_recommended_working_set_size` |

## Additional Transforms

### Forward-Mode Autodiff (JVP)

```python
out, tangent_out = mx.jvp(fn, primals, tangents)
```

Computes the Jacobian-vector product (forward-mode). Returns the function output
and the tangent of the output. Useful for directional derivatives.

### Reverse-Mode Autodiff (VJP)

```python
out, vjp_fn = mx.vjp(fn, primals, cotangents)
```

Computes the vector-Jacobian product (reverse-mode). Lower-level than `mx.grad`
-- use when you need explicit control over the backward pass.

### Custom Functions

```python
@mx.custom_function
def my_fn(x):
    return forward_result

@my_fn.vjp
def my_fn_vjp(primals, cotangent, output):
    return (grad_x,)
```

Define custom forward and backward passes. Pair with `mx.fast.metal_kernel` to
write GPU-accelerated ops with differentiable gradients. See
[references/custom-kernels.md](../references/custom-kernels.md).

### Compile Control

```python
mx.disable_compile()  # Globally disable compilation (for debugging)
mx.enable_compile()   # Re-enable compilation
```

Or set the environment variable `MLX_DISABLE_COMPILE=1`.

## Distributed API

MLX supports distributed communication via `mx.distributed`:

| Function | Description |
|----------|-------------|
| `mx.distributed.init()` | Initialize and return the global communication group |
| `mx.distributed.is_available()` | Check if distributed backend is available |
| `group.size()` | Number of processes in the group |
| `group.rank()` | Rank of the current process |
| `mx.distributed.all_sum(x, group)` | Sum across all processes |
| `mx.distributed.all_gather(x, group)` | Gather arrays from all processes |
| `mx.distributed.send(x, dst, group)` | Send to a specific rank |
| `mx.distributed.recv(shape, dtype, src, group)` | Receive from a specific rank |

## Quantization

### nn.quantize

```python
nn.quantize(model, group_size=64, bits=4, class_predicate=None)
```

Quantize model weights in-place. Replaces `nn.Linear` layers with
`nn.QuantizedLinear`. The `class_predicate` function controls which layers to
quantize (default: all `nn.Linear` and `nn.Embedding` layers).

```python
# Quantize all linear layers to 4-bit
nn.quantize(model)

# Quantize only layers matching a predicate
nn.quantize(model, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear) and "lm_head" not in p)
```

## Memory Management Complete API

| Function | Description |
|----------|-------------|
| `mx.metal.get_active_memory()` | Currently allocated bytes |
| `mx.metal.get_peak_memory()` | Peak allocation since last reset |
| `mx.metal.reset_peak_memory()` | Reset peak memory counter |
| `mx.metal.get_cache_memory()` | Cached (freed but held) bytes |
| `mx.metal.set_memory_limit(n)` | Soft limit; allocations beyond this may fail or page |
| `mx.metal.set_cache_limit(n)` | Max bytes to keep in the buffer cache |
| `mx.set_wired_limit(n)` | Pin memory in physical RAM (prevents paging for large models) |
| `mx.clear_cache()` | Release all cached buffers back to the system |
