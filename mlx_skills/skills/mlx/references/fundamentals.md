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
