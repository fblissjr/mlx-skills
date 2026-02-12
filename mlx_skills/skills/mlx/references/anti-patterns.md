# MLX Anti-Patterns

Common mistakes when writing MLX code, typically caused by NumPy/PyTorch
mental models. Each section shows the mistake, explains why it is wrong,
and provides the correct MLX idiom.

## Evaluation Mistakes

### Evaluating Inside Loops

```python
# BAD: evaluates on every iteration, killing performance
for x in output:
    do_something(x.item())  # .item() forces evaluation each time
```

This is equivalent to:
```python
for i in range(len(output)):
    x = output[i]
    mx.eval(x)          # Synchronous evaluation per element
    do_something(x.item())
```

```python
# GOOD: batch-convert first, then iterate in Python
for x in output.tolist():
    do_something(x)
```

Or better, do everything in MLX without leaving the framework.

### Evaluating Too Frequently in Training

```python
# BAD: multiple evaluations per step
for x, y in dataset:
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    mx.eval(loss)                    # Evaluation 1 -- unnecessary
    optimizer.update(model, grads)
    mx.eval(model, optimizer.state)  # Evaluation 2
```

```python
# GOOD: single evaluation at the end of the step
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

for x, y in dataset:
    loss = step(x, y)
    mx.eval(model.parameters(), optimizer.state)
```

### Breaking Async Pipeline

```python
# BAD: synchronous work on the computation stream stalls the pipeline
for out in generator():
    out = out * 2
    mx.eval(out)  # Blocks the generation stream!
```

```python
# GOOD: use a separate stream for the pipeline
def generator():
    with mx.stream(mx.new_stream(mx.gpu)):
        out = my_function()
        mx.async_eval(out)
        while True:
            out_next = my_function()
            mx.async_eval(out_next)
            mx.eval(out)
            yield out
            out = out_next
```

### Evaluating Inside Compiled Functions

```python
# BAD: mx.eval inside a compiled function breaks tracing
@mx.compile
def bad_fn(x):
    y = x * 2
    mx.eval(y)  # Forces evaluation, defeating compilation
    return y + 1
```

Compiled functions should be pure computation graphs with no side effects.

## Type Promotion Mistakes

### Accidental Float32 Promotion

```python
# BAD: mx.array(2.0) is float32, promotes fp16 input
x = my_fp16_tensor * mx.array(2.0)  # Result is float32!

# GOOD: Python scalar preserves dtype
x = my_fp16_tensor * 2.0  # Result is float16
```

### Default Dtype in Array Creation

```python
# BAD: mx.zeros defaults to float32
mask = mx.zeros(shape)
result = my_fp16_tensor * mask  # Promoted to float32!

# GOOD: match dtype explicitly
mask = mx.zeros(shape, dtype=my_fp16_tensor.dtype)
result = my_fp16_tensor * mask  # Stays float16
```

### Cross-Precision Operations

```python
# BAD: mixing bfloat16 and float16 promotes to float32
a = mx.array(1.0, mx.bfloat16)
b = mx.array(1.0, mx.float16)
c = a * b  # float32 -- neither half type can represent the other

# GOOD: pick one precision and stick with it
b = b.astype(mx.bfloat16)
c = a * b  # bfloat16
```

### Unnecessary Upcasting for Norms

```python
# BAD: manual upcast for normalization
x = mx.fast.rms_norm(x.astype(mx.float32), w, eps=eps).astype(x.dtype)

# GOOD: fast ops accumulate in higher precision internally
x = mx.fast.rms_norm(x, w, eps=eps)
```

Same for `mx.fast.layer_norm` and `mx.softmax(x, precise=True)`.

## Compilation Mistakes

For how `mx.compile` works (tracing, fusion, recompilation triggers), see
`references/fundamentals.md`. The mistakes below are the most common pitfalls.

### Recompiling on Every Call

```python
# BAD: changing Python scalar causes recompilation
@mx.compile
def fn(x, temperature):
    return x / temperature

for t in [0.5, 0.7, 0.9, 1.0]:
    fn(x, t)  # Recompiles 4 times!
```

```python
# GOOD: make varying values into mx.array
@mx.compile
def fn(x, temperature):
    return x / temperature

for t in [0.5, 0.7, 0.9, 1.0]:
    fn(x, mx.array(t))  # Compiles once, reuses
```

### Capturing Large Computations in Closures

```python
# BAD: y's entire computation is baked into the compiled graph
y = expensive_setup()

@mx.compile
def fn(x):
    return x + y  # Recomputes expensive_setup() in the compiled graph!
```

```python
# GOOD: declare as explicit input
from functools import partial

y = expensive_setup()

@partial(mx.compile, inputs=[y])
def fn(x):
    return x + y  # y is an external input, not recomputed
```

### Compiling Functions with Side Effects

```python
# BAD: stateful operations inside compiled functions
@mx.compile
def bad_train_step(x):
    loss = model(x)
    optimizer.update(model, mx.grad(model, loss_fn)(model, x))  # Mutates state
    return loss
```

Compiled functions should not mutate external state. Keep state updates outside
the compiled region.

## Memory Mistakes

### Holding Temporaries During Evaluation

```python
# BAD: grads reference held during mx.eval, can't reclaim memory
for x, y in dataset:
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)  # grads still in scope!
```

```python
# GOOD: wrap in function so grads is released before evaluation
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

for x, y in dataset:
    loss = step(x, y)
    mx.eval(model.parameters(), optimizer.state)
```

Or use explicit `del`:
```python
for x, y in dataset:
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    del grads  # Release before evaluation
    mx.eval(model.parameters(), optimizer.state)
```

### Loading Weights at Full Precision Then Casting

```python
# BAD: loads all weights as float32, THEN casts -- peak memory = 1.5x model size
weights = mx.load("model.safetensors")
mx.eval(weights)  # Everything materializes as float32
weights = {k: v.astype(mx.float16) for k, v in weights.items()}
```

```python
# GOOD: cast before evaluation -- lazy loading + cast fused
weights = mx.load("model.safetensors")
weights = {k: v.astype(mx.float16) for k, v in weights.items()}
mx.eval(weights)  # Loads as float32, casts immediately, reuses buffer
```

Peak memory reduced by nearly a third.

### Loading From File Handles

```python
# BAD: loading from file handle disables lazy loading
weights = mx.load(open("model.safetensors", "rb"))  # Entire file loaded!

# GOOD: pass string path for lazy loading
weights = mx.load("model.safetensors")  # Lazy -- loads on demand
```

### Not Clearing Cache in Long-Running Loops

```python
# BAD: variable-shape computations accumulate cached buffers
for batch in variable_length_batches:
    result = model(batch)
    mx.eval(result)
    # Memory grows as different-sized buffers accumulate in cache
```

```python
# GOOD: periodically clear the cache
for i, batch in enumerate(variable_length_batches):
    result = model(batch)
    mx.eval(result)
    if i % 256 == 0:
        mx.clear_cache()
```

## NumPy/PyTorch Habit Mistakes

### Using float64

```python
# BAD: float64 not supported on GPU
x = mx.array(1.0, mx.float64)  # Will fail or fall back to CPU

# GOOD: use float32 as the highest precision
x = mx.array(1.0, mx.float32)
```

### Expecting Eager Execution

```python
# BAD: printing mid-computation forces evaluation
x = model.layer1(input)
print(f"After layer 1: {x.shape}")  # Forces evaluation!
x = model.layer2(x)
```

Printing evaluates the array. For debugging, use shape inspection (which
does not require evaluation) or evaluate explicitly at intended boundaries.

### Using .backward() Pattern

```python
# BAD: PyTorch pattern
loss = model(x)
loss.backward()  # Does not exist in MLX!

# GOOD: functional transformation
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
```

### In-Place Mutation Expectations

```python
# MISLEADING: looks in-place but creates a new graph node
x[0] = 1.0  # New computation node, not a memory mutation

# This is correct syntax and works, but don't expect memory savings
# from "in-place" operations like you would in PyTorch
```

### Device Management

```python
# BAD: PyTorch pattern
x = x.to("mps")  # No device management needed!
x = x.cuda()      # Does not exist

# GOOD: just use the arrays -- unified memory handles it
x = mx.array([1, 2, 3])  # Works on GPU automatically
```

## Indexing Mistakes

### Fancy Indexing When take_along_axis Works

```python
# SLOW: fancy indexing
result = x[indices]

# FAST: explicit gather
result = mx.take_along_axis(x, indices[:, None], axis=0)
```

### Concatenation When Broadcasting Works

```python
# BAD: creates copies
result = mx.concatenate([a] * n)

# GOOD: uses broadcasting
result = mx.repeat(a, n)
```

## Matrix Multiplication Mistakes

### Transposing Wrong Side

```python
# For vector-matrix (single token through linear layer):
# SLOW
result = x @ W

# FAST -- transpose on the right side
result = x @ W.T

# For matrix-vector:
# SLOW
result = W.T @ x

# FAST -- no transpose needed
result = W @ x
```

### Not Using addmm for Bias

```python
# SLOW: separate matmul and add
result = x @ W.T + b

# FAST: fused operation
result = mx.addmm(b, x, W)
```

## QQLinear Mode Mistakes

### Forgetting Train Mode Before Fine-Tuning

```python
# BAD: weights stay quantized, no gradients flow
qlayer = nn.QQLinear.from_linear(layer, mode="nvfp4")
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)  # grads are zero!

# GOOD: call .train() so weights dequantize for training
qlayer = nn.QQLinear.from_linear(layer, mode="nvfp4")
model.train()  # Dequantizes QQLinear weights
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
```

### Using QQLinear When QuantizedLinear Suffices

If weights do not need to be trainable, prefer `nn.QuantizedLinear` -- it is
simpler and does not carry the train/eval mode complexity.

## Softmax Precision

```python
# BAD: manual upcast for softmax
scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)

# GOOD: use the precise flag
scores = mx.softmax(scores, axis=-1, precise=True)
```
