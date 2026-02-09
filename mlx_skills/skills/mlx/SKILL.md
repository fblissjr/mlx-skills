---
name: mlx
description: >
  Use when writing, debugging, reviewing, or analyzing MLX code. Triggers on
  "MLX", "mlx-lm", "mlx-vlm", "Apple silicon ML", "mx.array", "mx.compile",
  "mx.eval", or any project using the mlx framework. Covers core concepts,
  idiomatic patterns, ecosystem integration, and debugging.
---

# MLX

MLX is Apple's array framework for machine learning on Apple silicon. It looks
like NumPy and PyTorch but works fundamentally differently. You must understand
three things before writing any MLX code: lazy evaluation, unified memory, and
the compilation model.

## Core Concepts

### Lazy Evaluation

Every MLX operation builds a computation graph -- nothing executes until you
explicitly evaluate. This is the single most important difference from
NumPy/PyTorch.

```python
import mlx.core as mx

x = mx.array([1, 2, 3])
y = x + 1          # No computation yet -- just graph construction
z = y * 2          # Still no computation
mx.eval(z)         # NOW the entire graph executes
```

Evaluation happens explicitly via `mx.eval(...)` or implicitly when you call
`.item()`, `.tolist()`, convert to NumPy, or print. The evaluation granularity
matters:

- Evaluate too often: overhead from repeated graph dispatch
- Evaluate too rarely: graph gets large, high memory pressure

The right place to evaluate is at iteration boundaries: after one training step,
after generating one token, after one denoising step.

### Unified Memory

CPU and GPU share the same memory on Apple silicon. Arrays live in shared memory
and can be used by either processor without explicit transfers.

- No `.to(device)` or `.cuda()` calls
- No host-to-device copies
- Data stays in one place; the processor comes to the data
- Memory pressure is the main constraint (not VRAM vs RAM)

### Streams

MLX uses streams to order operations. All operations on the same stream execute
in order. Operations on different streams can run concurrently.

```python
# Default: all operations on the default GPU stream
s = mx.new_stream(mx.gpu)
with mx.stream(s):
    # Operations here go on stream s
    result = model(x)
```

The generation loop in mlx-lm uses a dedicated stream to pipeline graph
construction with computation via `mx.async_eval`.

### Compilation

`mx.compile` traces and fuses operations. Compiled functions run significantly
faster, especially for element-wise operations. Be aware of recompilation
triggers:

- **Shape changes** cause recompilation (default behavior). Use `shapeless=True`
  with caution.
- **Constant input changes** cause recompilation. Wrap varying scalars in
  `mx.array` to make them traceable inputs.
- **Closures** over `mx.array` values include the closed-over computation in
  the compiled graph. Pass arrays as explicit inputs or use `mx.compile(inputs=[...])`.

```python
from functools import partial

state = mx.array([1.0])

@partial(mx.compile, inputs=[state])
def step(x):
    return x + state  # state tracked as implicit input, not recomputed
```

### Function Transformations

MLX supports JAX-style function transformations:

| Transform | Purpose |
|-----------|---------|
| `mx.grad(fn)` | Gradient of fn w.r.t. first argument |
| `mx.value_and_grad(fn)` | Value and gradient together |
| `nn.value_and_grad(model, fn)` | Model-aware: gradients w.r.t. model params |
| `mx.vmap(fn)` | Vectorize fn over a batch dimension |
| `mx.compile(fn)` | Compile fn for fused execution |
| `mx.checkpoint(fn)` | Recompute activations in backward pass to save memory |

These compose: `mx.compile(mx.grad(fn))` works.

### Type Promotion

MLX promotes types to avoid precision loss. The critical rule: **Python scalars
are weakly typed**, `mx.array` scalars are strongly typed.

```python
x = mx.array([1.0], mx.float16)
x * 2.0             # float16 -- Python float is weak
x * mx.array(2.0)   # float32 -- mx.array default is float32, promotes!
```

Always use Python scalars for constants when working in half precision. If you
must use `mx.array`, match the dtype explicitly.

## Ecosystem

MLX has a clear hierarchy of trust:

| Layer | Package | Trust Level |
|-------|---------|-------------|
| Foundation | `mlx` (core) | Authoritative -- this IS the API |
| Gold Standard | `mlx-lm` | Official reference for LLM patterns |
| Functional | `mlx-vlm` | Third-party VLM; verify patterns against mlx-lm |

### mlx-lm (Gold Standard)

mlx-lm is the reference implementation for language models on MLX. When in
doubt about how to structure MLX code, look at mlx-lm first. Key patterns:

- **Model structure**: `ModelArgs` dataclass + `nn.Module` subclasses for
  Attention, MLP, TransformerBlock, Model
- **KV cache**: `KVCache` (standard), `RotatingKVCache` (sliding window),
  `QuantizedKVCache` (memory savings), `BatchKVCache` (batched generation)
- **Generation**: Async evaluation pipeline in `generate_step` with dedicated
  stream, prefill chunking, and `mx.async_eval` for latency hiding
- **Quantization**: `nn.QuantizedLinear` for weight quantization, quantized KV
  cache for activation memory
- **Fine-tuning**: LoRA via `LoRALinear.from_base()` wrapping existing layers

### mlx-vlm

Third-party vision-language model support. Uses mlx-lm patterns but adds vision
encoders. Verify its patterns match mlx-lm before adopting.

## Quick Reference: MLX vs Other Frameworks

| Concept | NumPy/PyTorch | MLX |
|---------|--------------|-----|
| Execution | Eager (immediate) | Lazy (deferred) |
| Evaluate | Automatic | `mx.eval()` or `mx.async_eval()` |
| Device transfer | `.cuda()`, `.to()` | Not needed (unified memory) |
| Gradients | `loss.backward()` | `mx.grad(fn)` / `nn.value_and_grad(model, fn)` |
| Compilation | `torch.compile` | `mx.compile` (explicit, composable) |
| RNG | Global state | Explicit key: `mx.random.key(seed)` |
| In-place ops | `x += 1` mutates | `x += 1` creates new node (immutable graphs) |
| Indexing | Full fancy indexing | Limited; prefer `mx.take_along_axis` |
| Normalization | Manual upcast needed | `mx.fast.rms_norm`, `mx.fast.layer_norm` accumulate in higher precision |

## Working with MLX Code

When writing or reviewing MLX code, check:

1. **Evaluation boundaries**: Is `mx.eval` called at the right granularity?
   Look for accidental evaluations (`.item()` in loops, NumPy conversions).
2. **Type promotion**: Are half-precision arrays accidentally promoted to
   float32 by `mx.array` scalar operations?
3. **Fast ops**: Use `mx.fast.scaled_dot_product_attention`,
   `mx.fast.rms_norm`, `mx.fast.layer_norm`, `mx.fast.rope` instead of
   manual implementations.
4. **Memory**: Is evaluation happening before temporaries are released? Are
   weights loaded lazily then cast before evaluation?
5. **Compilation**: Are compiled functions being recompiled unnecessarily?
   Check for changing shapes, constants, or captured arrays.

For **performance optimization**, load the `fast-mlx` skill which has detailed
profiling and optimization guides.

## References

Load these on demand for deeper guidance:

- `references/fundamentals.md` -- Lazy evaluation, unified memory, streams,
  compile, transformations, type system (detailed)
- `references/patterns.md` -- Idiomatic MLX patterns from mlx-lm: nn.Module,
  cache, attention, generation, quantization, LoRA, tree ops
- `references/anti-patterns.md` -- Common mistakes from NumPy/PyTorch habits
- `references/ecosystem.md` -- mlx-lm and mlx-vlm architecture details
- `references/debugging.md` -- Shape debugging, evaluation issues, memory
  profiling, common errors
