---
name: mlx
description: >
  Use when writing, debugging, reviewing, optimizing, or analyzing MLX code.
  Scan project imports for "import mlx", "from mlx", "mx.array", "mx.compile",
  "mx.eval" to determine applicability. Also triggers on "MLX", "Apple silicon
  ML", "nn.Module", "nn.Linear", "mlx.optimizers", "training loop", or any
  project using the mlx framework. Covers core concepts, nn module system,
  layers, optimizers, training patterns, and debugging. If the project also
  uses mlx-lm, load the mlx-lm skill. For performance optimization, load the
  fast-mlx skill.
---

# MLX

MLX is Apple's array framework for machine learning on Apple silicon. It looks
like NumPy and PyTorch but works fundamentally differently. You must understand
three things before writing any MLX code: lazy evaluation, unified memory, and
the compilation model.

## Core Concepts

### Lazy Evaluation

Every MLX operation builds a computation graph -- nothing executes until you
explicitly evaluate. Evaluate via `mx.eval(...)` or implicitly via `.item()`,
`.tolist()`, NumPy conversion, or printing. Evaluate at iteration boundaries:
after one training step, one token, or one denoising step.

For details on evaluation strategy, async evaluation, and the pipelining
pattern, see `references/fundamentals.md`.

### Unified Memory

CPU and GPU share the same memory on Apple silicon. No `.to(device)` or
`.cuda()` calls. Data stays in one place; the processor comes to the data.
Memory pressure is the main constraint.

### Compilation

`mx.compile` traces and fuses operations for faster execution. Be aware:
shape changes and constant input changes cause recompilation; closures over
`mx.array` values include the closed-over computation in the graph. For
details and examples, see `references/fundamentals.md`.

### Function Transformations

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

Python scalars are weakly typed, `mx.array` scalars are strongly typed.
Always use Python scalars for constants in half precision. For the full
promotion rules, see `references/fundamentals.md`.

## Ecosystem

| Layer | Package | Trust Level |
|-------|---------|-------------|
| Foundation | `mlx` (core) | Authoritative -- this IS the API |
| Gold Standard | `mlx-lm` | Official reference for LLM patterns |
| Functional | `mlx-vlm` | Third-party VLM; verify patterns against mlx-lm |

## Quick Reference: mx.fast Ops

| Op | Replaces |
|----|----------|
| `mx.fast.rms_norm` | Manual RMS normalization (accumulates in higher precision) |
| `mx.fast.layer_norm` | Manual layer normalization (accumulates in higher precision) |
| `mx.fast.rope` | Manual rotary position embedding |
| `mx.fast.scaled_dot_product_attention` | Manual attention computation |

Always prefer `mx.fast` ops over manual implementations.

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
- `references/nn-and-training.md` -- nn.Module system, all layers, losses,
  optimizers, schedulers, training loop patterns
- `references/anti-patterns.md` -- Common mistakes from NumPy/PyTorch habits
- `references/debugging.md` -- Shape debugging, evaluation issues, memory
  profiling, common errors

## Related Skills

- **`mlx-lm`** -- mlx-lm patterns: model architecture, generation, KV cache,
  quantization, LoRA fine-tuning, server
- **`fast-mlx`** -- Performance optimization: profiling, compilation tuning,
  memory reduction, async pipeline optimization
