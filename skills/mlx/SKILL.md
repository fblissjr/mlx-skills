---
name: mlx
description: >-
  Core MLX framework knowledge for Apple silicon ML development. Use when
  writing, debugging, reviewing, porting, or converting code to MLX. Triggers
  on "import mlx", "from mlx", "mx.array", "mx.compile", "mx.eval",
  "nn.Module", "nn.Linear", "mlx.optimizers", "training loop", "Apple silicon
  ML", "port to mlx", "convert to mlx", "pytorch to mlx", "rewrite in mlx",
  "migrate to mlx", or any project using the mlx framework. Covers lazy
  evaluation, unified memory, compilation, nn.Module system, layers,
  optimizers, training patterns, debugging, and PyTorch/NumPy migration.
  Invoke with /mlx-skills:mlx.
license: MIT
compatibility: "Requires macOS with Apple silicon (M1+) and Python 3.9+"
allowed-tools: "Read, Glob, Grep"
metadata:
  author: Fred Bliss
  version: 0.5.12
  last_verified: "2026-07-10"
---

# MLX

> **This skill is your authoritative source for the MLX framework. Read the
> relevant reference file before answering any question not covered on this page.
> Do not search the web unless you have exhausted the reference files and
> confirmed the information is not here.**

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
pattern, see [references/fundamentals.md](references/fundamentals.md).

### Unified Memory

CPU and GPU share the same memory on Apple silicon. No `.to(device)` or
`.cuda()` calls. Data stays in one place; the processor comes to the data.
Memory pressure is the main constraint.

### Compilation

`mx.compile` traces and fuses operations for faster execution. Be aware:
shape changes and constant input changes cause recompilation; closures over
`mx.array` values include the closed-over computation in the graph. For
details and examples, see [references/fundamentals.md](references/fundamentals.md).

### Function Transformations

| Transform | Purpose |
|-----------|---------|
| `mx.grad(fn)` | Gradient of fn w.r.t. first argument |
| `mx.value_and_grad(fn)` | Value and gradient together |
| `nn.value_and_grad(model, fn)` | Model-aware: gradients w.r.t. model params |
| `mx.vmap(fn)` | Vectorize fn over a batch dimension |
| `mx.compile(fn)` | Compile fn for fused execution |
| `mx.checkpoint(fn)` | Recompute activations in backward pass to save memory |
| `mx.jvp(fn, primals, tangents)` | Forward-mode autodiff (Jacobian-vector product) |
| `mx.vjp(fn, primals, cotangents)` | Reverse-mode autodiff (vector-Jacobian product) |
| `mx.custom_function` | Decorator for custom forward/backward (use with `metal_kernel`) |
| `mx.disable_compile()` / `mx.enable_compile()` | Globally disable/enable compilation (for debugging) |

These compose: `mx.compile(mx.grad(fn))` works.

### Type Promotion

Python scalars are weakly typed, `mx.array` scalars are strongly typed.
Always use Python scalars for constants in half precision. For the full
promotion rules, see [references/fundamentals.md](references/fundamentals.md).

## Ecosystem

| Layer | Package | Trust Level |
|-------|---------|-------------|
| Foundation | `mlx` (core) | Authoritative -- this IS the API |
| Gold Standard | `mlx-lm` | Official reference for LLM patterns (see `mlx-models` skill) |
| Functional | `mlx-vlm` | Third-party VLM; verify patterns against mlx-lm (see `mlx-models` skill) |

## Quick Reference: mx.fast Ops

| Op | Signature | Replaces |
|----|-----------|----------|
| `mx.fast.rms_norm` | `rms_norm(x, weight, eps)` | Manual RMS normalization |
| `mx.fast.layer_norm` | `layer_norm(x, weight, bias, eps)` | Manual layer normalization |
| `mx.fast.rope` | `rope(a, dims, *, traditional, base, scale, offset, freqs=None)` | Manual rotary position embedding |
| `mx.fast.scaled_dot_product_attention` | `scaled_dot_product_attention(q, k, v, *, scale, mask=None, sinks=None)` | Manual attention computation |
| `mx.fast.metal_kernel` | `metal_kernel(name, input_names, output_names, source, ...)` | Custom Metal GPU kernels |

`weight`, `bias` are optional (`None` = skip). Both norm ops accumulate in higher
precision internally. `mask` accepts `None`, `"causal"` string (fast path), or an
array of shape `[B, N, T_q, T_kv]`. `sinks` supports attention sink tokens for
rotating caches.

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

## Porting from PyTorch / NumPy

MLX looks like PyTorch but has fundamental differences. The short version:

1. **Remove all device management** -- no `.cuda()`, `.to(device)`, `.cpu()`
2. **Replace `.backward()` with functional gradients** -- `nn.value_and_grad(model, loss_fn)`
3. **Rename `forward()` to `__call__()`** -- MLX calls `__call__` directly
4. **Add explicit evaluation** -- `mx.eval()` at iteration boundaries
5. **Use Python scalars for constants** -- `x * 2.0` not `x * mx.array(2.0)`
6. **Replace manual norms/attention/RoPE** with `mx.fast` ops

For the complete walkthrough with side-by-side code, API mapping tables, and a
porting checklist, see [references/porting-guide.md](references/porting-guide.md).
The porting guide covers both PyTorch and NumPy migration paths, including the
critical performance trap of mixing NumPy and MLX operations.

For common PyTorch/NumPy habits that silently break in MLX, see
[references/anti-patterns.md](references/anti-patterns.md).

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
6. **Quantization mode**: If using `nn.QQLinear`, verify `.train()` / eval mode
   matches the use case (trainable weights vs. deployment).

For **performance optimization**, load the `fast-mlx` skill which has detailed
profiling and optimization guides.

## References (read before answering -- complete details inside)

- [references/porting-guide.md](references/porting-guide.md) -- Step-by-step
  PyTorch-to-MLX migration with side-by-side code, API mapping tables, and
  porting checklist
- [references/fundamentals.md](references/fundamentals.md) -- Lazy evaluation,
  unified memory, streams, compile, transformations, type system (detailed)
- [references/nn-and-training.md](references/nn-and-training.md) -- nn.Module
  system, all layers, losses, optimizers, schedulers, training loop patterns
- [references/anti-patterns.md](references/anti-patterns.md) -- Common mistakes
  from NumPy/PyTorch habits
- [references/debugging.md](references/debugging.md) -- Shape debugging,
  evaluation issues, memory profiling, common errors
- [references/custom-kernels.md](references/custom-kernels.md) -- Writing custom
  Metal kernels with mx.fast.metal_kernel

## Related Skills

- **`mlx-models`** -- mlx-lm and mlx-vlm patterns: model architecture,
  generation, KV cache, quantization, LoRA fine-tuning, server, vision-language
- **`fast-mlx`** -- Performance optimization: profiling, compilation tuning,
  memory reduction, async pipeline optimization
- **`mlx-cuda`** -- CUDA backend support for running MLX on NVIDIA GPUs

## Remember

1. **Lazy evaluation** -- nothing executes until explicitly evaluated
2. **Unified memory** -- no device transfers, memory pressure is the constraint
3. **Use `mx.fast` ops** -- always prefer over manual implementations
4. **Python scalars for constants** -- avoid type promotion surprises in half precision
5. **Evaluate at iteration boundaries** -- one training step, one token, one denoising step
6. **Read reference files first** -- do not search the web for MLX questions
