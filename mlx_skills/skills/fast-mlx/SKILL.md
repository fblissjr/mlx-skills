---
name: fast-mlx
description: >
  Optimize MLX code for performance and memory. Use when asked to speed up MLX
  models or algorithms, reduce latency/throughput bottlenecks, tune lazy
  evaluation, type promotion, fast ops, compilation, memory use, or profiling.
  For general MLX concepts and patterns, load the mlx skill first.
---

# Fast MLX

## Prerequisites

Load the `mlx` skill first if you need background on lazy evaluation, type
promotion, or other MLX fundamentals. Load the `mlx-lm` skill for language
model-specific patterns. This skill focuses specifically on performance
optimization.

## Workflow

1. Identify evaluation boundaries and unintended sync points (`mx.eval`,
   `.item()`, NumPy conversions, `print` on arrays).
2. Check dtype promotion and scalar usage; keep precision consistent with intent.
   Use Python scalars (not `mx.array`) for constants in half-precision code.
3. Replace manual implementations with `mx.fast` ops: `rms_norm`, `layer_norm`,
   `rope`, `scaled_dot_product_attention`.
4. Look for opportunities to compile functions of mostly element-wise operations.
   For models with fixed-shape inputs, compile the entire forward pass.
5. Review compilation strategy; avoid unnecessary recompiles from changing
   constants, shapes, or closure captures.
6. Reduce peak memory via lazy loading order, releasing temporaries before
   evaluation, and periodic `mx.clear_cache()`.
7. Suggest profiling steps if the bottleneck is unclear.

## Domain-Specific Guides

Pick the guide that matches your optimization target:

- **LLM inference/training**: Read `references/llm-optimization.md` for KV cache
  tuning, async generation, prefill chunking, batch generation, and speculative
  decoding.
- **Diffusion / DiT models**: Read `references/dit-optimization.md` for denoising
  step compilation, CFG batching, vision attention, and memory management.
- **General compute**: Read `references/compute-optimization.md` for matrix ops,
  element-wise fusion, vmap, streaming, and data pipelines.

## General References

- Read `references/fast-mlx-guide.md` for the comprehensive performance guide
  covering graph evaluation, type promotion, operations, compile, memory, and
  profiling. Use it as the source of truth.

## Output Expectations

- Provide concrete code changes with brief rationale.
- Call out changes that need user confirmation (e.g., enabling async evaluation
  or shapeless compile).
- Quantify expected improvements when possible (e.g., "eliminates N
  recompilations per step" or "reduces peak memory by ~30%").
