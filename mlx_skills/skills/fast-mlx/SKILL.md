---
name: fast-mlx
description: >
  Optimize MLX code for performance and memory on Apple silicon. Use when asked
  to speed up, optimize, profile, or benchmark MLX models or algorithms.
  Triggers on "optimize mlx", "speed up", "reduce latency", "throughput",
  "profiling", "benchmark", "memory optimization", "mx.compile tuning",
  "slow inference", "reduce memory", or "make it faster". Covers graph
  evaluation, type promotion, fast ops, compilation strategy, memory
  management, and domain-specific optimization for LLMs and diffusion models.
metadata:
  author: Fred Bliss
  version: 0.4.0
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

- **LLM inference/training**: Read [references/llm-optimization.md](references/llm-optimization.md)
  for KV cache tuning, async generation, prefill chunking, batch generation,
  and speculative decoding.
- **Diffusion / DiT models**: Read [references/dit-optimization.md](references/dit-optimization.md)
  for denoising step compilation, CFG batching, vision attention, and memory
  management.
- **General compute**: Read [references/compute-optimization.md](references/compute-optimization.md)
  for matrix ops, element-wise fusion, vmap, streaming, and data pipelines.

## General References

- Read [references/fast-mlx-guide.md](references/fast-mlx-guide.md) for the
  comprehensive performance guide covering graph evaluation, type promotion,
  operations, compile, memory, and profiling. Use it as the source of truth.

## Output Expectations

- Provide concrete code changes with brief rationale.
- Call out changes that need user confirmation (e.g., enabling async evaluation
  or shapeless compile).
- Quantify expected improvements when possible (e.g., "eliminates N
  recompilations per step" or "reduces peak memory by ~30%").

## Remember

1. **Profile before optimizing** -- identify the actual bottleneck first
2. **Check dtype promotion first** -- accidental float32 is the most common perf issue
3. **`mx.fast` ops before compile** -- fused ops give guaranteed gains
4. **Compile whole forward passes** -- when shapes are fixed, compile aggressively
5. **Release temporaries before evaluating** -- `del` intermediates to reduce peak memory
