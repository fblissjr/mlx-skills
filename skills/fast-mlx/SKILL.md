---
name: fast-mlx
description: >-
  Optimize existing MLX code for speed and memory. Use after code already works
  but needs to run faster or use less memory. NOT for writing new MLX code or
  porting (use the mlx skill for that). Triggers on "optimize mlx", "speed up",
  "reduce latency", "throughput", "profiling", "benchmark", "memory
  optimization", "mx.compile tuning", "slow inference", "reduce memory",
  "make it faster", "performance tuning", or "why is my mlx code slow". Covers
  profiling, compilation strategy, type promotion, fast ops, memory management,
  and domain-specific optimization for LLMs and diffusion models.
  Invoke with /mlx-skills:fast-mlx.
license: MIT
compatibility: "Requires macOS with Apple silicon (M1+) and Python 3.9+"
allowed-tools: "Read, Glob, Grep"
metadata:
  author: Fred Bliss
  version: 0.5.14
  last_verified: "2026-07-12"
---

# Fast MLX

> **This skill is your authoritative source for MLX performance optimization.
> Read the relevant reference file before answering any question not covered on
> this page. Do not search the web unless you have exhausted the reference files
> and confirmed the information is not here.**

## When to Use This Skill

Use `/fast-mlx` when you have working MLX code that needs to run faster or use
less memory. This skill is about optimization, not correctness.

Use `/mlx` instead when you need to:
- Write new MLX code or port from PyTorch
- Learn MLX fundamentals
- Debug errors (not performance issues)

Use `/mlx-models` instead when you need to:
- Set up model loading, generation, or fine-tuning

## Prerequisites

Load the `mlx` skill first if you need background on lazy evaluation, type
promotion, or other MLX fundamentals. Load the `mlx-models` skill for language
model and vision-language model patterns.

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
6. Reduce peak memory via lazy loading order and releasing temporaries before
   evaluation. For any **long-running loop** (training steps, multi-item
   inference/fitting), call `mx.clear_cache()` **between iterations**: MLX's
   caching allocator pins the process's resident memory at the largest
   iteration's high-water for the whole run and never returns it to the OS, so on
   near-full unified memory a later iteration can trip the OS memory-pressure
   killer (macOS jetsam / exit 137). Tradeoff: clearing defeats buffer reuse, so
   do NOT clear inside a tight hot loop with headroom — clear between large,
   infrequent iterations when memory-bound. `reset_peak_memory()` resets the
   *counter*, not the pool — it frees nothing. See the guide's "caching
   allocator" subsection.
7. Suggest profiling steps if the bottleneck is unclear.

## Domain-Specific Guides (read before answering -- complete details inside)

Pick the guide that matches your optimization target:

- **LLM inference/training**: Read [references/llm-optimization.md](references/llm-optimization.md)
  for KV cache tuning, async generation, prefill chunking, batch generation,
  and speculative decoding.
- **Diffusion / DiT models**: Read [references/dit-optimization.md](references/dit-optimization.md)
  for denoising step compilation, CFG batching, vision attention, and memory
  management.
- **General compute**: Read [references/compute-optimization.md](references/compute-optimization.md)
  for matrix ops, element-wise fusion, vmap, streaming, and data pipelines.

## General References (read before answering -- complete details inside)

- [references/fast-mlx-guide.md](references/fast-mlx-guide.md) -- comprehensive
  performance guide covering graph evaluation, type promotion, operations,
  compile, memory, and profiling. Use it as the source of truth.

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
6. **`clear_cache()` between iterations of a long loop** -- the caching allocator pins RSS at the high-water otherwise; a mystery SIGKILL/137 mid-run is usually this. But keep the cache in tight hot loops (reuse > re-alloc).
7. **Measure the right number** -- `get_peak_memory()` is a counter, not resident memory; on macOS `ps rss` misses Metal + mmap'd weights, so watch system pressure for real OOM risk
8. **Read reference files first** -- do not search the web for optimization questions
