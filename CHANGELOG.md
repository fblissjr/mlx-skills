# Changelog

## 0.2.0

### Added

- **mlx skill**: Comprehensive MLX skill covering core concepts, ecosystem,
  patterns, anti-patterns, and debugging
  - `SKILL.md` with lazy evaluation, unified memory, streams, compilation,
    type promotion, ecosystem hierarchy, and framework comparison table
  - `references/fundamentals.md` with deep coverage of evaluation, memory
    model, streams, compilation, function transformations, and type system
  - `references/patterns.md` with idiomatic patterns from mlx-lm: nn.Module,
    attention, KV cache, generation, quantization, LoRA, RoPE, sharding
  - `references/anti-patterns.md` with common mistakes from NumPy/PyTorch
    habits and their MLX-correct alternatives
  - `references/ecosystem.md` with mlx-lm and mlx-vlm architecture,
    loading flow, generation flow, and integration patterns
  - `references/debugging.md` with shape debugging, evaluation issue
    diagnosis, memory profiling, and common error resolution
  - `scripts/check_updates.py` for scanning coderef repos and generating
    structured update reports

- **fast-mlx enhancements**: Domain-specific optimization guides
  - `references/llm-optimization.md` covering KV cache selection and tuning,
    async generation pipeline, prefill chunking, batch generation, speculative
    decoding, and memory budgeting
  - `references/dit-optimization.md` covering denoising step compilation,
    CFG batching, vision attention, and diffusion memory management
  - `references/compute-optimization.md` covering matrix ops, element-wise
    fusion, vmap, streaming, data pipelines, and numerical stability

### Changed

- Updated `fast-mlx/SKILL.md` with cross-reference to mlx skill and pointers
  to domain-specific optimization guides
- Updated `README.md` with documentation for both skills and maintenance workflow
- Updated `pyproject.toml` description and bumped version to 0.2.0

## 0.1.0

### Added

- Initial release with `fast-mlx` skill for MLX performance optimization
- CLI installer supporting Codex, Claude, OpenCode, and custom destinations
