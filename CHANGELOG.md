# Changelog

## 0.4.0

### Added

- `nn.QQLinear` documentation: trainable quantized linear layer with `nvfp4`
  and `mxfp8` modes in nn-and-training.md and anti-patterns.md
- `Muon` optimizer documentation with MultiOptimizer pairing guidance
- `nn.init.sparse` and `nn.init.orthogonal` initializer documentation
- Distributed layers section: `AllToShardedLinear`, `ShardedToAllLinear`,
  quantized variants, and `nn.shard_linear` factory
- Multi-head Latent Attention (MLA) pattern from DeepSeek V3 in mlx-lm
  patterns.md with `MultiLinear` and compressed KV cache
- New cache types: `CacheList`, `ChunkedKVCache`, `ArraysCache`,
  `BatchRotatingKVCache` in mlx-lm SKILL.md and patterns.md
- Speculative decoding pattern with draft model in mlx-lm patterns.md and
  fast-mlx llm-optimization.md
- MLA cache optimization subsection in fast-mlx llm-optimization.md
- Tool calling section in mlx-lm architecture.md (tool_parsers/, chat_templates/)
- QQLinear mode checklist item in mlx SKILL.md

### Changed

- Updated `nn.QuantizedLinear` signature: new `mode` parameter (`"affine"`,
  `"mxfp4"`, `"nvfp4"`, `"mxfp8"`); `group_size`/`bits` now default based on mode
- Updated `nn.QuantizedEmbedding` with same `mode` parameter support
- Updated mlx-lm SDPA wrapper with `sinks` parameter for attention sinks
- Updated mlx-lm model architecture count from 40+ to 50+
- Updated mlx-vlm section: 48+ VLM architectures, processor-centric design,
  shared mlx-lm utilities (make_sampler, make_logits_processors)
- Updated speculative decoding in fast-mlx with `trim_prompt_cache` rewind
  mechanism and CLI parameters
- Added `val_batches=0` skip-validation note to fine-tuning flow

## 0.3.1

### Added

- `mlx_skills/validate.py` -- Validation script for skill plugin structure,
  frontmatter, word counts, reference file existence, and cross-references
- `mlx-skills-validate` CLI entrypoint in pyproject.toml
- `tests/` -- Test suite with pytest (conftest, test_validate, test_cli,
  test_skill_structure)
- pytest dev dependency and pytest config in pyproject.toml

### Changed

- Deduplicated `mlx/SKILL.md` (~970 words -> ~480 words): lazy eval,
  compilation, type promotion sections trimmed to summaries with pointers
  to `references/fundamentals.md`
- Deduplicated `mlx-lm/SKILL.md` (~860 words -> ~430 words): generation,
  KV cache, fine-tuning sections trimmed to summaries with pointers to
  `references/patterns.md`
- Added cross-references in `fast-mlx/references/fast-mlx-guide.md` (type
  promotion and compile sections point to `mlx` skill's fundamentals.md)
- Added cross-reference in `fast-mlx/references/llm-optimization.md` pointing
  to `mlx-lm` skill for architecture context
- Added cross-reference in `mlx/references/anti-patterns.md` compilation
  section pointing to fundamentals.md
- Relocated `mlx/scripts/check_updates.py` to `scripts/check_updates.py`
- `mlx/references/fundamentals.md`: genericized wired memory section
- `mlx/references/nn-and-training.md`: added mlx-lm cross-reference in
  transformer note; added `key` parameter comment in init_weights
- `mlx/references/debugging.md`: added URL currency note for Metal debugger
- `mlx-lm/references/patterns.md`: added routing comment to
  scaled_dot_product_attention; added prefill_step_size default note
- `mlx-lm/references/architecture.md`: added input/output shape comment to
  Model.__call__
- Updated README.md with validation section and corrected paths
- `scripts/check_updates.py`: removed hardcoded `coderef` directory convention;
  now fetches directly from GitHub by default (shallow bare clones to temp dir).
  `--repos-dir` or `MLX_SKILLS_REPOS` env var available for local clones.

### Removed

- `mlx/scripts/` directory (script moved to project-level `scripts/`)

## 0.3.0

### Added

- **mlx-lm skill**: Separate skill for Apple's official language model library
  - `SKILL.md` covering model architecture, generation pipelines, KV caching,
    quantization, fine-tuning, sampling, and server deployment
  - `references/patterns.md` with idiomatic mlx-lm patterns (moved from mlx skill)
  - `references/architecture.md` with mlx-lm directory structure, loading flow,
    generation flow, model registration, fine-tuning flow, and integration patterns

- **mlx nn-and-training reference**: `mlx/references/nn-and-training.md` covering
  the nn.Module system, building custom layers, all available layers (linear,
  conv, norm, activation, pooling, dropout, recurrent, transformer, embedding,
  positional, quantized), loss functions, parameter initialization, optimizers,
  learning rate schedulers, and training loop patterns (basic, compiled, gradient
  checkpointing, gradient accumulation, distributed)

### Changed

- `mlx` skill now covers MLX core only (removed mlx-lm specific content from
  SKILL.md, updated triggers to include nn.Module, nn.Linear, mlx.optimizers,
  training loop)
- `mlx/references/debugging.md` updated to remove mlx-lm specific sections
  (batch dimension, KV cache shapes, generation metrics)
- `mlx/references/anti-patterns.md` updated "Breaking Async Pipeline" comment
  to use generic "computation stream" language
- `mlx/scripts/check_updates.py` expanded WATCHED_FILES to cover nn layers,
  losses, init, optimizers, schedulers; updated suggested actions to reference
  both mlx and mlx-lm skill files
- `fast-mlx/SKILL.md` updated to cross-reference mlx-lm skill
- `README.md` updated with mlx-lm skill section and revised structure diagram

### Removed

- `mlx/references/ecosystem.md` (content distributed to mlx-lm skill)
- `mlx/references/patterns.md` (moved to mlx-lm skill)

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
  - `scripts/check_updates.py` for scanning upstream repos and generating
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
