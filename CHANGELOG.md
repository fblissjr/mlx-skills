# Changelog

## 0.5.3

### Added

- New `mlx-cuda` skill: separate skill for CUDA backend support on NVIDIA GPUs
  with backend detection, custom CUDA kernels (`cuda_kernel`,
  `precompiled_cuda_kernel`), and Metal-to-CUDA kernel migration guide
- `mx.linalg` section in fundamentals.md: 17 linear algebra functions (qr, svd,
  inv, pinv, cholesky, lu, eig, eigh, solve, cross, etc.)
- Function export/import section in fundamentals.md: `export_function`,
  `import_function`, `exporter` context manager, `export_to_dot` visualization
- FP8 quantization section in fundamentals.md: `mx.to_fp8`, `mx.from_fp8`,
  `mx.qqmm`, `mx.gather_qmm`, `mx.segmented_mm`
- Type inspection utilities in fundamentals.md: `mx.finfo`, `mx.iinfo`,
  `mx.issubdtype`
- Random distributions table in fundamentals.md: 10 distributions including
  `multivariate_normal`, `gumbel`, `laplace`, `permutation`, `categorical`
- Additional array operations table in fundamentals.md: `topk`, `median`,
  `cummax`, `cummin`, `logcumsumexp`, `unflatten`, `hadamard_transform`,
  `contiguous`, `as_strided`, `convolve`, `depends`, `view`, `kron`
- Missing distributed ops: `recv_like`, `sum_scatter`
- Missing memory API entries: `mx.metal.is_available()`,
  `mx.metal.clear_cache()`, `mx.metal.device_info()`
- `mx.fft` namespace in fundamentals.md: 14 FFT functions (fft, ifft, fft2,
  fftn, rfft, irfft, fftshift, etc.) with usage examples
- `sum_scatter` expanded documentation with pipeline parallelism pattern
- FFT mappings in porting-guide.md for both NumPy and PyTorch
- PyTorch porting mappings: `topk`, `median`, `cumsum`, `linalg.solve`,
  `linalg.eigh`, `linalg.svd`, `linalg.inv`, `linalg.cholesky`
- NumPy porting mappings: 23 additional operations including `argsort`,
  `median`, `cumsum`, `cumprod`, `pad`, `roll`, `repeat`, `meshgrid`,
  `convolve`, all linalg functions, and additional random distributions

### Changed

- Separated CUDA content from Metal content: removed cuda_kernel from
  custom-kernels.md (now Metal-only), removed CUDA references from mlx
  SKILL.md mx.fast table
- custom-kernels.md renamed to "Custom Metal Kernels" with cross-reference
  to mlx-cuda skill
- Type table updated: added `float64` (CPU only), `uint64`, clarified GPU
  limitations

## 0.5.2

### Fixed

- Removed non-existent `--tool-parser` CLI flag from server docs in
  cli-reference.md and architecture.md; tool parsing is configured via model
  chat template, not a CLI flag
- Fixed `nn.quantize` signature: added `mode` parameter, corrected
  `group_size`/`bits` defaults to `None` (mode-dependent)
- Added missing `all_max` and `all_min` to distributed API table
- Added missing benchmark CLI flags (`--prompt-tokens`, `--generation-tokens`,
  `--batch-size`, `--num-trials`)
- Added actual server flags (`--log-level`, `--chat-template`, `--pipeline`)

### Added

- NumPy-to-MLX porting section in porting-guide.md: performance trap, data
  boundary pattern, API mapping table (35+ operations), and NumPy patterns
  that need rethinking (float64, in-place mutation, eager assumptions)
- "Mixing NumPy and MLX" anti-pattern section in anti-patterns.md
- "Avoid NumPy in Hot Paths" note in fast-mlx-guide.md Operations section

### Changed

- Softened authority preamble in all three SKILL.md files: web search now
  allowed as last resort after exhausting reference files (was blanket
  prohibition)
- Replaced duplicated compile state capture section in fast-mlx-guide.md with
  cross-reference to mlx skill's nn-and-training.md (kept optimization-specific
  Dropout/random state tip)
- Replaced duplicated memory management API table in fast-mlx-guide.md with
  cross-reference to mlx skill's fundamentals.md (kept optimization-specific
  set_memory_limit vs set_cache_limit guidance)
- Added benchmarking recipes cross-reference from fast-mlx-guide.md to
  compute-optimization.md
- Updated mlx SKILL.md porting section to mention NumPy migration coverage
- Removed stale local directory references from changelog

## 0.5.1

### Added

- Authority preambles in all three SKILL.md files ("do not search the web")
  to prevent Claude from web-searching after skills load
- `skills/mlx/references/custom-kernels.md`: Metal and CUDA custom kernel
  guide with `mx.fast.metal_kernel`, `mx.fast.cuda_kernel`, atomic outputs,
  strided inputs, and `mx.custom_function` pairing
- `skills/mlx-lm/references/cli-reference.md`: complete CLI reference for
  all 17 mlx_lm subcommands
- mlx skill: mx.fast full function signatures with parameter tables, Device
  and Stream API reference, additional transforms (jvp, vjp, custom_function,
  disable_compile/enable_compile), expanded distributed API (send, recv,
  all_gather, is_available), nn.quantize API, complete memory management API
- mlx-lm skill: prompt caching API (make_prompt_cache, save_prompt_cache,
  load_prompt_cache), sampling details (make_sampler, make_logits_processors,
  XTC algorithm, repetition penalty), DoRA pattern with config, quantization
  techniques comparison (AWQ/GPTQ/DWQ/mixed), tool calling parsers table,
  CLI subcommand reference table
- fast-mlx skill: Metal debugger profiling (start_capture/stop_capture),
  benchmarking methodology guide, compile state capture for training loops
  (inputs=/outputs= params), complete memory management API with
  set_memory_limit vs set_cache_limit explanation, benchmarking recipes

### Changed

- Reference section headings in all SKILL.md files changed from "loaded on
  demand -- not in context until you open them" to "read before answering --
  complete details inside"
- Remember sections in all SKILL.md files now include "Read reference files
  first -- do not search the web"
- mlx SKILL.md mx.fast table expanded with full signatures, metal_kernel,
  and cuda_kernel entries
- mlx SKILL.md transforms table expanded with jvp, vjp, custom_function,
  disable_compile/enable_compile
- mlx-lm SKILL.md feature list expanded with prompt caching, tool calling,
  17 CLI subcommands, and AWQ/GPTQ/DWQ quantization
- Porting guide SDPA example annotated with full signature and GQA note

## 0.5.0

### Added

- Plugin format: `.claude-plugin/plugin.json` manifest and `marketplace.json`
  catalog for native Claude Code plugin installation
- `plugins/mlx-skills/` wrapper directory for marketplace compatibility
  (marketplace `source` cannot be `"."`; requires a subdirectory path)
- Plugin installation via `claude plugin add` (local path or GitHub)
- `allowed-tools: "Read, Glob, Grep"` in all SKILL.md frontmatter (knowledge
  skills should not write files)
- Invocation hints in all SKILL.md descriptions (`Invoke with /mlx-skills:mlx`)
- `>-` (folded, strip) YAML scalar style for descriptions (matches
  fb-claude-skills convention)

### Changed

- Skills moved from `mlx_skills/skills/` to top-level `skills/` directory
  (standard plugin convention); symlink at `mlx_skills/skills/` preserves
  backward compatibility for pip-based CLI installer
- README updated with plugin installation as primary method
- Validator `SKILLS_DIR` now resolves to top-level `skills/`

## 0.4.6

### Added

- `license: MIT` field in all three SKILL.md frontmatter blocks
- `compatibility` field in all three SKILL.md frontmatter blocks
- `metadata.last_verified` date in all three SKILL.md frontmatter blocks

### Changed

- Clarified on-demand reference loading in SKILL.md bodies (section headings
  now state "loaded on demand -- not in context until you open them")
- Fixed cross-reference pattern in `fast-mlx/references/fast-mlx-guide.md`:
  simplified `load the \`mlx\` skill's \`references/fundamentals.md\`` to
  standard `load the \`mlx\` skill` pattern (2 occurrences)
- Synced `metadata.version` to `0.4.6` in all three SKILL.md files (was 0.4.0)

## 0.4.5

### Changed

- Upstream sync (Mar 2-6, 2026): 80 commits across mlx/mlx-lm, 4 changes
  relevant to skill reference files
- `mlx-lm/references/patterns.md`: added cache `.nbytes` introspection with
  `prompt_cache_nbytes` for batch generation; added `CacheList.from_state()`
  for MLA prompt cache persistence; noted automatic `RotatingKVCache` selection
  for models with mixed sliding window / full attention layers
- `mlx-lm/references/architecture.md`: annotated cache.py with `.nbytes`;
  added `--prefill-step-size` CLI flag to generation flow
- `fast-mlx/references/llm-optimization.md`: added `--prefill-step-size` CLI
  note to chunked prefill section; added runtime cache memory monitoring
  subsection with `.nbytes` usage and `prompt_cache_nbytes` for load shedding

## 0.4.4

### Changed

- Upstream sync (Feb 23 - Mar 1, 2026): 19 commits across mlx/mlx-lm, 3
  changes relevant to skill reference files
- `mlx/references/nn-and-training.md`: added `Linear.to_quantized()` conversion
  path with `quantize_input` parameter for `QQLinear` (nvfp4/mxfp8 only)
- `mlx-lm/references/patterns.md`: added batch generation memory management
  patterns -- periodic `mx.clear_cache()` every 512 tokens and including all
  dependent arrays in `mx.async_eval()` to prevent graph node accumulation
- `fast-mlx/references/llm-optimization.md`: added graph management note for
  `mx.async_eval()` with good/bad code comparison; added periodic cache clearing
  subsection for custom batch generation loops

## 0.4.3

### Added

- `mlx/references/porting-guide.md`: comprehensive PyTorch-to-MLX migration
  guide with step-by-step walkthrough, side-by-side code examples, API mapping
  tables (40+ operations), layer equivalents, training loop conversion, and a
  porting completion checklist
- `/update-skills` project-level skill: maintainer workflow that runs the
  upstream scanner, analyzes diffs, smart-routes changes to the correct
  reference files, updates them, and validates
- `--diff-lines N` CLI arg for `scripts/check_updates.py`: truncate diffs
  to N lines per file (0 = unlimited)
- "Which Skill Do I Need?" quick-reference table in CLAUDE.md for skill routing

### Changed

- `mlx` skill: added porting/migration triggers ("port to mlx", "pytorch to mlx",
  "convert to mlx", "rewrite in mlx", "migrate to mlx") to description; added
  "Porting from PyTorch / NumPy" section to SKILL.md body pointing to new guide
- `mlx-lm` skill: added "When to Use This Skill" routing section; clarified
  description to distinguish "run existing HuggingFace models" (mlx-lm) from
  "write custom models / port from PyTorch" (mlx); added triggers "run llama",
  "run a model on my mac", "local LLM", "huggingface model"
- `fast-mlx` skill: added "When to Use This Skill" routing section; clarified
  description with "NOT for writing new MLX code or porting"; added triggers
  "performance tuning", "why is my mlx code slow"
- CLAUDE.md skills section rewritten with "Use for:" summaries and updated
  trigger lists
- `get_watched_file_diffs()` default is now unlimited (no truncation) since
  `--diff` is already opt-in; use `--diff-lines N` for explicit truncation
- `analyze_watched_files()` and `generate_report()` now accept `max_lines`
  parameter, passed through from CLI

## 0.4.2

### Added

- `--diff` flag for `scripts/check_updates.py`: includes unified diffs for
  watched files directly in the report, making it self-contained and actionable
- `get_watched_file_diffs()` function with configurable line-limit truncation
  (default 200 lines per file)
- `build_parser()` extracted from `main()` for testability
- `tests/test_check_updates.py` with 9 tests covering diff output, truncation,
  integration with `analyze_watched_files` and `generate_report`, and CLI parsing

## 0.4.1

### Added

- `metadata` field (author, version) in all three SKILL.md frontmatter blocks
- "Remember" summary sections at the end of each SKILL.md with key takeaways
- `last updated: YYYY-MM-DD` date stamps on all 10 reference files
- Validator checks: metadata field presence (warning), description length
  limit (1024 chars, error), reference file date presence (warning)
- Tests for new validation checks (metadata, description length, ref dates)
- Structural tests for metadata, description length, and reference file dates
  against actual skills

### Changed

- Rewritten SKILL.md descriptions using WHAT + WHEN + Capabilities formula;
  removed cross-reference directives from description fields (those belong
  in the SKILL.md body, not the routing description)
- Improved fast-mlx description with more natural-language trigger phrases
  ("slow inference", "make it faster", "benchmark", "reduce memory")
- Converted backtick reference paths to markdown links in all SKILL.md files
  (e.g., `references/file.md` to [references/file.md](references/file.md))
- `validate()` now returns `(errors, warnings)` tuple instead of just errors;
  warnings are advisory and do not cause validation failure

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
- `scripts/check_updates.py`: now fetches directly from GitHub by default
  (shallow bare clones to temp dir). `--repos-dir` or `MLX_SKILLS_REPOS` env
  var available for local clones.

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
