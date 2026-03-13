# MLX Skills - Development Guide

## Project Overview

This is a Claude Code plugin that teaches AI coding assistants how to write
correct, performant MLX code. Skills are markdown files with YAML frontmatter
in the top-level `skills/` directory, auto-discovered by the plugin system.

## Which Skill Do I Need?

| I want to... | Use | Command |
|--------------|-----|---------|
| Port PyTorch code to MLX | mlx | `/mlx-skills:mlx` |
| Port NumPy code to MLX | mlx | `/mlx-skills:mlx` |
| Write a custom MLX model | mlx | `/mlx-skills:mlx` |
| Learn MLX fundamentals | mlx | `/mlx-skills:mlx` |
| Write a training loop | mlx | `/mlx-skills:mlx` |
| Debug MLX errors | mlx | `/mlx-skills:mlx` |
| Run a HuggingFace model on my Mac | mlx-lm | `/mlx-skills:mlx-lm` |
| Fine-tune with LoRA | mlx-lm | `/mlx-skills:mlx-lm` |
| Quantize a model | mlx-lm | `/mlx-skills:mlx-lm` |
| Set up a local LLM server | mlx-lm | `/mlx-skills:mlx-lm` |
| Speed up my MLX code | fast-mlx | `/mlx-skills:fast-mlx` |
| Reduce memory usage | fast-mlx | `/mlx-skills:fast-mlx` |
| Profile performance | fast-mlx | `/mlx-skills:fast-mlx` |
| Run MLX on NVIDIA GPU | mlx-cuda | `/mlx-skills:mlx-cuda` |
| Write custom CUDA kernels | mlx-cuda | `/mlx-skills:mlx-cuda` |
| Port Metal kernels to CUDA | mlx-cuda | `/mlx-skills:mlx-cuda` |
| Update skills from upstream | update-skills | `/update-skills` |
| Verify content accuracy | review-content | `/review-content` |
| Bump version everywhere | sync-versions | `/sync-versions 0.5.4` |

## Skills and When They Load

There are four skills. Each has a `SKILL.md` (always loaded when triggered)
and `references/` files (loaded on demand).

### mlx (core framework)

**Use for:** Writing, porting, debugging, or learning MLX code.

**Triggers:** `import mlx`, `from mlx`, `mx.array`, `mx.compile`, `mx.eval`,
`nn.Module`, `nn.Linear`, `mlx.optimizers`, "port to mlx", "convert from
pytorch", "training loop", writing/debugging MLX code.

**Invocation:**
- Automatic: mention MLX concepts or work with MLX code
- Explicit: `/mlx-skills:mlx` (plugin) or `/mlx` (personal skill)
- From other skills: "For core MLX concepts, load the mlx skill"

**What it covers:** lazy evaluation, unified memory, compile, nn.Module system,
layers, optimizers, training patterns, debugging, PyTorch-to-MLX and
NumPy-to-MLX porting.

### mlx-lm (language models)

**Use for:** Running, fine-tuning, or serving existing HuggingFace models.

**Triggers:** `import mlx_lm`, `from mlx_lm import`, `stream_generate`,
`KVCache`, LoRA, quantization, GGUF, safetensors, "run llama", "run a model
on my mac", "local LLM", "huggingface model".

**Invocation:**
- Automatic: scan imports for `mlx_lm` usage
- Explicit: `/mlx-skills:mlx-lm` (plugin) or `/mlx-lm` (personal skill)
- From other skills: "load the mlx-lm skill for generation patterns"

**What it covers:** model loading, generation pipelines, KV cache,
quantization, fine-tuning, server deployment.

### fast-mlx (performance)

**Use for:** Optimizing working MLX code that needs to be faster or use less memory.

**Triggers:** "optimize mlx", "speed up", "reduce latency", "profiling",
`mx.compile`, `mx.metal`, memory optimization, "make it faster", "why is my
mlx code slow".

**Invocation:**
- Automatic: ask to optimize or profile MLX code
- Explicit: `/mlx-skills:fast-mlx` (plugin) or `/fast-mlx` (personal skill)
- From other skills: "For performance optimization, load the fast-mlx skill"

**What it covers:** graph evaluation, type promotion, fast ops, compilation,
memory management, profiling, LLM/diffusion-specific optimization.

### mlx-cuda (CUDA backend)

**Use for:** Running MLX on NVIDIA GPUs, writing custom CUDA kernels, porting
Metal kernels to CUDA.

**Triggers:** `mx.cuda`, `cuda_kernel`, `precompiled_cuda_kernel`, `nvidia`,
"run mlx on cuda", "NVIDIA GPU", "cuda backend".

**Invocation:**
- Automatic: mention CUDA or NVIDIA in MLX context
- Explicit: `/mlx-skills:mlx-cuda` (plugin) or `/mlx-cuda` (personal skill)
- From other skills: "For CUDA backend support, load the mlx-cuda skill"

**What it covers:** backend detection, custom CUDA kernels, precompiled kernels,
Metal-to-CUDA kernel migration, CUDA-specific differences.

## Usage Scenarios

### "Port my PyTorch model to MLX"
1. `/mlx` loads core skill with porting checklist and comparison table
2. Reference `porting-guide.md` for step-by-step migration with side-by-side
   PyTorch/MLX code and API mapping tables
3. Reference `anti-patterns.md` for PyTorch habits that break in MLX
4. Reference `nn-and-training.md` for MLX layer equivalents

### "Port my NumPy code to MLX"
1. `/mlx` loads core skill with NumPy porting mention
2. Reference `porting-guide.md` for NumPy-to-MLX API mapping, data boundary
   pattern, and patterns that need rethinking
3. Reference `anti-patterns.md` for "Mixing NumPy and MLX" performance trap

### "Optimize my MLX project"
1. Claude scans imports for `import mlx`, `from mlx_lm import`, etc.
2. Loads `mlx` skill for core patterns
3. Loads `mlx-lm` if mlx-lm imports are present
4. Loads `fast-mlx` for optimization guidance
5. Reviews code against anti-patterns and optimization checklist

### "Write an MLX training loop"
1. `/mlx` loads core skill
2. Reference `nn-and-training.md` for training loop patterns, optimizers, schedulers
3. Reference `fundamentals.md` for lazy evaluation and compile semantics

### "Debug my MLX model"
1. `/mlx` loads core skill
2. Reference `debugging.md` for shape debugging, memory profiling, common errors
3. Reference `anti-patterns.md` for common mistakes

### "Set up mlx-lm generation"
1. `/mlx-lm` loads language model skill
2. Reference `patterns.md` for generation pipeline, KV cache, sampling
3. Reference `architecture.md` for model loading flow

### "Speed up my LLM inference"
1. `/fast-mlx` loads optimization skill
2. Reference `llm-optimization.md` for KV cache tuning, prefill chunking, speculative decoding
3. `/mlx-lm` for generation pipeline patterns

### "Fine-tune a model with LoRA"
1. `/mlx-lm` loads language model skill
2. Reference `patterns.md` for LoRA patterns and quantization
3. Reference `architecture.md` for fine-tuning flow

### "Update skills from upstream" (maintainer)
1. `/update-skills` loads the maintainer workflow skill
2. Runs `scripts/check_updates.py --diff` to generate a change report
3. Analyzes diffs, routes changes to the right reference files
4. Updates reference files (not SKILL.md), validates, and reports

### "Check content accuracy" (maintainer)
1. `/review-content` loads the content accuracy checker
2. Parses reference files for documented APIs, CLI flags, signatures
3. Fetches upstream source for comparison
4. Reports mismatches -- does not auto-fix

### "Bump project version" (maintainer)
1. `/sync-versions 0.5.3` loads the version coordinator
2. Updates version in all 6 locations (pyproject.toml, 4 SKILL.md, plugin.json)
3. Updates `last_verified` dates, adds CHANGELOG section header
4. Runs validator and tests to confirm

## Development

### Key files

- `.claude-plugin/plugin.json` -- root plugin manifest
- `.claude-plugin/marketplace.json` -- marketplace catalog
- `plugins/mlx-skills/` -- marketplace plugin wrapper (symlinks to `skills/`)
- `skills/*/SKILL.md` -- skill definitions (YAML frontmatter + body)
- `skills/*/references/*.md` -- reference material (loaded on demand)
- `scripts/validate.py` -- skill structure validation
- `scripts/check_updates.py` -- upstream change scanner
- `.claude/skills/` -- project-level maintainer skills (not part of plugin)
- `tests/` -- pytest suite

### Commands

```
uv run python scripts/validate.py     # Validate skill structure
uv run pytest tests/                  # Run tests
uv run python scripts/check_updates.py --since 30days  # Check upstream changes
```

### Skill structure rules

- Skills are at top-level `skills/`
- Every skill directory must have a `SKILL.md` with YAML frontmatter
- Frontmatter must have `name` (matching directory name) and `description` fields
- `description` should follow WHAT + WHEN + Capabilities formula, list trigger keywords
- `description` should end with invocation hint (`Invoke with /mlx-skills:<name>`)
- `description` uses `>-` (folded, strip) YAML scalar style
- `metadata` block should have `author`, `version`, and `last_verified` fields
- Frontmatter also has `license: MIT`, `compatibility`, and `allowed-tools` fields
- `allowed-tools: "Read, Glob, Grep"` -- knowledge skills should not write files
- SKILL.md body must be under 5000 words (this is always loaded into context)
- Reference files go in `references/` and are loaded on demand
- Reference files should start with `last updated: YYYY-MM-DD`
- Cross-references use `load the \`skill-name\` skill` pattern
- Run `uv run python scripts/validate.py` after any changes to verify structure

### Content guidelines

- `mlx` skill: core MLX framework only (no mlx-lm specifics)
- `mlx-lm` skill: Apple's mlx-lm library only (generation, caching, fine-tuning)
- `fast-mlx` skill: performance optimization (profiling, compilation, memory)
- `mlx-cuda` skill: CUDA backend only (keep separate from Metal content)
- Avoid duplicating content across skills; use cross-references instead
- Code examples should be minimal and correct
- Keep SKILL.md concise; put details in reference files
- Backend-specific content goes in its own skill (avoids context cost for
  users on other platforms)
- In reference tables, omit trivially obvious 1:1 API mappings (e.g.,
  `np.linalg.svd` -> `mx.linalg.svd`); use prose instead
- `coderef/` contains upstream source checkouts for API gap analysis;
  compare `.pyi` files against skill content to find missing APIs
