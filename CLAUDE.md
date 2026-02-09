# MLX Skills - Development Guide

## Project Overview

This is a skills package that teaches AI coding assistants (Claude Code, Codex,
OpenCode) how to write correct, performant MLX code. Skills are markdown files
with YAML frontmatter that get installed into an assistant's skills directory.

## Skills and When They Load

There are three skills. Each has a `SKILL.md` (always loaded when triggered)
and `references/` files (loaded on demand).

### mlx (core framework)

**Triggers:** `import mlx`, `from mlx`, `mx.array`, `mx.compile`, `mx.eval`,
`nn.Module`, `nn.Linear`, `mlx.optimizers`, writing/debugging/optimizing MLX code.

**Invocation:**
- Automatic: mention MLX concepts or work with MLX code
- Explicit: `/mlx` or "load the mlx skill"
- From other skills: "For core MLX concepts, load the mlx skill"

**What it covers:** lazy evaluation, unified memory, compile, nn.Module system,
layers, optimizers, training patterns, debugging.

### mlx-lm (language models)

**Triggers:** `import mlx_lm`, `from mlx_lm import`, `stream_generate`,
`KVCache`, LoRA, quantization, GGUF, safetensors, LLM inference on Apple silicon.

**Invocation:**
- Automatic: scan imports for `mlx_lm` usage
- Explicit: `/mlx-lm` or "load the mlx-lm skill"
- From other skills: "load the mlx-lm skill for generation patterns"

**What it covers:** model architecture, generation pipelines, KV cache,
quantization, fine-tuning, server deployment.

### fast-mlx (performance)

**Triggers:** "optimize mlx", "speed up", "reduce latency", "profiling",
`mx.compile`, `mx.metal`, memory optimization.

**Invocation:**
- Automatic: ask to optimize or profile MLX code
- Explicit: `/fast-mlx` or "load the fast-mlx skill"
- From other skills: "For performance optimization, load the fast-mlx skill"

**What it covers:** graph evaluation, type promotion, fast ops, compilation,
memory management, profiling, LLM/diffusion-specific optimization.

## Usage Scenarios

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

## Development

### Key files

- `mlx_skills/cli.py` -- installer (`mlx-skills` entrypoint)
- `mlx_skills/validate.py` -- validation (`mlx-skills-validate` entrypoint)
- `mlx_skills/skills/*/SKILL.md` -- skill definitions (YAML frontmatter + body)
- `mlx_skills/skills/*/references/*.md` -- reference material (loaded on demand)
- `scripts/check_updates.py` -- upstream change scanner
- `tests/` -- pytest suite

### Commands

```
uv run mlx-skills --claude --force    # Install skills locally
uv run mlx-skills-validate            # Validate skill structure
uv run pytest tests/                  # Run tests (56 tests)
uv run python scripts/check_updates.py --since 30days  # Check upstream changes
```

### Skill structure rules

- Every skill directory must have a `SKILL.md` with YAML frontmatter
- Frontmatter must have `name` (matching directory name) and `description` fields
- `description` should list trigger keywords and when to use the skill
- SKILL.md body must be under 5000 words (this is always loaded into context)
- Reference files go in `references/` and are loaded on demand
- Cross-references use `load the \`skill-name\` skill` pattern
- Run `uv run mlx-skills-validate` after any changes to verify structure

### Content guidelines

- `mlx` skill: core MLX framework only (no mlx-lm specifics)
- `mlx-lm` skill: Apple's mlx-lm library only (generation, caching, fine-tuning)
- `fast-mlx` skill: performance optimization (profiling, compilation, memory)
- Avoid duplicating content across skills; use cross-references instead
- Code examples should be minimal and correct
- Keep SKILL.md concise; put details in reference files
