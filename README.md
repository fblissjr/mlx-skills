# MLX Skills

A collection of skills that teach AI coding assistants how MLX actually works --
lazy evaluation, unified memory, idiomatic patterns, and performance tuning.

## Skills

### mlx

Core MLX knowledge: concepts, nn module system, layers, optimizers, training
patterns, anti-patterns, and debugging. Triggers on any MLX-related work
(writing, debugging, reviewing, analyzing).

**Reference files loaded on demand:**
- `fundamentals.md` -- Lazy evaluation, unified memory, streams, compile, transformations, type system
- `nn-and-training.md` -- nn.Module system, all layers, losses, optimizers, schedulers, training loop patterns
- `anti-patterns.md` -- Common mistakes from NumPy/PyTorch habits
- `debugging.md` -- Shape debugging, evaluation issues, memory profiling, common errors

### mlx-lm

Language model patterns for mlx-lm, Apple's official LLM library. Triggers on
mlx-lm imports, model loading, generation, KV cache, LoRA, quantization.

**Reference files loaded on demand:**
- `patterns.md` -- Idiomatic mlx-lm patterns: nn.Module structure, attention, KV cache, generation, quantization, LoRA, RoPE, sharding
- `architecture.md` -- mlx-lm directory structure, model loading flow, generation flow, model registration, fine-tuning flow, server integration

### fast-mlx

Performance optimization for MLX code. Triggers on explicit optimization requests.

**Reference files:**
- `fast-mlx-guide.md` -- Comprehensive performance guide (graph evaluation, type promotion, ops, compile, memory, profiling)
- `llm-optimization.md` -- KV cache tuning, async generation, prefill chunking, batch generation, speculative decoding
- `dit-optimization.md` -- Denoising step compilation, CFG batching, vision attention, memory management
- `compute-optimization.md` -- Matrix ops, element-wise fusion, vmap, streaming, data pipelines

## Install

### One-liner (uv)

```
uvx --from git+https://github.com/awni/mlx-skills.git mlx-skills --claude --force
```

Other targets: `--codex`, `--opencode`, or `--dest /path/to/skills`.

### Manual

Copy the `mlx_skills/skills/mlx`, `mlx_skills/skills/mlx-lm`, and
`mlx_skills/skills/fast-mlx` directories into your assistant's skills directory.

## Validation

Validate skill structure, frontmatter, word counts, and cross-references:

```
uv run mlx-skills-validate
```

Run the test suite:

```
uv run pytest tests/
```

## Maintenance

The skills are based on patterns extracted from actual MLX source code. To check
for upstream changes that may require skill updates:

```
uv run python scripts/check_updates.py --since 30days
```

This fetches recent commits from the upstream GitHub repos (mlx, mlx-lm, mlx-vlm,
mlx-examples) and produces a structured report of watched file changes, potential
breaking changes, and suggested skill updates. Use `--repos` to scan specific
repos, or `--repos-dir` to point at existing local clones instead.

## Structure

```
mlx_skills/
  cli.py                  CLI installer
  validate.py             Skill validation script
  skills/
    mlx/
      SKILL.md              Core skill (loaded always)
      references/
        fundamentals.md     Deep dive: lazy eval, memory, streams, compile, types
        nn-and-training.md  nn.Module system, layers, losses, optimizers, training
        anti-patterns.md    Common mistakes
        debugging.md        Debugging and profiling guide
    mlx-lm/
      SKILL.md              Language model skill (loaded always)
      references/
        patterns.md         Idiomatic mlx-lm patterns
        architecture.md     mlx-lm architecture and integration
    fast-mlx/
      SKILL.md              Optimization skill (loaded always)
      references/
        fast-mlx-guide.md   Comprehensive performance guide
        llm-optimization.md LLM-specific optimization
        dit-optimization.md Diffusion model optimization
        compute-optimization.md  General compute optimization
scripts/
  check_updates.py          Maintenance: scan upstream repos for changes
tests/
  test_validate.py          Validation logic tests
  test_cli.py               CLI tests
  test_skill_structure.py   Structural integrity tests
```
