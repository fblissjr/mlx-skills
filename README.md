# MLX Skills

A collection of skills that teach AI coding assistants how MLX actually works --
lazy evaluation, unified memory, idiomatic patterns, and performance tuning.

Once installed, skills load automatically when you work with MLX code, or you
can invoke them explicitly.

## Install

### From local clone

```
cd mlx-skills
uv run mlx-skills --claude --force
```

Other targets: `--codex`, `--opencode`, or `--dest /path/to/skills`.

### Manual

Copy the `mlx_skills/skills/mlx`, `mlx_skills/skills/mlx-lm`, and
`mlx_skills/skills/fast-mlx` directories into your assistant's skills directory
(e.g., `~/.claude/skills/`).

## Usage

Skills load in two ways:

1. **Automatically** -- Claude scans your code for MLX imports and triggers
   the relevant skills based on what it finds.
2. **Explicitly** -- invoke a skill by name with `/mlx`, `/mlx-lm`, or
   `/fast-mlx`, or say "load the mlx skill" in your prompt.

### Scenarios

**"Review my MLX code for issues"**
- Claude detects `import mlx` and loads the `mlx` skill
- If it also finds `from mlx_lm import`, it loads `mlx-lm` too
- Anti-patterns and debugging references inform the review

**"Optimize my MLX inference server"**
- Say `/fast-mlx` or "optimize my MLX code"
- Claude loads `fast-mlx` for profiling, compilation, and memory guidance
- If mlx-lm is involved, it loads `mlx-lm` for generation pipeline patterns
- References: `llm-optimization.md`, `fast-mlx-guide.md`

**"Write a training loop"**
- Claude detects MLX usage and loads `mlx`
- References: `nn-and-training.md` for optimizers, schedulers, training patterns
- References: `fundamentals.md` for lazy evaluation and compile semantics

**"Set up mlx-lm text generation"**
- Say `/mlx-lm` or just work with mlx-lm code
- References: `patterns.md` for generation pipeline, KV cache, sampling
- References: `architecture.md` for model loading and server integration

**"Fine-tune a model with LoRA"**
- `/mlx-lm` for LoRA patterns and quantization
- References: `architecture.md` for the fine-tuning flow

**"Debug shape mismatches in my model"**
- `/mlx` loads core skill
- References: `debugging.md` for shape debugging, memory profiling, common errors
- References: `anti-patterns.md` for PyTorch/NumPy habits that break in MLX

**"Speed up my diffusion model"**
- `/fast-mlx` for optimization
- References: `dit-optimization.md` for denoising compilation, CFG batching,
  vision attention

### Tips

- You can load multiple skills in one session. They complement each other.
- Say "load the references for X" if Claude hasn't pulled in the detail you need.
- Skills work in any project -- they activate based on imports, not project structure.

## Skills

### mlx

Core MLX framework knowledge.

**Triggers:** `import mlx`, `mx.array`, `mx.compile`, `mx.eval`, `nn.Module`,
`nn.Linear`, `mlx.optimizers`, or any MLX-related work.

**Invoke:** `/mlx`

**Reference files (loaded on demand):**
- `fundamentals.md` -- Lazy evaluation, unified memory, streams, compile, transformations, type system
- `nn-and-training.md` -- nn.Module system, all layers, losses, optimizers, schedulers, training loop patterns
- `anti-patterns.md` -- Common mistakes from NumPy/PyTorch habits
- `debugging.md` -- Shape debugging, evaluation issues, memory profiling, common errors

### mlx-lm

Language model patterns for mlx-lm, Apple's official LLM library.

**Triggers:** `import mlx_lm`, `from mlx_lm import`, `stream_generate`,
`KVCache`, LoRA, quantization, GGUF, safetensors.

**Invoke:** `/mlx-lm`

**Reference files (loaded on demand):**
- `patterns.md` -- nn.Module structure, attention, KV cache, generation, quantization, LoRA, RoPE, sharding
- `architecture.md` -- Model loading flow, generation flow, model registration, fine-tuning flow, server integration

### fast-mlx

Performance optimization for MLX code.

**Triggers:** "optimize mlx", "speed up", "reduce latency", "profiling",
`mx.compile`, `mx.metal`, memory optimization.

**Invoke:** `/fast-mlx`

**Reference files (loaded on demand):**
- `fast-mlx-guide.md` -- Graph evaluation, type promotion, ops, compile, memory, profiling
- `llm-optimization.md` -- KV cache tuning, async generation, prefill chunking, batch generation, speculative decoding
- `dit-optimization.md` -- Denoising step compilation, CFG batching, vision attention, memory management
- `compute-optimization.md` -- Matrix ops, element-wise fusion, vmap, streaming, data pipelines

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
