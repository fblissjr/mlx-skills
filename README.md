# MLX Skills

A collection of skills that teach AI coding assistants how MLX actually works --
lazy evaluation, unified memory, idiomatic patterns, and performance tuning.

## Skills

### mlx

Core MLX knowledge: concepts, ecosystem, patterns, anti-patterns, and debugging.
Triggers on any MLX-related work (writing, debugging, reviewing, analyzing).

**Reference files loaded on demand:**
- `fundamentals.md` -- Lazy evaluation, unified memory, streams, compile, transformations, type system
- `patterns.md` -- Idiomatic patterns from mlx-lm: nn.Module, cache, attention, generation, quantization, LoRA
- `anti-patterns.md` -- Common mistakes from NumPy/PyTorch habits
- `ecosystem.md` -- mlx-lm and mlx-vlm architecture, integration patterns
- `debugging.md` -- Shape debugging, evaluation issues, memory profiling, common errors

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

Copy the `mlx_skills/skills/mlx` and `mlx_skills/skills/fast-mlx` directories
into your assistant's skills directory.

## Maintenance

The skills are based on patterns extracted from actual MLX source code. To check
for upstream changes that may require skill updates:

```
uv run python mlx_skills/skills/mlx/scripts/check_updates.py --since 30days
```

This scans the `coderef/` repos and produces a structured report of changes
affecting the skill content.

## Structure

```
mlx_skills/
  skills/
    mlx/
      SKILL.md              Core skill (loaded always)
      references/
        fundamentals.md     Deep dive: lazy eval, memory, streams, compile, types
        patterns.md         Idiomatic patterns from mlx-lm
        anti-patterns.md    Common mistakes
        ecosystem.md        mlx-lm and mlx-vlm architecture
        debugging.md        Debugging and profiling guide
      scripts/
        check_updates.py    Maintenance: scan coderef for changes
    fast-mlx/
      SKILL.md              Optimization skill (loaded always)
      references/
        fast-mlx-guide.md   Comprehensive performance guide
        llm-optimization.md LLM-specific optimization
        dit-optimization.md Diffusion model optimization
        compute-optimization.md  General compute optimization
```
