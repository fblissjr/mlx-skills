# MLX Skills

> Originally forked from [awni/mlx-skills](https://github.com/awni/mlx-skills)

A Claude Code plugin that teaches AI coding assistants how to write correct,
performant MLX code on Apple silicon -- lazy evaluation, unified memory,
idiomatic patterns, and performance tuning.

Built following the [skill best practices checklist](https://github.com/fblissjr/fb-claude-skills/blob/main/tools/skill-maintainer/references/best_practices.md)
from fb-claude-skills: token budgets, description precision, progressive
disclosure, and spec compliance.

## installation

### as a plugin (recommended)

```bash
claude plugin add fblissjr/mlx-skills
```

Or from a local clone:

```bash
claude plugin add /path/to/mlx-skills
```

### legacy CLI

```bash
uv run mlx-skills --claude --force
```

Other targets: `--codex`, `--opencode`, or `--dest /path/to/skills`.

### manual

Copy `skills/mlx`, `skills/mlx-lm`, and `skills/fast-mlx` into your
assistant's skills directory (e.g., `~/.claude/skills/`).

## skills

| Skill | Trigger | What it does |
|-------|---------|--------------|
| mlx | `import mlx`, `mx.array`, `nn.Module`, "port to mlx", "training loop" | Core framework: lazy eval, unified memory, compile, nn.Module, layers, optimizers, porting |
| mlx-lm | `import mlx_lm`, `stream_generate`, `KVCache`, "run llama", "local LLM" | Language models: loading, generation, KV cache, quantization, LoRA, serving |
| fast-mlx | "optimize mlx", "speed up", "profiling", "reduce memory" | Performance: graph eval, compile, memory management, LLM/diffusion optimization |

### invocation

Skills load automatically when Claude detects relevant imports or keywords.
Invoke explicitly with `/mlx`, `/mlx-lm`, or `/fast-mlx`.

### reference files (loaded on demand)

| Skill | Reference | Content |
|-------|-----------|---------|
| mlx | fundamentals.md | Lazy eval, unified memory, streams, compile, type system |
| mlx | nn-and-training.md | nn.Module, layers, losses, optimizers, schedulers, training loops |
| mlx | anti-patterns.md | NumPy/PyTorch habits that break in MLX |
| mlx | debugging.md | Shape debugging, memory profiling, common errors |
| mlx | porting-guide.md | PyTorch-to-MLX migration with API mapping tables |
| mlx-lm | patterns.md | Attention, KV cache, generation, quantization, LoRA, RoPE |
| mlx-lm | architecture.md | Model loading, generation flow, fine-tuning, server integration |
| fast-mlx | fast-mlx-guide.md | Graph eval, type promotion, ops, compile, memory, profiling |
| fast-mlx | llm-optimization.md | KV cache tuning, prefill chunking, speculative decoding |
| fast-mlx | dit-optimization.md | Denoising compilation, CFG batching, vision attention |
| fast-mlx | compute-optimization.md | Matrix ops, element-wise fusion, vmap, data pipelines |

## validation

```bash
uv run mlx-skills-validate            # validate skill structure
uv run pytest tests/                  # run test suite (90 tests)
```

## maintenance

Skills are based on patterns from actual MLX source code. Check for upstream
changes:

```bash
uv run python scripts/check_updates.py --since 30days
```

## structure

```
.claude-plugin/
  plugin.json               Root plugin manifest
  marketplace.json          Marketplace catalog
plugins/
  mlx-skills/
    .claude-plugin/
      plugin.json           Plugin manifest for marketplace
    skills -> ../../skills  Symlink to top-level skills
skills/                     Skill source (auto-discovered)
  mlx/
    SKILL.md
    references/
  mlx-lm/
    SKILL.md
    references/
  fast-mlx/
    SKILL.md
    references/
mlx_skills/                 Python package (legacy CLI)
  cli.py
  validate.py
  skills -> ../skills       Symlink for pip compat
scripts/
  check_updates.py          Upstream change scanner
tests/
```
