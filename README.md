<!-- last updated: 2026-04-18 -->

# MLX Skills

> Originally forked from [awni/mlx-skills](https://github.com/awni/mlx-skills)

A Claude Code plugin that teaches AI coding assistants how to write correct,
performant MLX code on Apple silicon -- lazy evaluation, unified memory,
idiomatic patterns, and performance tuning.

Built following the [skill best practices checklist](https://github.com/fblissjr/fb-claude-skills/blob/main/tools/skill-maintainer/references/best_practices.md)
from fb-claude-skills: token budgets, description precision, progressive
disclosure, and spec compliance.

## maintaining this plugin (read first)

MLX moves fast. Keeping these skills accurate is the whole point of this repo.
Use the slash commands -- don't drive the scripts by hand.

| When | Command | What it does |
|------|---------|--------------|
| Routine upstream sync | `/update-skills` | Full 6-step workflow: runs `check_updates.py --diff`, analyzes diffs, routes changes into `references/*.md`, validates, runs tests. This is the one you want 95% of the time. |
| Before a release | `/review-content` | Read-only audit of reference files vs. upstream source (CLI flags, signatures, API tables). Reports mismatches; does not auto-fix. |
| Bump version | `/sync-versions 0.5.10` | Atomically bumps all 8 version files, updates `last_verified`, adds CHANGELOG header, runs tests. |
| Quality check | `/skill-maintainer:quality` | Spec compliance, token budgets, body size, freshness, description quality. Safe to run anytime. Already wired into `/update-skills` and `/sync-versions`. |
| One-time setup (optional) | `/skill-maintainer:init-maintenance` | Adds `.skill-maintainer/` config. Enables `/skill-maintainer:maintain` (broader hygiene pass: Claude Code doc drift, tracked-repo pulls, best-practices review). |

`scripts/check_updates.py` is plumbing called by `/update-skills`. Running it
standalone only gives you a diff report -- no edits, no validation. Prefer
the slash command.

The maintainer slash commands (`/update-skills`, `/review-content`,
`/sync-versions`) live at `.claude/skills/*.md` in this repo and load
automatically when Claude Code is invoked from the project directory -- no
separate install step. `.claude/settings.local.json` is per-user and
gitignored.

Install the skill-maintainer plugin once (for `/skill-maintainer:quality`):

```bash
/plugin marketplace add fblissjr/fb-claude-skills
/plugin install skill-maintainer@fb-claude-skills
```

Enable the pre-commit version-bump gate once per clone:

```bash
git config core.hooksPath .githooks
```

The hook at `.githooks/pre-commit` blocks commits that touch shipped paths
(`skills/`, `scripts/`, `.claude-plugin/`, `plugins/`) unless
`pyproject.toml` version differs from HEAD. Commits limited to tests, docs,
`.claude/skills/`, or other non-shipped paths pass through. Bypass an
individual commit with `git commit --no-verify` when the change genuinely
doesn't warrant a bump.

## installation

### as a plugin (recommended)

```bash
claude plugin add fblissjr/mlx-skills
```

Or from a local clone:

```bash
claude plugin add /path/to/mlx-skills
```

### manual

Copy `skills/mlx`, `skills/mlx-models`, `skills/fast-mlx`, and `skills/mlx-cuda`
into your assistant's skills directory (e.g., `~/.claude/skills/`).

## skills

| Skill | Trigger | What it does |
|-------|---------|--------------|
| mlx | `import mlx`, `mx.array`, `nn.Module`, "port to mlx", "training loop" | Core framework: lazy eval, unified memory, compile, nn.Module, layers, optimizers, porting |
| mlx-models | `import mlx_lm`, `import mlx_vlm`, `KVCache`, "run llama", "local LLM", "VLM" | Models: loading, generation, KV cache, quantization, LoRA, serving, vision-language |
| fast-mlx | "optimize mlx", "speed up", "profiling", "reduce memory" | Performance: graph eval, compile, memory management, LLM/diffusion optimization |
| mlx-cuda | `mx.cuda`, `cuda_kernel`, "NVIDIA GPU", "run mlx on cuda" | CUDA backend: detection, custom CUDA kernels, Metal-to-CUDA porting |

### invocation

Skills load automatically when Claude detects relevant imports or keywords.

Explicit invocation depends on how you installed:

```bash
# Plugin install (namespaced)
/mlx-skills:mlx
/mlx-skills:mlx-models
/mlx-skills:fast-mlx
/mlx-skills:mlx-cuda

# Legacy CLI / manual install (personal skills)
/mlx
/mlx-models
/fast-mlx
/mlx-cuda
```

### reference files (loaded on demand)

| Skill | Reference | Content |
|-------|-----------|---------|
| mlx | fundamentals.md | Lazy eval, unified memory, streams, compile, type system |
| mlx | nn-and-training.md | nn.Module, layers, losses, optimizers, schedulers, training loops |
| mlx | anti-patterns.md | NumPy/PyTorch habits that break in MLX |
| mlx | debugging.md | Shape debugging, memory profiling, common errors |
| mlx | porting-guide.md | PyTorch-to-MLX migration with API mapping tables |
| mlx | custom-kernels.md | Custom Metal kernels with mx.fast.metal_kernel |
| mlx-models | patterns.md | Attention, KV cache, generation, quantization, LoRA, RoPE |
| mlx-models | architecture.md | Model loading, generation flow, fine-tuning |
| mlx-models | serving.md | mlx-lm and mlx-vlm server architectures, deployment |
| mlx-models | vlm.md | Vision-language models, TurboQuant, Gemma 4 multimodal |
| mlx-models | cli-reference.md | mlx_lm (17 subcommands) and mlx_vlm CLI reference |
| fast-mlx | fast-mlx-guide.md | Graph eval, type promotion, ops, compile, memory, profiling |
| fast-mlx | llm-optimization.md | KV cache tuning, prefill chunking, speculative decoding |
| fast-mlx | dit-optimization.md | Denoising compilation, CFG batching, vision attention |
| fast-mlx | compute-optimization.md | Matrix ops, element-wise fusion, vmap, data pipelines |

## validation

`/skill-maintainer:quality` is the primary validator (see install note in
"maintaining this plugin" above). It checks spec compliance, token budgets,
body size, freshness, and description quality.

Pytest covers version consistency and reference freshness:

```bash
uv run pytest tests/                  # run test suite
```

## release checklist

The full path from "upstream MLX changed" to "shipped a new version":

1. `/update-skills` -- sync reference files with upstream (runs scanner,
   routes diffs, edits refs, validates).
2. `/review-content` -- sanity check reference accuracy before cutting.
3. `/sync-versions X.Y.Z` -- bump all 8 version files, add CHANGELOG header.
4. Fill in CHANGELOG.md entries for the new version.
5. `PYTEST_STRICT=1 uv run pytest tests/` -- final gate (version consistency
   + fails on stale references >45 days).
6. Commit. The pre-commit hook verifies the version bump.

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
  mlx-models/
    SKILL.md
    references/
  fast-mlx/
    SKILL.md
    references/
  mlx-cuda/
    SKILL.md
.claude/
  skills/                   Maintainer workflows (tracked):
    update-skills.md          upstream sync
    review-content.md         accuracy audit
    sync-versions.md          version bump
  settings.local.json       Per-user permissions (gitignored)
.githooks/
  pre-commit                Version-bump gate
scripts/
  check_updates.py          Upstream change scanner (plumbing)
tests/
CHANGELOG.md
pyproject.toml
```
