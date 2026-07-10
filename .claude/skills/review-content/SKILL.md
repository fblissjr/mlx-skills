---
name: review-content
description: >
  Verify skill reference files against upstream MLX and mlx-lm source code.
  Cross-checks function signatures, CLI flags, API tables, and code examples
  for accuracy. Reports mismatches but does not auto-fix. Use when preparing
  a release, after upstream syncs, or when suspecting content drift.
---

# Review Content Accuracy Against Upstream

This skill systematically cross-checks documented APIs, CLI flags, and function
signatures in skill reference files against upstream MLX and mlx-lm source code.
It produces a report of mismatches but does NOT auto-fix anything.

## Step 1 -- Identify what to verify

Read all reference files and extract verifiable claims:

- `skills/mlx/references/` -- fundamentals.md, nn-and-training.md,
  anti-patterns.md, debugging.md, porting-guide.md
- `skills/mlx-models/references/` -- patterns.md, architecture.md, cli-reference.md,
  serving.md, vlm.md
- `skills/fast-mlx/references/` -- fast-mlx-guide.md, llm-optimization.md,
  dit-optimization.md, compute-optimization.md

For each file, extract:

- **Function signatures** -- parameter names, defaults, types
- **CLI flags** -- flag names, defaults, descriptions
- **API tables** -- class/function names, parameter lists
- **Code examples** -- imports, function calls, argument usage

Focus on the areas most prone to drift: CLI flags with specific defaults,
function signatures with parameter names, and API tables listing specific
entries.

## Step 2 -- Fetch upstream source for comparison

Use one of these approaches to get current upstream source:

### Option A: Use check_updates.py infrastructure

```
uv run python scripts/check_updates.py --repos mlx mlx-lm --diff --since 1days
```

This clones/fetches repos. After running, the repos are available at the path
shown in the output (or at `MLX_SKILLS_REPOS` if set).

### Option B: Fetch specific files from GitHub

Use WebFetch to retrieve specific source files directly:

- `https://raw.githubusercontent.com/ml-explore/mlx/main/python/mlx/nn/layers/*.py`
- `https://raw.githubusercontent.com/ml-explore/mlx-examples/main/llms/mlx_lm/generate.py`
- etc.

### Key upstream files to check against

| Documented area | Upstream source |
|----------------|-----------------|
| Layer signatures (nn.Linear, etc.) | `mlx/python/mlx/nn/layers/*.py` |
| Init functions | `mlx/python/mlx/nn/init.py` |
| Optimizer signatures | `mlx/python/mlx/optimizers/*.py` |
| Generation API | `mlx-lm/mlx_lm/generate.py` |
| Server CLI flags (text) | `mlx-lm/mlx_lm/server.py` |
| Server CLI flags (VLM) | `mlx-vlm/mlx_vlm/server.py` |
| Benchmark CLI flags | `mlx-lm/mlx_lm/benchmark.py` |
| Convert CLI flags | `mlx-lm/mlx_lm/convert.py` |
| All mlx-lm subcommands | `mlx-lm/mlx_lm/` entry points |
| TurboQuant | `mlx-vlm/mlx_vlm/turboquant.py` |
| VLM generation | `mlx-vlm/mlx_vlm/generate.py` |
| VLM model base | `mlx-vlm/mlx_vlm/models/base.py` |

## Step 3 -- Cross-check and report

For each documented claim, compare against upstream source and categorize:

### Mismatch types

- **Wrong default**: documented default differs from source
- **Missing parameter**: source has a param not documented
- **Extra parameter**: documented param not in source (removed?)
- **Wrong signature**: parameter order, names, or types differ
- **Non-existent flag**: CLI flag documented but not in argparse
- **Missing flag**: argparse flag not documented
- **Stale example**: code example uses removed/changed API

### Report format

Output a structured report grouped by reference file:

```
## skills/mlx-models/references/cli-reference.md

### MISMATCH: Wrong default (line 42)
- Documented: `--max-tokens` default is 256
- Upstream: `--max-tokens` default is 512
- Source: mlx_lm/generate.py:87

### MISSING: New parameter (not documented)
- `--top-k` added in mlx_lm/generate.py:91
- Should be added to the generation flags table

## skills/mlx/references/nn-and-training.md

### OK: All layer signatures verified
(no mismatches found)
```

### Summary

End with a summary:

```
## Summary
- Files checked: 8
- Mismatches found: 3
- Missing APIs: 1
- Stale content: 0
- Files with no issues: 5
```

## Step 4 -- Do NOT auto-fix

This skill is report-only. After reviewing the report:

- Use `/update-skills` to apply fixes via the standard update workflow
- Or fix manually with targeted edits
- Or note items as intentional omissions (not everything upstream needs documenting)

## Guardrails

- **Read-only** -- never modify skill files during a review
- **Be specific** -- include line numbers and upstream source locations
- **Skip internals** -- don't flag undocumented private/internal APIs
- **Skip trivial** -- don't flag docstring-only or formatting differences
- **Focus on user-facing** -- CLI flags, public API signatures, documented examples
- **Note confidence** -- if upstream source is ambiguous, say so
