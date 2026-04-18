---
name: update-skills
description: >
  Maintainer workflow: scan upstream MLX repos for changes, analyze diffs,
  update skill reference files, and validate. Use when asked to update skills
  from upstream, sync with latest MLX changes, or run /update-skills.
---

# Update Skills from Upstream Changes

This skill automates the process of keeping MLX skill reference files in sync
with upstream MLX repositories. It runs the change scanner, analyzes what
changed, routes updates to the right reference files, and validates.

## Step 1 -- Generate the upstream change report

Run the scanner with diffs enabled:

```
uv run python scripts/check_updates.py --diff --since {N}days
```

Default to 30 days. The user can specify a different window (e.g., `--since 7days`,
`--since 2025-01-01`). To scan specific repos only, add `--repos mlx mlx-lm`.

Use `--diff-lines N` to truncate large diffs if the output is unmanageable.

Read the full report output before proceeding.

## Step 2 -- Analyze what changed

For each watched file that has diffs in the report, categorize the changes:

- **New APIs**: New classes, functions, or methods added
- **Changed signatures**: Parameters added/removed/renamed, new defaults
- **Removed/deprecated**: Functions or classes removed, deprecated, or renamed
- **Behavioral changes**: New default values, changed semantics, new modes
- **Documentation changes**: Updated docs (`.rst` files in the watched list)

Skip changes that are purely internal (private methods, test-only code,
formatting/style changes with no functional impact).

## Step 3 -- Route changes to reference files (smart matching)

Do NOT use a static mapping. Instead:

1. Read the headings and structure of each reference file in the affected skill:
   - `skills/mlx/references/` -- fundamentals.md, nn-and-training.md,
     anti-patterns.md, debugging.md, porting-guide.md, custom-kernels.md
   - `skills/mlx-models/references/` -- patterns.md, architecture.md, cli-reference.md,
     serving.md, vlm.md
   - `skills/fast-mlx/references/` -- fast-mlx-guide.md,
     llm-optimization.md, dit-optimization.md, compute-optimization.md

2. Match each upstream change to the reference file whose content covers that topic.
   Every file in `scripts/check_updates.py` `WATCHED_FILES` has a routing rule
   here. The `TestRoutingCoverage` pytest class fails if any watched file is
   missing from this list.

   **mlx core (nn layers -> `mlx/references/nn-and-training.md`):**
   - `nn/layers/linear.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/normalization.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/transformer.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/activations.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/convolution.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/recurrent.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/positional_encoding.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/quantized.py` -> `mlx/references/nn-and-training.md`
   - `nn/layers/embedding.py` -> `mlx/references/nn-and-training.md`
   - `nn/losses.py` -> `mlx/references/nn-and-training.md`
   - `nn/init.py` -> `mlx/references/nn-and-training.md`
   - `nn/utils.py` -> `mlx/references/nn-and-training.md`

   **mlx core (optimizers -> `mlx/references/nn-and-training.md`):**
   - `optimizers/__init__.py` -> `mlx/references/nn-and-training.md`
   - `optimizers/optimizers.py` -> `mlx/references/nn-and-training.md`
   - `optimizers/schedulers.py` -> `mlx/references/nn-and-training.md`

   **mlx core (runtime + docs -> `mlx/references/fundamentals.md`):**
   - `python/mlx/utils.py` (mlx core) -> `mlx/references/fundamentals.md`
   - `docs/src/usage/lazy_evaluation.rst` -> `mlx/references/fundamentals.md`
   - `docs/src/usage/compile.rst` -> `mlx/references/fundamentals.md`
   - `docs/src/usage/unified_memory.rst` -> `mlx/references/fundamentals.md`

   **mlx-lm (generation + models -> `mlx-models/references/patterns.md`):**
   - `mlx_lm/generate.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/models/cache.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/models/base.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/models/llama.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/utils.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/tuner/lora.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/tuner/trainer.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/sample_utils.py` -> `mlx-models/references/patterns.md`
   - `mlx_lm/tokenizer_utils.py` -> `mlx-models/references/patterns.md`

   **mlx-lm (server + conversion):**
   - `mlx_lm/server.py` -> `mlx-models/references/serving.md`
   - `mlx_lm/convert.py` -> `mlx-models/references/cli-reference.md`

   **mlx-vlm -> `mlx-models/references/vlm.md` (unless server):**
   - `mlx_vlm/utils.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/generate.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/turboquant.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/models/base.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/models/cache.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/lora.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/video_generate.py` -> `mlx-models/references/vlm.md`
   - `mlx_vlm/server.py` -> `mlx-models/references/serving.md`

3. If a significant new area isn't covered by any existing reference file,
   propose creating a new one -- but ask the user for confirmation first.

**Note on mlx-cuda:** The `mlx-cuda` skill exists but CUDA upstream files are
not currently tracked in `WATCHED_FILES`. It's lower priority -- Metal is the
primary backend. If CUDA tracking is added later, route changes to
`skills/mlx-cuda/SKILL.md` (no references directory exists today).

## Step 4 -- Update reference files

For each target reference file:

1. Read the full file content
2. Apply changes based on what was categorized in Step 2:

   **New APIs:**
   - Add to the appropriate section with a minimal, correct code example
   - Follow the existing style of the file (heading levels, formatting)
   - Place in logical order relative to existing content

   **Changed signatures:**
   - Update existing code examples to use the new signature
   - Update parameter descriptions
   - If a new parameter enables important behavior, add a brief note

   **Removed/deprecated:**
   - Remove from reference files if fully removed upstream
   - If deprecated but still functional, add a brief deprecation note so users
     of the old API know what to migrate to
   - Remove stale code examples that use the old API

   **Stale content:**
   - Remove content that no longer matches upstream reality
   - Don't just accumulate -- actively prune outdated information

3. Update the `last updated: YYYY-MM-DD` date at the top of each modified file

## Step 5 -- Guard SKILL.md size

NEVER add detailed content to SKILL.md files. They are always loaded into
context and must stay concise.

- SKILL.md body must stay under 5000 words (the validator enforces this)
- SKILL.md should only contain summaries with pointers to reference files
- If a SKILL.md mentions something that changed upstream, update the mention
  but keep it brief (one line, not a paragraph)
- All detailed content goes in reference files

## Step 6 -- Validate and report

After making changes:

1. Run validation: `/skill-maintainer:quality` (or `uv run agentskills validate skills/` if plugin not installed)
2. Run tests: `uv run pytest tests/ -v`
3. Report to the user:
   - Which reference files were updated
   - Summary of what changed in each
   - Any new reference files proposed (if applicable)
   - Validation and test results

## Guardrails

- **Reference files are the target**, not SKILL.md
- **Remove stale content** -- don't just add; prune what's outdated
- **Keep code examples minimal and correct** -- just enough to show usage
- **Preserve existing structure** -- match heading levels, formatting style
- **When in doubt, skip it** -- if a change isn't clearly relevant to what
  the skills teach, leave it out
- **Ask before creating** new reference files
- **Cross-reference** other skills instead of duplicating content
  (use "load the `skill-name` skill" pattern)
- **Don't duplicate** content across skills; if mlx-models skill's patterns.md already
  covers cache changes, don't also add them to mlx fundamentals.md
