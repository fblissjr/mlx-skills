<!-- last updated: 2026-04-18 -->

# MLX Skills - Development Guide

## Project Overview

This is a Claude Code plugin that teaches AI coding assistants how to write
correct, performant MLX code. Skills are markdown files with YAML frontmatter
in the top-level `skills/` directory, auto-discovered by the plugin system.

## Maintenance Routines (Read First)

Keeping reference files in sync with upstream MLX is the primary maintenance
burden. Use the slash commands in this order. Do NOT drive `check_updates.py`
by hand -- it only produces a diff report (Step 1 of a 6-step workflow).

### Routine upstream sync

```
/update-skills
```

Runs `scripts/check_updates.py --diff`, categorizes diffs (new APIs, changed
signatures, removals, behavior changes), routes each change to the right
`references/*.md` file, edits in place, then validates with
`/skill-maintainer:quality` + `uv run pytest tests/`.

The skill is defined at `.claude/skills/update-skills.md` and is the source of
truth for the routing rules (which upstream path maps to which reference file).

### Pre-release accuracy audit

```
/review-content
```

Read-only cross-check of reference files against upstream source (CLI flags,
function signatures, API tables). Reports mismatches but does not edit
anything. Use before `/sync-versions` to catch drift the scanner missed.

### Version bump

```
/sync-versions 0.5.10
```

Atomically updates all 8 version locations (see "Version files" below),
refreshes `last_verified` dates, adds a CHANGELOG.md section header, runs
pytest. Skill definition: `.claude/skills/sync-versions.md`. Required before
committing any change under `skills/`, `scripts/`, `.claude-plugin/`, or
`plugins/` (enforced by `.githooks/pre-commit`).

### Always-available hygiene check

```
/skill-maintainer:quality
```

Generic skill linter from the skill-maintainer plugin: spec compliance, token
budgets (4K warn, 8K critical), body size (<500 lines), freshness (<30 days),
description WHAT+WHEN quality. Already invoked by `/update-skills` and
`/sync-versions`. Safe to run anytime.

### Optional: broader maintenance pass

```
/skill-maintainer:init-maintenance   # one-time setup
/skill-maintainer:maintain            # 4-phase run
```

`init-maintenance` creates `.skill-maintainer/` with a `config.json` listing
Claude Code doc URLs and tracked repos. `maintain` then does:
1. Pull tracked local repos (e.g., `coderef/` clones).
2. Check Claude Code upstream docs (skills/plugins/hooks guides) for drift.
3. Run quality report.
4. Propose best-practices updates.

This is complementary to `/update-skills` -- it does NOT track MLX source
files. Use it to catch changes in Claude Code authoring conventions, not MLX
APIs. Skip it unless you want broader hygiene.

### Version bump gate (pre-commit hook)

Commits touching `skills/`, `scripts/`, `.claude-plugin/`, or `plugins/`
require a `pyproject.toml` version bump. Enforced by `.githooks/pre-commit`
(enable once: `git config core.hooksPath .githooks`). Commits limited to
tests, docs, `.claude/skills/`, or `internal/` are unaffected. Bypass with
`--no-verify` if a shipped-path change doesn't warrant a user-visible bump
(e.g., comment fix).

The hook only checks that `pyproject.toml` version changed. `TestVersionConsistency`
in the pytest suite verifies all 8 version files agree -- so the intended
flow is: `/sync-versions X.Y.Z` (updates all 8 locations) then commit.

### Release checklist

1. `/update-skills` -- sync upstream.
2. `/review-content` -- audit accuracy.
3. `/sync-versions X.Y.Z` -- bump version (required by pre-commit hook).
4. Fill in CHANGELOG.md entries under the new header.
5. `PYTEST_STRICT=1 uv run pytest tests/` -- final gate (ref freshness).
6. Commit.

## Which Skill Do I Need?

| I want to... | Use | Command |
|--------------|-----|---------|
| Port PyTorch code to MLX | mlx | `/mlx-skills:mlx` |
| Port NumPy code to MLX | mlx | `/mlx-skills:mlx` |
| Write a custom MLX model | mlx | `/mlx-skills:mlx` |
| Learn MLX fundamentals | mlx | `/mlx-skills:mlx` |
| Write a training loop | mlx | `/mlx-skills:mlx` |
| Debug MLX errors | mlx | `/mlx-skills:mlx` |
| Run a HuggingFace model on my Mac | mlx-models | `/mlx-skills:mlx-models` |
| Fine-tune with LoRA | mlx-models | `/mlx-skills:mlx-models` |
| Quantize a model | mlx-models | `/mlx-skills:mlx-models` |
| Set up a local LLM server | mlx-models | `/mlx-skills:mlx-models` |
| Run a vision-language model (VLM) | mlx-models | `/mlx-skills:mlx-models` |
| Use mlx-vlm or TurboQuant | mlx-models | `/mlx-skills:mlx-models` |
| Process images/audio with a model | mlx-models | `/mlx-skills:mlx-models` |
| Speed up my MLX code | fast-mlx | `/mlx-skills:fast-mlx` |
| Reduce memory usage | fast-mlx | `/mlx-skills:fast-mlx` |
| Profile performance | fast-mlx | `/mlx-skills:fast-mlx` |
| Run MLX on NVIDIA GPU | mlx-cuda | `/mlx-skills:mlx-cuda` |
| Write custom CUDA kernels | mlx-cuda | `/mlx-skills:mlx-cuda` |
| Port Metal kernels to CUDA | mlx-cuda | `/mlx-skills:mlx-cuda` |
| Update skills from upstream | update-skills | `/update-skills` |
| Verify content accuracy | review-content | `/review-content` |
| Bump version everywhere | sync-versions | `/sync-versions 0.5.10` |

## Skills and When They Load

There are 4 skills. Each has a `SKILL.md` (always loaded when triggered)
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

### mlx-models (language models + vision-language models)

**Use for:** Running, fine-tuning, or serving existing HuggingFace models
(text LLMs via mlx-lm, vision-language models via mlx-vlm).

**Triggers:** `import mlx_lm`, `from mlx_lm import`, `import mlx_vlm`,
`from mlx_vlm import`, `stream_generate`, `KVCache`, LoRA, quantization,
GGUF, safetensors, "run llama", "run a model on my mac", "local LLM",
"huggingface model", "vision-language model", "VLM", "TurboQuant",
"multimodal", "VisionFeatureCache", "mlx server".

**Invocation:**
- Automatic: scan imports for `mlx_lm` or `mlx_vlm` usage
- Explicit: `/mlx-skills:mlx-models` (plugin) or `/mlx-models` (personal skill)
- From other skills: "load the mlx-models skill for generation patterns"

**What it covers:** model loading, generation pipelines, KV cache,
quantization, fine-tuning, server deployment (both mlx-lm and mlx-vlm),
vision-language models, TurboQuant.

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
3. Loads `mlx-models` if mlx-lm or mlx-vlm imports are present
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
1. `/mlx-models` loads the model inference skill
2. Reference `patterns.md` for generation pipeline, KV cache, sampling
3. Reference `architecture.md` for model loading flow

### "Speed up my LLM inference"
1. `/fast-mlx` loads optimization skill
2. Reference `llm-optimization.md` for KV cache tuning, prefill chunking, speculative decoding
3. `/mlx-models` for generation pipeline patterns

### "Fine-tune a model with LoRA"
1. `/mlx-models` loads the model inference skill
2. Reference `patterns.md` for LoRA patterns and quantization
3. Reference `architecture.md` for fine-tuning flow

### "Run a vision-language model"
1. `/mlx-models` loads the model inference skill
2. Reference `vlm.md` for mlx-vlm architecture, model catalog, TurboQuant
3. Reference `patterns.md` for Gemma 4 text-side patterns (shared KV, MoE)
4. Reference `architecture.md` for mlx-vlm cross-reference

### "Deploy a local model server"
1. `/mlx-models` loads the model inference skill
2. Reference `serving.md` for both mlx-lm and mlx-vlm server architectures
3. Reference `cli-reference.md` for server CLI flags

### "Update skills from upstream" (maintainer)
1. `/update-skills` loads the maintainer workflow skill
2. Runs `scripts/check_updates.py --diff` to generate a change report
3. Analyzes diffs, routes changes to the right reference files (per the
   routing table in `.claude/skills/update-skills.md` Step 3)
4. Updates reference files (not SKILL.md), refreshes `last updated` headers,
   validates with `/skill-maintainer:quality`, runs `uv run pytest tests/`
5. Does NOT bump versions or touch CHANGELOG. Those are `/sync-versions` --
   run it separately once upstream sync is reviewed and ready to ship.

### "Check content accuracy" (maintainer)
1. `/review-content` loads the content accuracy checker
2. Parses reference files for documented APIs, CLI flags, signatures
3. Fetches upstream source for comparison
4. Reports mismatches -- does not auto-fix

### "Bump project version" (maintainer)
1. `/sync-versions 0.5.10` loads the version coordinator
2. Updates version in all 8 locations (pyproject.toml, 4 SKILL.md,
   .claude-plugin/plugin.json, .claude-plugin/marketplace.json,
   plugins/mlx-skills/.claude-plugin/plugin.json)
3. Updates `last_verified` dates, adds CHANGELOG section header
4. Runs validator and tests to confirm

## Development

### Key files

- `.claude-plugin/plugin.json` -- root plugin manifest
- `.claude-plugin/marketplace.json` -- marketplace catalog
- `plugins/mlx-skills/` -- marketplace plugin wrapper. `skills/` inside it is
  a REAL copy of top-level `skills/`, regenerated from the canonical source
  by `scripts/sync_plugin_skills.py`. Not a symlink: Claude's plugin cache
  preserves symlinks without dereferencing, so a link pointing outside the
  plugin subtree resolves to nothing on installs, which is why Claude
  Desktop showed "no skills" through 0.5.5-0.5.9. Has its own
  `.claude-plugin/plugin.json` that needs separate version bumps.
- `skills/*/SKILL.md` -- skill definitions (YAML frontmatter + body). Edit
  these; the plugin copy is downstream.
- `skills/*/references/*.md` -- reference material (loaded on demand)
- `scripts/sync_plugin_skills.py` -- mirrors top-level `skills/` into
  `plugins/mlx-skills/skills/`. `--check` mode exits non-zero on drift and
  is wired into pytest + the pre-commit hook.
- `scripts/check_updates.py` -- upstream change scanner
- `.claude/skills/` -- project-level maintainer skills (tracked in git,
  loaded by Claude Code when working in this repo; not part of the
  installable plugin manifest). Contains `update-skills.md`,
  `review-content.md`, `sync-versions.md`.
- `.claude/settings.json` -- shared Claude Code config (tracked).
  Configures the PostToolUse hook that auto-syncs the plugin mirror
  after Edit/Write/MultiEdit under `skills/`.
- `.claude/settings.local.json` -- per-user overrides (gitignored).
- `hooks/sync_plugin_skills_on_edit.py` -- the PostToolUse hook body.
  Reads tool payload on stdin; if a file under `skills/` was touched,
  runs `scripts/sync_plugin_skills.py`. Silent on no-op.
- `.githooks/pre-commit` -- version-bump gate AND mirror-drift gate.
  Enable per clone with `git config core.hooksPath .githooks`.
- `tests/` -- pytest suite
- `internal/` -- gitignored scratch / session logs
  (`internal/log/log_YYYY-MM-DD.md`)

### Defense layers for the plugin mirror

Claude Desktop broke through 0.5.5-0.5.9 because a symlink pointing outside
the plugin subtree dangled on install. Four overlapping gates now prevent
the whole class of bug:

1. **PostToolUse hook** (`hooks/sync_plugin_skills_on_edit.py`) -- auto-runs
   the sync script when Claude edits anything under `skills/` in-session.
2. **pytest** -- `TestPluginSkillsMirror` asserts no symlink + byte-parity;
   `TestInstallSmoke` walks the whole plugin tree for ANY symlink and
   validates every plugin-side `SKILL.md` frontmatter; `TestManifestSchemas`
   validates `plugin.json` and `marketplace.json` parse and have required
   keys.
3. **Pre-commit hook** (`.githooks/pre-commit`) -- blocks commits touching
   `skills/` or the mirror if the sync script's `--check` mode reports
   drift.
4. **`/sync-versions` skill** -- Step 3i runs the sync after bumping SKILL.md
   versions; Step 4 re-runs it after updating `last_verified` dates.

### Commands

```
uv run pytest tests/                              # Run tests (version consistency, staleness, plugin mirror)
uv run python scripts/sync_plugin_skills.py      # Refresh plugins/mlx-skills/skills/ from skills/
uv run python scripts/sync_plugin_skills.py --check  # Verify mirror is current (CI + pre-commit use this)
uv run python scripts/check_updates.py --since 30days  # Plumbing: diff report only.
                                                        # Prefer /update-skills.
```

For the full maintenance workflow, see "Maintenance Routines" at the top of
this file. Do not invoke `check_updates.py` as a substitute for `/update-skills`.

### Session gotchas

- `/sync-versions` refers to the project-local skill at
  `.claude/skills/sync-versions.md` (knows this repo's 8-file layout). Do
  NOT invoke `/skill-maintainer:sync-versions` -- that skill uses a
  different, generic layout and will skip most of the locations.
- Sandbox blocks `uv run ...` (cache writes under `~/.cache/uv`),
  `git commit` with SSH signing via 1Password (socket unreachable), and
  `git config` (writes to `.git/config`). Symptoms: "Operation not
  permitted" or "Could not connect to socket." Retry the Bash call with
  `dangerouslyDisableSandbox: true`.
- The project security hook (`hooks/security_reminder_hook.py`)
  pattern-matches the literal `eval` + open-paren sequence and blocks
  Edits whose old/new strings contain `mx.eval` followed by `(`. This is a
  false positive -- MLX's array evaluator is not Python's builtin. Narrow
  the Edit so that substring isn't in either side of the diff, or split
  the edit into chunks that avoid it.
- The Write tool may report success but the file evaporates between
  commands when writing to a newly-created directory outside the
  sandbox's writable paths. Symptom: `ls` shows the dir missing even
  though Write said "File created successfully". Fix: create via
  `cat > <path> <<EOF ... EOF` in a Bash call with
  `dangerouslyDisableSandbox: true`, then Write-tool edits to the
  same path will persist normally.
- A failing PostToolUse hook configured in `.claude/settings.json`
  blocks ALL subsequent Edit/Write/MultiEdit calls in the session,
  not just the one that triggered it. If you see
  "PostToolUse:Edit hook blocking error" on unrelated edits, the
  quickest recovery is `rm .claude/settings.json` (or rename it),
  fix the hook body, then restore the config.

### Version files (ALL must match on every release)

1. `pyproject.toml` -- `version` field
2. `skills/mlx/SKILL.md` -- `metadata.version`
3. `skills/mlx-models/SKILL.md` -- `metadata.version`
4. `skills/fast-mlx/SKILL.md` -- `metadata.version`
5. `skills/mlx-cuda/SKILL.md` -- `metadata.version`
6. `.claude-plugin/plugin.json` -- `version` field
7. `.claude-plugin/marketplace.json` -- `plugins[0].version`
8. `plugins/mlx-skills/.claude-plugin/plugin.json` -- `version` field

`uv run pytest tests/` will fail if any version is out of sync.
Always run tests after version bumps.

Skill validation is handled by the skill-maintainer plugin (`/skill-maintainer:quality`). Install via: `/plugin marketplace add fblissjr/fb-claude-skills` then `/plugin install skill-maintainer@fb-claude-skills`.

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
- Run `/skill-maintainer:quality` after any changes to verify structure (requires skill-maintainer plugin)

### Content guidelines

- `mlx` skill: core MLX framework only (no mlx-lm specifics)
- `mlx-models` skill: mlx-lm and mlx-vlm (generation, caching, fine-tuning, serving, VLMs)
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
