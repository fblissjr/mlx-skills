---
name: sync-versions
description: >
  Bump the project version across all 8 locations (pyproject.toml, four
  SKILL.md metadata blocks, plugin.json, marketplace.json, plugins wrapper
  plugin.json), update last_verified dates, and add a CHANGELOG.md section
  header. Pass the target version as an argument (e.g., /sync-versions 0.5.7).
---

# Sync Versions Across All Locations

This skill bumps the project version atomically across all 8 locations where
version is tracked, updates `last_verified` dates, adds a CHANGELOG section,
and validates.

## Step 1 -- Determine target version

If the user passed a version argument (e.g., `/sync-versions 0.5.7`), use that.
Otherwise, ask what version to bump to.

Read the current version from `pyproject.toml` to confirm the starting point.

## Step 2 -- Validate the version

- Must be valid semver (X.Y.Z)
- Must be higher than the current version
- If the target equals the current version, report "already at X.Y.Z" and stop
- Do NOT bump major version without explicit user confirmation

## Step 3 -- Update all 8 locations

Update the version string in each of these files:

### 3a. pyproject.toml

```
version = "X.Y.Z"
```

### 3b. skills/mlx/SKILL.md

In the YAML frontmatter `metadata` block:

```yaml
metadata:
  version: "X.Y.Z"
```

### 3c. skills/mlx-models/SKILL.md

Same as 3b.

### 3d. skills/fast-mlx/SKILL.md

Same as 3b.

### 3e. skills/mlx-cuda/SKILL.md

Same as 3b.

### 3f. .claude-plugin/plugin.json

```json
"version": "X.Y.Z"
```

### 3g. .claude-plugin/marketplace.json

In `plugins[0].version`:

```json
"version": "X.Y.Z"
```

### 3h. plugins/mlx-skills/.claude-plugin/plugin.json

```json
"version": "X.Y.Z"
```

### 3i. Mirror the edited SKILL.md files into the plugin copy

Steps 3b-3e edited the canonical `skills/*/SKILL.md` files. The plugin at
`plugins/mlx-skills/skills/` is a real-file mirror that now has stale versions.
Run the sync script to propagate:

```
uv run python scripts/sync_plugin_skills.py
```

The pre-commit hook and the `TestPluginSkillsMirror` pytest will block the
release if this step is skipped.

## Step 4 -- Update last_verified dates

Set `last_verified` to today's date (YYYY-MM-DD) in the `metadata` block of
all four SKILL.md files:

```yaml
metadata:
  last_verified: "YYYY-MM-DD"
```

Then re-run `uv run python scripts/sync_plugin_skills.py` so the plugin
copy picks up the new dates too.

## Step 5 -- Update CHANGELOG.md

Add a new section header at the top of the changelog (after any existing header):

```markdown
## X.Y.Z

- (changes go here)
```

If there is already a section for this version, skip this step.

Do NOT fill in changelog entries -- the user will do that. Just add the header.

## Step 6 -- Update memory

Update the version number in `/Users/fredbliss/.claude/projects/-Users-fredbliss-claude-mlx-skills/memory/MEMORY.md`
under the "Versioning" section to reflect the new version.

## Step 7 -- Validate

Run tests in **strict mode** to confirm nothing broke. `PYTEST_STRICT=1`
promotes `TestReferenceStaleness` from a warning to a hard failure -- a
release should not ship with reference files >45 days stale.

```
PYTEST_STRICT=1 uv run pytest tests/ -q
```

The following tests enforce release invariants:

- `TestVersionConsistency::test_all_versions_match` -- all version files agree
- `TestReferenceStaleness` -- fails (under strict mode) if any reference file is stale
- `TestVersionFilesDocumented` -- doc counts match `VERSION_FILES`
- `TestSkillListDocumented` -- doc counts match `SKILL_NAMES`
- `TestRoutingCoverage` -- every `WATCHED_FILES` entry has a routing rule

If `TestReferenceStaleness` fails, run `/update-skills` first to refresh
references, then retry `/sync-versions`.

## Step 8 -- Report

List exactly what was changed:

```
Version bumped: 0.5.6 -> 0.5.7

Updated files:
  - pyproject.toml (version)
  - skills/mlx/SKILL.md (metadata.version, last_verified)
  - skills/mlx-models/SKILL.md (metadata.version, last_verified)
  - skills/fast-mlx/SKILL.md (metadata.version, last_verified)
  - skills/mlx-cuda/SKILL.md (metadata.version, last_verified)
  - .claude-plugin/plugin.json (version)
  - .claude-plugin/marketplace.json (plugins[0].version)
  - plugins/mlx-skills/.claude-plugin/plugin.json (version)
  - CHANGELOG.md (new section header)

Validation: passed
Tests: passed
```

## Guardrails

- **Atomic** -- update all 8 locations or none (if any edit fails, stop and report)
- **No major bumps** without explicit user confirmation
- **No changelog entries** -- just the section header; user fills in details
- **Validate after** -- always run tests to catch version mismatches
- **Do not commit** -- the user decides when to commit
