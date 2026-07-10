"""Structural validation tests for skill files.

Replaces the removed validate.py with proper pytest coverage:
- Version consistency across all locations
- SKILL.md frontmatter validity
- Reference file existence and headers
- Cross-reference validity between skills
- SKILL.md word count guard
- Stale skill path detection
- Reference file staleness warnings (strict mode via PYTEST_STRICT=1)
- Routing coverage: every WATCHED_FILES entry has a rule in update-skills/SKILL.md
- Doc-count invariants: skill count and version-file count claims in docs
  must match the actual constants.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import warnings
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
SKILL_NAMES = sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir() and (p / "SKILL.md").exists())


def _load_script_module(stem: str):
    """Load scripts/<stem>.py as a module.

    scripts/ has no __init__.py, so use importlib. Cached via sys.modules.
    """
    if stem in sys.modules:
        return sys.modules[stem]
    path = ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def _load_check_updates():
    return _load_script_module("check_updates")


def _strict_mode() -> bool:
    """PYTEST_STRICT=1 gates release-blocking checks (staleness, etc).

    /sync-versions sets this so a release fails loudly if references are stale.
    Routine dev runs with strict off so warnings don't block iteration.
    """
    return os.environ.get("PYTEST_STRICT", "").strip() in ("1", "true", "True")

REQUIRED_FRONTMATTER = {"name", "description", "license", "metadata"}
REQUIRED_METADATA = {"author", "version", "last_verified"}
MAX_BODY_WORDS = 5000

# All locations where the version is tracked
VERSION_FILES = [
    ROOT / "pyproject.toml",
    ROOT / ".claude-plugin" / "plugin.json",
    ROOT / ".claude-plugin" / "marketplace.json",
    ROOT / "plugins" / "mlx-skills" / ".claude-plugin" / "plugin.json",
] + [SKILLS_DIR / name / "SKILL.md" for name in SKILL_NAMES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Returns (frontmatter_dict, body_text). Handles the >- folded scalar
    style used for descriptions by joining continuation lines.
    """
    text = path.read_text()
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    raw = text[3:end].strip()
    body = text[end + 3:].strip()

    fm: dict[str, str] = {}
    metadata: dict[str, str] = {}
    current_key = ""
    in_metadata = False
    in_multiline = False
    multiline_parts: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # Metadata sub-keys (2+ space indent under metadata:)
        if in_metadata and indent >= 2 and ":" in stripped:
            k, _, v = stripped.partition(":")
            metadata[k.strip()] = v.strip().strip('"')
            continue

        # Continuation of a folded scalar (2+ space indent)
        if in_multiline and indent >= 2:
            multiline_parts.append(stripped)
            continue

        # End of metadata or multiline on unindented line
        if in_metadata and indent == 0:
            in_metadata = False
        if in_multiline and indent == 0:
            fm[current_key] = " ".join(multiline_parts)
            in_multiline = False
            multiline_parts = []

        if not stripped or ":" not in stripped:
            continue

        k, _, v = stripped.partition(":")
        k = k.strip()
        v = v.strip().strip('"')

        if k == "metadata":
            in_metadata = True
            in_multiline = False
            continue

        if v in (">-", ">", "|", "|-"):
            current_key = k
            in_multiline = True
            in_metadata = False
            multiline_parts = []
            continue

        in_metadata = False
        in_multiline = False
        fm[k] = v

    if in_multiline:
        fm[current_key] = " ".join(multiline_parts)

    fm["metadata"] = metadata  # type: ignore[assignment]
    return fm, body


def _extract_version(path: Path) -> str:
    """Extract version string from any of the tracked files."""
    text = path.read_text()
    name = path.name

    if name == "SKILL.md":
        fm, _ = _parse_frontmatter(path)
        metadata = fm.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata.get("version", "")
        return ""
    elif name == "pyproject.toml":
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    elif name.endswith(".json"):
        m = re.search(r'"version"\s*:\s*"([^"]+)"', text)
    else:
        return ""

    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVersionConsistency:
    """All version locations must agree."""

    def test_all_versions_match(self):
        versions = {}
        for path in VERSION_FILES:
            if path.exists():
                versions[str(path.relative_to(ROOT))] = _extract_version(path)

        unique = set(versions.values())
        assert len(unique) == 1, (
            f"Version mismatch across files:\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(versions.items()))
        )


class TestFrontmatterValidity:
    """Every SKILL.md must have valid frontmatter."""

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_required_fields(self, skill_name: str):
        path = SKILLS_DIR / skill_name / "SKILL.md"
        fm, _ = _parse_frontmatter(path)
        missing = REQUIRED_FRONTMATTER - fm.keys()
        assert not missing, f"{skill_name}/SKILL.md missing frontmatter fields: {missing}"

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_required_metadata(self, skill_name: str):
        path = SKILLS_DIR / skill_name / "SKILL.md"
        fm, _ = _parse_frontmatter(path)
        metadata = fm.get("metadata", {})
        assert isinstance(metadata, dict), f"{skill_name}: metadata is not a dict"
        missing = REQUIRED_METADATA - metadata.keys()
        assert not missing, f"{skill_name}/SKILL.md missing metadata fields: {missing}"

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_name_matches_directory(self, skill_name: str):
        path = SKILLS_DIR / skill_name / "SKILL.md"
        fm, _ = _parse_frontmatter(path)
        assert fm.get("name") == skill_name, (
            f"Frontmatter name '{fm.get('name')}' does not match directory '{skill_name}'"
        )

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_description_present(self, skill_name: str):
        path = SKILLS_DIR / skill_name / "SKILL.md"
        fm, _ = _parse_frontmatter(path)
        desc = fm.get("description", "")
        assert len(desc) > 50, f"{skill_name}: description too short ({len(desc)} chars)"


class TestReferenceFiles:
    """Reference files must exist and have proper headers."""

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_referenced_files_exist(self, skill_name: str):
        skill_md = SKILLS_DIR / skill_name / "SKILL.md"
        text = skill_md.read_text()
        # Match [references/foo.md](references/foo.md) or (references/foo.md)
        refs = re.findall(r'\(references/([^)]+\.md)\)', text)
        missing = []
        for ref in refs:
            ref_path = SKILLS_DIR / skill_name / "references" / ref
            if not ref_path.exists():
                missing.append(ref)
        assert not missing, f"{skill_name}: missing reference files: {missing}"

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_reference_date_headers(self, skill_name: str):
        ref_dir = SKILLS_DIR / skill_name / "references"
        if not ref_dir.exists():
            pytest.skip(f"{skill_name} has no references/ directory")
        for ref_file in sorted(ref_dir.glob("*.md")):
            first_line = ref_file.read_text().splitlines()[0]
            assert re.match(r"last updated: \d{4}-\d{2}-\d{2}", first_line), (
                f"{skill_name}/references/{ref_file.name}: "
                f"first line must be 'last updated: YYYY-MM-DD', got: {first_line!r}"
            )


class TestCrossReferences:
    """Cross-references between skills must point to valid skills."""

    def test_all_cross_refs_valid(self):
        pattern = re.compile(r'load the `([^`]+)` skill')
        invalid = []
        for skill_name in SKILL_NAMES:
            skill_dir = SKILLS_DIR / skill_name
            for md_file in skill_dir.rglob("*.md"):
                text = md_file.read_text()
                for match in pattern.finditer(text):
                    ref = match.group(1)
                    if ref not in SKILL_NAMES:
                        rel = md_file.relative_to(ROOT)
                        invalid.append(f"{rel}: references non-existent skill '{ref}'")
        assert not invalid, "Invalid cross-references:\n" + "\n".join(invalid)


class TestStaleSkillPaths:
    """Catch references to skill directories that no longer exist."""

    # Directories and files to skip (historical logs, changelogs, URLs)
    _SKIP = ("CHANGELOG.md", ".venv", "node_modules", ".git", "internal/")

    def test_no_stale_skill_directory_refs(self):
        """Scan project .md and .py files for paths like skills/<name>/
        where <name> is not a current skill directory.

        Catches stale references after skill renames or removals."""
        # Match relative paths like skills/<name>/SKILL.md or
        # skills/<name>/references/. Excludes URLs (preceded by //) and
        # absolute paths (preceded by /).
        pattern = re.compile(r'(?<!/)(?<!//)skills/([a-z][a-z0-9-]+)/(?:SKILL|references)')
        stale = []
        for ext in ("*.md", "*.py"):
            for f in ROOT.rglob(ext):
                rel = str(f.relative_to(ROOT))
                if any(skip in rel for skip in self._SKIP):
                    continue
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    for m in pattern.finditer(line):
                        ref_name = m.group(1)
                        if ref_name not in SKILL_NAMES:
                            stale.append(f"{rel}:{i}: references 'skills/{ref_name}/' but no such skill exists")
        assert not stale, (
            "Stale skill directory references found:\n" + "\n".join(stale)
        )


class TestReferenceStaleness:
    """Warn when reference files haven't been updated in over 45 days.

    Normal mode: emits a warning. PYTEST_STRICT=1: fails the test. /sync-versions
    runs with PYTEST_STRICT=1 so a release is gated on reference freshness.
    """

    MAX_AGE_DAYS = 45

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_reference_dates_not_stale(self, skill_name: str):
        ref_dir = SKILLS_DIR / skill_name / "references"
        if not ref_dir.exists():
            pytest.skip(f"{skill_name} has no references/ directory")
        stale = []
        for ref in sorted(ref_dir.glob("*.md")):
            first_line = ref.read_text().splitlines()[0]
            m = re.match(r"last updated: (\d{4}-\d{2}-\d{2})", first_line)
            if m:
                age = (date.today() - date.fromisoformat(m.group(1))).days
                if age > self.MAX_AGE_DAYS:
                    stale.append(f"{ref.name}: {age} days old")
        if stale:
            msg = (
                f"{skill_name} has stale references (>{self.MAX_AGE_DAYS} days):\n"
                + "\n".join(f"  {s}" for s in stale)
            )
            if _strict_mode():
                pytest.fail(msg)
            else:
                warnings.warn(msg, stacklevel=1)


class TestWordCount:
    """SKILL.md body must stay under the word limit."""

    @pytest.mark.parametrize("skill_name", SKILL_NAMES)
    def test_body_under_limit(self, skill_name: str):
        path = SKILLS_DIR / skill_name / "SKILL.md"
        _, body = _parse_frontmatter(path)
        word_count = len(body.split())
        assert word_count <= MAX_BODY_WORDS, (
            f"{skill_name}/SKILL.md body is {word_count} words (limit: {MAX_BODY_WORDS})"
        )


# Drift-detection gates: invariants between machine-readable sources
# (WATCHED_FILES, VERSION_FILES, SKILL_NAMES) and the prose that documents
# them. These fail loudly so a maintainer can't silently add a watched file
# without a routing rule, a skill without docs, or a version location without
# touching /sync-versions.

UPDATE_SKILLS_PATH = ROOT / ".claude" / "skills" / "update-skills" / "SKILL.md"
SYNC_VERSIONS_PATH = ROOT / ".claude" / "skills" / "sync-versions" / "SKILL.md"
CLAUDE_MD_PATH = ROOT / "CLAUDE.md"
README_PATH = ROOT / "README.md"
PROJECT_SKILLS_DIR = ROOT / ".claude" / "skills"
REQUIRED_PROJECT_SKILL_FRONTMATTER = {"name", "description"}
# Non-skill entries allowed to sit in .claude/skills/. A leading "." or "_"
# marks scratch/tool-generated content (matches this file's own _foo naming
# convention); README.md is the one expected non-skill filename.
_ALLOWED_NON_SKILL_FILES = {"README.md"}
PROJECT_SKILL_NAMES = sorted(
    p.name for p in PROJECT_SKILLS_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
)


class TestProjectSkillsFormat:
    """Project-level skills must use the <name>/SKILL.md directory format.

    A flat `.claude/skills/<name>.md` file is silently never discovered by
    Claude Code's skill auto-discovery -- it doesn't error, it just never
    appears as an invocable skill/command. All three maintainer skills
    shipped this way undetected until fixed. Guard against regressing.
    """

    def test_no_flat_md_files(self):
        flat_files = sorted(
            p.name for p in PROJECT_SKILLS_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".md"
            and p.name not in _ALLOWED_NON_SKILL_FILES
        )
        assert not flat_files, (
            "Flat .md files directly under .claude/skills/ are never auto-discovered:\n"
            + "\n".join(f"  {f}" for f in flat_files)
            + "\nMove each into .claude/skills/<name>/SKILL.md."
        )

    def test_every_subdirectory_has_skill_md(self):
        missing = []
        for p in sorted(PROJECT_SKILLS_DIR.iterdir()):
            if not p.is_dir() or p.name.startswith((".", "_")):
                continue
            # Exact-case, file-only membership check: Path.exists() alone
            # would accept a directory named SKILL.md, and would silently
            # case-fold on case-insensitive filesystems (e.g. macOS APFS).
            entry_names = {child.name for child in p.iterdir() if child.is_file()}
            if "SKILL.md" not in entry_names:
                missing.append(p.name)
        assert not missing, (
            "Project-level skill directories missing a SKILL.md entrypoint:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    def test_frontmatter_has_required_fields(self):
        problems = []
        for name in PROJECT_SKILL_NAMES:
            skill_md = PROJECT_SKILLS_DIR / name / "SKILL.md"
            if not skill_md.is_file():
                continue  # reported by test_every_subdirectory_has_skill_md
            frontmatter, _ = _parse_frontmatter(skill_md)
            missing_fields = REQUIRED_PROJECT_SKILL_FRONTMATTER - frontmatter.keys()
            if missing_fields:
                problems.append(f"{name}/SKILL.md: missing {sorted(missing_fields)}")
            elif frontmatter.get("name") != name:
                problems.append(
                    f"{name}/SKILL.md: name '{frontmatter.get('name')}' != directory '{name}'"
                )
        assert not problems, (
            "Project-level skill frontmatter problems:\n"
            + "\n".join(f"  {p}" for p in problems)
        )


class TestProjectSkillListDocumented:
    """The .claude/skills/ list/count in README.md and CLAUDE.md must match disk."""

    def test_skill_count_matches_docs(self):
        expected = len(PROJECT_SKILL_NAMES)
        claim_pattern = re.compile(r"\ball (\d+) maintainer skills\b", re.IGNORECASE)
        mismatches = []
        for doc in (CLAUDE_MD_PATH, README_PATH):
            text = doc.read_text()
            for match in claim_pattern.finditer(text):
                claim = int(match.group(1))
                if claim != expected:
                    rel = doc.relative_to(ROOT)
                    mismatches.append(
                        f"{rel}: claims '{match.group(0)}' but .claude/skills/ has {expected} entries"
                    )
        assert not mismatches, (
            "Documentation numbers out of sync with .claude/skills/:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )

    def test_all_project_skills_mentioned_in_docs(self):
        missing = []
        for doc in (CLAUDE_MD_PATH, README_PATH):
            text = doc.read_text()
            for name in PROJECT_SKILL_NAMES:
                if name not in text:
                    missing.append(f"{doc.relative_to(ROOT)}: missing '{name}'")
        assert not missing, (
            "Project-level skills not mentioned in docs:\n" + "\n".join(f"  {m}" for m in missing)
        )


# Routing prose refers to watched paths with `python/mlx/` stripped. Match
# against both the full and stripped form.
_PATH_PREFIXES = ("python/mlx/",)


def _watched_path_candidates(path: str) -> list[str]:
    """Return the path forms a routing rule might use to refer to this file."""
    candidates = [path]
    for prefix in _PATH_PREFIXES:
        if path.startswith(prefix):
            candidates.append(path[len(prefix):])
    return candidates


class TestRoutingCoverage:
    """Every watched upstream file needs a routing rule in update-skills/SKILL.md."""

    def test_every_watched_file_has_routing_rule(self):
        """WATCHED_FILES -> update-skills/SKILL.md routing coverage.

        For each file in scripts/check_updates.py WATCHED_FILES, at least one
        backticked form of its path must appear in the routing section.
        """
        mod = _load_check_updates()
        routing_text = UPDATE_SKILLS_PATH.read_text()

        missing = []
        for repo, files in mod.WATCHED_FILES.items():
            for path in files:
                candidates = _watched_path_candidates(path)
                if not any(f"`{c}`" in routing_text for c in candidates):
                    missing.append(f"{repo}: {path}")

        assert not missing, (
            "WATCHED_FILES entries with no routing rule in update-skills/SKILL.md:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nAdd a rule under Step 3 or remove the entry from WATCHED_FILES."
        )

    def test_routing_targets_exist(self):
        """Every routing arrow `-> target/path.md` must resolve on disk.

        Matches lines like:  `source.py` -> `mlx/references/patterns.md`
        The target path is relative to skills/.
        """
        routing_text = UPDATE_SKILLS_PATH.read_text()
        arrow_pattern = re.compile(r"->\s*`([^`]+\.md)`")
        targets = set(arrow_pattern.findall(routing_text))

        missing = []
        for target in sorted(targets):
            # Targets are like "mlx/references/patterns.md" (relative to skills/)
            # or "skills/mlx-cuda/SKILL.md" (absolute-from-root, rare).
            if target.startswith("skills/"):
                resolved = ROOT / target
            else:
                resolved = SKILLS_DIR / target
            if not resolved.exists():
                missing.append(f"{target} (expected at {resolved.relative_to(ROOT)})")

        assert not missing, (
            "Routing targets that don't exist on disk:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


class TestVersionFilesDocumented:
    """Numeric claims about version-file count must match VERSION_FILES."""

    def test_version_file_count_matches_docs(self):
        """`all N version files` / `N locations` claims must equal len(VERSION_FILES)."""
        expected = len(VERSION_FILES)
        claim_pattern = re.compile(
            r"\ball (\d+) (?:version files?|locations?|version locations?)\b",
            re.IGNORECASE,
        )

        mismatches = []
        for doc in (CLAUDE_MD_PATH, README_PATH, SYNC_VERSIONS_PATH, UPDATE_SKILLS_PATH):
            if not doc.exists():
                continue
            text = doc.read_text()
            for match in claim_pattern.finditer(text):
                claim = int(match.group(1))
                if claim != expected:
                    rel = doc.relative_to(ROOT)
                    mismatches.append(
                        f"{rel}: claims '{match.group(0)}' but VERSION_FILES has {expected} entries"
                    )

        assert not mismatches, (
            "Documentation numbers out of sync with VERSION_FILES:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )

    def test_sync_versions_skill_lists_all_files(self):
        """Every VERSION_FILES entry (by relative path) must appear in sync-versions/SKILL.md."""
        text = SYNC_VERSIONS_PATH.read_text()
        missing = []
        for path in VERSION_FILES:
            rel = str(path.relative_to(ROOT))
            if rel not in text:
                missing.append(rel)

        assert not missing, (
            "VERSION_FILES entries not mentioned in sync-versions/SKILL.md:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nAdd each missing path to Step 3 of the skill."
        )


class TestPluginSkillsMirror:
    """plugins/mlx-skills/skills/ must be a byte-identical copy of skills/.

    Claude's plugin cache preserves symlinks without dereferencing, so any
    symlink pointing outside the plugin subtree resolves to nothing at
    runtime. The mirror must be real files, synced via
    scripts/sync_plugin_skills.py.
    """

    PLUGIN_SKILLS_DIR = ROOT / "plugins" / "mlx-skills" / "skills"

    def test_not_a_symlink(self):
        """The plugin skills directory must be a real directory, not a link."""
        assert self.PLUGIN_SKILLS_DIR.exists(), (
            f"{self.PLUGIN_SKILLS_DIR.relative_to(ROOT)} is missing. "
            f"Run: uv run python scripts/sync_plugin_skills.py"
        )
        assert not self.PLUGIN_SKILLS_DIR.is_symlink(), (
            f"{self.PLUGIN_SKILLS_DIR.relative_to(ROOT)} is a symlink. "
            f"Claude's plugin cache does not dereference symlinks whose "
            f"targets sit outside the plugin subtree, so the installed "
            f"plugin ends up with no skills. Replace with a real copy: "
            f"rm that path, then run "
            f"uv run python scripts/sync_plugin_skills.py"
        )

    def test_plugin_skills_match_source(self):
        """Byte-level parity with top-level skills/."""
        sync = _load_script_module("sync_plugin_skills")
        added, removed, changed = sync._diff(sync.SOURCE, sync.TARGET)
        assert not (added or removed or changed), (
            "plugins/mlx-skills/skills/ has drifted from skills/. "
            "Run: uv run python scripts/sync_plugin_skills.py\n"
            + sync._format_report(added, removed, changed)
        )


class TestInstallSmoke:
    """Verifies the plugin subtree Claude would actually ship:

    - no symlinks anywhere (any symlink whose target sits outside the
      subtree will dangle on install)
    - every plugin-side SKILL.md has valid frontmatter
    """

    PLUGIN_ROOT = ROOT / "plugins" / "mlx-skills"

    def test_no_symlinks_anywhere_in_plugin_tree(self):
        links = [p for p in self.PLUGIN_ROOT.rglob("*") if p.is_symlink()]
        assert not links, (
            "Symlinks found under plugins/mlx-skills/ (will dangle after "
            "install because Claude's plugin cache does not dereference "
            "symlinks whose targets sit outside the plugin subtree):\n"
            + "\n".join(f"  {p.relative_to(ROOT)} -> {os.readlink(p)}" for p in links)
            + "\nReplace each with real files. If the source lives in "
            "skills/, run: uv run python scripts/sync_plugin_skills.py"
        )

    # NB: no separate frontmatter check for the plugin copy -- TestPluginSkillsMirror
    # guarantees byte parity with skills/, and TestFrontmatterValidity validates
    # skills/, so plugin-side validity is transitive.


class TestManifestSchemas:
    """Plugin manifests must be valid JSON with the fields Claude needs.

    A typo in .claude-plugin/plugin.json or marketplace.json surfaces at
    install time with zero diagnostic information. These tests catch
    malformed manifests pre-commit.
    """

    PLUGIN_MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
    MARKETPLACE_MANIFEST = ROOT / ".claude-plugin" / "marketplace.json"
    PLUGIN_WRAPPER_MANIFEST = (
        ROOT / "plugins" / "mlx-skills" / ".claude-plugin" / "plugin.json"
    )

    PLUGIN_REQUIRED = {"name", "version", "description", "author"}
    MARKETPLACE_REQUIRED = {"name", "owner", "plugins"}
    MARKETPLACE_PLUGIN_REQUIRED = {"name", "source", "description", "version"}

    def _load(self, path: Path) -> dict:
        import orjson

        try:
            return orjson.loads(path.read_bytes())
        except orjson.JSONDecodeError as exc:
            pytest.fail(f"{path.relative_to(ROOT)}: invalid JSON -- {exc}")

    @pytest.mark.parametrize(
        "manifest_attr",
        ["PLUGIN_MANIFEST", "PLUGIN_WRAPPER_MANIFEST"],
    )
    def test_plugin_manifest_required_fields(self, manifest_attr: str):
        path = getattr(self, manifest_attr)
        data = self._load(path)
        missing = self.PLUGIN_REQUIRED - data.keys()
        assert not missing, (
            f"{path.relative_to(ROOT)}: missing required fields {missing}"
        )

    def test_marketplace_manifest_required_fields(self):
        data = self._load(self.MARKETPLACE_MANIFEST)
        missing = self.MARKETPLACE_REQUIRED - data.keys()
        assert not missing, (
            f"{self.MARKETPLACE_MANIFEST.relative_to(ROOT)}: "
            f"missing required fields {missing}"
        )

        plugins = data.get("plugins", [])
        assert plugins, "marketplace.json has empty plugins list"
        for i, plugin in enumerate(plugins):
            plugin_missing = self.MARKETPLACE_PLUGIN_REQUIRED - plugin.keys()
            assert not plugin_missing, (
                f"marketplace.json plugins[{i}]: missing {plugin_missing}"
            )

    def test_marketplace_source_resolves(self):
        """plugins[].source must be a path that actually exists on disk."""
        data = self._load(self.MARKETPLACE_MANIFEST)
        missing = []
        for i, plugin in enumerate(data.get("plugins", [])):
            src = plugin.get("source", "")
            resolved = (ROOT / src).resolve()
            if not resolved.exists():
                missing.append(f"plugins[{i}].source = {src!r} -> {resolved}")
        assert not missing, (
            "marketplace.json sources that don't resolve on disk:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


class TestSkillListDocumented:
    """Numeric claims about skill count must match len(SKILL_NAMES)."""

    def test_skill_count_matches_docs(self):
        """`N skills` claims in CLAUDE.md and README.md must equal len(SKILL_NAMES).

        Only numeric claims are checked; written-out numbers ('four skills')
        should be normalized to digits so drift is detectable.
        """
        expected = len(SKILL_NAMES)
        # Require a word boundary before the number and a word boundary / space
        # before 'skill' so we don't match e.g. '0.5.8' or 'all 4 SKILL.md'.
        claim_pattern = re.compile(r"\b(\d+) skills\b", re.IGNORECASE)

        mismatches = []
        for doc in (CLAUDE_MD_PATH, README_PATH):
            if not doc.exists():
                continue
            text = doc.read_text()
            for match in claim_pattern.finditer(text):
                claim = int(match.group(1))
                if claim != expected:
                    rel = doc.relative_to(ROOT)
                    mismatches.append(
                        f"{rel}: claims '{match.group(0)}' but SKILL_NAMES has {expected} entries"
                    )

        assert not mismatches, (
            "Documentation numbers out of sync with SKILL_NAMES:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )

    def test_all_skills_mentioned_in_claude_md(self):
        """Every SKILL_NAMES entry must be named in CLAUDE.md.

        Catches the case where a new skill was added under skills/ but
        CLAUDE.md's routing tables weren't updated.
        """
        text = CLAUDE_MD_PATH.read_text()
        missing = [name for name in SKILL_NAMES if name not in text]
        assert not missing, (
            f"Skills not mentioned in CLAUDE.md: {missing}.\n"
            "Add to the 'Which Skill Do I Need?' and 'Skills and When They Load' sections."
        )
