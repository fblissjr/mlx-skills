"""Structural validation tests for skill files.

Replaces the removed validate.py with proper pytest coverage:
- Version consistency across all locations
- SKILL.md frontmatter validity
- Reference file existence and headers
- Cross-reference validity between skills
- SKILL.md word count guard
- Stale skill path detection
- Reference file staleness warnings (strict mode via PYTEST_STRICT=1)
- Routing coverage: every WATCHED_FILES entry has a rule in update-skills.md
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


def _load_check_updates():
    """Import scripts/check_updates.py as a module.

    scripts/ has no __init__.py, so load via importlib. Cached via sys.modules.
    """
    if "check_updates" in sys.modules:
        return sys.modules["check_updates"]
    path = ROOT / "scripts" / "check_updates.py"
    spec = importlib.util.spec_from_file_location("check_updates", path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_updates"] = module
    spec.loader.exec_module(module)
    return module


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

UPDATE_SKILLS_PATH = ROOT / ".claude" / "skills" / "update-skills.md"
SYNC_VERSIONS_PATH = ROOT / ".claude" / "skills" / "sync-versions.md"
CLAUDE_MD_PATH = ROOT / "CLAUDE.md"
README_PATH = ROOT / "README.md"

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
    """Every watched upstream file needs a routing rule in update-skills.md."""

    def test_every_watched_file_has_routing_rule(self):
        """WATCHED_FILES -> update-skills.md routing coverage.

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
            "WATCHED_FILES entries with no routing rule in update-skills.md:\n"
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
        """Every VERSION_FILES entry (by relative path) must appear in sync-versions.md."""
        text = SYNC_VERSIONS_PATH.read_text()
        missing = []
        for path in VERSION_FILES:
            rel = str(path.relative_to(ROOT))
            if rel not in text:
                missing.append(rel)

        assert not missing, (
            "VERSION_FILES entries not mentioned in sync-versions.md:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nAdd each missing path to Step 3 of the skill."
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
