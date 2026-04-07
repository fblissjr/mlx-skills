"""Structural validation tests for skill files.

Replaces the removed validate.py with proper pytest coverage:
- Version consistency across all locations
- SKILL.md frontmatter validity
- Reference file existence and headers
- Cross-reference validity between skills
- SKILL.md word count guard
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
SKILL_NAMES = sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir() and (p / "SKILL.md").exists())

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
