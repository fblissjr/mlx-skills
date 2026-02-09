"""Structural integrity tests against the actual skills in the repository.

These tests run against the real skill files on disk to catch regressions
in frontmatter, reference paths, word counts, and cross-references.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_skills.validate import (
    _find_cross_refs,
    _parse_frontmatter,
    _word_count,
    validate,
    MAX_BODY_WORDS,
    SKILLS_DIR,
)


# The actual skills directory in this repo
REPO_SKILLS = SKILLS_DIR


def _skill_dirs() -> list[Path]:
    """Return all skill directories in the repo."""
    return sorted(
        p for p in REPO_SKILLS.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


# ---------------------------------------------------------------------------
# Parametrize over each actual skill
# ---------------------------------------------------------------------------

_SKILL_NAMES = [d.name for d in _skill_dirs()]


@pytest.fixture(params=_SKILL_NAMES)
def skill_dir(request: pytest.FixtureRequest) -> Path:
    return REPO_SKILLS / request.param


class TestSkillMdExists:
    def test_has_skill_md(self, skill_dir: Path):
        assert (skill_dir / "SKILL.md").exists(), (
            f"{skill_dir.name} is missing SKILL.md"
        )


class TestFrontmatter:
    def test_has_name(self, skill_dir: Path):
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        fields, _ = _parse_frontmatter(text)
        assert "name" in fields, f"{skill_dir.name}/SKILL.md missing 'name'"

    def test_has_description(self, skill_dir: Path):
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        fields, _ = _parse_frontmatter(text)
        assert "description" in fields, (
            f"{skill_dir.name}/SKILL.md missing 'description'"
        )

    def test_name_matches_directory(self, skill_dir: Path):
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        fields, _ = _parse_frontmatter(text)
        assert fields.get("name") == skill_dir.name, (
            f"Frontmatter name '{fields.get('name')}' does not match "
            f"directory name '{skill_dir.name}'"
        )


class TestWordCounts:
    def test_body_under_limit(self, skill_dir: Path):
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        _, body = _parse_frontmatter(text)
        wc = _word_count(body)
        assert wc <= MAX_BODY_WORDS, (
            f"{skill_dir.name}/SKILL.md body is {wc} words (limit {MAX_BODY_WORDS})"
        )


class TestReferenceFilesExist:
    def test_all_reference_files_on_disk(self, skill_dir: Path):
        refs_dir = skill_dir / "references"
        if not refs_dir.is_dir():
            pytest.skip(f"{skill_dir.name} has no references/ directory")
        for ref_file in sorted(refs_dir.iterdir()):
            assert ref_file.is_file(), (
                f"{skill_dir.name}/references/{ref_file.name} is not a file"
            )


class TestCrossReferences:
    def test_all_cross_refs_resolve(self, skill_dir: Path):
        all_skill_names = {d.name for d in _skill_dirs()}
        md_files = list(skill_dir.rglob("*.md"))
        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            for ref in _find_cross_refs(text):
                assert ref in all_skill_names, (
                    f"{md_file.relative_to(REPO_SKILLS)} references "
                    f"unknown skill `{ref}`"
                )


class TestFullValidation:
    """Run the full validate() function against the real skills."""

    def test_all_skills_pass(self):
        errors = validate(REPO_SKILLS)
        assert errors == [], (
            "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
