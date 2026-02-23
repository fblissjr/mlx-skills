"""Tests for mlx_skills.validate module."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from mlx_skills.validate import (
    _find_cross_refs,
    _has_last_updated,
    _parse_frontmatter,
    _word_count,
    validate,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_single_line(self):
        text = "---\nname: test\ndescription: A test.\n---\n\nBody here."
        fields, body = _parse_frontmatter(text)
        assert fields["name"] == "test"
        assert fields["description"] == "A test."
        assert "Body here." in body

    def test_valid_multiline_description(self):
        text = dedent("""\
            ---
            name: mlx
            description: >
              Use when writing MLX code.
              Covers core concepts.
            ---

            # Body
        """)
        fields, body = _parse_frontmatter(text)
        assert fields["name"] == "mlx"
        assert "Use when writing MLX code." in fields["description"]
        assert "Covers core concepts." in fields["description"]

    def test_missing_name(self):
        text = "---\ndescription: Has desc but no name.\n---\n\nBody."
        fields, _ = _parse_frontmatter(text)
        assert "name" not in fields
        assert "description" in fields

    def test_missing_description(self):
        text = "---\nname: test\n---\n\nBody."
        fields, _ = _parse_frontmatter(text)
        assert "name" in fields
        assert "description" not in fields

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nSome content."
        fields, body = _parse_frontmatter(text)
        assert fields == {}
        assert body == text

    def test_metadata_present(self):
        text = dedent("""\
            ---
            name: test
            description: A test.
            metadata:
              author: Test
              version: 1.0.0
            ---

            Body.
        """)
        fields, _ = _parse_frontmatter(text)
        assert "metadata" in fields

    def test_metadata_absent(self):
        text = "---\nname: test\ndescription: A test.\n---\n\nBody."
        fields, _ = _parse_frontmatter(text)
        assert "metadata" not in fields


class TestWordCount:
    def test_simple(self):
        assert _word_count("one two three") == 3

    def test_empty(self):
        # "".split() returns [], so len is 0
        assert _word_count("") == 0

    def test_multiline(self):
        assert _word_count("one\ntwo\nthree four") == 4


class TestFindCrossRefs:
    def test_load_the_pattern(self):
        text = "Load the `mlx` skill first."
        refs = _find_cross_refs(text)
        assert "mlx" in refs

    def test_backtick_skill_pattern(self):
        text = "The `fast-mlx` skill covers optimization."
        refs = _find_cross_refs(text)
        assert "fast-mlx" in refs

    def test_multiple_refs(self):
        text = (
            "Load the `mlx` skill. "
            "The `mlx-lm` skill has patterns. "
            "Load the `fast-mlx` skill too."
        )
        refs = _find_cross_refs(text)
        assert set(refs) == {"mlx", "mlx-lm", "fast-mlx"}

    def test_no_refs(self):
        text = "No cross-references here."
        refs = _find_cross_refs(text)
        assert refs == []

    def test_deduplication(self):
        text = (
            "Load the `mlx` skill. Also the `mlx` skill again."
        )
        refs = _find_cross_refs(text)
        assert refs.count("mlx") == 1


class TestHasLastUpdated:
    def test_valid_date(self):
        assert _has_last_updated("last updated: 2026-02-23\n\n# Title")

    def test_no_date(self):
        assert not _has_last_updated("# Title\n\nSome content.")

    def test_malformed_date(self):
        assert not _has_last_updated("last updated: yesterday\n\n# Title")

    def test_date_not_first_line(self):
        assert _has_last_updated("# Title\n\nlast updated: 2026-01-01")


# ---------------------------------------------------------------------------
# Integration tests for validate()
# ---------------------------------------------------------------------------


class TestValidateValid:
    def test_valid_skill_passes(self, tmp_skills: Path):
        errors, warnings = validate(tmp_skills)
        assert errors == []

    def test_multi_skill_cross_refs_pass(self, multi_skill_dir: Path):
        errors, warnings = validate(multi_skill_dir)
        assert errors == []


class TestValidateMissingSkillMd:
    def test_missing_skill_md(self, missing_skill_md: Path):
        errors, warnings = validate(missing_skill_md)
        assert len(errors) == 1
        assert "Missing SKILL.md" in errors[0]


class TestValidateBadFrontmatter:
    def test_missing_description(self, bad_frontmatter: Path):
        errors, warnings = validate(bad_frontmatter)
        assert any("missing 'description' field" in e for e in errors)

    def test_missing_both_fields(self, tmp_path: Path):
        skill_dir = tmp_path / "empty-fm"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n---\n\nBody.\n", encoding="utf-8"
        )
        errors, warnings = validate(tmp_path)
        assert any("missing 'name' field" in e for e in errors)
        assert any("missing 'description' field" in e for e in errors)


class TestValidateWordCount:
    def test_over_limit(self, over_word_count: Path):
        errors, warnings = validate(over_word_count)
        assert len(errors) == 1
        assert "5500 words" in errors[0] or "words" in errors[0]
        assert "limit" in errors[0]


class TestValidateCrossRefs:
    def test_broken_cross_ref(self, broken_cross_ref: Path):
        errors, warnings = validate(broken_cross_ref)
        assert len(errors) == 1
        assert "nonexistent" in errors[0]
        assert "unknown skill" in errors[0]


class TestValidateEmpty:
    def test_no_skill_dirs(self, tmp_path: Path):
        errors, warnings = validate(tmp_path)
        assert len(errors) == 1
        assert "No skill directories found" in errors[0]


# ---------------------------------------------------------------------------
# Tests for new validation checks
# ---------------------------------------------------------------------------


class TestValidateMetadataWarning:
    def test_missing_metadata_warns(self, tmp_path: Path):
        skill_dir = tmp_path / "no-meta"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-meta\ndescription: No metadata.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        errors, warnings = validate(tmp_path)
        assert errors == []
        assert any("missing 'metadata' field" in w for w in warnings)

    def test_has_metadata_no_warning(self, tmp_path: Path):
        skill_dir = tmp_path / "with-meta"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            dedent("""\
                ---
                name: with-meta
                description: Has metadata.
                metadata:
                  author: Test
                  version: 1.0.0
                ---

                Body.
            """),
            encoding="utf-8",
        )
        errors, warnings = validate(tmp_path)
        assert errors == []
        assert not any("missing 'metadata'" in w for w in warnings)


class TestValidateDescriptionLength:
    def test_over_limit(self, tmp_path: Path):
        skill_dir = tmp_path / "long-desc"
        skill_dir.mkdir()
        long_desc = "x " * 600  # 1200 chars
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: long-desc\ndescription: {long_desc}\n---\n\nBody.\n",
            encoding="utf-8",
        )
        errors, warnings = validate(tmp_path)
        assert any("description is" in e and "limit 1024" in e for e in errors)

    def test_under_limit(self, tmp_path: Path):
        skill_dir = tmp_path / "short-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: short-desc\ndescription: Short.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        errors, warnings = validate(tmp_path)
        assert not any("description is" in e for e in errors)


class TestValidateRefDateWarning:
    def test_missing_date_warns(self, tmp_path: Path):
        skill_dir = tmp_path / "no-date"
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-date\ndescription: Test.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        (refs_dir / "guide.md").write_text(
            "# Guide\n\nNo date here.\n",
            encoding="utf-8",
        )
        errors, warnings = validate(tmp_path)
        assert errors == []
        assert any("missing 'last updated' date" in w for w in warnings)

    def test_has_date_no_warning(self, tmp_path: Path):
        skill_dir = tmp_path / "has-date"
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: has-date\ndescription: Test.\n---\n\nBody.\n",
            encoding="utf-8",
        )
        (refs_dir / "guide.md").write_text(
            "last updated: 2026-02-23\n\n# Guide\n\nContent.\n",
            encoding="utf-8",
        )
        errors, warnings = validate(tmp_path)
        assert errors == []
        assert not any("missing 'last updated'" in w for w in warnings)
