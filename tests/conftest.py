"""Shared fixtures for skill validation tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture()
def tmp_skills(tmp_path: Path) -> Path:
    """Create a temporary skills directory with a single valid skill."""
    skill_dir = tmp_path / "valid-skill"
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True)

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
            ---
            name: valid-skill
            description: A valid test skill for unit testing.
            ---

            # Valid Skill

            This is a valid skill body with a reasonable word count.
        """),
        encoding="utf-8",
    )

    (refs_dir / "guide.md").write_text(
        "# Guide\n\nSome reference content.\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def multi_skill_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with multiple skills that cross-reference."""
    # -- alpha skill --
    alpha = tmp_path / "alpha"
    alpha_refs = alpha / "references"
    alpha_refs.mkdir(parents=True)

    (alpha / "SKILL.md").write_text(
        dedent("""\
            ---
            name: alpha
            description: The alpha skill.
            ---

            # Alpha

            For beta patterns, load the `beta` skill.
        """),
        encoding="utf-8",
    )
    (alpha_refs / "basics.md").write_text("# Basics\n", encoding="utf-8")

    # -- beta skill --
    beta = tmp_path / "beta"
    beta_refs = beta / "references"
    beta_refs.mkdir(parents=True)

    (beta / "SKILL.md").write_text(
        dedent("""\
            ---
            name: beta
            description: The beta skill.
            ---

            # Beta

            The `alpha` skill has fundamentals.
        """),
        encoding="utf-8",
    )
    (beta_refs / "patterns.md").write_text("# Patterns\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def missing_skill_md(tmp_path: Path) -> Path:
    """A skill directory that is missing its SKILL.md file."""
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir()
    # No SKILL.md created
    return tmp_path


@pytest.fixture()
def bad_frontmatter(tmp_path: Path) -> Path:
    """A skill with SKILL.md that has missing frontmatter fields."""
    skill_dir = tmp_path / "bad-fm"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
            ---
            name: bad-fm
            ---

            # Bad Frontmatter

            Missing description field.
        """),
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def over_word_count(tmp_path: Path) -> Path:
    """A skill whose SKILL.md body exceeds the word limit."""
    skill_dir = tmp_path / "wordy"
    skill_dir.mkdir()

    body_words = " ".join(["word"] * 5500)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: wordy\ndescription: A wordy skill.\n---\n\n{body_words}\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def broken_cross_ref(tmp_path: Path) -> Path:
    """A skill that references a non-existent skill."""
    skill_dir = tmp_path / "lonely"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(
        dedent("""\
            ---
            name: lonely
            description: A skill with a broken cross-reference.
            ---

            # Lonely

            Load the `nonexistent` skill for more details.
        """),
        encoding="utf-8",
    )

    return tmp_path
