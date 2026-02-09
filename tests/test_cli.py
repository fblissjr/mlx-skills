"""Tests for mlx_skills.cli module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_skills.cli import _parse_args, _install_to


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_codex_flag(self):
        args = _parse_args(["--codex"])
        assert args.codex is True
        assert args.claude is False
        assert args.opencode is False
        assert args.dest is None

    def test_claude_flag(self):
        args = _parse_args(["--claude"])
        assert args.claude is True

    def test_opencode_flag(self):
        args = _parse_args(["--opencode"])
        assert args.opencode is True

    def test_dest_flag(self):
        args = _parse_args(["--dest", "/tmp/claude/test-skills"])
        assert args.dest == Path("/tmp/claude/test-skills")

    def test_force_flag(self):
        args = _parse_args(["--codex", "--force"])
        assert args.force is True

    def test_multiple_targets(self):
        args = _parse_args(["--codex", "--claude"])
        assert args.codex is True
        assert args.claude is True

    def test_no_destination_exits(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_force_alone_exits(self):
        with pytest.raises(SystemExit):
            _parse_args(["--force"])

    def test_all_targets(self):
        args = _parse_args(["--codex", "--claude", "--opencode"])
        assert args.codex is True
        assert args.claude is True
        assert args.opencode is True

    def test_dest_with_force(self):
        args = _parse_args(["--dest", "/tmp/claude/out", "--force"])
        assert args.dest == Path("/tmp/claude/out")
        assert args.force is True


# ---------------------------------------------------------------------------
# Installation tests
# ---------------------------------------------------------------------------


class TestInstallTo:
    def test_install_creates_skill_dirs(self, tmp_path: Path):
        """Install skills into a temporary directory and verify structure."""
        # Build a fake skills_root with one skill
        src = tmp_path / "src"
        skill = src / "test-skill"
        refs = skill / "references"
        refs.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: test\n---\n", encoding="utf-8")
        (refs / "guide.md").write_text("# Guide\n", encoding="utf-8")

        dest = tmp_path / "dest"
        _install_to(dest, src, force=False)

        assert (dest / "test-skill" / "SKILL.md").exists()
        assert (dest / "test-skill" / "references" / "guide.md").exists()

    def test_install_without_force_conflicts(self, tmp_path: Path):
        """Raise SystemExit when skills already exist and --force is not set."""
        src = tmp_path / "src"
        skill = src / "my-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("content", encoding="utf-8")

        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "my-skill").mkdir()

        with pytest.raises(SystemExit, match="--force"):
            _install_to(dest, src, force=False)

    def test_install_with_force_overwrites(self, tmp_path: Path):
        """With --force, existing skill directories are replaced."""
        src = tmp_path / "src"
        skill = src / "my-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("new content", encoding="utf-8")

        dest = tmp_path / "dest"
        existing = dest / "my-skill"
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text("old content", encoding="utf-8")

        _install_to(dest, src, force=True)

        assert (dest / "my-skill" / "SKILL.md").read_text() == "new content"
