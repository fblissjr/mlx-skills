"""Tests for check_updates.py --diff functionality."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a real git repo with commits for testing diff output."""
    repo = tmp_path / "test-repo"
    repo.mkdir()

    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    _git("init")
    _git("config", "user.email", "test@test.com")
    _git("config", "user.name", "Test")

    # Initial commit with a watched file
    watched = repo / "python" / "mlx" / "nn" / "layers"
    watched.mkdir(parents=True)
    linear = watched / "linear.py"
    linear.write_text("class Linear:\n    pass\n")
    _git("add", ".")
    _git("commit", "-m", "initial commit")

    # Second commit that modifies the watched file
    linear.write_text("class Linear:\n    def __init__(self, in_features, out_features):\n        pass\n")
    _git("add", ".")
    _git("commit", "-m", "add Linear.__init__")

    # Third commit with another change
    linear.write_text(
        "class Linear:\n"
        "    def __init__(self, in_features, out_features):\n"
        "        self.weight = None\n"
        "    def __call__(self, x):\n"
        "        return x\n"
    )
    _git("add", ".")
    _git("commit", "-m", "add weight and __call__")

    return repo


@pytest.fixture()
def repos_dir(git_repo: Path) -> Path:
    """Wrap the git repo in a repos dir structure expected by check_updates."""
    repos = git_repo.parent / "repos"
    repos.mkdir()
    # Symlink the repo as "mlx" so WATCHED_FILES["mlx"] matches
    (repos / "mlx").symlink_to(git_repo)
    return repos


class TestGetWatchedFileDiffs:
    """Tests for get_watched_file_diffs()."""

    def test_returns_diff_content(self, git_repo: Path) -> None:
        from scripts.check_updates import get_watched_file_diffs

        diff = get_watched_file_diffs(
            git_repo, "1year", "python/mlx/nn/layers/linear.py"
        )
        assert diff is not None
        assert "Linear" in diff
        assert "def __init__" in diff

    def test_returns_empty_for_unwatched_file(self, git_repo: Path) -> None:
        from scripts.check_updates import get_watched_file_diffs

        diff = get_watched_file_diffs(
            git_repo, "1year", "nonexistent/file.py"
        )
        assert diff == ""

    def test_default_returns_full_output(self, git_repo: Path) -> None:
        """Default max_lines=0 means no truncation."""
        from scripts.check_updates import get_watched_file_diffs

        diff = get_watched_file_diffs(
            git_repo, "1year", "python/mlx/nn/layers/linear.py"
        )
        assert diff != ""
        assert "truncated" not in diff.lower()

    def test_truncates_when_max_lines_set(self, git_repo: Path) -> None:
        from scripts.check_updates import get_watched_file_diffs

        diff = get_watched_file_diffs(
            git_repo, "1year", "python/mlx/nn/layers/linear.py", max_lines=5
        )
        lines = diff.strip().splitlines()
        # Should be at most 5 lines + truncation notice
        assert len(lines) <= 6
        assert "truncated" in lines[-1].lower()


class TestAnalyzeWatchedFilesWithDiffs:
    """Tests for analyze_watched_files() with include_diffs=True."""

    def test_without_diffs(self, repos_dir: Path) -> None:
        from scripts.check_updates import analyze_watched_files, get_changed_files

        changed = get_changed_files(repos_dir / "mlx", "1year")
        hits = analyze_watched_files(repos_dir / "mlx", "mlx", changed)
        # Should have hits but no diff blocks
        assert any("`python/mlx/nn/layers/linear.py`" in h for h in hits)
        assert not any("```diff" in h for h in hits)

    def test_with_diffs(self, repos_dir: Path) -> None:
        from scripts.check_updates import analyze_watched_files, get_changed_files

        changed = get_changed_files(repos_dir / "mlx", "1year")
        hits = analyze_watched_files(
            repos_dir / "mlx", "mlx", changed, include_diffs=True, since="1year"
        )
        # Should have the file hit AND a diff block
        combined = "\n".join(hits)
        assert "`python/mlx/nn/layers/linear.py`" in combined
        assert "```diff" in combined
        assert "```\n" in combined or combined.endswith("```")


class TestGenerateReportWithDiffs:
    """Tests for generate_report() with include_diffs=True."""

    def test_report_includes_diffs_when_flag_set(self, repos_dir: Path) -> None:
        from scripts.check_updates import generate_report

        report = generate_report(repos_dir, "1year", repos=["mlx"], include_diffs=True)
        assert "```diff" in report

    def test_report_excludes_diffs_by_default(self, repos_dir: Path) -> None:
        from scripts.check_updates import generate_report

        report = generate_report(repos_dir, "1year", repos=["mlx"])
        assert "```diff" not in report


class TestCliDiffArg:
    """Test that --diff CLI arg is parsed correctly."""

    def test_diff_flag_exists(self) -> None:
        from scripts.check_updates import build_parser

        parser = build_parser()
        args = parser.parse_args(["--diff"])
        assert args.diff is True

    def test_diff_flag_default_false(self) -> None:
        from scripts.check_updates import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.diff is False

    def test_diff_lines_arg(self) -> None:
        from scripts.check_updates import build_parser

        parser = build_parser()
        args = parser.parse_args(["--diff-lines", "50"])
        assert args.diff_lines == 50

    def test_diff_lines_default_zero(self) -> None:
        from scripts.check_updates import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.diff_lines == 0


class TestAnalyzeWatchedFilesMaxLines:
    """Tests for max_lines passthrough in analyze_watched_files."""

    def test_max_lines_passed_to_diffs(self, repos_dir: Path) -> None:
        from scripts.check_updates import analyze_watched_files, get_changed_files

        changed = get_changed_files(repos_dir / "mlx", "1year")
        hits = analyze_watched_files(
            repos_dir / "mlx", "mlx", changed,
            include_diffs=True, since="1year", max_lines=5,
        )
        combined = "\n".join(hits)
        assert "```diff" in combined
        # With max_lines=5 on a small diff, it should truncate
        assert "truncated" in combined.lower()
