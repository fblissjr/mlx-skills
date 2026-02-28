#!/usr/bin/env python3
"""
Scan upstream MLX repos for recent changes and produce a structured update report.

By default, fetches directly from GitHub (shallow bare clones scoped to the
time window). No local setup required.

Override with --repos-dir to use existing local clones instead.

Usage:
    uv run python scripts/check_updates.py --since 30days
    uv run python scripts/check_updates.py --since 2024-01-15 --repos mlx mlx-lm
    uv run python scripts/check_updates.py --repos-dir ~/code/upstream --since 30days

Output: Markdown report with watched file changes, potential breaking changes,
notable commits, and suggested skill updates.
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


DEFAULT_REPOS = {
    "mlx": "https://github.com/ml-explore/mlx.git",
    "mlx-lm": "https://github.com/ml-explore/mlx-lm.git",
    "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm.git",
    "mlx-examples": "https://github.com/ml-explore/mlx-examples.git",
}

# Files to watch closely for pattern changes
WATCHED_FILES = {
    "mlx": [
        # NN layers
        "python/mlx/nn/layers/linear.py",
        "python/mlx/nn/layers/normalization.py",
        "python/mlx/nn/layers/transformer.py",
        "python/mlx/nn/layers/activations.py",
        "python/mlx/nn/layers/convolution.py",
        "python/mlx/nn/layers/recurrent.py",
        "python/mlx/nn/layers/positional_encoding.py",
        "python/mlx/nn/layers/quantized.py",
        "python/mlx/nn/layers/embedding.py",
        # NN utilities
        "python/mlx/nn/losses.py",
        "python/mlx/nn/init.py",
        "python/mlx/nn/utils.py",
        # Optimizers
        "python/mlx/optimizers/__init__.py",
        "python/mlx/optimizers/optimizers.py",
        "python/mlx/optimizers/schedulers.py",
        # Core
        "python/mlx/utils.py",
        # Docs
        "docs/src/usage/lazy_evaluation.rst",
        "docs/src/usage/compile.rst",
        "docs/src/usage/unified_memory.rst",
    ],
    "mlx-lm": [
        "mlx_lm/generate.py",
        "mlx_lm/models/cache.py",
        "mlx_lm/models/base.py",
        "mlx_lm/utils.py",
        "mlx_lm/tuner/lora.py",
        "mlx_lm/tuner/trainer.py",
        "mlx_lm/models/llama.py",
        "mlx_lm/sample_utils.py",
    ],
    "mlx-vlm": [
        "mlx_vlm/utils.py",
    ],
    "mlx-examples": [],
}

DEPRECATION_KEYWORDS = [
    "deprecat",
    "removed",
    "breaking",
    "rename",
    "replace",
    "migrate",
    "backward compat",
]


MAX_DIFF_LINES = 200


def get_watched_file_diffs(
    repo_path: Path, since: str, file_path: str, max_lines: int = 0
) -> str:
    """Get unified diff output for a watched file within the time range.

    Returns the patch text. When max_lines > 0, truncates to that many lines.
    When max_lines is 0 (default), returns full output.
    Returns empty string if no diffs are found.
    """
    output = run_git(
        repo_path, "log", f"--since={since}", "-p", "--", file_path
    )
    if not output:
        return ""

    if max_lines <= 0:
        return output

    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output

    truncated = "\n".join(lines[:max_lines])
    truncated += f"\n... truncated ({len(lines) - max_lines} lines omitted)"
    return truncated


def clone_repo(url: str, dest: Path, since: str) -> bool:
    """Shallow bare clone scoped to the time window. Returns True on success."""
    result = subprocess.run(
        [
            "git", "clone", "--bare", "--single-branch",
            "--shallow-since", since, url, str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0


def run_git(repo_path: Path, *args: str) -> Optional[str]:
    """Run a git command in the given repo path."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_recent_commits(repo_path: Path, since: str) -> list[dict]:
    """Get recent commits with metadata."""
    output = run_git(
        repo_path,
        "log",
        f"--since={since}",
        "--format=%H|%s|%an|%ai",
        "--no-merges",
    )
    if not output:
        return []

    commits = []
    for line in output.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0][:12],
                "subject": parts[1],
                "author": parts[2],
                "date": parts[3][:10],
            })
    return commits


def get_changed_files(repo_path: Path, since: str) -> list[str]:
    """Get list of files changed since the given date."""
    output = run_git(
        repo_path,
        "log",
        f"--since={since}",
        "--name-only",
        "--format=",
        "--no-merges",
    )
    if not output:
        return []

    files = set()
    for line in output.splitlines():
        line = line.strip()
        if line:
            files.add(line)
    return sorted(files)


def extract_public_api(file_path: Path) -> list[str]:
    """Extract public function and class names from a Python file using AST."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            names.append(f"def {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    methods.append(item.name)
            methods_str = f" [{', '.join(methods)}]" if methods else ""
            names.append(f"class {node.name}{methods_str}")
        elif isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            names.append(f"async def {node.name}(...)")
    return names


def check_deprecations(commits: list[dict]) -> list[str]:
    """Check commit messages for deprecation-related keywords."""
    flagged = []
    for commit in commits:
        subject_lower = commit["subject"].lower()
        for keyword in DEPRECATION_KEYWORDS:
            if keyword in subject_lower:
                flagged.append(
                    f"  - [{commit['hash']}] {commit['subject']}"
                )
                break
    return flagged


def analyze_watched_files(
    repo_path: Path,
    repo_name: str,
    changed_files: list[str],
    include_diffs: bool = False,
    since: str = "",
    max_lines: int = 0,
) -> list[str]:
    """Check if any watched files were modified."""
    watched = WATCHED_FILES.get(repo_name, [])
    if not watched:
        return []

    hits = []
    for wf in watched:
        if wf in changed_files:
            file_path = repo_path / wf
            if file_path.exists() and file_path.suffix == ".py":
                api = extract_public_api(file_path)
                api_str = f" -- Public API: {', '.join(api[:5])}" if api else ""
                if len(api) > 5:
                    api_str += f" (+{len(api) - 5} more)"
                hits.append(f"  - `{wf}`{api_str}")
            else:
                hits.append(f"  - `{wf}`")

            if include_diffs and since:
                diff = get_watched_file_diffs(repo_path, since, wf, max_lines=max_lines)
                if diff:
                    hits.append("")
                    hits.append("```diff")
                    hits.append(diff)
                    hits.append("```")
                    hits.append("")
    return hits


def generate_report(
    repos_dir: Path,
    since: str,
    repos: Optional[list[str]] = None,
    include_diffs: bool = False,
    max_lines: int = 0,
) -> str:
    """Generate the full update report."""
    lines = [
        "# MLX Skill Update Report",
        "",
        f"Changes since: {since}",
        "",
    ]

    available_repos = sorted(
        d.name for d in repos_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if repos:
        target_repos = [r for r in repos if r in available_repos]
    else:
        target_repos = available_repos

    if not target_repos:
        lines.append(f"No repos found in {repos_dir}")
        return "\n".join(lines)

    has_updates = False

    for repo_name in target_repos:
        repo_path = repos_dir / repo_name
        commits = get_recent_commits(repo_path, since)
        changed_files = get_changed_files(repo_path, since)

        if not commits:
            continue

        has_updates = True
        py_files = [f for f in changed_files if f.endswith(".py")]

        lines.append(f"## {repo_name}")
        lines.append("")
        lines.append(f"**{len(commits)} commits**, **{len(changed_files)} files changed** ({len(py_files)} Python)")
        lines.append("")

        # Watched file changes
        watched_hits = analyze_watched_files(
            repo_path, repo_name, changed_files,
            include_diffs=include_diffs, since=since,
            max_lines=max_lines,
        )
        if watched_hits:
            lines.append("### Watched File Changes")
            lines.append("")
            lines.extend(watched_hits)
            lines.append("")

        # Deprecation / breaking changes
        deprecations = check_deprecations(commits)
        if deprecations:
            lines.append("### Potential Breaking Changes")
            lines.append("")
            lines.extend(deprecations)
            lines.append("")

        # Notable commits (first 15)
        lines.append("### Recent Commits")
        lines.append("")
        for commit in commits[:15]:
            lines.append(
                f"  - `{commit['hash']}` {commit['subject']} "
                f"({commit['author']}, {commit['date']})"
            )
        if len(commits) > 15:
            lines.append(f"  - ... and {len(commits) - 15} more")
        lines.append("")

        # Changed Python files of interest
        if py_files:
            model_files = [f for f in py_files if "/models/" in f]
            tuner_files = [f for f in py_files if "/tuner/" in f]
            other_files = [f for f in py_files if f not in model_files and f not in tuner_files]

            if model_files:
                lines.append("### Model Files Changed")
                lines.append("")
                for f in model_files[:20]:
                    lines.append(f"  - `{f}`")
                if len(model_files) > 20:
                    lines.append(f"  - ... and {len(model_files) - 20} more")
                lines.append("")

            if tuner_files:
                lines.append("### Tuner Files Changed")
                lines.append("")
                for f in tuner_files:
                    lines.append(f"  - `{f}`")
                lines.append("")

            if other_files:
                lines.append("### Other Python Files Changed")
                lines.append("")
                for f in other_files[:15]:
                    lines.append(f"  - `{f}`")
                if len(other_files) > 15:
                    lines.append(f"  - ... and {len(other_files) - 15} more")
                lines.append("")

    if not has_updates:
        lines.append("No changes found in the specified time range.")

    # Suggested actions
    lines.append("---")
    lines.append("")
    lines.append("## Suggested Actions")
    lines.append("")
    lines.append("Review the watched file changes above and update the following")
    lines.append("skill reference files as needed:")
    lines.append("")
    lines.append("- `mlx/references/nn-and-training.md` -- if nn layers, losses, optimizers, or schedulers changed")
    lines.append("- `mlx/references/fundamentals.md` -- if core MLX APIs changed")
    lines.append("- `mlx/references/anti-patterns.md` -- if new footguns were discovered")
    lines.append("- `mlx-lm/references/patterns.md` -- if model patterns changed (cache, attention, generation)")
    lines.append("- `mlx-lm/references/architecture.md` -- if loading, generation flow, or model registration changed")
    lines.append("- `fast-mlx/references/*.md` -- if optimization techniques changed")
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Scan upstream MLX repos for recent changes affecting skills."
    )
    parser.add_argument(
        "--since",
        default="30days",
        help="Time range for git log (e.g., '30days', '2024-01-15'). Default: 30days",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        choices=list(DEFAULT_REPOS.keys()),
        help="Specific repos to scan (default: all)",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=os.environ.get("MLX_SKILLS_REPOS"),
        help=(
            "Use local git clones instead of fetching from GitHub. "
            "Can also be set via MLX_SKILLS_REPOS env var."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="Include unified diffs for watched files in the report",
    )
    parser.add_argument(
        "--diff-lines",
        type=int,
        default=0,
        help="Truncate diffs to N lines per file (0 = unlimited, default: 0)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.repos_dir:
        # Local mode: use existing clones
        repos_dir = Path(args.repos_dir).expanduser().resolve()
        if not repos_dir.is_dir():
            print(f"Error: directory not found at {repos_dir}", file=sys.stderr)
            sys.exit(1)
        report = generate_report(repos_dir, args.since, args.repos, include_diffs=args.diff, max_lines=args.diff_lines)
    else:
        # Remote mode: shallow-clone from GitHub to a temp directory
        targets = DEFAULT_REPOS
        if args.repos:
            targets = {k: v for k, v in targets.items() if k in args.repos}

        tmp_dir = Path(tempfile.mkdtemp(prefix="mlx-skills-"))
        try:
            for name, url in targets.items():
                print(f"Fetching {name}...", file=sys.stderr, end=" ", flush=True)
                dest = tmp_dir / name
                if clone_repo(url, dest, args.since):
                    print("done", file=sys.stderr)
                else:
                    print("skipped (no recent commits or clone failed)", file=sys.stderr)
            print("", file=sys.stderr)
            report = generate_report(tmp_dir, args.since, args.repos, include_diffs=args.diff, max_lines=args.diff_lines)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
