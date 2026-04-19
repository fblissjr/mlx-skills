"""Mirror top-level skills/ into plugins/mlx-skills/skills/.

Claude's plugin cache preserves symlinks without dereferencing, and
installed plugins cannot reference files outside their own subtree. So the
plugin copy must be real files, not a symlink to the canonical source.

The canonical edit target is top-level skills/ (scripts, tests, and
maintainer workflows reference it). Running this script propagates edits
to the plugin subtree.

Usage
-----
    uv run python scripts/sync_plugin_skills.py          # copy + report
    uv run python scripts/sync_plugin_skills.py --check  # exit 1 if drift
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "skills"
TARGET = ROOT / "plugins" / "mlx-skills" / "skills"


def _iter_relative_files(base: Path) -> set[Path]:
    """Return the set of file paths under `base`, relative to `base`."""
    return {p.relative_to(base) for p in base.rglob("*") if p.is_file()}


def _diff(source: Path, target: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (added, removed, changed) paths relative to source/target.

    - added: exists in source, missing in target
    - removed: exists in target, missing in source
    - changed: exists in both but content differs
    """
    if not target.exists():
        return sorted(_iter_relative_files(source)), [], []

    source_files = _iter_relative_files(source)
    target_files = _iter_relative_files(target)

    added = sorted(source_files - target_files)
    removed = sorted(target_files - source_files)
    changed: list[Path] = []

    for rel in sorted(source_files & target_files):
        if not filecmp.cmp(source / rel, target / rel, shallow=False):
            changed.append(rel)

    return added, removed, changed


def _sync(source: Path, target: Path) -> None:
    """Make target an exact copy of source."""
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _format_report(added: list[Path], removed: list[Path], changed: list[Path]) -> str:
    lines = []
    if added:
        lines.append(f"  added ({len(added)}):")
        lines.extend(f"    + {p}" for p in added)
    if removed:
        lines.append(f"  removed ({len(removed)}):")
        lines.extend(f"    - {p}" for p in removed)
    if changed:
        lines.append(f"  changed ({len(changed)}):")
        lines.extend(f"    ~ {p}" for p in changed)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if source and target differ. Does not write.",
    )
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"source does not exist: {SOURCE}", file=sys.stderr)
        return 2

    added, removed, changed = _diff(SOURCE, TARGET)
    in_sync = not (added or removed or changed)

    if args.check:
        if in_sync:
            print(f"plugin skills are in sync with {SOURCE.relative_to(ROOT)}")
            return 0
        print(
            f"plugin skills drifted from {SOURCE.relative_to(ROOT)}:\n"
            f"{_format_report(added, removed, changed)}\n\n"
            f"Run: uv run python scripts/sync_plugin_skills.py",
            file=sys.stderr,
        )
        return 1

    if in_sync:
        print(f"plugin skills already in sync with {SOURCE.relative_to(ROOT)}")
        return 0

    _sync(SOURCE, TARGET)
    print(
        f"synced {SOURCE.relative_to(ROOT)} -> {TARGET.relative_to(ROOT)}\n"
        f"{_format_report(added, removed, changed)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
