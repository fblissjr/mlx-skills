from __future__ import annotations

import argparse
import importlib.resources as resources
import shutil
import sys
from pathlib import Path


TARGETS = {
    "codex": Path("~/.codex/skills"),
    "claude": Path("~/.claude/skills"),
    "opencode": Path("~/.config/opencode/skills"),
}


def _skill_root() -> resources.abc.Traversable:
    return resources.files("mlx_skills").joinpath("skills")


def _install_to(target: Path, skills_root: Path, force: bool) -> None:
    target = target.expanduser()
    target.mkdir(parents=True, exist_ok=True)

    skill_dirs = [path for path in sorted(skills_root.iterdir()) if path.is_dir()]

    if not force:
        conflicts = [target / skill.name for skill in skill_dirs if (target / skill.name).exists()]
        if conflicts:
            conflict_list = "\n".join(str(path) for path in conflicts)
            raise SystemExit(
                "Destination already has skills:\n"
                f"{conflict_list}\n"
                "Re-run with --force to overwrite."
            )

    for skill_dir in skill_dirs:
        dest = target / skill_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill_dir, dest)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install MLX skills into supported assistants."
    )
    parser.add_argument("--codex", action="store_true", help="Install for Codex.")
    parser.add_argument("--claude", action="store_true", help="Install for Claude.")
    parser.add_argument("--opencode", action="store_true", help="Install for OpenCode.")
    parser.add_argument(
        "--dest",
        type=Path,
        help="Install into a custom destination (path to skills directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing skills in the destination.",
    )
    args = parser.parse_args(argv)

    if not (args.codex or args.claude or args.opencode or args.dest):
        parser.error("Pick a destination via --codex, --claude, --opencode, or --dest.")

    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    skills_root = _skill_root()

    targets = []
    if args.codex:
        targets.append(TARGETS["codex"])
    if args.claude:
        targets.append(TARGETS["claude"])
    if args.opencode:
        targets.append(TARGETS["opencode"])
    if args.dest:
        targets.append(args.dest)

    with resources.as_file(skills_root) as skills_path:
        for target in targets:
            _install_to(target, skills_path, args.force)


if __name__ == "__main__":
    main()
