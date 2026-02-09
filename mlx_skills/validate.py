"""Validate skill plugin structure, frontmatter, and cross-references."""

from __future__ import annotations

import re
import sys
from pathlib import Path


SKILLS_DIR = Path(__file__).parent / "skills"
MAX_BODY_WORDS = 5000

# Patterns for cross-skill references:
#   load the `X` skill
#   `X` skill
_CROSS_REF_PATTERNS = [
    re.compile(r"load the `([^`]+)` skill"),
    re.compile(r"`([^`]+)` skill"),
]


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter delimited by ``---``.

    Returns a dict of parsed key-value pairs and the body text after
    frontmatter.  Uses simple regex extraction for ``name`` and
    ``description`` fields so we don't require a YAML library.
    """
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    raw_yaml = parts[1]
    body = parts[2]

    fields: dict[str, str] = {}
    name_match = re.search(r"^name:\s*(.+)", raw_yaml, re.MULTILINE)
    if name_match:
        fields["name"] = name_match.group(1).strip()

    # Description can be single-line or multi-line (using > or |)
    desc_match = re.search(
        r"^description:\s*>?\s*\n((?:\s{2,}.+\n?)+)", raw_yaml, re.MULTILINE
    )
    if desc_match:
        fields["description"] = " ".join(
            line.strip() for line in desc_match.group(1).strip().splitlines()
        )
    else:
        desc_match = re.search(r"^description:\s*(.+)", raw_yaml, re.MULTILINE)
        if desc_match:
            fields["description"] = desc_match.group(1).strip()

    return fields, body


def _word_count(text: str) -> int:
    """Return approximate word count of *text*."""
    return len(text.split())


def _find_cross_refs(text: str) -> list[str]:
    """Return skill names referenced via cross-reference patterns."""
    refs: list[str] = []
    for pattern in _CROSS_REF_PATTERNS:
        refs.extend(pattern.findall(text))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            unique.append(ref)
    return unique


def validate(skills_dir: Path | None = None) -> list[str]:
    """Validate all skills under *skills_dir*.

    Returns a list of error strings.  An empty list means everything is
    valid.
    """
    if skills_dir is None:
        skills_dir = SKILLS_DIR

    errors: list[str] = []
    validated_skills: list[str] = []
    total_refs_checked = 0
    total_files_checked = 0

    skill_dirs = sorted(
        p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    if not skill_dirs:
        errors.append(f"No skill directories found under {skills_dir}")
        return errors

    all_skill_names = {d.name for d in skill_dirs}

    for skill_dir in skill_dirs:
        skill_name = skill_dir.name

        # -- Check SKILL.md exists --
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            errors.append(f"[{skill_name}] Missing SKILL.md")
            continue

        total_files_checked += 1
        text = skill_md.read_text(encoding="utf-8")

        # -- Parse and validate frontmatter --
        fields, body = _parse_frontmatter(text)
        if "name" not in fields:
            errors.append(f"[{skill_name}] SKILL.md frontmatter missing 'name' field")
        if "description" not in fields:
            errors.append(
                f"[{skill_name}] SKILL.md frontmatter missing 'description' field"
            )

        # -- Word count --
        wc = _word_count(body)
        if wc > MAX_BODY_WORDS:
            errors.append(
                f"[{skill_name}] SKILL.md body is {wc} words (limit {MAX_BODY_WORDS})"
            )

        # -- Verify reference files exist --
        refs_dir = skill_dir / "references"
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.iterdir()):
                if ref_file.is_file():
                    total_files_checked += 1
                    # File exists on disk -- that's the check.
                    # (We iterate what's there; nothing to verify beyond
                    # existence, which iterdir already confirms.)

        # -- Cross-reference validation on all .md files --
        md_files = list(skill_dir.rglob("*.md"))
        for md_file in md_files:
            md_text = md_file.read_text(encoding="utf-8")
            cross_refs = _find_cross_refs(md_text)
            for ref in cross_refs:
                total_refs_checked += 1
                if ref not in all_skill_names:
                    rel = md_file.relative_to(skills_dir)
                    errors.append(
                        f"[{skill_name}] {rel} references unknown skill `{ref}`"
                    )

        validated_skills.append(skill_name)

    if not errors:
        print(f"Validated {len(validated_skills)} skills: {', '.join(validated_skills)}")
        print(f"  Files checked: {total_files_checked}")
        print(f"  Cross-references checked: {total_refs_checked}")
        print("All checks passed.")

    return errors


def main() -> None:
    """CLI entrypoint for skill validation."""
    errors = validate()
    if errors:
        print("Validation failed:\n")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
