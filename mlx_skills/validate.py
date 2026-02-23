"""Validate skill plugin structure, frontmatter, and cross-references."""

from __future__ import annotations

import re
import sys
from pathlib import Path


SKILLS_DIR = Path(__file__).parent / "skills"
MAX_BODY_WORDS = 5000
MAX_DESCRIPTION_LENGTH = 1024

# Patterns for cross-skill references:
#   load the `X` skill
#   `X` skill
_CROSS_REF_PATTERNS = [
    re.compile(r"load the `([^`]+)` skill"),
    re.compile(r"`([^`]+)` skill"),
]

_LAST_UPDATED_PATTERN = re.compile(r"^last updated:\s*\d{4}-\d{2}-\d{2}", re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter delimited by ``---``.

    Returns a dict of parsed key-value pairs and the body text after
    frontmatter.  Uses simple regex extraction for ``name``,
    ``description``, and ``metadata`` fields so we don't require a YAML
    library.
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

    # metadata block (just check presence, not contents)
    if re.search(r"^metadata:\s*$", raw_yaml, re.MULTILINE):
        fields["metadata"] = "present"

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


def _has_last_updated(text: str) -> bool:
    """Check whether *text* contains a ``last updated: YYYY-MM-DD`` line."""
    return bool(_LAST_UPDATED_PATTERN.search(text))


def validate(skills_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Validate all skills under *skills_dir*.

    Returns a tuple of ``(errors, warnings)``.  An empty errors list
    means all required checks passed.  Warnings are advisory.
    """
    if skills_dir is None:
        skills_dir = SKILLS_DIR

    errors: list[str] = []
    warnings: list[str] = []
    validated_skills: list[str] = []
    total_refs_checked = 0
    total_files_checked = 0

    skill_dirs = sorted(
        p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    if not skill_dirs:
        errors.append(f"No skill directories found under {skills_dir}")
        return errors, warnings

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

        # -- Metadata check (warning) --
        if "metadata" not in fields:
            warnings.append(
                f"[{skill_name}] SKILL.md frontmatter missing 'metadata' field"
            )

        # -- Description length --
        desc = fields.get("description", "")
        if len(desc) > MAX_DESCRIPTION_LENGTH:
            errors.append(
                f"[{skill_name}] SKILL.md description is {len(desc)} chars "
                f"(limit {MAX_DESCRIPTION_LENGTH})"
            )

        # -- Word count --
        wc = _word_count(body)
        if wc > MAX_BODY_WORDS:
            errors.append(
                f"[{skill_name}] SKILL.md body is {wc} words (limit {MAX_BODY_WORDS})"
            )

        # -- Verify reference files exist and have dates --
        refs_dir = skill_dir / "references"
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.iterdir()):
                if ref_file.is_file() and ref_file.suffix == ".md":
                    total_files_checked += 1
                    ref_text = ref_file.read_text(encoding="utf-8")
                    if not _has_last_updated(ref_text):
                        rel = ref_file.relative_to(skills_dir)
                        warnings.append(
                            f"[{skill_name}] {rel} missing 'last updated' date"
                        )

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
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for w in warnings:
                print(f"    - {w}")
        print("All checks passed.")

    return errors, warnings


def main() -> None:
    """CLI entrypoint for skill validation."""
    errors, warnings = validate()
    if errors:
        print("Validation failed:\n")
        for err in errors:
            print(f"  - {err}")
        if warnings:
            print("\nWarnings:\n")
            for w in warnings:
                print(f"  - {w}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
