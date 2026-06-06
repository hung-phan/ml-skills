#!/usr/bin/env python3
"""
Print a flat manifest of every wiki topic by reading each SKILL.md's YAML
frontmatter. On-demand only — never writes to disk. Run when you need to
discover topics without loading full pages.

Usage (run from the skill root, `skills/system-review/`):
    python3 scripts/extract-manifest.py                    # all topics
    python3 scripts/extract-manifest.py caching            # one keyword
    python3 scripts/extract-manifest.py caching cdn        # AND — both must match
    python3 scripts/extract-manifest.py --any caching cdn  # OR  — either matches

Each keyword is a case-insensitive substring matched against `name` +
`description`.
"""

from __future__ import annotations

import sys
from pathlib import Path

WIKI_ROOT = Path(__file__).resolve().parent.parent / "references"


def parse_frontmatter(path: Path) -> dict[str, str] | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    fields: dict[str, str] = {}
    current: str | None = None
    for line in text[4:end].splitlines():
        if line and not line.startswith(" ") and ":" in line:
            key, _, value = line.partition(":")
            current = key.strip()
            fields[current] = value.strip()
        elif current and line.startswith(" "):
            fields[current] = (fields[current] + " " + line.strip()).strip()
    return fields


def parse_args(argv: list[str]) -> tuple[bool, list[str]]:
    """Returns (any_mode, keywords). any_mode=False means AND."""
    any_mode = False
    keywords: list[str] = []
    for arg in argv:
        if arg == "--any":
            any_mode = True
        elif arg == "--all":
            any_mode = False
        else:
            keywords.append(arg.lower())
    return any_mode, keywords


def matches(haystack: str, keywords: list[str], any_mode: bool) -> bool:
    if not keywords:
        return True
    test = any if any_mode else all
    return test(kw in haystack for kw in keywords)


def main() -> int:
    if not WIKI_ROOT.exists():
        print(f"ERROR: wiki root not found: {WIKI_ROOT}", file=sys.stderr)
        return 1

    any_mode, keywords = parse_args(sys.argv[1:])

    by_category: dict[str, list[tuple[str, str, Path]]] = {}
    for skill in sorted(WIKI_ROOT.glob("*/*/SKILL.md")):
        fm = parse_frontmatter(skill)
        if not fm or "name" not in fm or "description" not in fm:
            print(f"WARN: missing/invalid frontmatter: {skill}", file=sys.stderr)
            continue
        name, desc = fm["name"], fm["description"]
        haystack = f"{name}\n{desc}".lower()
        if not matches(haystack, keywords, any_mode):
            continue
        rel = skill.relative_to(WIKI_ROOT).parent
        by_category.setdefault(skill.parent.parent.name, []).append((name, desc, rel))

    total = 0
    for category in sorted(by_category):
        print(f"## {category}")
        for name, desc, rel in sorted(by_category[category]):
            print(f"- **{name}** (`{rel}/`) — {desc}")
            total += 1
        print()
    if keywords:
        mode = "ANY" if any_mode else "ALL"
        print(f"# {total} topic(s) matching {mode} of {keywords!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
