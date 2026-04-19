"""PostToolUse hook: auto-sync plugins/mlx-skills/skills/ after edits to skills/.

Runs when Claude edits a file under top-level skills/. Invokes the sync
script in-process (stdlib-only, no subprocess) so the plugin mirror stays
current in-session. Silent on no-op. Does not block tool use on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PREFIX = ROOT / "skills"
MIRROR_PREFIX = ROOT / "plugins" / "mlx-skills" / "skills"

sys.path.insert(0, str(ROOT / "scripts"))
import sync_plugin_skills  # noqa: E402


def main() -> int:
    try:
        payload = orjson.loads(sys.stdin.buffer.read() or b"{}")
    except orjson.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return 0

    raw = tool_input.get("file_path")
    if not isinstance(raw, str):
        return 0

    try:
        resolved = Path(raw).resolve()
    except (OSError, RuntimeError):
        return 0

    if not resolved.is_relative_to(SOURCE_PREFIX):
        return 0
    if resolved.is_relative_to(MIRROR_PREFIX):
        return 0

    try:
        added, removed, changed = sync_plugin_skills._diff(
            sync_plugin_skills.SOURCE, sync_plugin_skills.TARGET
        )
        if not (added or removed or changed):
            return 0
        sync_plugin_skills._sync(sync_plugin_skills.SOURCE, sync_plugin_skills.TARGET)
    except Exception as exc:
        print(f"[sync_plugin_skills hook] sync failed: {exc}", file=sys.stderr)
        return 0

    print(
        "[sync_plugin_skills hook] "
        + sync_plugin_skills._format_report(added, removed, changed).strip(),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
