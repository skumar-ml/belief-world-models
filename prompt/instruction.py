"""Load instruction text, expanding {snippet} placeholders at runtime.

Composition files under prompt/instructions/ reference named snippets
(e.g. {task_instr}). Placeholders with an underscore are treated as snippet
names and must exist; brace tokens like {obj} are left as literal text.
"""

from __future__ import annotations

import os
import re

_PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_SNIPPET_ROOT = os.path.join(os.path.dirname(__file__), "snippets")
_MAX_EXPAND_PASSES = 8


def _pack_from_path(path: str) -> str | None:
    """'babyai_react_inst.txt' -> 'babyai' when that snippet pack exists."""
    stem = os.path.basename(path).split("_", 1)[0]
    pack_dir = os.path.join(_SNIPPET_ROOT, stem)
    return stem if os.path.isdir(pack_dir) else None


def _load_snippets(pack: str) -> dict[str, str]:
    """Read every *.txt in prompt/snippets/<pack>/ (body has no trailing newline)."""
    pack_dir = os.path.join(_SNIPPET_ROOT, pack)
    snippets = {}
    for name in os.listdir(pack_dir):
        if not name.endswith(".txt"):
            continue
        key = name[:-4]
        with open(os.path.join(pack_dir, name)) as f:
            snippets[key] = f.read().replace("\r\n", "\n").strip("\n")
    return snippets


def expand_snippets(text: str, pack: str) -> str:
    """Replace {name} with snippet bodies; nested refs expand in later passes."""
    snippets = _load_snippets(pack)
    current = text.replace("\r\n", "\n")
    for _ in range(_MAX_EXPAND_PASSES):
        missing = []

        def _sub(match: re.Match) -> str:
            name = match.group(1)
            if name in snippets:
                return snippets[name]
            if "_" in name:
                missing.append(name)
            return match.group(0)

        nxt = _PLACEHOLDER_RE.sub(_sub, current)
        if missing:
            raise KeyError(
                f"Unknown instruction snippet(s) in pack '{pack}': {sorted(set(missing))}"
            )
        if nxt == current:
            return current.strip("\n") + "\n"
        current = nxt
    raise RuntimeError(f"Instruction snippet expansion did not settle for pack '{pack}'")


def load_instruction(path: str) -> str:
    """Read an instruction file and expand snippet placeholders when a pack exists."""
    with open(path) as f:
        text = f.read()
    pack = _pack_from_path(path)
    if pack is None:
        return text
    return expand_snippets(text, pack)
