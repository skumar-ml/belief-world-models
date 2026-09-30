#!/usr/bin/env python3
"""Rewrite query-based SciWorld WM ICL into always-on push traces.

Drops `query` turns, merges the following thought, and appends a
`[World model]` line (room, inventory, focus, where-is on the goal target
and stated-location objects) after each observation by replaying the
conversation through SciWorldBeliefState.
"""

from __future__ import annotations

import json
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wm.sciworld_belief import SciWorldBeliefState  # noqa: E402
from wm.sciworld_goal import parse_goal, parse_stated_locations  # noqa: E402
from wm.sciworld_query import format_push  # noqa: E402

_QUERY_RE = re.compile(r"^\s*query\b", re.IGNORECASE)
_ASK_RE = re.compile(
    r"\s*Before I start wandering between rooms, I'll ask the world model "
    r"where .+? is most likely to be\.\s*",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(r"^(Thought|Reflection):\s*", re.IGNORECASE)

_PAIRS = [
    (
        "prompt/icl_examples/sciworld_react_wm_pertask_icl.json",
        "prompt/icl_examples/sciworld_react_wm_push_pertask_icl.json",
    ),
    (
        "prompt/icl_examples/sciworld_reflact_wm_pertask_icl.json",
        "prompt/icl_examples/sciworld_reflact_wm_push_pertask_icl.json",
    ),
]


def _action(content: str) -> str | None:
    m = re.search(r"Action:\s?(.*)", content or "", re.DOTALL)
    if not m:
        return None
    return m.group(1).strip().split("\n", 1)[0].strip() or None


def _split_thought_action(content: str) -> tuple[str, str | None]:
    m = re.search(r"\nAction:\s?", content)
    if not m:
        return content.strip(), None
    return content[: m.start()].strip(), content[m.end() :].strip()


def _merge_query_followup(query_msg: str, followup_msg: str) -> str:
    q_thought, _ = _split_thought_action(query_msg)
    f_thought, f_action = _split_thought_action(followup_msg)
    q_thought = re.sub(r"\s+", " ", _ASK_RE.sub(" ", q_thought)).strip()
    q_thought = q_thought.replace(
        "I'll ask the world model where",
        "The world model shows where",
    )
    prefix_m = _PREFIX_RE.match(q_thought) or _PREFIX_RE.match(f_thought)
    prefix = prefix_m.group(0) if prefix_m else "Thought: "
    q_body = _PREFIX_RE.sub("", q_thought).strip()
    f_body = _PREFIX_RE.sub("", f_thought).strip()
    body = f"{q_body} {f_body}".strip() if q_body else f_body
    if f_action is None:
        return f"{prefix}{body}"
    return f"{prefix}{body}\nAction: {f_action}"


def _drop_queries(msgs: list[dict]) -> list[dict]:
    out = [msgs[0]]
    i = 1
    while i < len(msgs):
        msg = msgs[i]
        action = _action(msg["content"]) if msg.get("role") == "assistant" else None
        if action is not None and _QUERY_RE.match(action):
            i += 2  # skip query assistant + query observation
            if i < len(msgs) and msgs[i].get("role") == "assistant":
                out.append({
                    "role": "assistant",
                    "content": _merge_query_followup(msg["content"], msgs[i]["content"]),
                })
                i += 1
            continue
        out.append(msg)
        i += 1
    return out


def _init_belief(first_user: str) -> SciWorldBeliefState | None:
    try:
        return SciWorldBeliefState(
            parse_goal(first_user),
            stated_locations=parse_stated_locations(first_user),
        )
    except Exception:
        return None


def _obs_body(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("Observation:"):
        text = text[len("Observation:") :].strip()
    return text.split("\n[World model]")[0].strip()


def _with_push(content: str, belief: SciWorldBeliefState) -> str:
    return f"{content.rstrip()}\n[World model] {format_push(belief)}"


def _replay(msgs: list[dict]) -> list[dict]:
    belief = _init_belief(msgs[0]["content"])
    if belief is None:
        return msgs
    out = [{"role": "user", "content": _with_push(msgs[0]["content"], belief)}]
    last_action = None
    for msg in msgs[1:]:
        if msg["role"] == "assistant":
            last_action = _action(msg["content"])
            out.append(msg)
            continue
        raw = _obs_body(msg["content"])
        if last_action is not None:
            belief.observe(last_action, raw)
        last_action = None
        out.append({"role": "user", "content": _with_push(msg["content"], belief)})
    return out


def convert_file(src: str, dst: str) -> None:
    with open(os.path.join(_ROOT, src)) as f:
        data = json.load(f)
    converted = {
        family: [_replay(_drop_queries(conv)) for conv in convs]
        for family, convs in data.items()
    }
    out_path = os.path.join(_ROOT, dst)
    with open(out_path, "w") as f:
        json.dump(converted, f, indent=2)
        f.write("\n")
    n = sum(len(v) for v in converted.values())
    print(f"wrote {dst} ({n} conversations, {len(converted)} task types)")


def main() -> None:
    for src, dst in _PAIRS:
        convert_file(src, dst)


if __name__ == "__main__":
    main()
