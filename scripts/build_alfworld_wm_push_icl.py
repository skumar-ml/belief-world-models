#!/usr/bin/env python3
"""Rewrite query-based ALFWorld WM ICL into always-on push traces.

Drops `query` turns, merges the following go-to thought, and appends a
`[World model]` line (location, inventory, where-is) after each observation
by replaying the conversation through BeliefState.
"""

from __future__ import annotations

import json
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.alfworld_parse import parse_action  # noqa: E402
from wm.belief import BeliefState  # noqa: E402
from wm.goal_parser import parse_goal  # noqa: E402
from wm.query import format_push  # noqa: E402
from wm.scene import RECEP_BLOB_RE, parse_receptacles  # noqa: E402

_QUERY_RE = re.compile(r"^\s*query\b", re.IGNORECASE)
_ASK_RE = re.compile(
    r"\s*To narrow my search, I will first ask the world model where a \w+ is likely to be\.\s*",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(r"^(Thought|Reflection):\s*", re.IGNORECASE)

_PAIRS = [
    (
        "prompt/icl_examples/alfworld_react_wm_icl.json",
        "prompt/icl_examples/alfworld_react_wm_push_icl.json",
    ),
    (
        "prompt/icl_examples/alfworld_reflact_wm_icl.json",
        "prompt/icl_examples/alfworld_reflact_wm_push_icl.json",
    ),
]


def _action(content: str) -> str | None:
    try:
        return parse_action(content)
    except Exception:
        return None


def _split_thought_action(content: str) -> tuple[str, str | None]:
    m = re.search(r"\nAction:\s?", content)
    if not m:
        return content.strip(), None
    return content[: m.start()].strip(), content[m.end() :].strip()


def _merge_query_followup(query_msg: str, followup_msg: str) -> str:
    q_thought, _ = _split_thought_action(query_msg)
    f_thought, f_action = _split_thought_action(followup_msg)
    q_thought = re.sub(r"\s+", " ", _ASK_RE.sub(" ", q_thought)).strip()
    f_thought = f_thought.replace("gives likely locations", "shows likely locations")
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


def _init_belief(first_user: str) -> BeliefState:
    blob = RECEP_BLOB_RE.search(first_user)
    receptacles = parse_receptacles(blob.group(1)) if blob else []
    goal_match = re.search(r"(Your task is to:.*)$", first_user, re.DOTALL)
    goal_line = goal_match.group(1).strip() if goal_match else first_user
    return BeliefState(parse_goal(goal_line), receptacles, track_belief=True)


def _obs_body(content: str) -> str:
    text = content.strip()
    if text.startswith("Observation:"):
        text = text[len("Observation:") :].strip()
    return text.split("\n[World model]")[0].strip()


def _with_push(content: str, belief: BeliefState) -> str:
    return f"{content.rstrip()}\n[World model] {format_push(belief)}"


def _replay(msgs: list[dict]) -> list[dict]:
    belief = _init_belief(msgs[0]["content"])
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
    print(f"wrote {dst} ({n} conversations)")


def main() -> None:
    for src, dst in _PAIRS:
        convert_file(src, dst)


if __name__ == "__main__":
    main()
