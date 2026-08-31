"""World-model query answering for SciWorld.

Supported queries (case-insensitive, text after the `query` keyword):
  - "where is <object>"              -> ranked candidate rooms
  - "what is in <room>"              -> observed contents of a searched room
  - "have i searched <room>"         -> searched flag
  - "state" / "status"               -> compact structured dump
"""

import re
from typing import List, Tuple

from wm.sciworld_belief import SciWorldBeliefState

_HELP = ("Unrecognized query. Try: 'query where is <object>', "
         "'query what is in <room>', 'query searched <room>', 'query state'.")

_MASS_THRESHOLD = 0.90
_TRAILING_FILLER = ("again", "now", "please", "still", "yet", "then", "next")


def _strip_referent(text: str) -> str:
    ref = text.strip().strip("?.!,").strip()
    changed = True
    while changed:
        changed = False
        for w in _TRAILING_FILLER:
            if ref.lower().endswith(" " + w):
                ref = ref[: -(len(w) + 1)].rstrip()
                changed = True
    return ref


def _format_ranking(ranking: List[Tuple[str, float]]) -> str:
    shown, cum = [], 0.0
    for i, (room, p) in enumerate(ranking):
        shown.append(f"{room} ({p:.2f})")
        cum += p
        if cum >= _MASS_THRESHOLD and i + 1 < len(ranking):
            break
    parts = ", ".join(shown)
    remaining = 1.0 - cum
    if len(ranking) - len(shown) > 0 and remaining > 1e-9:
        parts += f", and other rooms ({remaining * 100:.0f}%)"
    return parts


def _fmt_inventory(belief: SciWorldBeliefState) -> str:
    if not belief.inventory:
        return "nothing"
    parts = []
    for name in sorted(belief.inventory):
        if name not in belief.held_contents:
            parts.append(name)
            continue
        contents = belief.held_contents[name]
        if contents is None:
            parts.append(f"{name} (contents unknown)")
        elif contents:
            parts.append(f"{name} (containing {', '.join(sorted(contents))})")
        else:
            parts.append(f"{name} (empty)")
    return ", ".join(parts)


def state_dump(belief: SciWorldBeliefState) -> str:
    loc = belief.agent_location or "unknown"
    inv = _fmt_inventory(belief)
    focus = belief.focus or "nothing"
    searched = sorted(r for r, rm in belief.rooms.items() if rm.searched)
    return (f"You are in the {loc}, holding {inv}, focused on {focus}. "
            f"Rooms searched: {', '.join(searched) or 'none'}.")


def answer_query(belief: SciWorldBeliefState, query: str) -> str:
    ql = query.lower().strip()

    m = re.match(r"where (?:is|are)\s+(?:a |an |some |the )?(.+)", ql)
    if m:
        obj_type = _strip_referent(m.group(1))
        if belief.is_room(obj_type):
            return (f"{obj_type} is a room, not an object. You can teleport to it, or "
                    f"try 'query what is in {obj_type}' / 'query searched {obj_type}'.")
        loc = belief.container_of(obj_type)
        if loc:
            room, container = loc
            return f"{obj_type} is in the {room}, in the {container}."
        if not belief.has_information(obj_type):
            return "The world model cannot help you with this query."
        ranking = belief.where_is(obj_type)
        if not ranking:
            return f"No candidate rooms remain for {obj_type} (all likely rooms searched)."
        return f"{obj_type} is most likely in: {_format_ranking(ranking)}."

    m = re.match(r"what(?:'s| is| are)?\s+in\s+(?:the )?(.+)", ql)
    if m:
        room = _strip_referent(m.group(1))
        contents = belief.contents_of(room)
        if contents is None:
            return f"{room} has not been searched yet."
        return f"{room} contains: {', '.join(sorted(contents)) or 'nothing'}."

    m = re.match(r"(?:have i )?searched\s+(?:the )?(.+)", ql)
    if m:
        room = _strip_referent(m.group(1))
        return f"{room}: {'searched' if belief.is_searched(room) else 'not searched yet'}."

    if ql in ("state", "status"):
        return state_dump(belief)

    return _HELP
