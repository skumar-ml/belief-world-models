"""World-model query answering for BabyAI-Text.

Supported queries (case-insensitive, text after the `query` keyword):
  - "where is <object>"     -> last-seen cell (relative) or residual ranking
  - "what is at <place>" / "what is in <place>"
  - "have i searched <place>" / "searched <place>"
  - "what have i seen"
  - "state" / "status"

Places are egocentric: "1 step forward", "2 steps back and 1 right", "front".
"""

import re
from typing import List, Optional, Tuple

from wm.babyai_belief import BabyAIBeliefState
from wm.babyai_scene import INVENTORY, OTHER_ROOM, UNSEEN, parse_cell_key, referent_key


_HELP = (
    "Unrecognized query. Try: 'query where is <object>', "
    "'query what is at <1 step forward>', 'query searched <place>', "
    "'query what have I seen', 'query state'."
)
_MASS_THRESHOLD = 0.90
_RE_OFFSET = re.compile(r"(\d+) steps? (forward|back|left|right)", re.IGNORECASE)


def format_relative(forward: int, right: int) -> str:
    """Render an (forward, right) offset the way BabyAI-Text would."""
    parts = []
    if forward > 0:
        parts.append(f"{forward} step{'s' if forward != 1 else ''} forward")
    elif forward < 0:
        parts.append(f"{-forward} step{'s' if forward != -1 else ''} back")
    if right > 0:
        parts.append(f"{right} step{'s' if right != 1 else ''} right")
    elif right < 0:
        parts.append(f"{-right} step{'s' if right != -1 else ''} left")
    if not parts:
        return "your current cell"
    return " and ".join(parts)


def _atom_phrase(belief: BabyAIBeliefState, atom: str) -> str:
    if atom == INVENTORY:
        return "in your inventory"
    if atom == UNSEEN:
        return "not yet seen"
    if atom == OTHER_ROOM:
        return "the other room"
    xy = parse_cell_key(atom)
    if xy is None:
        return atom
    fwd, right = belief.abs_to_rel(xy)
    return format_relative(fwd, right)


def _format_ranking(belief: BabyAIBeliefState, ranking: List[Tuple[str, float]]) -> str:
    shown, cum = [], 0.0
    for i, (atom, p) in enumerate(ranking):
        shown.append(f"{_atom_phrase(belief, atom)} ({p:.2f})")
        cum += p
        if cum >= _MASS_THRESHOLD and i + 1 < len(ranking):
            break
    parts = ", ".join(shown)
    remaining = 1.0 - cum
    if len(ranking) - len(shown) > 0 and remaining > 1e-9:
        parts += f", and other places ({remaining * 100:.0f}%)"
    return parts


def _parse_place(text: str) -> Optional[Tuple[int, int]]:
    """Parse an egocentric place into (forward, right), or None."""
    blob = (text or "").strip().strip("?.!,").lower()
    if blob in ("front", "in front", "in front of me", "1 step forward"):
        return 1, 0
    if blob in ("here", "current cell", "your current cell"):
        return 0, 0
    hits = list(_RE_OFFSET.finditer(blob))
    if not hits:
        return None
    forward, right = 0, 0
    for m in hits:
        n, axis = int(m.group(1)), m.group(2).lower()
        if axis == "forward":
            forward += n
        elif axis == "back":
            forward -= n
        elif axis == "right":
            right += n
        else:
            right -= n
    return forward, right


def _normalize_object(text: str) -> str:
    ref = text.strip().strip("?.!,").lower()
    ref = re.sub(r"^(a |an |the |some )", "", ref)
    ref = re.sub(r"\s+", " ", ref).replace("gray", "grey")
    return ref


def _cell_label(fact) -> str:
    if fact.kind == "empty":
        return "empty"
    if fact.kind == "wall":
        return "a wall"
    if fact.kind == "door":
        state = f"{fact.door_state} " if fact.door_state else ""
        color = f"{fact.color} " if fact.color else ""
        return f"a {state}{color}door".replace("  ", " ")
    color = f"{fact.color} " if fact.color else ""
    return f"a {color}{fact.type or 'object'}".strip()


def state_dump(belief: BabyAIBeliefState) -> str:
    inv = (
        f"a {referent_key(*belief.inventory)}"
        if belief.inventory
        else "nothing"
    )
    seen = belief.last_seen_objects()
    if seen:
        seen_txt = ", ".join(f"{k} at {_atom_phrase(belief, a)}" for k, a in seen)
    else:
        seen_txt = "none"
    unseen = False
    if belief.track_belief:
        unseen = any(
            b.located_at is None and b.dist.get(UNSEEN, 0.0) > 0
            for b in belief.beliefs.values()
        )
    residual = " UNSEEN still has mass." if unseen else ""
    return (
        f"You are at start-frame ({belief.x}, {belief.y}), facing {belief.heading_name()}, "
        f"holding {inv}. Last seen: {seen_txt}.{residual}"
    )


def answer_query(belief: BabyAIBeliefState, query: str) -> str:
    ql = query.lower().strip()

    if ql in ("what have i seen", "what have I seen", "seen"):
        seen = belief.last_seen_objects()
        if not seen:
            return "You have not seen any objects yet."
        return "You have seen: " + ", ".join(
            f"{k} at {_atom_phrase(belief, a)}" for k, a in seen
        ) + "."

    m = re.match(r"where (?:is|are)\s+(?:a |an |some |the )?(.+)", ql)
    if m:
        key = _normalize_object(m.group(1))
        status, atom = belief.locate_referent(key)
        if status == "inventory":
            return f"You are holding the {key}."
        if status == "located" and atom:
            return f"The {key} is at {_atom_phrase(belief, atom)}."
        ranking = belief.where_is(key)
        if not ranking:
            return f"{key} has not been observed yet. Explore to find it."
        if not getattr(belief, "track_belief", True):
            if ranking and ranking[0][0] != UNSEEN:
                return f"{key} has been observed at: {', '.join(_atom_phrase(belief, a) for a, _ in ranking)}."
            return f"{key} has not been observed yet. Explore to find it."
        if len(ranking) == 1 and ranking[0][0] == UNSEEN:
            return f"The {key} has not been seen yet."
        return f"{key} is most likely at: {_format_ranking(belief, ranking)}."

    m = re.match(r"what(?:'s| is| are)?\s+(?:at|in)\s+(?:the )?(.+)", ql)
    if m:
        place = _parse_place(m.group(1))
        if place is None:
            return f"Could not parse place {m.group(1)!r}. Try '1 step forward'."
        pos = belief.rel_to_abs(*place)
        fact = belief.contents_of(pos)
        if fact is None:
            return f"{format_relative(*place)} has not been observed yet."
        return f"{format_relative(*place)} is {_cell_label(fact)}."

    m = re.match(r"(?:have i )?searched\s+(?:the )?(.+)", ql)
    if m:
        place = _parse_place(m.group(1))
        if place is None:
            return f"Could not parse place {m.group(1)!r}. Try '1 step forward'."
        pos = belief.rel_to_abs(*place)
        label = format_relative(*place)
        return f"{label}: {'searched' if belief.is_searched(pos) else 'not searched yet'}."

    if ql in ("state", "status"):
        return state_dump(belief)

    return _HELP
