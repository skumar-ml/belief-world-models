"""Query answering for the BabyAI world model.

`where is` does not dump the 36-cell categorical: after localization it reports
remaining count, nearest cells, and which side of the agent still has mass.
`map` prints the pinned 6x6.
"""

import re
from typing import List, Tuple

from wm.babyai_prior_belief import BabyAIPriorBeliefState
from wm.babyai_query import (
    _HELP as _MEMORY_HELP,
    _cell_label,
    _normalize_object,
    _parse_place,
    format_relative,
)
from wm.babyai_scene import INVENTORY, UNSEEN, parse_cell_key, referent_key


_HELP = (
    _MEMORY_HELP[:-1]
    + ", 'query map'."
)
_NEAREST_K = 5
_QUADRANT_LABEL = {
    "front": "in front of you",
    "back": "behind you",
    "left": "to your left",
    "right": "to your right",
    "here": "under you",
}


def _atom_phrase(belief: BabyAIPriorBeliefState, atom: str) -> str:
    if atom == INVENTORY:
        return "in your inventory"
    if atom == UNSEEN:
        return "not yet seen"
    xy = parse_cell_key(atom)
    if xy is None:
        return atom
    return format_relative(*belief.abs_to_rel(xy))


def _format_nearest(belief: BabyAIPriorBeliefState, ranking: List[Tuple[str, float]]) -> str:
    shown = ranking[:_NEAREST_K]
    parts = [_atom_phrase(belief, atom) for atom, _ in shown]
    extra = len(ranking) - len(shown)
    text = ", ".join(parts)
    if extra > 0:
        text += f", and {extra} farther cell{'s' if extra != 1 else ''}"
    return text


def _format_quadrants(belief: BabyAIPriorBeliefState, query_key: str) -> str:
    buckets = belief.quadrant_mass(query_key)
    ranked = sorted(
        ((name, p) for name, p in buckets.items() if p > 0.04),
        key=lambda kv: -kv[1],
    )
    if not ranked:
        return ""
    bits = [f"{_QUADRANT_LABEL[name]} ({p:.0%})" for name, p in ranked]
    return " Mass is " + ", ".join(bits) + "."


_MAP_LEGEND = (
    "Legend: ^>v< you (facing), # wall, . empty/ruled out, "
    "? candidate, letter = seen object."
)


def _holding_line(belief: BabyAIPriorBeliefState) -> str:
    if belief.inventory:
        return f"Holding a {referent_key(*belief.inventory)}."
    return "Holding nothing."


def _format_map(belief: BabyAIPriorBeliefState) -> str:
    if not belief.room_localized:
        return (
            "Room box is not localized yet. Face a wall ahead and a wall "
            "to one side, then query map again. "
            "Legend: ^>v< you, # wall, . empty/ruled out, ? candidate, "
            "letter = seen object."
        )
    return f"{belief.ascii_map()}\n{_MAP_LEGEND}"


def _state_dump(belief: BabyAIPriorBeliefState) -> str:
    """Map plus inventory only — no pose / candidate-count dump."""
    return f"{_holding_line(belief)}\n{_format_map(belief)}"


def _where_is(belief: BabyAIPriorBeliefState, key: str) -> str:
    status, atom = belief.locate_referent(key)
    if status == "inventory":
        return f"You are holding the {key}."
    if status == "located" and atom:
        return f"The {key} is at {_atom_phrase(belief, atom)}."
    ranking = belief.where_is(key)
    if ranking and len(ranking) == 1 and ranking[0][0] == INVENTORY:
        return f"You are holding the {key}."
    if not belief.room_localized:
        return (
            f"The {key} has not been seen. The 6x6 room is not localized yet "
            "(observe a wall in front and a wall to the left or right)."
        )
    nearest = belief.remaining_ranking(key)
    if not nearest:
        return f"No candidate cells remain for the {key}."
    n = len(nearest)
    p = nearest[0][1]
    body = (
        f"The {key} has not been seen. {n} candidate cell{'s' if n != 1 else ''} "
        f"remain (about {p:.0%} each). Nearest: {_format_nearest(belief, nearest)}."
    )
    return body + _format_quadrants(belief, key)


def _what_is_at(belief: BabyAIPriorBeliefState, place: Tuple[int, int]) -> str:
    pos = belief.rel_to_abs(*place)
    label = format_relative(*place)
    fact = belief.contents_of(pos)
    if fact is not None:
        return f"{label} is {_cell_label(fact)}."
    masses = belief.cell_mass(pos)
    if masses:
        refs = ", ".join(f"{k} ({p:.0%})" for k, p in masses)
        return f"{label} has not been observed; still a candidate for {refs}."
    if belief.room_localized and not belief.room.contains_interior(pos):
        return f"{label} is outside the room."
    if belief.room_localized:
        return f"{label} is ruled out for tracked objects."
    return f"{label} has not been observed yet."


def answer_query(belief: BabyAIPriorBeliefState, query: str) -> str:
    ql = query.lower().strip()

    if ql in ("what have i seen", "what have I seen", "seen"):
        seen = belief.last_seen_objects()
        if not seen:
            return "You have not seen any objects yet."
        return "You have seen: " + ", ".join(
            f"{k} at {_atom_phrase(belief, a)}" for k, a in seen
        ) + "."

    if ql in ("map", "room", "grid"):
        return _format_map(belief)

    m = re.match(r"where (?:is|are)\s+(?:a |an |some |the )?(.+)", ql)
    if m:
        return _where_is(belief, _normalize_object(m.group(1)))

    m = re.match(r"what(?:'s| is| are)?\s+(?:at|in)\s+(?:the )?(.+)", ql)
    if m:
        place = _parse_place(m.group(1))
        if place is None:
            return f"Could not parse place {m.group(1)!r}. Try '1 step forward'."
        return _what_is_at(belief, place)

    m = re.match(r"(?:have i )?searched\s+(?:the )?(.+)", ql)
    if m:
        place = _parse_place(m.group(1))
        if place is None:
            return f"Could not parse place {m.group(1)!r}. Try '1 step forward'."
        pos = belief.rel_to_abs(*place)
        label = format_relative(*place)
        if belief.is_searched(pos):
            return f"{label}: searched."
        masses = belief.cell_mass(pos)
        if masses:
            return f"{label}: not searched yet (still a candidate)."
        return f"{label}: not searched yet."

    if ql in ("state", "status"):
        return _state_dump(belief)

    return _HELP
