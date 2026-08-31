"""World-model query answering (alfworld-free, so it's unit-testable on any node).

The WM env (envs/wm_alfworld_env.py) delegates query handling here; tests import the
same function, so there is a single source of truth for the query grammar + responses.

Supported queries (case-insensitive, text after the `query` keyword):
  - "where is <object_type>"                 -> ranked non-searched receptacles
  - "where is <object_type> <n>"             -> that specific instance's location only
                                                (or held / not-yet-located)
  - "what is in <receptacle>" / "what's in …" -> observed contents, or not-searched
  - "have i searched <receptacle>" / "searched …"
  - "state" / "status"                        -> compact structured dump

Design contract: wiki/syntheses/m3-stage1-spec.md.
"""

import re
from typing import List, Tuple

from wm.belief import BeliefState

_HELP = ("Unrecognized query. Try: 'query where is <object>', "
         "'query what is in <receptacle>', 'query searched <receptacle>', 'query state'.")

# Cumulative-probability threshold for how many receptacles to enumerate in a
# where-is answer: list them in ranked order until we've covered this much mass.
_MASS_THRESHOLD = 0.90


def _format_ranking(ranking: List[Tuple[str, float]]) -> str:
    """Enumerate receptacles (highest-probability first) until cumulative mass crosses
    _MASS_THRESHOLD; summarize any remaining tail as "and other receptacles (X%)".

    `ranking` is already sorted best-first and contains only non-zero candidates.
    """
    shown, cum = [], 0.0
    for i, (recep, p) in enumerate(ranking):
        shown.append(f"{recep} ({p:.2f})")
        cum += p
        # Stop once we've covered the threshold — but only if a tail actually remains.
        if cum >= _MASS_THRESHOLD and i + 1 < len(ranking):
            break
    parts = ", ".join(shown)
    remaining = 1.0 - cum
    n_remaining = len(ranking) - len(shown)
    # Honestly flag unenumerated locations when any meaningful tail mass is left.
    if n_remaining > 0 and remaining > 1e-9:
        parts += f", and other receptacles ({remaining * 100:.0f}%)"
    return parts


def state_dump(belief: BeliefState) -> str:
    loc = belief.agent_location or "unknown"
    inv = ", ".join(sorted(belief.inventory)) or "nothing"
    known = sorted(o.id for o in belief.objects.values())
    searched = sorted(r for r, rec in belief.receptacles.items() if rec.searched)
    return (f"You are at {loc}, holding {inv}. "
            f"Known objects: {', '.join(known) or 'none'}. "
            f"Searched receptacles: {', '.join(searched) or 'none'}.")


def state_dump_internal(belief: BeliefState) -> str:
    """Internal-state-only push: just the agent's location and inventory.

    External state (objects seen, searched receptacles, where-is) stays query-able.
    """
    loc = belief.agent_location or "unknown"
    inv = ", ".join(sorted(belief.inventory)) or "nothing"
    return f"You are at {loc}, holding {inv}."


def answer_query(belief: BeliefState, query: str) -> str:
    ql = query.lower().strip()

    # Capture the object type and an OPTIONAL trailing instance number
    # ("peppershaker" -> type-level; "peppershaker 2" -> that instance only).
    m = re.match(r"where (?:is|are)\s+(?:a |an |some |the )?([a-z]+)(?:\s+(\d+))?", ql)
    if m:
        obj_type, instance_num = m.group(1), m.group(2)

        # Reject receptacle queries — receptacles are fixed/known, not searched-for.
        if obj_type in belief.receptacle_types():
            return (f"{obj_type} is a receptacle, not an object. "
                    f"Try 'query what is in {obj_type} 1' or 'query searched {obj_type} 1'.")

        # Instance-specific query: answer only for that instance, from tracked state.
        if instance_num is not None:
            obj_id = f"{obj_type} {instance_num}"
            status, loc = belief.locate_instance(obj_id)
            if status == "inventory":
                return f"You are holding {obj_id}."
            if status == "located":
                return f"{obj_id} is at {loc}."
            return f"{obj_id} has not been located yet."

        # Type-level query: rank likely receptacles from the belief distribution.
        ranking = belief.where_is(obj_type)
        if not ranking:
            # Deterministic-only WM: nothing observed yet (no prior to fall back on).
            if not getattr(belief, "track_belief", True):
                return f"{obj_type} has not been observed yet. Search receptacles to find it."
            return f"No candidate locations remain for {obj_type} (all likely receptacles searched)."
        # Deterministic-only WM reports known locations, not a probability estimate.
        if not getattr(belief, "track_belief", True):
            return f"{obj_type} has been observed at: {', '.join(r for r, _ in ranking)}."
        return f"{obj_type} is most likely at: {_format_ranking(ranking)}."

    m = re.match(r"what(?:'s| is| are)?\s+in\s+(?:the )?(.+)", ql)
    if m:
        rid = m.group(1).strip()
        contents = belief.contents_of(rid)
        if contents is None:
            return f"{rid} has not been searched yet."
        return f"{rid} contains: {', '.join(sorted(contents)) or 'nothing'}."

    m = re.match(r"(?:have i )?searched\s+(?:the )?(.+)", ql)
    if m:
        rid = m.group(1).strip()
        return f"{rid}: {'searched' if belief.is_searched(rid) else 'not searched yet'}."

    if ql in ("state", "status"):
        return state_dump(belief)

    return _HELP
