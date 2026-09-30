"""Oracle action-validity checks for BabyAI-Text, using the MiniGrid grid.

Unlike the ALFWorld / SciWorld belief WMs (parse-from-text), this is an oracle:
it reads env.unwrapped (pose, carrying, cell in front). Used by the WALL-E gate
so a rejected action never steps the simulator.

Also flags illegal movement aliases (go backward / go left / go right) that are
not MiniGrid verbs but should still get [World model] feedback instead of a
generic invalid-action no-op.

Scope (v1):
  - go forward into a non-walkable cell (wall, closed/locked door, object)
  - pick up when the front cell is empty, not pickup-able, or hands are full
  - drop when not carrying, or when the front cell is occupied
  - toggle when the front cell is empty
  - go backward / go left / go right (not MiniGrid verbs; tell the agent to turn first)

MiniGrid pickup is the cell IN FRONT of the agent, not underfoot. Only key /
ball / box implement can_pickup(); door and wall do not.
"""

import re


_VALID_ACTIONS = "turn left, turn right, go forward, pick up, drop, toggle"

# "go backward", "go back", "move backwards", optional "N steps".
_BACKWARD_RE = re.compile(
    r"^(?:go|move|walk|step)\s+back(?:wards?)?(?:\s+\d+\s+steps?)?$|^back(?:wards?)?$",
    re.IGNORECASE,
)

# "go right", "move left", "go left twice", "go right 2 steps".
_LATERAL_RE = re.compile(
    r"^(?:go|move|walk|step)\s+(left|right)\b",
    re.IGNORECASE,
)


def _is_backward_action(action: str) -> bool:
    """True for go-back aliases that are not in the six MiniGrid verbs."""
    return bool(_BACKWARD_RE.match((action or "").strip()))


def _lateral_direction(action: str):
    """'left' / 'right' for go-left/go-right aliases, else None."""
    m = _LATERAL_RE.match((action or "").strip())
    return m.group(1).lower() if m else None


def _turn_then_forward_feedback(direction: str) -> str:
    """Shared hint: list legal verbs, then turn to face `direction` and go forward."""
    return (
        f"You cannot go {direction}. Valid actions are: {_VALID_ACTIONS}. "
        f"To go {direction}, you must turn to face that direction, then go forward."
    )


def _core(env):
    """Return the unwrapped MiniGrid env (gymnasium wrappers sit on top)."""
    return getattr(env, "unwrapped", env)


def _obj_phrase(cell) -> str:
    """Short English name for a MiniGrid cell (color + type, plus door state)."""
    if cell.type == "door":
        if cell.is_locked:
            state = "locked"
        elif cell.is_open:
            state = "open"
        else:
            state = "closed"
        return f"{state} {cell.color} door"
    return f"{cell.color} {cell.type}"


def _blocker_phrase(cell) -> str:
    """'a wall' or 'a grey key' for occupancy feedback."""
    if cell.type == "wall":
        return "a wall"
    return f"a {_obj_phrase(cell)}"


def check_babyai_action(env, action: str):
    """Return rejection feedback, or None if the action may execute.

    In-space commands: turns always pass. Drop / toggle are gated when they
    would be MiniGrid no-ops (empty hands, occupied front, empty front).
    Out-of-space aliases: go backward / go left / go right are rejected with a
    turn-then-forward hint. Other unknown verbs return None so the env can use
    its invalid-action path.
    """
    if _is_backward_action(action):
        return _turn_then_forward_feedback("backward")
    lateral = _lateral_direction(action)
    if lateral is not None:
        return _turn_then_forward_feedback(lateral)

    core = _core(env)
    fwd_cell = core.grid.get(*core.front_pos)

    # Blocked forward: MiniGrid only moves if the front cell is empty or can_overlap.
    if action == "go forward":
        if fwd_cell is not None and not fwd_cell.can_overlap():
            return (
                f"You cannot go forward because there is {_blocker_phrase(fwd_cell)} "
                f"in the way."
            )
        return None

    # Pickup: front cell, at most one carried object, only key/ball/box.
    if action == "pick up":
        if fwd_cell is None:
            return (
                "You cannot pick up anything because there is no object in front of you."
            )
        if not fwd_cell.can_pickup():
            return f"You cannot pick up the {_obj_phrase(fwd_cell)}."
        if core.carrying is not None:
            return (
                f"You cannot pick up the {_obj_phrase(fwd_cell)} because you are "
                f"already carrying a {_obj_phrase(core.carrying)}."
            )
        return None

    # Drop: must be carrying, and the cell in front must be empty (MiniGrid no-op otherwise).
    if action == "drop":
        if core.carrying is None:
            return "You cannot drop anything because you are not holding an object."
        if fwd_cell is not None:
            return (
                f"You cannot drop the {_obj_phrase(core.carrying)} because there is "
                f"{_blocker_phrase(fwd_cell)} in the way."
            )
        return None

    # Toggle: MiniGrid only toggles the cell in front (door / box). Empty front is a no-op.
    if action == "toggle":
        if fwd_cell is None:
            return (
                "You cannot toggle anything because there is no object in front of you."
            )
        return None

    return None
