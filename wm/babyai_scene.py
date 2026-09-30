"""Parse BabyAI-Text observations into carry / wall-ray / object sightings.

Pure text -> structure. Templates are MiniGrid `gen_graph` (English) after
BALROG's `You see ` strip in `envs/babyai_parse.format_text_obs`:

    You carry a grey box
    a wall 2 steps forward
    a wall 3 steps left
    a red ball 1 step forward
    a green key 1 step right and 2 steps forward
    a closed yellow door 2 steps forward and 1 step right
    an open blue door 1 step forward
    a locked green door 3 steps left

Offsets are egocentric (forward / left / right). The belief layer converts
them into the start-pose frame. Unrecognized lines are dropped.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


COLORS = ("red", "green", "blue", "purple", "yellow", "grey", "gray")
OBJECT_TYPES = ("key", "ball", "box", "door")
DOOR_STATES = ("open", "closed", "locked")

# Residual belief atoms (not grid cells).
UNSEEN = "UNSEEN"
OTHER_ROOM = "OTHER_ROOM"
INVENTORY = "inventory"

_RE_CARRY = re.compile(r"^You carry a (\w+) (\w+)\s*$", re.IGNORECASE)
_RE_WALL = re.compile(r"^a wall (\d+) steps? (forward|left|right)\s*$", re.IGNORECASE)
_RE_SIGHTING = re.compile(
    r"^an? (?:(open|closed|locked) )?(\w+) (key|ball|box|door) (.+)$",
    re.IGNORECASE,
)
_RE_OFFSET = re.compile(r"(\d+) steps? (forward|left|right)", re.IGNORECASE)


def normalize_color(color: Optional[str]) -> Optional[str]:
    """Canonicalize grey/gray; return None if missing."""
    if not color:
        return None
    c = color.strip().lower()
    return "grey" if c == "gray" else c


def referent_key(color: Optional[str], obj_type: str) -> str:
    """'green' + 'key' -> 'green key'; uncolored door -> 'door'."""
    t = obj_type.strip().lower()
    c = normalize_color(color)
    return f"{c} {t}" if c else t


def cell_key(x: int, y: int) -> str:
    """Stable atom for a start-frame cell in the categorical belief."""
    return f"{x},{y}"


def parse_cell_key(atom: str) -> Optional[Tuple[int, int]]:
    """Inverse of cell_key, or None if `atom` is a residual bucket."""
    m = re.fullmatch(r"(-?\d+),(-?\d+)", atom)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


@dataclass
class Sighting:
    """One FoV line: a wall ray or a visible object/door."""

    kind: str                          # wall | object | door
    forward: int                       # steps along heading
    right: int                         # +right / -left
    color: Optional[str] = None
    type: Optional[str] = None
    door_state: Optional[str] = None
    is_wall_ray: bool = False          # mark empties 1..N-1 on this axis


@dataclass
class ParsedObs:
    """Structured body of one BabyAI-Text observation (no 'Observation:' prefix)."""

    carrying: Optional[Tuple[str, str]] = None   # (color, type)
    sightings: List[Sighting] = None

    def __post_init__(self):
        if self.sightings is None:
            self.sightings = []


def _strip_obs(observation: str) -> str:
    """Drop the wrapper prefix and leftover 'You see ' if a caller forgot."""
    obs = (observation or "").strip()
    if obs.lower().startswith("observation:"):
        obs = obs.split(":", 1)[1].strip()
    return obs.replace("You see ", "")


def _parse_offsets(blob: str) -> Optional[Tuple[int, int]]:
    """Turn '2 steps forward and 1 step right' into (forward, right)."""
    hits = list(_RE_OFFSET.finditer(blob or ""))
    if not hits:
        return None
    forward, right = 0, 0
    for m in hits:
        n, axis = int(m.group(1)), m.group(2).lower()
        if axis == "forward":
            forward += n
        elif axis == "right":
            right += n
        else:
            right -= n
    return forward, right


def parse_observation(observation: str) -> ParsedObs:
    """Parse carry + wall rays + object/door lines; ignore the rest."""
    parsed = ParsedObs()
    for raw in _strip_obs(observation).splitlines():
        line = raw.strip()
        if not line:
            continue

        m = _RE_CARRY.match(line)
        if m:
            parsed.carrying = (normalize_color(m.group(1)), m.group(2).lower())
            continue

        m = _RE_WALL.match(line)
        if m:
            n, axis = int(m.group(1)), m.group(2).lower()
            fwd, right = (n, 0) if axis == "forward" else (0, n if axis == "right" else -n)
            parsed.sightings.append(
                Sighting(kind="wall", forward=fwd, right=right, type="wall", is_wall_ray=True)
            )
            continue

        m = _RE_SIGHTING.match(line)
        if m:
            state, color, typ, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            offs = _parse_offsets(rest)
            if offs is None:
                continue
            fwd, right = offs
            typ = typ.lower()
            kind = "door" if typ == "door" else "object"
            door_state = state.lower() if state and kind == "door" else None
            parsed.sightings.append(
                Sighting(
                    kind=kind,
                    forward=fwd,
                    right=right,
                    color=normalize_color(color),
                    type=typ,
                    door_state=door_state,
                )
            )
    return parsed


def is_uninformative(observation: str) -> bool:
    """True for format errors, invalid actions, or WALL-E rejections."""
    obs = (observation or "").strip()
    if obs.startswith("[World model]"):
        return True
    low = obs.lower()
    return "error input" in low or "nothing happens" in low
