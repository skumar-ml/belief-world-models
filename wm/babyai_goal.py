"""Parse a BabyAI-Text mission string into a structured goal.

Surface forms from MiniGrid `verifier.py` (MixedTrainLocal, locations=False):

    go to the green key
    pick up the red ball
    open the yellow door
    open a door
    put the green box next to the purple ball
    pick up the blue key, then go to the red ball
    go to the red ball after you pick up the blue key

Color may be absent (`open a door`). Location suffixes (`in front of you`)
are stripped if they appear. Unparseable missions raise ValueError so the
WM env can disable itself for that episode.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from wm.babyai_scene import normalize_color


_ART = r"(?:a |an |the )?"
_COLOR = r"(?:red|green|blue|purple|yellow|grey|gray)"
_TYPE = r"(?:key|ball|box|door)"
_LOC_TAIL = r"\s+(?:in front of you|behind you|on your (?:left|right))"


@dataclass
class Referent:
    """A mission object: optional color + MiniGrid type."""

    color: Optional[str]
    type: str

    @property
    def key(self) -> str:
        return f"{self.color} {self.type}" if self.color else self.type


@dataclass
class BabyAIGoal:
    """Parsed mission. `seq_order` is 'then' or 'after' for pick-up-seq-go-to."""

    act: str                           # goto | pickup | open | putnext | seq
    referents: List[Referent]
    seq_order: Optional[str] = None


def _ref(color: Optional[str], typ: str) -> Referent:
    return Referent(color=normalize_color(color), type=typ.lower())


def _strip_mission(mission: str) -> str:
    text = (mission or "").strip()
    m = re.search(r"your task is to:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip().rstrip(".").lower()
    text = re.sub(_LOC_TAIL, "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_goal(mission: str) -> BabyAIGoal:
    """Parse a mission (with or without the 'Your task is to:' prefix)."""
    g = _strip_mission(mission)
    if not g:
        raise ValueError(f"empty BabyAI mission: {mission!r}")

    # Seq: "pick up A, then go to B"
    m = re.match(
        rf"pick up {_ART}(?:({_COLOR}) )?({_TYPE}), then go to {_ART}(?:({_COLOR}) )?({_TYPE})$",
        g,
    )
    if m:
        return BabyAIGoal(
            act="seq",
            referents=[_ref(m.group(1), m.group(2)), _ref(m.group(3), m.group(4))],
            seq_order="then",
        )

    # Seq: "go to B after you pick up A"
    m = re.match(
        rf"go to {_ART}(?:({_COLOR}) )?({_TYPE}) after you pick up {_ART}(?:({_COLOR}) )?({_TYPE})$",
        g,
    )
    if m:
        return BabyAIGoal(
            act="seq",
            referents=[_ref(m.group(3), m.group(4)), _ref(m.group(1), m.group(2))],
            seq_order="after",
        )

    # Put-next: "put A next to B"
    m = re.match(
        rf"put {_ART}(?:({_COLOR}) )?({_TYPE}) next to {_ART}(?:({_COLOR}) )?({_TYPE})$",
        g,
    )
    if m:
        return BabyAIGoal(
            act="putnext",
            referents=[_ref(m.group(1), m.group(2)), _ref(m.group(3), m.group(4))],
        )

    # Open: color optional so "open a door" is allowed.
    m = re.match(rf"open {_ART}(?:({_COLOR}) )?(door)$", g)
    if m:
        return BabyAIGoal(act="open", referents=[_ref(m.group(1), m.group(2))])

    m = re.match(rf"pick up {_ART}(?:({_COLOR}) )?({_TYPE})$", g)
    if m:
        return BabyAIGoal(act="pickup", referents=[_ref(m.group(1), m.group(2))])

    m = re.match(rf"go to {_ART}(?:({_COLOR}) )?({_TYPE})$", g)
    if m:
        return BabyAIGoal(act="goto", referents=[_ref(m.group(1), m.group(2))])

    raise ValueError(f"unparseable BabyAI mission: {mission!r}")
