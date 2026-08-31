"""Parse an ALFWorld goal line into a structured Goal tuple.

Goal-line phrasings seen in the actual ALFWorld dataset (both verb-first and
adjective forms occur for the transform tasks):
  pick_and_place        : "put some spraybottle on toilet."
  pick_clean_then_place : "put a clean lettuce in diningtable."
  pick_heat_then_place  : "heat some egg and put it in diningtable."  /  "put a hot mug in coffeemachine."
  pick_cool_then_place  : "cool some pan and put it in stoveburner."  /  "put a cool mug in coffeemachine."
  pick_two_obj          : "put two creditcard in dresser."
  look_at_obj           : "look at bowl under the desklamp."  /  "examine a cd with a desklamp."

We extract (act, target_type, transform, dest_recep_type, count). Object/receptacle
words in the goal line are already lowercased single tokens (e.g. "spraybottle",
"diningtable"), matching the affinity-prior keys. Note the adjective for "heat" is
"hot" (not "heated"); we normalize it to the transform verb "heat".

Design contract: wiki/syntheses/m3-stage1-spec.md.
"""

import re
from typing import Optional

from wm.belief import Goal


_GOAL_PREFIX = re.compile(r"your task is to:\s*(.+?)\.?\s*$", re.IGNORECASE)

# Article alternation ("a"/"an"/"some"/"the"), reused across the goal patterns.
_ART = r"(?:a |an |some |the )?"
# Adjective -> transform-verb map; "hot" is the adjective form of "heat".
_ADJ = {"clean": "clean", "hot": "heat", "cool": "cool"}


def _strip_prefix(goal_line: str) -> str:
    m = _GOAL_PREFIX.search(goal_line.strip())
    return (m.group(1) if m else goal_line).strip().rstrip(".").lower()


def _normalize_slice(g: str) -> str:
    """Collapse sliced-object phrasings to the base object type, since the affinity
    prior and observations key on the base type (e.g. "apple", not "apple slice").
    "slice of apple" -> "apple"; "sliced apple" -> "apple"."""
    g = re.sub(r"slice of (\w+)", r"\1", g)
    g = re.sub(r"sliced (\w+)", r"\1", g)
    return g


def parse_goal(goal_line: str) -> Goal:
    """Parse a raw goal line (with or without the 'Your task is to:' prefix).

    Raises ValueError on garbled / sentinel goals; callers (e.g. WMAlfWorldEnv)
    are expected to catch this and degrade gracefully (run without the WM).
    """
    g = _normalize_slice(_strip_prefix(goal_line))

    # Sentinel for malformed/missing goals in the dataset — fail soft.
    if g == "unknown goal":
        raise ValueError(f"Unparseable ALFWorld goal sentinel: {goal_line!r}")

    # look_at_obj: "look at <obj> under the <lamp>" OR "examine <obj> with a <lamp>"
    m = re.match(rf"look at {_ART}(\w+) under {_ART}(\w+)", g) \
        or re.match(rf"examine {_ART}(\w+) with {_ART}(\w+)", g)
    if m:
        return Goal(act="look", target_type=m.group(1), transform=None,
                    dest_recep_type=m.group(2), count=1)

    # transform-then-place, VERB-first: "<clean|heat|cool> some <obj> and put it in/on <recep>"
    m = re.match(rf"(clean|heat|cool) {_ART}(\w+) and put it (?:in|on) {_ART}(\w+)", g)
    if m:
        verb = m.group(1)
        return Goal(act=f"pick_{verb}_then_place", target_type=m.group(2),
                    transform=verb, dest_recep_type=m.group(3), count=1)

    # pick_two: "put two <obj> in/on <recep>" OR "find two <obj> and put them in/on <recep>"
    # (the "find two ... them" form is the dominant picktwo phrasing in the dataset).
    m = re.match(rf"put two (\w+) (?:in|on) {_ART}(\w+)", g) \
        or re.match(rf"find two (\w+) and put them (?:in|on) {_ART}(\w+)", g)
    if m:
        return Goal(act="picktwo", target_type=m.group(1), transform=None,
                    dest_recep_type=m.group(2), count=2)

    # transform-then-place, ADJECTIVE form: "put a <clean|hot|cool> <obj> in/on <recep>".
    # Must precede the bare catch-all, else the adjective is mis-read as the object.
    m = re.match(rf"put {_ART}(clean|hot|cool) (\w+) (?:in|on) {_ART}(\w+)", g)
    if m:
        verb = _ADJ[m.group(1)]
        return Goal(act=f"pick_{verb}_then_place", target_type=m.group(2),
                    transform=verb, dest_recep_type=m.group(3), count=1)

    # pick_and_place: "put some <obj> in/on <recep>" OR "find some <obj> and put it in/on <recep>"
    m = re.match(rf"put {_ART}(\w+) (?:in|on) {_ART}(\w+)", g) \
        or re.match(rf"find {_ART}(\w+) and put it (?:in|on) {_ART}(\w+)", g)
    if m:
        return Goal(act="pick_and_place", target_type=m.group(1), transform=None,
                    dest_recep_type=m.group(2), count=1)

    raise ValueError(f"Could not parse goal line: {goal_line!r} (normalized: {g!r})")
