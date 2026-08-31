"""Parse a SciWorld task description into (target_type, named_location?).

The SciWorld analogue of `wm/goal_parser.py`, but far simpler: SciWorld goals are
free-form English. We extract two things the location router needs:

  - `target_type`: the object the agent must find/act on (keys the room prior).
  - `named_location`: a room the task description explicitly states the target is in
    (the "task description is a location oracle" refinement) — used to seed the belief
    as a point mass, overriding the prior. None if not stated.

Verb/phrase patterns transcribed from the real taskDesc strings across the 24 test
task types. Design contract: wiki/syntheses/sciworld-wm-build-scope.md.
"""

import re
from dataclasses import dataclass
from typing import Optional

from wm.sciworld_scene import ROOMS


@dataclass
class SciWorldGoal:
    target_type: str                    # e.g. "water", "animal", "aluminum foil"
    named_location: Optional[str] = None  # a room stated in the desc, or None


# Ordered target-extraction patterns; first match wins. Each captures the target
# phrase, which is then cleaned (article/qualifier stripping) by _clean_target.
_TARGET_PATTERNS = [
    r"\bboil\s+(.+?)[.,]",                       # "boil water."
    r"\bmelt\s+(.+?)[.,]",                        # "melt water."
    r"\bfreeze\s+(.+?)[.,]",                      # "freeze X."
    r"\bchange the state of matter of\s+(.+?)[.,]",
    r"\bfind the (.+?) with\b",                   # "find the animal with the longest life span"
    r"\bfind a\(n\)\s+(.+?)[.,]",                 # "find a(n) animal."
    r"\bfind (?:a|an)\s+(.+?)[.,]",
    r"\bgrow (?:a|an)\s+(.+?)(?: from seed)?[.,]",  # "grow a apple plant from seed."
    # "life stages of the X" must precede the generic "focus on" so it keys on the
    # organism ("beaver") rather than the phrase "3 life stages of the beaver".
    r"\blife stages of the\s+(.+?)[.,]",          # "life stages of the turtle"
    # "focus on X" names the findable object in most tasks; placed before `create`
    # so e.g. "turn on the red light bulb ... create an electrical circuit" keys on
    # "red light bulb" (findable) rather than the created-circuit clause. Kept AFTER
    # the specific verbs so "boil ice" keys on "ice" (not "the substance") and life
    # stages keys on the organism.
    r"\bfocus on (?:the\s+|a\s+|an\s+)?(.+?)[.,]",
    r"\bcreate\s+(.+?)[.,]",                      # "create green paint."
    r"\bdetermine if\s+(.+?)\s+is\b",             # "determine if aluminum foil is ..."
    r"\bmeasure the (?:temperature|melting point) of\s+(.+?)[.,]",
]

# Explicit location-oracle phrases; capture the room word(s). "the" is optional and a
# bare room word is allowed so "located outside" (no article) is caught; "near" is
# included alongside "in"/"around" so "found near the kitchen" matches.
_LOCATION_PATTERNS = [
    r"located (?:around|in|near)?\s*(?:the\s+)?(.+?)[.,]",  # "located around the workshop" / "located outside"
    r"can be found (?:in|near|around)\s+(?:the\s+)?(.+?)[.,]",  # "found near the kitchen"
    r"are in the '?(.+?)'? location",             # "The animals are in the 'outside' location"
    r"is (?:around|in|near)\s+(?:the\s+)?(.+?)[.,]",
]


def parse_goal(task_desc: str) -> SciWorldGoal:
    """Extract (target_type, named_location) from a task description.

    Raises ValueError if no target type can be identified, so the caller can degrade
    to plain ReflAct (WM disabled) for goals we don't understand.
    """
    desc = " ".join((task_desc or "").replace("Task Description:", "").split())

    target = None
    for pat in _TARGET_PATTERNS:
        m = re.search(pat, desc, re.IGNORECASE)
        if m:
            target = _clean_target(m.group(1))
            if target:
                break
    if not target:
        raise ValueError(f"could not parse target type from: {desc[:120]!r}")

    return SciWorldGoal(target_type=target, named_location=_parse_location(desc, target))


def _parse_location(desc: str, target: str) -> Optional[str]:
    """Return a room explicitly named as the TARGET's location, or None.

    Guards:
    - Destination rooms ("move it to the red box in the kitchen") use 'move ... to',
      which none of the location patterns match.
    - Subject mismatch: a clause like "unknown substance B ... is located around the
      living room. First, focus on the thermometer" names the SUBSTANCE's location,
      not the target (thermometer). We only accept the location if the target word(s)
      appear in the clause's subject (the text just before the location phrase), so
      the substance's location isn't mis-applied to the thermometer target. Falls
      back to the prior (e.g. thermometer -> kitchen) when the subject differs.
    """
    trigger = re.compile(r"\b(located|can be found|are in the|is (?:around|in|near))\b",
                         re.IGNORECASE)
    for pat in _LOCATION_PATTERNS:
        m = re.search(pat, desc, re.IGNORECASE)
        if not m:
            continue
        room = m.group(1).strip().lower().strip("'\"")
        if room not in ROOMS:
            continue
        # Subject = up to ~60 chars of desc before the location trigger word.
        tm = trigger.search(desc, 0, m.start() + 1) or trigger.search(desc)
        subject = desc[max(0, (tm.start() if tm else m.start()) - 60): (tm.start() if tm else m.start())].lower()
        # Accept if a content word of the target appears in the subject, OR the
        # subject is "seed(s)" — for grow tasks ("Seeds can be found in the X")
        # the seed is the findable precursor of the target plant/fruit.
        target_words = [w for w in target.lower().split() if len(w) > 2]
        subject_matches = any(w in subject for w in target_words) if target_words else True
        if not subject_matches and "seed" not in subject:
            continue
        return room
    return None


# Location clauses, capturing BOTH the subject object phrase and the room. Unlike
# _LOCATION_PATTERNS (which only capture the room for the target), these bind the
# subject noun phrase before the trigger so we can seed a belief for ANY named object
# (e.g. "unknown substance B is located around the bathroom" -> substance, not target).
# The subject is greedy-but-bounded: the shortest run of words ending at the trigger.
_STATED_LOCATION_PATTERNS = [
    # "The animals are in the 'outside' location."
    r"(?:the\s+)?([\w ]+?)\s+are in the '?([\w ]+?)'? location",
    # "... of X, which is located around the workshop." (inline measure-task form)
    r"(?:the\s+)?([\w ]+?),?\s+which is located (?:around|in|near)?\s*(?:the\s+)?([\w ]+?)[.,]",
    # "The sodium chloride is located around the workshop." / "The plants are located outside."
    r"(?:the\s+)?([\w ]+?)\s+(?:is|are) located (?:around|in|near)?\s*(?:the\s+)?([\w ]+?)[.,]",
    # "Seeds can be found in the living room."
    r"(?:the\s+)?([\w ]+?)\s+can be found (?:in|near|around)\s+(?:the\s+)?([\w ]+?)[.,]",
    # "The boxes are in the kitchen." (bare "is/are in [the] <room>")
    r"(?:the\s+)?([\w ]+?)\s+(?:is|are) (?:around|in|near)\s+(?:the\s+)?([\w ]+?)[.,]",
]

# Subject phrases that are not real findable objects (pronouns / task filler). If a clause's
# captured subject is one of these we skip it rather than seed a junk belief.
_SUBJECT_STOPWORDS = {"it", "they", "them", "this", "that", "which", "object", "objects",
                      "target", "task"}


def parse_stated_locations(task_desc: str) -> dict:
    """Map every object phrase the task description explicitly places in a room -> room.

    Complements `parse_goal.named_location` (target-only): scans ALL location clauses and
    binds the subject noun phrase, so non-target objects the agent must still find (the
    substance in a measure task, the seeds in a grow task) get their stated location too.
    Returns {object_phrase: room} for rooms in the fixed set; empty if none stated.

    Only the raw taskDesc should be passed -- NOT the full prompt -- so injected MPO
    meta-plan lines ("teleport to the room where X is located (e.g. ...)") are never
    scanned (they also contain "located").
    """
    desc = " ".join((task_desc or "").replace("Task Description:", "").split())
    out: dict = {}
    for pat in _STATED_LOCATION_PATTERNS:
        for m in re.finditer(pat, desc, re.IGNORECASE):
            subject = _clean_target(m.group(1))
            room = m.group(2).strip().lower().strip("'\"")
            if room not in ROOMS or not subject:
                continue
            # Keep only the last 1-3 words of the subject (the object head), dropping any
            # leading verb clause the greedy capture swept in ("your task is to measure the
            # temperature of unknown substance b" -> "unknown substance b").
            subject = _subject_head(subject)
            if not subject or subject in _SUBJECT_STOPWORDS:
                continue
            out.setdefault(subject, room)  # first (leftmost) clause wins per object
    return out


# Verb/preposition words that mark the end of a leading clause; the object head is the
# run of content words AFTER the last of these in a greedy subject capture.
_SUBJECT_LEAD_MARKERS = ("of", "on", "is", "are", "measure", "temperature", "melting",
                         "point", "the", "your", "task", "to", "find", "determine")


def _subject_head(subject: str) -> str:
    """Reduce a greedy subject capture to the object head phrase.

    "your task is to measure the temperature of unknown substance b" -> "unknown substance b".
    Drops everything up to and including the last leading marker word, keeping the tail.
    """
    words = subject.split()
    last_marker = -1
    for i, w in enumerate(words):
        if w in _SUBJECT_LEAD_MARKERS:
            last_marker = i
    head = words[last_marker + 1:] if last_marker >= 0 else words
    return " ".join(head).strip()


def _clean_target(text: str) -> str:
    """Normalize a captured target phrase to a prior key.

    'a apple plant' -> 'apple plant'; 'the aluminum foil' -> 'aluminum foil'.
    Trailing task qualifiers ('thing', 'substance') are kept as-is — they still match
    the prior's fallback (uniform) and the scene matcher tolerates a miss.
    """
    t = text.strip().lower()
    t = re.sub(r"^(a\(n\)|a|an|the|some)\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,")
    return t
