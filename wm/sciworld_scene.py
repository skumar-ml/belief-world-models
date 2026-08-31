"""Parse SciWorld observation text into room / container / object structure.

The SciWorld analogue of `wm/scene.py`. Unlike ALFWorld's flat one-line-per-receptacle
grammar, a SciWorld `look around` is a multi-line room dump with nested containment
(`counter -> bowl -> apple`). Objects carry multi-word surface referents ("red apple",
"aluminum foil", "apple seed") and generally NO numeric id, so we key on referent
substrings rather than ALFWorld's `<type> <num>`.

Two settings are supported. Under `openContainers` a single `look around` reveals every
container's nested contents (surface `On/In the X is:` lines). Under the harder
`closedContainers` setting containers start CLOSED — the dump shows `the X door is closed.`
with contents hidden — and the agent must `open X` then `look in X` (two actions) to reveal
contents. This module parses all three events (closed-in-dump, opened, look-inside) so the
belief layer can track container search; the parsing is a strict superset, so it is inert
under openContainers (no `door is closed.` lines appear).

This module is pure text->structure; the belief update that consumes it lives in
`wm/sciworld_belief.py`. Design contract: wiki/syntheses/sciworld-wm-build-scope.md.
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# The fixed 10-room set (SciWorld building layout, from BuildingMaker.scala).
ROOMS = [
    "kitchen", "bathroom", "living room", "bedroom", "workshop",
    "greenhouse", "art studio", "foundry", "outside", "hallway",
]

# Room header of a `look around`: "This room is called the kitchen. In it, you see:"
_RE_ROOM_HEADER = re.compile(r"This room is called the (.+?)\. In it, you see", re.IGNORECASE)
# Agent movement result: "You teleport to the art studio."
_RE_TELEPORT = re.compile(r"You teleport to the (.+?)\.?$", re.IGNORECASE)
# Also catch the action itself ("teleport to art studio") when we only have the action.
_RE_TELEPORT_ACTION = re.compile(r"^\s*teleport to (?:the )?(.+?)\s*$", re.IGNORECASE)
# Inventory pickup: "You move the seed jar to the inventory."  (SciWorld uses `move`)
_RE_MOVE_TO_INV = re.compile(r"You move the (.+?) to the inventory\.?$", re.IGNORECASE)
# Put-down: "You move the lead to the blast furnace." — obj leaves inventory (dest captured
# separately so the caller can exclude "to the inventory", which is a pickup, not a put-down).
_RE_MOVE_TO_DEST = re.compile(r"You move the (.+?) to the (.+?)\.?$", re.IGNORECASE)
# Pour of a named substance out of inventory: "You pour the mercury into the glass cup."
# NOT "You pour the contents of the X into ..." — there the held container stays in hand.
_RE_POUR = re.compile(r"You pour the (?!contents of )(.+?) into the (.+?)\.?$", re.IGNORECASE)
# Focus: "You focus on the sodium chloride."
_RE_FOCUS = re.compile(r"You focus on the (.+?)\.?$", re.IGNORECASE)
# No-op marker: no state change.
_NOOP_MARKER = "No known action matches that input"

# Inline containment, possibly nested: "a bowl (containing a red apple, a banana)".
_RE_CONTAINING = re.compile(r"\(containing (.+?)\)", re.IGNORECASE)
# Surface-listed contents: "On the counter is: ...", "In the sink is: ...".
_RE_ON_IN_IS = re.compile(r"(?:On|In) the .+? is:\s*(.+?)\.?$", re.IGNORECASE)

# closedContainers grammar (verified, ScienceWorld 1.2.3):
#   in a room dump:  "a cupboard. The cupboard door is closed. "  (contents hidden)
#                    "a oven, which is turned off. The oven door is closed. "
#   `open X` result: "The cupboard is now open."                  (NO contents)
#   `look in X`:     "Inside the cupboard is: \n\t<items>"         (reveals contents)
# The door line may carry a trailing clause, so match up to " door is (closed|open)."
_RE_DOOR_STATE = re.compile(r"\bThe (.+?) door is (closed|open)\.", re.IGNORECASE)
_RE_OPENED = re.compile(r"\bThe (.+?) is now open\.", re.IGNORECASE)
_RE_LOOK_INSIDE = re.compile(r"\bInside the (.+?) is:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_room_name(observation: str) -> Optional[str]:
    """Return the room named by a `look around` header, or None."""
    m = _RE_ROOM_HEADER.search(observation)
    return _canon_room(m.group(1)) if m else None


def parse_agent_move(action: str, observation: str) -> Optional[str]:
    """Return the room the agent moved to this turn (from teleport), or None.

    Prefers the observation ("You teleport to the X.") but falls back to the action
    text ("teleport to X") when the observation is uninformative.
    """
    m = _RE_TELEPORT.search(observation or "")
    if m:
        return _canon_room(m.group(1))
    m = _RE_TELEPORT_ACTION.match(action or "")
    if m:
        return _canon_room(m.group(1))
    return None


def parse_pickup(observation: str) -> Optional[str]:
    """Object referent moved to inventory this turn, or None."""
    m = _RE_MOVE_TO_INV.search(observation or "")
    return _clean_referent(m.group(1)) if m else None


def parse_putdown(observation: str) -> Optional[str]:
    """Object referent that left the inventory this turn (moved/poured elsewhere), or None.

    Fires on `move OBJ to <dest>` where dest is not the inventory, and on a direct
    `pour OBJ into <dest>` (not `pour the contents of <container>`, which keeps the
    held container in hand). Counterpart to `parse_pickup`.
    """
    m = _RE_MOVE_TO_DEST.search(observation or "")
    if m and _clean_referent(m.group(2)) != "inventory":
        return _clean_referent(m.group(1))
    m = _RE_POUR.search(observation or "")
    if m:
        return _clean_referent(m.group(1))
    return None


def parse_focus(observation: str) -> Optional[str]:
    """Object referent focused on this turn, or None."""
    m = _RE_FOCUS.search(observation or "")
    return _clean_referent(m.group(1)) if m else None


def is_noop(observation: str) -> bool:
    return _NOOP_MARKER in (observation or "")


def is_room_dump(observation: str) -> bool:
    """True if this observation is a full `look around` (reveals a room's contents)."""
    return _RE_ROOM_HEADER.search(observation or "") is not None


def objects_in_dump(observation: str) -> List[str]:
    """All object/substance referents mentioned in a `look around` dump.

    Flattens nested containment (`(containing ...)`) and surface lists (`On the X is:`)
    into one referent list. Excludes structural furniture words is NOT attempted here —
    the caller matches referents against a target type by substring, so listing a
    superset (including containers like "bowl") is harmless.
    """
    if not is_room_dump(observation):
        return []
    refs: List[str] = []
    # Nested "(containing a, b, c)" groups.
    for m in _RE_CONTAINING.finditer(observation):
        refs.extend(_split_items(m.group(1)))
    # Surface "On/In the X is: a, b, c" lists (one per line).
    for line in observation.split("\n"):
        m = _RE_ON_IN_IS.search(line.strip())
        if m:
            refs.extend(_split_items(m.group(1)))
    # Also the top-level items directly named in the dump body (e.g. "a lighter",
    # "a substance called soap") — lines that start with an article after a tab.
    for line in observation.split("\n"):
        s = line.strip()
        item = _leading_item(s)
        if item:
            refs.append(item)
    # De-dup, keep order.
    seen, out = set(), []
    for r in refs:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def container_contents(observation: str) -> List[Tuple[str, List[str]]]:
    """(container_referent, [item referents]) pairs from surface `On/In the X is:` lines.

    Used for the `what is in <container>` query. Inline `(containing ...)` is not
    attributed to a named container here (kept simple for the first build).
    """
    out: List[Tuple[str, List[str]]] = []
    for line in observation.split("\n"):
        s = line.strip()
        mfull = re.search(r"(?:On|In) the (.+?) is:\s*(.+?)\.?$", s, re.IGNORECASE)
        if mfull:
            out.append((_clean_referent(mfull.group(1)), _split_items(mfull.group(2))))
    return out


# Inline "a <container> (containing <items>)" — the form container_contents skips. The
# name is a short referent, not a whole clause: no '.', ':', or comma between "a" and the
# "(containing ...)", so it won't swallow "table. On the table is: a glass cup".
_RE_INLINE_CONTAINER = re.compile(r"\ba ([^.:,()]+?) \(containing (.+?)\)", re.IGNORECASE)
# Pour OUT of a held container: "You pour the contents of the X into the Y."
_RE_POUR_CONTENTS = re.compile(
    r"You pour the contents of the (.+?) into the (.+?)\.?$", re.IGNORECASE)


def containers_by_state(observation: str) -> Tuple[List[str], List[str]]:
    """(closed_containers, open_containers) named by `door is closed/open` lines in a dump.

    closedContainers only: an openContainers dump has no `door is closed.` lines, so both
    lists come back empty and the caller's closed-container logic is inert. Referents are
    cleaned (article stripped, lowercased). Drawers never carry a door line, so they are
    naturally excluded (DEPTH-1: drawers are inert scenery)."""
    closed, opened = [], []
    for m in _RE_DOOR_STATE.finditer(observation or ""):
        ref = _clean_referent(m.group(1))
        (closed if m.group(2).lower() == "closed" else opened).append(ref)
    return closed, opened


def parse_opened(action: str, observation: str) -> Optional[str]:
    """Container referent just opened this turn (`The X is now open.`), or None.

    The open action reveals NO contents by itself; the caller marks the container open
    (removing it from the closed set) but does NOT eliminate it from any belief -- that
    waits for a subsequent `look in X`."""
    m = _RE_OPENED.search(observation or "")
    return _clean_referent(m.group(1)) if m else None


def parse_look_inside(observation: str) -> Optional[Tuple[str, List[str]]]:
    """(container, [item referents]) from a `look in X` result (`Inside the X is: ...`).

    This is the elimination trigger for the container belief: the contents are now known,
    so the target is either present (collapse) or confirmed absent (eliminate). Tolerates
    a bare `a drawer` line among the items without recursing (DEPTH-1). Returns None if the
    observation is not a look-inside result."""
    m = _RE_LOOK_INSIDE.search(observation or "")
    if not m:
        return None
    container = _clean_referent(m.group(1))
    # Items follow on tab-indented lines; also tolerate a comma/`and` list on one line.
    body = m.group(2)
    items: List[str] = []
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            continue
        item = _leading_item(s)
        if item:
            items.append(item)
        else:
            items.extend(_split_items(s))
        # Nested inline "(containing ...)" contents surface alongside the container name.
        for cm in _RE_CONTAINING.finditer(s):
            items.extend(_split_items(cm.group(1)))
    # De-dup, keep order.
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return container, out


def inline_contents(observation: str) -> Dict[str, Set[str]]:
    """container referent -> contents parsed from inline `(containing ...)` phrases.

    Handles `a wood cup (containing blue paint)`, which container_contents() ignores.
    Last occurrence wins if the same referent appears twice (ambiguous same-named cups
    collapse to one key -- best-effort; callers treat unknown as such)."""
    out: Dict[str, Set[str]] = {}
    for m in _RE_INLINE_CONTAINER.finditer(observation or ""):
        ref = _clean_referent(m.group(1))
        out[ref] = set(_split_items(m.group(2)))
    return out


def parse_pour_contents(observation: str) -> Optional[Tuple[str, str]]:
    """(source, dest) for `You pour the contents of the X into the Y.`, else None.

    Complements parse_putdown/_RE_POUR (which exclude the `contents of` form): here the
    held source container stays in hand but is emptied into dest."""
    m = _RE_POUR_CONTENTS.search(observation or "")
    if m:
        return _clean_referent(m.group(1)), _clean_referent(m.group(2))
    return None


def parse_pour_named(observation: str) -> Optional[Tuple[str, str]]:
    """(substance, dest) for `You pour the <subst> into the Y.` (not `contents of`), else None.

    The named substance is added to dest's contents when dest is a held container."""
    m = _RE_POUR.search(observation or "")
    if m:
        return _clean_referent(m.group(1)), _clean_referent(m.group(2))
    return None


# ------------------------------------------------------------------- helpers
def _canon_room(name: str) -> Optional[str]:
    """Normalize a parsed room name to one of ROOMS (or None if unrecognized)."""
    n = name.strip().lower()
    return n if n in ROOMS else None


def _split_items(blob: str) -> List[str]:
    """Split a 'a X, a Y, and a Z' / 'nothing' item list into referents."""
    b = blob.strip()
    if b.lower().startswith("nothing"):
        return []
    # Split on commas and the final 'and'; strip nested "(containing ...)" first so a
    # container's own name is captured, its contents handled by the _RE_CONTAINING pass.
    b = _RE_CONTAINING.sub("", b)
    parts = re.split(r",\s*|\s+and\s+", b)
    return [r for r in (_clean_referent(p) for p in parts) if r]


def _leading_item(line: str) -> Optional[str]:
    """Extract a top-level item referent from a dump body line.

    Handles 'a lighter', 'a substance called soap', 'a stopwatch, which is deactivated.'
    Skips doors, the agent, air, and pure structural lines.
    """
    if not re.match(r"^(a|an|the)\s", line, re.IGNORECASE):
        return None
    low = line.lower()
    if low.startswith(("a door", "the agent", "a substance called air")):
        return None
    # Cut at the first clause boundary (comma / period / '. On' / '. In' / '(containing').
    head = re.split(r"[,.]|\(", line, maxsplit=1)[0]
    return _clean_referent(head)


def _clean_referent(text: str) -> str:
    """Strip leading article and 'substance called', collapse whitespace, lowercase."""
    t = text.strip().lower()
    t = re.sub(r"^(a|an|the|some)\s+", "", t)
    t = re.sub(r"^substance called\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    return t
