"""Object -> room prior for the SciWorld location-WM router (first build).

The SciWorld analogue of `wm.affinity_prior`. ALFWorld inverts an in-package
constraint table (`VAL_RECEPTACLE_OBJECTS`); SciWorld has no such Python table —
placement lives in the game's Scala source. We therefore ship a transcribed table
(`sciworld_room_prior.json`), keyed on the surface object-type token as it appears
in `look around` observations, mapping each type to the UNION of rooms it can occupy
across all task generators + the base scene (v1.1.3).

Scope of the first build (SK 2026-07-09): LOCATION only, ROOM level only (the
container sub-level is dropped), and the prior is marginalized over task family — it
is NOT keyed on task type. `P(room | object_type)` is UNIFORM over the type's room
union. With most types occupying a single room this is (near) one-hot, so the WM acts
as a grounded location router; the multi-room types (salt, forks, seed jar, paints, …)
are where it carries real uncertainty for the belief update to resolve.

See wiki/syntheses/sciworld-wm-design.md and memory sciworld-placement-model.md.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List


_PRIOR_PATH = os.path.join(os.path.dirname(__file__), "sciworld_room_prior.json")


@lru_cache(maxsize=1)
def _object_to_rooms() -> Dict[str, tuple]:
    """object_type (lowercased) -> tuple of candidate rooms.

    Loads the transcribed table once and caches it. `_`-prefixed keys are
    provenance/comment entries and are skipped.
    """
    with open(_PRIOR_PATH) as f:
        raw = json.load(f)
    return {
        k.lower(): tuple(v)
        for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, list)
    }


def candidate_rooms(object_type: str) -> tuple:
    """Rooms an object of this type may occupy (union over task families).

    Exact match first; if that misses, drop leading qualifier words one at a time
    ("red light bulb" -> "light bulb" -> "bulb") so color/adjective-qualified targets
    still hit the canonical prior key. Returns an empty tuple for unknown types
    (caller treats that as "no prior knowledge" -> uniform over all present rooms).
    """
    table = _object_to_rooms()
    toks = object_type.lower().strip().split()
    for i in range(len(toks)):
        hit = table.get(" ".join(toks[i:]))
        if hit:
            return hit
    return ()


def uniform_room_prior(
    object_type: str,
    present_rooms: List[str],
) -> Dict[str, float]:
    """Room-level prior: uniform over the object's candidate rooms present in the scene.

    Mirrors `wm.affinity_prior.uniform_prior` but at the room level. Collect every
    present room in this type's candidate union, then spread probability uniformly
    over that set. Falls back to uniform over ALL present rooms if the type is unknown
    or none of its candidate rooms are present (so the belief is never empty).

    Args:
        object_type: e.g. "salt", "metal fork" (case-insensitive).
        present_rooms: room names in the scene (usually the fixed 10).

    Returns:
        {room: probability}, summing to 1.0 over `present_rooms`.
    """
    if not present_rooms:
        return {}

    allowed = set(candidate_rooms(object_type))
    candidates = [r for r in present_rooms if r in allowed]

    # Fallback: unknown type, or none of its candidate rooms are in this scene.
    if not candidates:
        candidates = list(present_rooms)

    p = 1.0 / len(candidates)
    return {r: (p if r in candidates else 0.0) for r in present_rooms}
